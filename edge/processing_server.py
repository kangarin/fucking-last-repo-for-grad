# processing_server.py
import torch
import cv2
import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit
import threading
import time
from pathlib import Path
import sys
import base64
import requests
from collections import deque

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

# Add YOLOv5 to Python path
sys.path.append(str(project_root / 'edge' / 'yolov5'))

from models.common import AutoShape
from models.experimental import attempt_load

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', ping_timeout=600)

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.next_model = None
        self.model_lock = threading.Lock()
        self.models_config = config.get_models_config()
        self.models_dir = Path(self.models_config['weights_dir'])
        # 添加每个模型对应的mAP值
        self.model_maps = {
            'n': 25.7,  # YOLOv5n
            's': 37.4,  # YOLOv5s
            'm': 45.2,  # YOLOv5m
            'l': 49.0,  # YOLOv5l
            'x': 50.7   # YOLOv5x
        }
        self.current_map = None  # 当前模型的mAP
        self.stats = {
            'accuracy': None,
            'latency': None,
            'queue_length': None
        }
        self.stats_lock = threading.Lock()

    def load_model(self, model_name):
        weight_file = self.models_dir / f'yolov5{model_name}.pt'
        if not weight_file.exists():
            raise ValueError(f"Model weights not found: {weight_file}")
            
        model = attempt_load(weight_file)
        model = AutoShape(model)
        if torch.cuda.is_available():
            model = model.cuda()
            
        return model
        
    def switch_model(self, new_model_name):
        if new_model_name not in self.models_config['allowed_sizes']:
            raise ValueError(f"Invalid model size: {new_model_name}")
            
        print(f"Preparing to switch to model: yolov5{new_model_name}")
        try:
            new_model = self.load_model(new_model_name)
            with self.model_lock:
                self.next_model = new_model
                self.current_map = self.model_maps[new_model_name]
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
            
    def get_active_model(self):
        with self.model_lock:
            if self.next_model is not None:
                self.current_model = self.next_model
                self.next_model = None
                print("Model switched successfully")
            return self.current_model

class DetectionProcessor:
    def __init__(self):
        self.model_manager = ModelManager()
        self.active_streams = set()
        display_config = config.get_cloud_display_config()
        self.display_server_url = display_config['url']
        
        # Initialize frame queue with maxlen=10
        self.frame_queue = deque(maxlen=10)
        self.queue_lock = threading.Lock()
        
        # Initialize processing thread
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.running = True
        
        # Initialize default model
        default_model = config.get_models_config()['default']
        print(f"Loading initial model (yolov5{default_model})...")
        self.model_manager.switch_model(default_model)
        
        # Start processing thread
        self.processing_thread.start()
    
    def add_frame_to_queue(self, stream_id, frame_id, timestamps, frame):
        """Add a frame to the processing queue"""
        with self.queue_lock:
            # If queue is at max length, oldest frame will automatically be dropped
            self.frame_queue.append({
                'stream_id': stream_id,
                'frame_id': frame_id,
                'timestamps': timestamps,
                'frame': frame
            })
    
    def _process_queue(self):
        """Background thread for processing frames from queue"""
        while self.running:
            frame_data = None
            
            # Get frame from queue with lock
            with self.queue_lock:
                if len(self.frame_queue) > 0:
                    frame_data = self.frame_queue.popleft()
            
            # Process frame if we got one
            if frame_data is not None:
                self._process_single_frame(**frame_data)
            else:
                # No frames to process, sleep briefly
                time.sleep(0.01)
    
    def _process_single_frame(self, stream_id, frame_id, timestamps, frame):
        """Process a single frame and send results to display server"""
        try:
            if stream_id not in self.active_streams:
                return
                
            # Get current model and run inference
            model = self.model_manager.get_active_model()
            if model is not None:
                # Perform detection
                results = model(frame)
                rendered_frame = results.render()[0]
                
                # Add processing completion timestamp
                timestamps['processed'] = time.time()

                # 更新统计信息
                with self.model_manager.stats_lock:
                    self.update_statistics(timestamps)
                
                # Encode processed frame
                _, buffer = cv2.imencode('.jpg', rendered_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send results to display server
                response = requests.post(f"{self.display_server_url}/update_detection", json={
                    'stream_id': stream_id,
                    'frame_id': frame_id,
                    'timestamps': timestamps,
                    'image': img_base64
                })
                
                if response.status_code != 200:
                    print(f"Failed to send detection result: {response.text}")
                    
        except Exception as e:
            print(f"Error processing frame: {e}")

    def update_statistics(self, timestamps):
        self.model_manager.stats['accuracy'] = self.model_manager.current_map
        self.model_manager.stats['latency'] = timestamps['processed'] - timestamps['received']
        self.model_manager.stats['queue_length'] = len(self.frame_queue)
        print(f"Accuracy: {self.model_manager.stats['accuracy']:.1f} mAP, "
        f"Latency: {self.model_manager.stats['latency']:.3f}s, "
        f"Queue Length: {self.model_manager.stats['queue_length']}")
    
    def stop(self):
        """Stop the processing thread"""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join()

# Create processor instance
processor = DetectionProcessor()

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    processor.active_streams.discard(request.sid)

@socketio.on('start_stream')
def handle_start_stream(data):
    stream_id = data.get('stream_id', request.sid)
    processor.active_streams.add(stream_id)
    emit('stream_started', {'stream_id': stream_id})
    print(f"Stream started: {stream_id}")

@socketio.on('stop_stream')
def handle_stop_stream(data):
    stream_id = data.get('stream_id', request.sid)
    processor.active_streams.discard(stream_id)
    emit('stream_stopped', {'stream_id': stream_id})
    print(f"Stream stopped: {stream_id}")

@socketio.on('process_frame')
def handle_frame(data):
    try:
        stream_id = data.get('stream_id', request.sid)
        if stream_id not in processor.active_streams:
            emit('error', {'message': 'Stream not active'})
            return
            
        # Add received timestamp
        timestamps = data.get('timestamps', {})
        timestamps['received'] = time.time()
        
        # Decode image data
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            emit('error', {'message': 'Invalid image data'})
            return
        
        # Use new queue method instead of direct processing
        processor.add_frame_to_queue(
            stream_id=stream_id,
            frame_id=data.get('frame_id'),
            timestamps=timestamps,
            frame=frame
        )
        
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('switch_model')
def handle_switch_model(data):
    try:
        model_name = data.get('model_name')
        allowed_sizes = config.get_models_config()['allowed_sizes']
        if model_name not in allowed_sizes:
            emit('error', {'message': f'Invalid model name. Must be one of: {allowed_sizes}'})
            return
            
        success = processor.model_manager.switch_model(model_name)
        if success:
            emit('model_switched', {'model_name': f'yolov5{model_name}'})
        else:
            emit('error', {'message': 'Failed to switch model'})
            
    except Exception as e:
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    try:
        server_config = config.get_edge_processing_config()
        socketio.run(app, 
                    host=server_config['host'], 
                    port=server_config['port'], 
                    debug=False)
    finally:
        processor.stop()