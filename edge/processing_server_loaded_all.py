# 一次性加载所有模型到内存，切换模型时直接从内存中获取模型
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify
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
        self.models = {}  # 存储所有加载的模型
        self.current_model_name = None
        self.model_lock = threading.Lock()
        self.models_config = config.get_models_config()
        self.models_dir = Path(self.models_config['weights_dir'])
        
        # mAP values for each model
        self.model_maps = {
            'n': 25.7,  # YOLOv5n
            's': 37.4,  # YOLOv5s
            'm': 45.2,  # YOLOv5m
            'l': 49.0,  # YOLOv5l
            'x': 50.7   # YOLOv5x
        }
        self.current_map = None
        self.stats = {
            'accuracy': 0,
            'latency': 0,
            'queue_length': 0,
            'model_name': 's',
            'avg_confidence': 0,
            'avg_size': 0,
            'total_targets': 0,
            'brightness': 0,
            'contrast': 0,
            'entropy': 0,
        }
        self.stats_lock = threading.Lock()
        
        # 初始化时加载所有模型
        self._load_all_models()
        
        # 设置默认模型
        default_model = self.models_config['default']
        self.current_model_name = default_model
        self.current_map = self.model_maps.get(default_model, 0)

    def _load_all_models(self):
        """一次性加载所有模型到内存"""
        print("Loading all YOLOv5 models...")
        for model_size in self.models_config['allowed_sizes']:
            try:
                weight_file = self.models_dir / f'yolov5{model_size}.pt'
                if not weight_file.exists():
                    print(f"Warning: Model weights not found: {weight_file}")
                    continue
                    
                print(f"Loading YOLOv5{model_size}...")
                model = attempt_load(weight_file)
                model = AutoShape(model)
                if torch.cuda.is_available():
                    model = model.cuda()
                
                self.models[model_size] = model
                print(f"Successfully loaded YOLOv5{model_size}")
                
            except Exception as e:
                print(f"Failed to load YOLOv5{model_size}: {e}")
                
        print("Finished loading all models")

    def switch_model(self, new_model_name):
        """切换到指定的模型"""
        if new_model_name not in self.models_config['allowed_sizes']:
            raise ValueError(f"Invalid model size: {new_model_name}")
            
        if new_model_name not in self.models:
            raise ValueError(f"Model YOLOv5{new_model_name} not loaded")
            
        with self.model_lock:
            self.current_model_name = new_model_name
            self.current_map = self.model_maps.get(new_model_name, 0)
            print(f"Switched to model: YOLOv5{new_model_name}")
        return True

    def get_active_model(self):
        """获取当前活动的模型"""
        with self.model_lock:
            return self.models.get(self.current_model_name)

class DetectionProcessor:
    def __init__(self):
        self.model_manager = ModelManager()
        self.active_streams = set()
        display_config = config.get_cloud_display_config()
        self.display_server_url = display_config['url']
        
        # Initialize frame queue with maxlen
        max_queue_length = config.get_queue_max_length()
        self.frame_queue = deque(maxlen=max_queue_length)
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
            timestamps['start_processing'] = time.time()
            # Get current model and run inference
            model = self.model_manager.get_active_model()
            if model is not None:

                # 计算亮度和对比度
                # 转换为灰度图像
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = gray.mean()  # 平均亮度
                contrast = gray.std()     # 对比度(使用标准差)

                # Perform detection
                results = model(frame)

                total_targets = len(results.pred[0])  # 检测到的目标数量

                # 计算平均置信度
                if len(results.pred[0]) > 0:  # 如果有检测到的物体
                    confidences = results.pred[0][:, 4].cpu().numpy()  # 提取置信度值
                    avg_conf = float(confidences.mean())  # 计算平均值
                else:
                    avg_conf = 0.0  # 如果没有检测到物体，置信度为0

                # 计算平均大小
                if len(results.pred[0]) > 0: # 如果有检测到的物体
                    boxes = results.pred[0][:, :4].cpu().numpy() # 提取边界框坐标
                    sizes = (boxes[:, 2:] - boxes[:, :2]).mean(axis=0) 
                    avg_size = sizes.mean() # 计算平均大小
                else:
                    avg_size = 0.0 # 如果没有检测到物体，大小为0

                # 计算图像熵
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([img_gray], [0], None, [256], [0,256])
                hist = hist / (frame.shape[0] * frame.shape[1])
                entropy = -np.sum(hist * np.log2(hist + 1e-7))

                rendered_frame = results.render()[0]
                
                # Add processing completion timestamp
                timestamps['processed'] = time.time()

                # 更新统计信息
                with self.model_manager.stats_lock:
                    self.update_statistics(timestamps, avg_conf, avg_size, brightness, contrast, entropy, total_targets)
                
                # Encode processed frame
                _, buffer = cv2.imencode('.jpg', rendered_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send results to display server
                response = requests.post(f"{self.display_server_url}/update_detection", json={
                    'stream_id': stream_id,
                    'frame_id': frame_id,
                    'timestamps': timestamps,
                    'image': img_base64,
                    'model': f'yolov5{self.model_manager.current_model_name}'
                })
                
                if response.status_code != 200:
                    print(f"Failed to send detection result: {response.text}")
                    
        except Exception as e:
            print(f"Error processing frame: {e}")

    def update_statistics(self, timestamps, avg_conf, avg_size, brightness, contrast, entropy, total_targets):
        self.model_manager.stats['accuracy'] = self.model_manager.current_map
        self.model_manager.stats['latency'] = timestamps['processed'] - timestamps['received']
        self.model_manager.stats['queue_length'] = len(self.frame_queue)
        self.model_manager.stats['model_name'] = self.model_manager.current_model_name
        self.model_manager.stats['avg_confidence'] = float(avg_conf)
        self.model_manager.stats['avg_size'] = float(avg_size)
        self.model_manager.stats['brightness'] = float(brightness)
        self.model_manager.stats['contrast'] = float(contrast)
        self.model_manager.stats['entropy'] = float(entropy)
        self.model_manager.stats['total_targets'] = float(total_targets)

        print(f"Accuracy: {self.model_manager.stats['accuracy']:.1f} mAP, "
        f"Latency: {self.model_manager.stats['latency']:.3f}s, "
        f"Queue Length: {self.model_manager.stats['queue_length']}, "
        f"Model: yolov5{self.model_manager.stats['model_name']}, "
        f"Avg Confidence: {self.model_manager.stats['avg_confidence']:.2f}, "
        f"Avg Size: {self.model_manager.stats['avg_size']:.2f}, "
        f"Brightness: {self.model_manager.stats['brightness']:.2f}, "
        f"Contrast: {self.model_manager.stats['contrast']:.2f}, "
        f"Entropy: {self.model_manager.stats['entropy']:.2f}, "
        f"Total Targets: {self.model_manager.stats['total_targets']:.2f}"
        )
    
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

# 添加获取状态的路由
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'})

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """获取当前系统状态统计信息"""
    with processor.model_manager.stats_lock:
        stats = {
            'stats': {
                'accuracy': processor.model_manager.stats['accuracy'],
                'latency': processor.model_manager.stats['latency'],
                'queue_length': processor.model_manager.stats['queue_length'],
                'model_name': processor.model_manager.stats['model_name'],
                'avg_confidence': processor.model_manager.stats['avg_confidence'],
                'avg_size': processor.model_manager.stats['avg_size'],
                'brightness': processor.model_manager.stats['brightness'],
                'contrast': processor.model_manager.stats['contrast'],
                'entropy': processor.model_manager.stats['entropy'],
                'total_targets': processor.model_manager.stats['total_targets']
            },
            'timestamp': time.time()
        }
        print(stats)
        return jsonify(stats)
    
if __name__ == '__main__':
    try:
        server_config = config.get_edge_processing_config()
        socketio.run(app, 
                    host=server_config['host'], 
                    port=server_config['port'], 
                    debug=False)
    finally:
        processor.stop()