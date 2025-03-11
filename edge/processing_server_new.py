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
from dataclasses import dataclass

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

# First, let's add the new classes to the codebase

from dataclasses import dataclass
from collections import deque
import time
import threading

@dataclass
class StatsEntry:
    # The timestamp of the inference
    timestamp: float = 0.0
    # The queue length at the time of the inference
    queue_length: int = 0
    # The model index of the inference
    cur_model_index: str = "s"  # Changed to string to match the existing code
    # The accuracy of the model
    cur_model_accuracy: float = 0.0
    # The processing latency of the inference
    processing_latency: float = 0.0  # Changed to float
    # The number of detected targets
    target_nums: int = 0
    # The average confidence of the detected targets
    avg_confidence: float = 0.0
    # The standard deviation of the confidence of the detected targets
    std_confidence: float = 0.0
    # The average size of the detected targets
    avg_size: float = 0.0
    # The standard deviation of the size of the detected targets
    std_size: float = 0.0
    # The brightness of the image
    brightness: float = 0.0
    # The contrast of the image
    contrast: float = 0.0
    # The entropy of the image (from original code)
    entropy: float = 0.0


    def __str__(self) -> str:
        return (
            f"Stats Entry:\n"
            f"  Timestamp: {self.timestamp:.2f}\n"
            f"  Queue Length: {self.queue_length}\n"
            f"  Model Index: {self.cur_model_index}\n"
            f"  Model Accuracy: {self.cur_model_accuracy:.2f}\n"
            f"  Processing Latency: {self.processing_latency:.4f}s\n"
            f"  Targets: {self.target_nums}\n"
            f"  Confidence: {self.avg_confidence:.2f}±{self.std_confidence:.2f}\n"
            f"  Size: {self.avg_size:.2f}±{self.std_size:.2f}\n"
            f"  Image Stats: brightness={self.brightness:.1f}, contrast={self.contrast:.1f}, entropy={self.entropy:.1f}"
        )
    
class StatsManager:
    def __init__(self, time_window: float = 30.0, csv_path: str = None):
        # initialize a deque to store the stats
        self.stats = deque()
        self.time_window = time_window
        self.csv_path = csv_path
        self.lock = threading.Lock()
        
        # Initialize the CSV file if a path is provided
        if self.csv_path:
            with open(self.csv_path, 'w') as f:
                f.write('timestamp,queue_length,cur_model_index,cur_model_accuracy,processing_latency,target_nums,avg_confidence,std_confidence,avg_size,std_size,brightness,contrast,entropy\n')

    def update_stats(self, entry: StatsEntry):
        '''
        Update the stats.
        Called when a new inference is done.
        Remove the outdated stats and append the new stats.
        '''
        with self.lock:
            # append the new stats
            self.stats.append(entry)
            # remove the outdated stats
            current_time = entry.timestamp
            while self.stats and current_time - self.stats[0].timestamp > self.time_window:
                self.stats.popleft()

            # Write to CSV if path is provided
            if self.csv_path:
                with open(self.csv_path, 'a') as f:
                    f.write(f'{entry.timestamp},{entry.queue_length},{entry.cur_model_index},{entry.cur_model_accuracy},{entry.processing_latency},{entry.target_nums},{entry.avg_confidence},{entry.std_confidence},{entry.avg_size},{entry.std_size},{entry.brightness},{entry.contrast},{entry.entropy}\n')
    
    def get_latest_stats(self, nums: int = 1):
        '''
        Get the latest statistics
        '''
        with self.lock:
            if not self.stats:
                return None
            # 如果请求的数量大于当前的数量，在列表前面补默认值
            elif len(self.stats) < nums:
                return [StatsEntry()] * (nums - len(self.stats)) + list(self.stats)
            else:
                return list(self.stats)[-nums:]  # 返回最新的nums个元素
            
    def get_interval_stats(self, nums: int = 1, interval: float = 1.0):
        '''
        Get the statistics at intervals
        Returns a list of stats, one for each interval, starting from the oldest to newest
        If no stats are available for an interval, None is used as placeholder
        '''
        with self.lock:
            if not self.stats:
                return [None] * nums
                
            result = [None] * nums
            current_time = self.stats[-1].timestamp
            
            # 使用二分查找优化查找过程
            stats_list = list(self.stats)  # 转换为列表以支持索引访问
            
            for i in range(nums):
                target_time = current_time - i * interval
                
                # 二分查找找到最接近且不大于目标时间的统计数据
                left, right = 0, len(stats_list) - 1
                closest_index = -1
                
                while left <= right:
                    mid = (left + right) // 2
                    if stats_list[mid].timestamp <= target_time:
                        closest_index = mid
                        left = mid + 1
                    else:
                        right = mid - 1
                
                if closest_index != -1:
                    result[nums - 1 - i] = stats_list[closest_index]  # 确保按时间从老到新排序
                    
            return result
            

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
        
        # Create a StatsManager instance
        stats_path = Path(config.get_stats_path()) if hasattr(config, 'get_stats_path') else None
        self.stats_manager = StatsManager(time_window=60.0, csv_path=stats_path)  # 60 seconds window
        
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

                # 计算平均置信度和标准差
                if len(results.pred[0]) > 0:  # 如果有检测到的物体
                    confidences = results.pred[0][:, 4].cpu().numpy()  # 提取置信度值
                    avg_conf = float(confidences.mean())  # 计算平均值
                    std_conf = float(confidences.std()) if len(confidences) > 1 else 0.0  # 计算标准差
                else:
                    avg_conf = 0.0  # 如果没有检测到物体，置信度为0
                    std_conf = 0.0  # 标准差也为0

                # 计算平均大小和标准差
                if len(results.pred[0]) > 0:  # 如果有检测到的物体
                    image_height, image_width = frame.shape[:2]
                    image_area = image_height * image_width
                    box_areas = (results.pred[0][:, 2] - results.pred[0][:, 0]) * (results.pred[0][:, 3] - results.pred[0][:, 1]) 
                    relative_areas = box_areas / image_area
                    sizes = np.sqrt(relative_areas)
                    avg_size = float(sizes.mean())  # 计算平均大小
                    std_size = float(sizes.std()) if len(sizes) > 1 else 0.0  # 计算标准差
                else:
                    avg_size = 0.0  # 如果没有检测到物体，大小为0
                    std_size = 0.0  # 标准差也为0

                # 计算图像熵
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([img_gray], [0], None, [256], [0,256])
                hist = hist / (frame.shape[0] * frame.shape[1])
                entropy = -np.sum(hist * np.log2(hist + 1e-7))
                entropy = float(entropy)

                rendered_frame = results.render()[0]
                
                # Add processing completion timestamp
                timestamps['processed'] = time.time()

                # Create a StatsEntry and update the StatsManager
                stats_entry = StatsEntry(
                    timestamp=timestamps['processed'],
                    queue_length=len(self.frame_queue),
                    cur_model_index=self.model_manager.current_model_name,
                    cur_model_accuracy=self.model_manager.current_map,
                    processing_latency=timestamps['processed'] - timestamps['start_processing'],
                    target_nums=total_targets,
                    avg_confidence=avg_conf,
                    std_confidence=std_conf,
                    avg_size=avg_size,
                    std_size=std_size,
                    brightness=brightness,
                    contrast=contrast,
                    entropy=entropy
                )
                
                # Update the stats manager
                self.model_manager.stats_manager.update_stats(stats_entry)
                
                # Print detailed stats for debugging
                print(f"Accuracy: {self.model_manager.current_map:.1f} mAP, "
                      f"Latency: {timestamps['processed'] - timestamps['received']:.3f}s, "
                      f"Processing Latency: {timestamps['processed'] - timestamps['start_processing']:.3f}s, "
                      f"Queue Length: {len(self.frame_queue)}, "
                      f"Model: yolov5{self.model_manager.current_model_name}, "
                      f"Avg Confidence: {avg_conf:.2f}, "
                      f"Avg Size: {avg_size:.2f}, "
                      f"Brightness: {brightness:.2f}, "
                      f"Contrast: {contrast:.2f}, "
                      f"Entropy: {entropy:.2f}, "
                      f"Total Targets: {total_targets}"
                     )
                
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

        # Get target fps
        target_fps = data.get('target_fps', 0)
        # Store target FPS in stats if needed for reporting
        
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
            emit('model_switched', {'model_name': f'{model_name}'})
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
    nums = request.args.get('nums', default=1, type=int)
    
    # Ensure nums is at least 1
    nums = max(1, nums)

    # Get the latest stats
    stats = processor.model_manager.stats_manager.get_latest_stats(nums)
    if stats is None:
        return jsonify([])
    
    # Convert StatsEntry objects to dictionaries
    stats_dicts = [s.__dict__ for s in stats]
    return jsonify(stats_dicts)


@app.route('/get_interval_stats', methods=['GET'])
def get_interval_stats():
    """获取当前系统状态统计信息，按照时间窗口的方式，从最后一个状态开始，每隔interval秒取一个状态"""
    nums = request.args.get('nums', default=1, type=int)
    interval = request.args.get('interval', default=1.0, type=float)

    nums = max(1, nums)
    interval = max(0.1, interval)

    # Get the interval stats
    stats = processor.model_manager.stats_manager.get_interval_stats(nums, interval)
    if stats is None:
        return jsonify([])
    
    # Convert StatsEntry objects to dictionaries
    stats_dicts = [s.__dict__ for s in stats]
    return jsonify(stats_dicts)

if __name__ == '__main__':
    try:
        server_config = config.get_edge_processing_config()
        socketio.run(app, 
                    host=server_config['host'], 
                    port=server_config['port'], 
                    debug=False)
    finally:
        processor.stop()