from socketio import Client
import cv2
import base64
import time
import numpy as np
from pathlib import Path
import sys
import os
import random
import math

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

class DynamicStreamClient:
    def __init__(self, stream_id=None, fps_bias_k=0.5):
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        
        stream_config = config.get_dynamic_stream_config()
        self.stream_id = stream_id or stream_config['default_id']
        self.fps_config = stream_config['fps_control']
        self.fps_bias_k = fps_bias_k  # 控制FPS采样的偏向程度
        
        # 初始化周期状态
        self.current_fps_range = self.generate_biased_fps_range()
        self.next_fps_range = None
        self.is_transitioning = False
        self.transition_progress = 0
        self.transition_duration = 0
        
        self.setup_socket_events()
        
    def setup_socket_events(self):
        @self.sio.event
        def connect():
            print(f"Connected to processing server: {self.processing_server_url}")
            
        @self.sio.event
        def connect_error(data):
            print(f"Connection error: {data}")
            
        @self.sio.event
        def disconnect():
            print("Disconnected from processing server")
            
        @self.sio.on('stream_started')
        def on_stream_started(data):
            print(f"Stream started: {data['stream_id']}")
            
        @self.sio.on('error')
        def on_error(data):
            print(f"Error: {data['message']}")

    def get_video_files(self, folder_path):
        """Get all video files from the specified folder"""
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv')
        video_files = []
        
        for file in os.listdir(folder_path):
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(folder_path, file))
                
        return sorted(video_files)

    def generate_biased_fps_range(self):
        """使用加权的方式生成FPS范围"""
        min_fps = self.fps_config['min_fps']
        max_fps = self.fps_config['max_fps']
        k = self.fps_bias_k
        
        # 将范围分成N个区间
        N = 10
        intervals = np.linspace(min_fps, max_fps, N)
        
        # 生成递减的权重
        weights = np.array([(N-i)**k for i in range(N-1)])
        weights = weights / np.sum(weights)  # 归一化
        
        # 从区间中选择两个不同的值
        selected_intervals = np.random.choice(range(N-1), size=2, replace=False, p=weights)
        
        # 在选中的区间内均匀采样
        fps1 = random.uniform(intervals[selected_intervals[0]], intervals[selected_intervals[0]+1])
        fps2 = random.uniform(intervals[selected_intervals[1]], intervals[selected_intervals[1]+1])
        
        return {
            'min_fps': min(fps1, fps2),
            'max_fps': max(fps1, fps2)
        }

    def calculate_target_fps(self, elapsed_time, cycle_duration):
        """计算目标FPS，包含周期转换的平滑处理"""
        if self.is_transitioning:
            # 在转换期间进行线性插值
            old_fps = self._calculate_sine_wave_fps(elapsed_time, cycle_duration, self.current_fps_range)
            new_fps = self._calculate_sine_wave_fps(elapsed_time, cycle_duration, self.next_fps_range)
            return old_fps * (1 - self.transition_progress) + new_fps * self.transition_progress
        else:
            return self._calculate_sine_wave_fps(elapsed_time, cycle_duration, self.current_fps_range)
    
    def _calculate_sine_wave_fps(self, elapsed_time, cycle_duration, fps_range):
        """计算基于正弦波的FPS"""
        phase = (elapsed_time % cycle_duration) * (2 * math.pi / cycle_duration)
        sine_val = math.sin(phase)
        fps_range_size = fps_range['max_fps'] - fps_range['min_fps']
        return (sine_val + 1) / 2 * fps_range_size + fps_range['min_fps']
    
    def start_streaming(self, video_source=None):
        """Start streaming videos from the specified source folder"""
        stream_config = config.get_dynamic_stream_config()
        if video_source is None:
            video_source = stream_config['source_path']
            
        if os.path.isdir(video_source):
            video_files = self.get_video_files(video_source)
            if not video_files:
                raise Exception(f"No video files found in directory: {video_source}")
        else:
            video_files = [video_source]
            
        try:
            print(f"Connecting to {self.processing_server_url}")
            self.sio.connect(self.processing_server_url, wait_timeout=10)
            self.sio.emit('start_stream', {'stream_id': self.stream_id})
            
            frame_count = 0
            cycle_start_time = time.time()
            current_cycle_duration = random.uniform(
                self.fps_config['min_duration'],
                self.fps_config['max_duration']
            )
            next_cycle_duration = None
            
            while True:  # Outer loop for continuous folder processing
                for video_file in video_files:
                    print(f"Processing video: {video_file}")
                    cap = cv2.VideoCapture(video_file)
                    
                    if not cap.isOpened():
                        print(f"Failed to open video file: {video_file}")
                        continue
                    
                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    start_time = time.time()
                    
                    try:
                        while True:
                            current_time = time.time()
                            cycle_elapsed = current_time - cycle_start_time
                            
                            # 检查是否需要准备下一个周期
                            remaining_time = current_cycle_duration - cycle_elapsed
                            if remaining_time <= current_cycle_duration * 0.1 and not self.is_transitioning:
                                # 开始转换准备
                                self.is_transitioning = True
                                self.transition_progress = 0
                                self.next_fps_range = self.generate_biased_fps_range()
                                next_cycle_duration = random.uniform(
                                    self.fps_config['min_duration'],
                                    self.fps_config['max_duration']
                                )
                                self.transition_duration = (current_cycle_duration + next_cycle_duration) * 0.1
                                print(f"Preparing transition to new FPS range: {self.next_fps_range}")
                            
                            # 处理转换进度
                            if self.is_transitioning:
                                self.transition_progress = min(1.0, (current_cycle_duration - remaining_time) / self.transition_duration)
                                if self.transition_progress >= 1.0:
                                    # 完成转换
                                    self.current_fps_range = self.next_fps_range
                                    self.next_fps_range = None
                                    self.is_transitioning = False
                                    cycle_start_time = current_time
                                    current_cycle_duration = next_cycle_duration
                                    next_cycle_duration = None
                                    print(f"New cycle started with duration: {current_cycle_duration:.2f}s")
                                    print(f"New FPS range: {self.current_fps_range['min_fps']:.2f} - {self.current_fps_range['max_fps']:.2f}")
                                    cycle_elapsed = 0
                            
                            # 计算目标FPS
                            target_fps = self.calculate_target_fps(cycle_elapsed, current_cycle_duration)
                            
                            # 计算帧位置和跳帧
                            elapsed_time = current_time - start_time
                            target_frame = int(elapsed_time * video_fps)
                            skip_frames = max(1, int(video_fps / target_fps))
                            next_frame = (target_frame // skip_frames) * skip_frames
                            
                            if next_frame >= total_frames:
                                print(f"Finished processing: {video_file}")
                                break
                                
                            # 设置位置并读取帧
                            cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # 编码并发送帧
                            _, buffer = cv2.imencode('.jpg', frame)
                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            self.sio.emit('process_frame', {
                                'stream_id': self.stream_id,
                                'frame_id': frame_count,
                                'timestamps': {
                                    'generated': time.time(),
                                },
                                'target_fps': target_fps,
                                'image': img_base64
                            })
                            
                            frame_count += 1
                            
                            # 控制帧率
                            processing_time = time.time() - current_time
                            sleep_time = max(0, (1.0 / target_fps) - processing_time)
                            time.sleep(sleep_time)
                            
                    finally:
                        cap.release()
                        
                print("Finished processing all videos, restarting...")
                cycle_start_time = time.time()
                
        except Exception as e:
            print(f"Streaming error: {e}")
            raise e
        finally:
            self.stop_streaming()
    
    def stop_streaming(self):
        """Stop the stream and disconnect"""
        try:
            if self.sio.connected:
                self.sio.emit('stop_stream', {'stream_id': self.stream_id})
                self.sio.disconnect()
        except Exception as e:
            print(f"Error stopping stream: {e}")

if __name__ == '__main__':
    # Create client and start streaming using config defaults
    client = DynamicStreamClient(fps_bias_k=0.5)  # 可以通过参数调整FPS分布的偏向程度
    client.start_streaming()