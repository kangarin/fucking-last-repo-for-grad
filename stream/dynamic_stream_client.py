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
    def __init__(self, stream_id=None):
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        
        stream_config = config.get_dynamic_stream_config()
        self.stream_id = stream_id or stream_config['default_id']
        self.fps_config = stream_config['fps_control']
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

    def calculate_target_fps(self, elapsed_time, cycle_duration):
        """Calculate target FPS based on sine wave within current cycle"""
        phase = (elapsed_time % cycle_duration) * (2 * math.pi / cycle_duration)
        sine_val = math.sin(phase)
        
        # Transform sine wave output [-1, 1] to [min_fps, max_fps]
        fps_range = self.fps_config['max_fps'] - self.fps_config['min_fps']
        target_fps = (sine_val + 1) / 2 * fps_range + self.fps_config['min_fps']
        
        return target_fps
    
    def start_streaming(self, video_source=None):
        """Start streaming videos from the specified source folder"""
        stream_config = config.get_dynamic_stream_config()
        if video_source is None:
            video_source = stream_config['source_path']
            
        # Check if video_source is a directory
        if os.path.isdir(video_source):
            video_files = self.get_video_files(video_source)
            if not video_files:
                raise Exception(f"No video files found in directory: {video_source}")
        else:
            video_files = [video_source]
            
        try:
            # Connect to processing server
            print(f"Connecting to {self.processing_server_url}")
            self.sio.connect(self.processing_server_url, wait_timeout=10)
            
            # Start stream
            self.sio.emit('start_stream', {'stream_id': self.stream_id})
            
            frame_count = 0
            cycle_start_time = time.time()
            cycle_duration = random.uniform(
                self.fps_config['min_duration'],
                self.fps_config['max_duration']
            )
            
            while True:  # Outer loop for continuous folder processing
                for video_file in video_files:
                    print(f"Processing video: {video_file}")
                    cap = cv2.VideoCapture(video_file)
                    
                    if not cap.isOpened():
                        print(f"Failed to open video file: {video_file}")
                        continue
                    
                    # Get video properties
                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    start_time = time.time()
                    
                    try:
                        while True:
                            current_time = time.time()
                            
                            # Check if we need a new cycle
                            cycle_elapsed = current_time - cycle_start_time
                            if cycle_elapsed >= cycle_duration:
                                cycle_start_time = current_time
                                cycle_duration = random.uniform(
                                    self.fps_config['min_duration'],
                                    self.fps_config['max_duration']
                                )
                                print(f"New cycle started with duration: {cycle_duration:.2f}s")
                                cycle_elapsed = 0
                            
                            # Calculate target FPS
                            target_fps = self.calculate_target_fps(cycle_elapsed, cycle_duration)
                            
                            # Calculate what frame we should be at in real time
                            elapsed_time = current_time - start_time
                            target_frame = int(elapsed_time * video_fps)
                            
                            # Calculate how many frames to skip based on ratio of video FPS to target FPS
                            skip_frames = max(1, int(video_fps / target_fps))
                            
                            # Calculate the next frame to show
                            next_frame = (target_frame // skip_frames) * skip_frames
                            
                            if next_frame >= total_frames:
                                print(f"Finished processing: {video_file}")
                                break
                                
                            # Set position and read frame
                            cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Encode frame
                            _, buffer = cv2.imencode('.jpg', frame)
                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Send frame
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
                            
                            # Calculate sleep time to maintain target FPS
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
    client = DynamicStreamClient()
    client.start_streaming()