# stream_client.py
from socketio import Client
import cv2
import base64
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

class StreamClient:
    def __init__(self, stream_id=None):
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        
        stream_config = config.get_stream_config()
        self.stream_id = stream_id or stream_config['default_id']
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
    
    def start_streaming(self, video_source=None):
        """Start streaming video from the specified source"""
        stream_config = config.get_stream_config()
        if video_source is None:
            video_source = stream_config['source_path']
            
        cap = None
        try:
            # First try to open video source before connecting
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                raise Exception(f"Failed to open video source: {video_source}")
            
            # Connect to processing server
            print(f"Connecting to {self.processing_server_url}")
            self.sio.connect(self.processing_server_url, wait_timeout=10)
            
            # Start stream
            self.sio.emit('start_stream', {'stream_id': self.stream_id})
            
            print(f"Started streaming from {video_source}")
            
            frame_delay = stream_config['frame_delay']
            frame_count = 0
            
            while True:
                # ret, frame = cap.read()
                # if not ret:
                #     print("End of video stream")
                #     break
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream, restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到视频开始
                    ret, frame = cap.read()  # 重新读取第一帧
                    if not ret:  # 如果还是失败，那么可能是视频文件损坏
                        print("Failed to restart video stream")
                        break
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame with timestamps
                self.sio.emit('process_frame', {
                    'stream_id': self.stream_id,
                    'frame_id': frame_count,
                    'timestamps': {
                        'generated': time.time(),
                    },
                    'image': img_base64
                })
                
                frame_count += 1
                time.sleep(frame_delay)
                
        except Exception as e:
            print(f"Streaming error: {e}")
        finally:
            self.stop_streaming()
            if cap is not None and cap.isOpened():
                cap.release()
    
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
    client = StreamClient()
    client.start_streaming()