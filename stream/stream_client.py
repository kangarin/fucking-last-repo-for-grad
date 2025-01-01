from socketio import Client
import cv2
import base64
import time
import numpy as np
from pathlib import Path
import sys
import os

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

    def get_video_files(self, folder_path):
        """Get all video files from the specified folder"""
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv')
        video_files = []
        
        for file in os.listdir(folder_path):
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(folder_path, file))
                
        return sorted(video_files)  # Sort files to ensure consistent order
    
    def start_streaming(self, video_source=None):
        """Start streaming videos from the specified source folder"""
        stream_config = config.get_stream_config()
        if video_source is None:
            video_source = stream_config['source_path']
            
        # Check if video_source is a directory
        if os.path.isdir(video_source):
            video_files = self.get_video_files(video_source)
            if not video_files:
                raise Exception(f"No video files found in directory: {video_source}")
        else:
            # If it's a single file, put it in a list
            video_files = [video_source]
            
        try:
            # Connect to processing server
            print(f"Connecting to {self.processing_server_url}")
            self.sio.connect(self.processing_server_url, wait_timeout=10)
            
            # Start stream
            self.sio.emit('start_stream', {'stream_id': self.stream_id})
            
            frame_delay = stream_config['frame_delay']
            frame_count = 0
            start_time = time.time()
            
            while True:  # Outer loop for continuous folder processing
                for video_file in video_files:
                    print(f"Processing video: {video_file}")
                    cap = cv2.VideoCapture(video_file)
                    
                    if not cap.isOpened():
                        print(f"Failed to open video file: {video_file}")
                        continue
                    
                    # Get video properties
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_duration = total_frames / fps
                    
                    try:
                        while True:
                            # Calculate which frame we should be at based on elapsed time
                            current_time = time.time()
                            elapsed_time = current_time - start_time
                            
                            # Calculate the ideal frame number based on elapsed time
                            ideal_frame = int(elapsed_time * fps)
                            
                            # If we're behind where we should be, skip frames to catch up
                            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            if ideal_frame > current_frame:
                                # Skip to the ideal frame
                                cap.set(cv2.CAP_PROP_POS_FRAMES, ideal_frame)
                            
                            # Read the frame
                            ret, frame = cap.read()
                            if not ret:
                                print(f"Finished processing: {video_file}")
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
                            
                            # If we've reached the end of video duration, reset the start time
                            if elapsed_time >= video_duration:
                                start_time = time.time()
                            
                    finally:
                        cap.release()
                        
                print("Finished processing all videos, restarting...")
                start_time = time.time()  # Reset start time for next iteration
                
        except Exception as e:
            print(f"Streaming error: {e}")
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
    client = StreamClient()
    client.start_streaming()