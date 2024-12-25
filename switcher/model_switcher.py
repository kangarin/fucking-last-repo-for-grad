# edge/model_switcher.py
import time
import random
from pathlib import Path
import sys
from socketio import Client

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

class ModelSwitcher:
    def __init__(self):
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        
        # 获取允许的模型列表
        models_config = config.get_models_config()
        self.allowed_models = models_config['allowed_sizes']
        print(f"Available models: {self.allowed_models}")
        
        # 设置Socket.IO事件处理
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
            
        @self.sio.on('model_switched')
        def on_model_switched(data):
            print(f"Model successfully switched to: {data['model_name']}")
            
        @self.sio.on('error')
        def on_error(data):
            print(f"Error: {data['message']}")

    def connect_to_server(self):
        """连接到处理服务器"""
        try:
            print(f"Connecting to {self.processing_server_url}")
            self.sio.connect(self.processing_server_url, wait_timeout=10)
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False

    def switch_model(self, model_name):
        """向处理服务器发送模型切换请求"""
        try:
            if not self.sio.connected:
                if not self.connect_to_server():
                    return False
            
            self.sio.emit('switch_model', {'model_name': model_name})
            return True
        except Exception as e:
            print(f"Error switching model: {e}")
            return False

    def random_switch_loop(self, interval=10):
        """每隔指定时间随机切换一次模型"""
        if not self.connect_to_server():
            print("Failed to connect to processing server")
            return
            
        print(f"Starting model switching loop with {interval} second interval")
        current_model = None
        
        while True:
            try:
                # 随机选择一个不同的模型
                available_models = [m for m in self.allowed_models if m != current_model]
                if not available_models:
                    available_models = self.allowed_models
                
                next_model = random.choice(available_models)
                print(f"\nSwitching from yolov5{current_model} to yolov5{next_model}")
                
                if self.switch_model(next_model):
                    current_model = next_model
                
                # 等待指定时间
                print(f"Waiting {interval} seconds before next switch...")
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\nStopping model switcher...")
                self.sio.disconnect()
                break
            except Exception as e:
                print(f"Error in switching loop: {e}")
                time.sleep(interval)

if __name__ == '__main__':
    switcher = ModelSwitcher()
    # 启动随机切换循环，每10秒切换一次模型
    switcher.random_switch_loop(interval=10)