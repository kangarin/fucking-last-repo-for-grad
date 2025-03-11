import time
import random
import requests
from socketio import Client
from pathlib import Path
import sys
from pprint import pprint

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

class ModelSwitcher:
    def __init__(self):
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        # 直接使用配置中的 URL
        self.http_url = self.processing_server_url
        
        # 获取允许的模型列表
        models_config = config.get_models_config()
        self.allowed_models = models_config['allowed_sizes']
        
        # 决策相关参数
        self.decision_interval = 2  # 决策周期(秒), 改为5秒
        
        # 可用模型列表
        self.available_models = ['n', 's', 'm', 'l', 'x']
        
        # 设置Socket.IO事件处理
        self.setup_socket_events()

    def setup_socket_events(self):
        """设置Socket.IO事件处理器"""
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

    def get_current_stats(self):
        """获取当前系统状态"""
        try:
            # 使用新的 API 接口, 请求最新的一条数据
            response = requests.get(f"{self.http_url}/get_stats?nums=1")
            if response.status_code == 200:
                data = response.json()
                print("Data:")
                pprint(data)
                return data[0]
            else:
                print(f"Failed to get stats. Status code: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Network error while getting stats: {e}")
            return None
        except ValueError as e:
            print(f"Invalid JSON response: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error getting stats: {e}")
            return None

    def switch_model(self, model_name):
        """向处理服务器发送模型切换请求"""
        try:
            if not self.sio.connected:
                if not self.connect_to_server():
                    return False
            
            self.sio.emit('switch_model', {'model_name': model_name})
            print(f"Switching model to: yolov5{model_name}")
            return True
        except Exception as e:
            print(f"Error switching model: {e}")
            return False

    def make_decision(self, stats):
        """随机选择一个不同于当前模型的新模型"""
        if not stats:
            return None
            
        current_model = stats['cur_model_index']
        # 创建一个不包含当前模型的列表
        available_models = [m for m in self.available_models if m != current_model]
        
        # 从可用模型中随机选择一个
        if available_models:
            return random.choice(available_models)
                
        return None  # 如果没有其他可用模型，保持当前模型不变

    def adaptive_switch_loop(self):
        """基于随机选择进行模型切换"""
        if not self.connect_to_server():
            print("Failed to connect to processing server")
            return
            
        print(f"Starting random model switching loop with {self.decision_interval} second interval")
        
        while True:
            try:
                # 获取当前状态
                stats = self.get_current_stats()
                if stats:

                    # 随机选择新模型
                    next_model = self.make_decision(stats)
                    if next_model:
                        print(f"Switching to yolov5{next_model}")
                        self.switch_model(next_model)
                    else:
                        print("Keeping current model")
                
                # 等待下一个决策周期
                time.sleep(self.decision_interval)
                
            except KeyboardInterrupt:
                print("\nStopping model switcher...")
                self.sio.disconnect()
                break
            except Exception as e:
                print(f"Error in switching loop: {e}")
                time.sleep(self.decision_interval)

if __name__ == '__main__':
    switcher = ModelSwitcher()
    # 启动随机切换循环
    switcher.adaptive_switch_loop()