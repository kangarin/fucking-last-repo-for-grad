import time
import requests
from socketio import Client
from collections import deque
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

class ModelSwitcher:
    def __init__(self):
        # 保持原有的初始化
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        
        models_config = config.get_models_config()
        self.allowed_models = models_config['allowed_sizes']
        
        # 决策相关参数
        self.decision_interval = 10  # 决策周期(秒)
        self.latency_threshold = 0.5  # 延迟阈值(秒)
        self.models_by_performance = ['n', 's', 'm', 'l', 'x']  # 从轻到重排序
        
        # 切换控制
        self.switch_cooldown = 2  # 切换冷却期(决策周期数)
        self.last_switch = 0
        self.current_step = 0
        
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
        try:
            print(f"Connecting to {self.processing_server_url}")
            self.sio.connect(self.processing_server_url, wait_timeout=10)
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False

    def get_current_stats(self):
        try:
            response = requests.get(f"{self.http_url}/get_stats")
            if response.status_code == 200:
                return response.json().get('stats')
            return None
        except Exception as e:
            print(f"Error getting stats: {e}")
            return None

    def switch_model(self, model_name):
        try:
            if not self.sio.connected and not self.connect_to_server():
                return False
            self.sio.emit('switch_model', {'model_name': model_name})
            return True
        except Exception as e:
            print(f"Error switching model: {e}")
            return False

    def make_decision(self, stats):
        """基于延迟做出模型切换决策"""
        self.current_step += 1
        
        # 检查冷却期
        if not stats or (self.current_step - self.last_switch) < self.switch_cooldown:
            return None
            
        current_model = stats['model_name']
        current_idx = self.models_by_performance.index(current_model)
        current_latency = stats['latency']
        
        # 简单的阈值判断
        if current_latency > self.latency_threshold and current_idx > 0:
            # 延迟过高，切换到更轻量级的模型
            self.last_switch = self.current_step
            return self.models_by_performance[current_idx - 1]
            
        if current_latency < self.latency_threshold * 0.7 and current_idx < len(self.models_by_performance) - 1:
            # 延迟充分低，可以尝试切换到更重的模型
            self.last_switch = self.current_step
            return self.models_by_performance[current_idx + 1]
            
        return None

    def adaptive_switch_loop(self):
        if not self.connect_to_server():
            print("Failed to connect to processing server")
            return
            
        print(f"Starting adaptive model switching loop with {self.decision_interval} second interval")
        
        while True:
            try:
                stats = self.get_current_stats()
                if stats:
                    print(f"\nCurrent stats: Accuracy={stats['accuracy']:.1f} mAP, "
                          f"Latency={stats['latency']:.3f}s, "
                          f"Model=yolov5{stats['model_name']}")
                    
                    next_model = self.make_decision(stats)
                    if next_model and next_model != stats['model_name']:
                        print(f"Switching to yolov5{next_model}")
                        self.switch_model(next_model)
                    else:
                        print("Keeping current model")
                
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
    # 启动自适应切换循环
    switcher.adaptive_switch_loop()