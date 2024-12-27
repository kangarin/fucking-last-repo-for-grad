# edge/model_switcher.py
import time
import requests
from socketio import Client
from pathlib import Path
import sys

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
        self.decision_interval = 10  # 决策周期(秒)
        self.latency_threshold = 0.5  # 延迟阈值(秒)
        self.queue_threshold = 5  # 队列长度阈值
        
        # 模型性能排序(按mAP从低到高)
        self.models_by_performance = ['n', 's', 'm', 'l', 'x']
        
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
            response = requests.get(f"{self.http_url}/get_stats")
            if response.status_code == 200:
                data = response.json()
                return data.get('stats')
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
            return True
        except Exception as e:
            print(f"Error switching model: {e}")
            return False

    def make_decision(self, stats):
        """基于系统状态做出模型切换决策"""
        if not stats:
            return None
            
        current_model = stats['model_name']
        current_idx = self.models_by_performance.index(current_model)
        
        # 判断是否需要切换到更轻量级模型
        if (stats['latency'] > self.latency_threshold or 
            stats['queue_length'] > self.queue_threshold):
            if current_idx > 0:  # 还能切换到更轻量级的模型
                return self.models_by_performance[current_idx - 1]
                
        # 判断是否可以切换到更重的模型
        elif (stats['latency'] < self.latency_threshold / 2 and 
              stats['queue_length'] < self.queue_threshold / 2):
            if current_idx < len(self.models_by_performance) - 1:
                return self.models_by_performance[current_idx + 1]
                
        return None  # 保持当前模型不变

    def adaptive_switch_loop(self):
        """基于系统状态自适应切换模型"""
        if not self.connect_to_server():
            print("Failed to connect to processing server")
            return
            
        print(f"Starting adaptive model switching loop with {self.decision_interval} second interval")
        
        while True:
            try:
                # 获取当前状态
                stats = self.get_current_stats()
                if stats:
                    print(f"\nCurrent stats: Accuracy={stats['accuracy']:.1f} mAP, "
                          f"Latency={stats['latency']:.3f}s, "
                          f"Queue={stats['queue_length']}, "
                          f"Model=yolov5{stats['model_name']}")
                    
                    # 做出决策
                    next_model = self.make_decision(stats)
                    if next_model and next_model != stats['model_name']:
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
    # 启动自适应切换循环
    switcher.adaptive_switch_loop()