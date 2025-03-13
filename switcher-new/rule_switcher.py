import time
import requests
from socketio import Client
import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

class SimpleModelSwitcher:
    def __init__(self):
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        
        # 获取允许的模型列表
        models_config = config.get_models_config()
        self.allowed_models = models_config['allowed_sizes']
        
        # 决策相关参数
        self.decision_interval = 2  # 决策周期(秒)
        
        # 可用模型列表，按性能从低到高排序
        self.available_models = ['n', 's', 'm', 'l', 'x']
        
        # 简化的队列阈值
        self.queue_threshold = (config.get_queue_high_threshold_length() + config.get_queue_low_threshold_length()) // 2
        
        # 压力计数器
        self.pressure = 0
        
        # 稳定性计数器和阈值
        self.stability_counter = 0
        self.stability_threshold = 3  # 连续3次无压力才升级模型
        
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
            # 重置所有计数器
            self.pressure = 0
            self.stability_counter = 0
            
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
            response = requests.get(f"{self.http_url}/get_stats?nums=1")
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
            return None
        except Exception as e:
            print(f"Error getting stats: {e}")
            return None

    def get_model_index(self, model_name):
        """获取模型在可用模型列表中的索引"""
        try:
            return self.available_models.index(model_name)
        except ValueError:
            return -1

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
        """基于简化规则做出模型切换决策"""
        if not stats:
            return None
        
        current_model = stats['cur_model_index']
        current_model_idx = self.get_model_index(current_model)
        queue_length = stats['queue_length']
        
        # 更新压力计数器 - 快减慢增模式
        if queue_length > self.queue_threshold:
            # 慢增：每次只增加1
            self.pressure += 1
            print(f"System under pressure: {self.pressure}")
        else:
            # 快减：直接重置为0
            self.pressure = 0
            print(f"System pressure fully released: pressure={self.pressure}")
        
        # 如果有任何压力，判断是否需要立即降级
        if self.pressure > 0:
            # 对于任何非零压力，都考虑降级
            if current_model_idx > 0:  # 确保有更轻量的模型可用
                next_model = self.available_models[current_model_idx - 1]
                print(f"Pressure detected ({self.pressure}). Downgrading from yolov5{current_model} to yolov5{next_model}")
                return next_model
            else:
                print("Already using lightest model, cannot downgrade further")
                self.pressure = 0  # 重置压力计数器
                return None
        
        # 如果系统无压力，谨慎尝试升级到更高级模型（增加稳定性计数器）
        if self.pressure == 0:
            # 增加稳定性计数器
            self.stability_counter += 1
            print(f"System stable: stability={self.stability_counter}/{self.stability_threshold}")
            
            # 只有连续达到稳定阈值次数才升级
            if self.stability_counter >= self.stability_threshold:
                if current_model_idx < len(self.available_models) - 1:  # 确保有更高级的模型可用
                    next_model = self.available_models[current_model_idx + 1]
                    print(f"System consistently stable. Upgrading model from yolov5{current_model} to yolov5{next_model}")
                    self.stability_counter = 0  # 重置稳定计数器
                    return next_model
                else:
                    print("Already using highest model, cannot upgrade further")
                    self.stability_counter = 0  # 重置稳定计数器
                    return None
        else:
            # 如果有任何压力，重置稳定性计数器
            self.stability_counter = 0
        
        # 其他情况保持当前模型
        return None

    def adaptive_switch_loop(self):
        """基于简化规则进行模型切换的主循环"""
        if not self.connect_to_server():
            print("Failed to connect to processing server")
            return
            
        print(f"Starting simple model switching loop with {self.decision_interval} second interval")
        print(f"Queue threshold: {self.queue_threshold}, Stability threshold: {self.stability_threshold}")
        print(f"Strategy: 快减慢增 - Fast downgrade when under pressure, slow and careful upgrade when stable")
        
        while True:
            try:
                # 获取当前状态
                stats = self.get_current_stats()
                if stats:
                    # 做出决策
                    next_model = self.make_decision(stats)
                    if next_model:
                        # 执行模型切换
                        self.switch_model(next_model)
                    else:
                        print(f"Keeping current model: yolov5{stats['cur_model_index']}")
                
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
    switcher = SimpleModelSwitcher()
    # 启动简化的切换循环
    switcher.adaptive_switch_loop()