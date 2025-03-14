import time
import requests
from socketio import Client

from pathlib import Path
import sys
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config
from config import config

class OptimizedModelSwitcher:
    def __init__(self):
        # 初始化Socket.IO客户端
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        
        # 可用模型列表，按性能从低到高排序
        self.available_models = ['n', 's', 'm', 'l', 'x']
        
        # 队列阈值 - 使用配置的高低阈值的平均值
        self.queue_threshold = (config.get_queue_high_threshold_length() + config.get_queue_low_threshold_length()) // 2
        
        # 稳定性相关参数
        self.is_stable = False  # 当前是否稳定
        self.stability_counter = 0  # 连续稳定周期计数
        self.base_stability_threshold = 3  # 基础稳定阈值
        
        # 冷却机制参数
        self.cooling_factor = 1  # 冷却系数，初始为1
        self.max_cooling_factor = 5  # 最大冷却系数
        self.cooling_recovery_rate = 1  # 每次稳定周期恢复的冷却系数
        
        # 决策间隔
        self.decision_interval = 2  # 决策周期(秒)
        
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

    def switch_model(self, model_name, is_downgrade=False):
        """向处理服务器发送模型切换请求"""
        try:
            if not self.sio.connected:
                if not self.connect_to_server():
                    return False
            
            # 更新冷却系数
            if is_downgrade:
                # 每次降级增加冷却系数，但不超过最大值
                self.cooling_factor = min(self.max_cooling_factor, self.cooling_factor + 1)
                print(f"Downgrade detected! New cooling factor: {self.cooling_factor}")
            else:
                # 如果是升级，重置冷却因子
                self.cooling_factor = 1
            
            self.sio.emit('switch_model', {'model_name': model_name})
            print(f"Switching model to: yolov5{model_name}")
            return True
        except Exception as e:
            print(f"Error switching model: {e}")
            return False

    def make_decision(self, stats):
        """基于简化规则做出模型切换决策"""
        if not stats:
            return None, False
        
        current_model = stats['cur_model_index']
        current_model_idx = self.get_model_index(current_model)
        queue_length = stats['queue_length']
        
        # 更新系统稳定状态
        is_currently_stable = queue_length <= self.queue_threshold
        
        # 如果系统不稳定(队列超过阈值)，立即考虑降级
        if not is_currently_stable:
            self.stability_counter = 0  # 重置稳定计数
            
            # 如果有更轻量级的模型可用，则降级
            if current_model_idx > 0:
                next_model = self.available_models[current_model_idx - 1]
                print(f"Queue length ({queue_length}) exceeds threshold ({self.queue_threshold}). Downgrading from yolov5{current_model} to yolov5{next_model}")
                return next_model, True  # 返回模型名和降级标志
            else:
                print(f"Queue length ({queue_length}) exceeds threshold, but already using lightest model.")
                return None, False
        
        # 如果系统稳定(队列低于阈值)，考虑是否可以升级
        else:
            # 增加稳定计数
            self.stability_counter += 1
            print(f"System stable: stability={self.stability_counter}/{self.base_stability_threshold * self.cooling_factor}")
            
            # 如果冷却系数大于1，且稳定计数为偶数，则降低冷却系数
            if self.cooling_factor > 1 and self.stability_counter % 2 == 0:
                self.cooling_factor = max(1, self.cooling_factor - self.cooling_recovery_rate)
                print(f"Cooling factor reduced to: {self.cooling_factor}")
            
            # 只有当连续稳定次数超过阈值(考虑冷却因子)时，才升级
            current_stability_threshold = int(self.base_stability_threshold * self.cooling_factor)
            if self.stability_counter >= current_stability_threshold:
                # 如果有更高级的模型可用，则升级
                if current_model_idx < len(self.available_models) - 1:
                    next_model = self.available_models[current_model_idx + 1]
                    print(f"System consistently stable ({self.stability_counter}/{current_stability_threshold}). Upgrading from yolov5{current_model} to yolov5{next_model}")
                    self.stability_counter = 0  # 重置稳定计数
                    return next_model, False  # 返回模型名和降级标志
                else:
                    print("System stable, but already using highest model.")
                    self.stability_counter = 0  # 重置稳定计数
            
        # 保持当前模型
        return None, False

    def adaptive_switch_loop(self):
        """模型切换的主循环"""
        if not self.connect_to_server():
            print("Failed to connect to processing server")
            return
            
        print(f"Starting model switching loop with {self.decision_interval} second interval")
        print(f"Queue threshold: {self.queue_threshold}, Base stability threshold: {self.base_stability_threshold}")
        print(f"Cooling mechanism: Max factor {self.max_cooling_factor}, Recovery rate {self.cooling_recovery_rate}")
        
        while True:
            try:
                # 获取当前状态
                stats = self.get_current_stats()
                if stats:
                    # 做出决策
                    next_model, is_downgrade = self.make_decision(stats)
                    if next_model:
                        # 执行模型切换
                        self.switch_model(next_model, is_downgrade)
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
    switcher = OptimizedModelSwitcher()
    # 启动切换循环
    switcher.adaptive_switch_loop()