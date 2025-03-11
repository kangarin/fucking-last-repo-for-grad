import time
import requests
from socketio import Client
from pathlib import Path
import sys
from pprint import pprint

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

class HeuristicModelSwitcher:
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
        self.decision_interval = 2  # 决策周期(秒)
        
        # 可用模型列表，按性能从低到高排序
        self.available_models = ['n', 's', 'm', 'l', 'x']
        
        # 队列阈值参数
        self.queue_max_length = config.get_queue_max_length()
        self.queue_low_threshold_length = config.get_queue_low_threshold_length()
        self.queue_high_threshold_length = config.get_queue_high_threshold_length()
        
        # 压力计数器和阈值
        self.pressure = 0
        self.pressure_threshold = 3  # 连续3次超过阈值则降级模型
        self.pressure_release_threshold = 5  # 连续5次低于阈值则尝试升级模型
        self.release_counter = 0
        
        # 历史状态记录
        self.history_stats = []
        self.history_size = 10  # 保留最近10条记录用于趋势分析
        
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
            # 重置压力计数器
            self.pressure = 0
            self.release_counter = 0
            
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
            # 使用 API 接口, 请求最新的一条数据
            response = requests.get(f"{self.http_url}/get_stats?nums=1")
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    current_stats = data[0]
                    self.update_history(current_stats)
                    print("Current Stats:")
                    pprint(current_stats)
                    return current_stats
                else:
                    print("No stats data received")
                    return None
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
    
    def update_history(self, stats):
        """更新历史状态记录"""
        self.history_stats.append(stats)
        if len(self.history_stats) > self.history_size:
            self.history_stats.pop(0)  # 移除最旧的记录

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

    def analyze_queue_trend(self):
        """分析队列长度趋势"""
        if len(self.history_stats) < 3:
            return "stable"  # 数据不足，认为稳定
            
        # 计算队列长度的增长率
        recent_queues = [stats['queue_length'] for stats in self.history_stats[-3:]]
        
        # 队列快速增长
        if recent_queues[2] > recent_queues[1] > recent_queues[0]:
            return "rapid_increase"
        # 队列持续增长
        elif recent_queues[2] > recent_queues[0]:
            return "increase"
        # 队列持续下降
        elif recent_queues[2] < recent_queues[0]:
            return "decrease"
        else:
            return "stable"

    def make_decision(self, stats):
        """基于启发式规则做出模型切换决策"""
        if not stats:
            return None
        
        current_model = stats['cur_model_index']
        current_model_idx = self.get_model_index(current_model)
        queue_length = stats['queue_length']
        processing_latency = stats['processing_latency']
        
        # 分析系统状态并更新压力计数器
        if queue_length > self.queue_high_threshold_length:
            self.pressure += 1
            self.release_counter = 0
            print(f"System under pressure: {self.pressure}/{self.pressure_threshold}")
        elif queue_length < self.queue_low_threshold_length:
            self.pressure = max(0, self.pressure - 1)
            self.release_counter += 1
            print(f"System pressure easing: pressure={self.pressure}, release={self.release_counter}/{self.pressure_release_threshold}")
        else:
            # 在中间区域保持当前状态
            pass
        
        # 分析队列趋势
        queue_trend = self.analyze_queue_trend()
        print(f"Queue trend: {queue_trend}")
        
        # 如果队列长度达到最大值，立即降级到最轻量模型
        if queue_length >= self.queue_max_length:
            lightest_model = self.available_models[0]
            print(f"Queue reached maximum length. Emergency downgrade to lightest model: yolov5{lightest_model}")
            return lightest_model
        
        # 如果压力持续增加，降级到更轻量级模型
        if self.pressure >= self.pressure_threshold:
            if current_model_idx > 0:  # 确保有更轻量的模型可用
                next_model = self.available_models[current_model_idx - 1]
                print(f"Sustained pressure detected. Downgrading model from yolov5{current_model} to yolov5{next_model}")
                return next_model
            else:
                print("Already using lightest model, cannot downgrade further")
                self.pressure = 0  # 重置压力计数器
                return None
        
        # 如果队列趋势快速增长，提前降级
        if queue_trend == "rapid_increase" and current_model_idx > 0:
            next_model = self.available_models[current_model_idx - 1]
            print(f"Queue rapidly increasing. Preemptively downgrading to yolov5{next_model}")
            return next_model
        
        # 如果系统长时间低压力，尝试升级到更高级模型
        if self.release_counter >= self.pressure_release_threshold:
            if current_model_idx < len(self.available_models) - 1:  # 确保有更高级的模型可用
                next_model = self.available_models[current_model_idx + 1]
                print(f"System stable with low pressure. Upgrading model from yolov5{current_model} to yolov5{next_model}")
                return next_model
            else:
                print("Already using highest model, cannot upgrade further")
                self.release_counter = 0  # 重置释放计数器
                return None
        
        # 其他情况保持当前模型
        return None

    def adaptive_switch_loop(self):
        """基于启发式规则进行模型切换的主循环"""
        if not self.connect_to_server():
            print("Failed to connect to processing server")
            return
            
        print(f"Starting heuristic model switching loop with {self.decision_interval} second interval")
        print(f"Queue thresholds - Low: {self.queue_low_threshold_length}, High: {self.queue_high_threshold_length}, Max: {self.queue_max_length}")
        
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
    switcher = HeuristicModelSwitcher()
    # 启动启发式切换循环
    switcher.adaptive_switch_loop()