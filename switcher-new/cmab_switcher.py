import time
import requests
import numpy as np
from socketio import Client
from pathlib import Path
import sys
from pprint import pprint

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

class ThompsonSamplingModelSwitcher:
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
        
        # 历史状态记录
        self.history_stats = []
        self.history_size = 10  # 保留最近10条记录用于趋势分析
        
        # Thompson Sampling 参数
        self.arms = {model: {'alpha': 1.0, 'beta': 1.0, 'count': 0, 'sum_reward': 0} 
                    for model in self.available_models}
        
        # 当前使用的模型
        self.current_model = None
        
        # 探索率 (epsilon-greedy 策略的备选项)
        self.exploration_rate = 0.1
        
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
            self.current_model = data['model_name']
            
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
        
        # 如果这是第一次获取统计数据，记录当前模型
        if self.current_model is None:
            self.current_model = stats['cur_model_index']

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

    def calculate_reward(self, stats):
        """计算模型的奖励值"""
        queue_ratio = stats['queue_length'] / self.queue_high_threshold_length
        
        # 权重计算
        w1 = max(1 - queue_ratio, 0)  # 准确率权重
        w2 = queue_ratio  # 延迟权重
        
        # 奖励计算
        reward = w1 * (stats['cur_model_accuracy']/100.0 + stats['avg_confidence']) - \
                 w2 * (stats['processing_latency'])
                 
        print(f"Calculated reward: {reward:.4f} (w1={w1:.2f}, w2={w2:.2f})")
        return reward

    def update_thompson_sampling(self, model, reward):
        """更新Thompson Sampling的参数"""
        arm = self.arms[model]
        arm['count'] += 1
        arm['sum_reward'] += reward
        
        # 更新Beta分布的参数
        # 我们将奖励值规范化到[0,1]范围，假设最大奖励为2.0（准确率1.0+置信度1.0）
        normalized_reward = min(max(reward / 2.0, 0), 1)
        
        # 增量更新
        arm['alpha'] += normalized_reward
        arm['beta'] += (1 - normalized_reward)
        
        print(f"Updated arm {model}: alpha={arm['alpha']:.2f}, beta={arm['beta']:.2f}, avg_reward={arm['sum_reward']/arm['count']:.4f}")

    def select_model_thompson_sampling(self):
        """使用Thompson Sampling策略选择模型"""
        # 从每个臂的Beta分布中采样
        samples = {model: np.random.beta(arm['alpha'], arm['beta']) 
                  for model, arm in self.arms.items()}
        
        # 检查是否有模型样本值特别高（超过队列长度阈值时的紧急情况）
        if self.history_stats and self.history_stats[-1]['queue_length'] >= self.queue_max_length:
            # 紧急情况，选择最轻量的模型
            selected_model = self.available_models[0]
            print(f"EMERGENCY: Queue length exceeded maximum. Selecting lightest model: yolov5{selected_model}")
        else:
            # 选择样本值最高的模型
            selected_model = max(samples, key=samples.get)
            print(f"Thompson sampling selected model: yolov5{selected_model} (sample={samples[selected_model]:.4f})")
            
            # 打印所有模型的样本值，便于调试
            for model, sample in samples.items():
                print(f"  yolov5{model}: sample={sample:.4f}, alpha={self.arms[model]['alpha']:.2f}, beta={self.arms[model]['beta']:.2f}")
        
        return selected_model

    def cmab_switch_loop(self):
        """基于Contextual Multi-Armed Bandit进行模型切换的主循环"""
        if not self.connect_to_server():
            print("Failed to connect to processing server")
            return
            
        print(f"Starting Thompson Sampling model switching loop with {self.decision_interval} second interval")
        print(f"Queue thresholds - Low: {self.queue_low_threshold_length}, High: {self.queue_high_threshold_length}, Max: {self.queue_max_length}")
        
        while True:
            try:
                # 获取当前状态
                stats = self.get_current_stats()
                if stats:
                    # 如果已经有当前模型，计算奖励并更新参数
                    if self.current_model:
                        reward = self.calculate_reward(stats)
                        self.update_thompson_sampling(self.current_model, reward)
                    
                    # 选择下一个模型
                    next_model = self.select_model_thompson_sampling()
                    
                    # 如果选择了不同的模型，则切换
                    if next_model != self.current_model:
                        self.switch_model(next_model)
                    else:
                        print(f"Keeping current model: yolov5{self.current_model}")
                
                # 等待下一个决策周期
                time.sleep(self.decision_interval)
                
            except KeyboardInterrupt:
                print("\nStopping model switcher...")
                self.sio.disconnect()
                break
            except Exception as e:
                print(f"Error in switching loop: {e}")
                import traceback
                traceback.print_exc()  # 打印完整的堆栈跟踪
                time.sleep(self.decision_interval)

if __name__ == '__main__':
    switcher = ThompsonSamplingModelSwitcher()
    # 启动Thompson Sampling切换循环
    switcher.cmab_switch_loop()