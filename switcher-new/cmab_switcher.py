import time
import requests
import numpy as np
from socketio import Client
from pathlib import Path
import sys
from pprint import pprint
from scipy import linalg

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
        
        # 当前使用的模型
        self.current_model = None
        
        # 探索增强参数
        self.exploration_rate = 0.9  # 随机探索率
        self.decay_factor = 0.995    # 探索率的衰减因子
        self.min_exploration_rate = 0.1  # 最小探索率
        self.force_exploration_count = 15  # 强制定期探索计数器
        self.exploration_counter = 0      # 当前计数
        
        # 噪声方差和正则化参数
        self.noise_variance = 0.1    # 假设的噪声方差
        self.lambda_reg = 1.0        # 正则化参数
        
        # Thompson Sampling 参数
        self.context_dimension = 13  # 特征维度
        self.models = {}
        self.init_thompson_sampling_models()
        
        # 设置Socket.IO事件处理
        self.setup_socket_events()
        
    def init_thompson_sampling_models(self):
        """初始化Thompson Sampling模型参数"""
        for model in self.available_models:
            # 参数先验：均值向量和协方差矩阵
            self.models[model] = {
                # 参数均值向量 (零向量)
                'mu': np.zeros(self.context_dimension),
                
                # 参数协方差矩阵 (单位矩阵乘以正则化参数的逆)
                'Sigma': np.eye(self.context_dimension) / self.lambda_reg,
                
                # 足够统计量: X^T X
                'precision': self.lambda_reg * np.eye(self.context_dimension),
                
                # 足够统计量: X^T y
                'precision_mean': np.zeros(self.context_dimension),
                
                # 观察计数
                'count': 0,
                
                # 观察的奖励总和
                'sum_reward': 0,
                
                # 奖励样本方差估计
                'reward_variance': 1.0
            }

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
            if 'model_name' in data:
                model_name = data['model_name']
                self.current_model = model_name
            
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
    
    def extract_features(self, stats):
        """从统计数据中提取和归一化特征"""
        # 确保所有特征都是浮点数并且进行适当归一化
        features = np.array([
            float(stats['queue_length']) / self.queue_high_threshold_length,  # 队列长度的归一化值
            float(stats['processing_latency']),
            float(stats['total_latency']),
            float(stats['target_nums']) / 10.0,  # 假设平均目标数量不超过10
            float(stats['avg_confidence']),  # 已经是0-1之间
            float(stats['std_confidence']),  # 已经是较小值
            float(stats['avg_size']),  # 已经是0-1之间
            float(stats['std_size']),  # 已经是较小值
            float(stats['brightness']) / 255.0,  # 亮度归一化到0-1
            float(stats['contrast']) / 255.0,  # 对比度归一化到0-1
            float(stats['entropy']) / 10.0,  # 熵通常在0-10之间
            float(stats['cur_model_accuracy']) / 100.0,  # 准确率归一化到0-1
            float(stats['fps'])
        ])
        
        # 增加一个常数特征，相当于偏置项
        # features = np.append(features, 1.0)
        
        # 打印特征向量，便于调试
        feature_names = [
            'queue_length_norm', 'processing_latency', 'total_latency', 'target_nums_norm', 
            'avg_confidence', 'std_confidence', 'avg_size', 'std_size',
            'brightness_norm', 'contrast_norm', 'entropy_norm', 'model_accuracy_norm', 'fps'
        ]
        
        print("Extracted features:")
        for name, value in zip(feature_names, features):
            print(f"  {name}: {value:.4f}")
        
        return features
    
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
            print(f"Switching model to: {model_name}")
            return True
        except Exception as e:
            print(f"Error switching model: {e}")
            return False

    def calculate_reward(self, stats):
        """计算模型的奖励值"""
        queue_ratio = stats['queue_length'] / self.queue_low_threshold_length
        
        # 权重计算
        w1 = max(1 - queue_ratio, 0)  # 准确率权重
        w2 = queue_ratio  # 延迟权重
        
        # 奖励计算
        reward = w1 * (stats['cur_model_accuracy']/100.0 + stats['avg_confidence']) - \
                 w2 * (stats['processing_latency'])
                 
        print(f"Calculated reward: {reward:.4f} (w1={w1:.2f}, w2={w2:.2f})")
        return reward

    def update_thompson_sampling(self, model_name, context, reward):
        """更新Thompson Sampling模型的参数"""
        model_data = self.models[model_name]
        
        # 累计计数和奖励
        model_data['count'] += 1
        model_data['sum_reward'] += reward
        
        # 更新协方差和精度矩阵
        context_2d = context.reshape(-1, 1)  # 列向量
        
        # 更新精度矩阵 (X^T X)
        model_data['precision'] += context_2d @ context_2d.T
        
        # 更新精度均值 (X^T y)
        model_data['precision_mean'] += context * reward
        
        # 重新计算均值向量和协方差矩阵
        try:
            # 计算协方差矩阵 (Sigma)
            model_data['Sigma'] = np.linalg.inv(model_data['precision'])
            
            # 计算均值向量 (mu = Sigma * precision_mean)
            model_data['mu'] = model_data['Sigma'] @ model_data['precision_mean']
            
            # 更新奖励方差估计
            if model_data['count'] > 1:
                avg_reward = model_data['sum_reward'] / model_data['count']
                # 简单估计方差 (可以使用更复杂的方法)
                model_data['reward_variance'] = max(0.1, self.noise_variance)  # 使用一个下限
            
            print(f"Updated Thompson Sampling for model {model_name}:")
            print(f"  count={model_data['count']}, avg_reward={model_data['sum_reward']/model_data['count']:.4f}")
            print(f"  mu_norm={np.linalg.norm(model_data['mu']):.4f}, var={model_data['reward_variance']:.4f}")
        except np.linalg.LinAlgError:
            print(f"Warning: Could not invert precision matrix for model {model_name}. Using previous values.")

    def sample_parameter(self, model_name):
        """从模型的后验分布中采样参数"""
        model_data = self.models[model_name]
        
        try:
            # 从多元正态分布中采样
            # 使用Cholesky分解进行采样，更加数值稳定
            L = linalg.cholesky(model_data['Sigma'], lower=True)
            
            # 采样标准正态分布
            standard_normal = np.random.standard_normal(self.context_dimension)
            
            # 变换为目标分布
            theta_sample = model_data['mu'] + L @ standard_normal
            
            return theta_sample
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Warning: Sampling error for model {model_name}: {e}")
            print("Using mean vector instead of sampling")
            # 如果采样失败，返回均值向量
            return model_data['mu']

    def select_model_thompson_sampling(self, context):
        """使用Thompson Sampling策略选择模型"""
        # 检查是否是强制探索回合
        self.exploration_counter += 1
        force_exploration = self.exploration_counter >= self.force_exploration_count
        
        # 如果是强制探索回合，重置计数器并随机选择
        if force_exploration:
            self.exploration_counter = 0
            # 随机选择一个模型
            selected_model = np.random.choice(self.available_models)
            print(f"FORCED EXPLORATION: Randomly selected model: yolov5{selected_model}")
            return selected_model
        
        # 检查是否有紧急情况
        if self.history_stats and self.history_stats[-1]['queue_length'] >= self.queue_max_length:
            # 紧急情况，选择最轻量的模型
            selected_model = self.available_models[0]
            print(f"EMERGENCY: Queue length exceeded maximum. Selecting lightest model: yolov5{selected_model}")
            return selected_model
            
        # 随机探索分支 (epsilon-greedy)
        if np.random.random() < self.exploration_rate:
            # 随机选择一个模型
            selected_model = np.random.choice(self.available_models)
            print(f"EXPLORATION: Randomly selected model: yolov5{selected_model}")
            
            # 衰减探索率，但不低于最小值
            self.exploration_rate = max(self.exploration_rate * self.decay_factor, 
                                       self.min_exploration_rate)
            print(f"Exploration rate decayed to {self.exploration_rate:.4f}")
            
            return selected_model
        
        # Thompson Sampling 分支
        expected_rewards = {}
        sampled_params = {}
        
        # 为每个模型采样参数并计算期望奖励
        for model_name in self.available_models:
            # 从后验分布采样参数向量
            theta = self.sample_parameter(model_name)
            sampled_params[model_name] = theta
            
            # 计算期望奖励
            expected_reward = np.dot(theta, context)
            expected_rewards[model_name] = expected_reward
        
        # 选择期望奖励最高的模型
        selected_model = max(expected_rewards, key=expected_rewards.get)
        
        print(f"Thompson sampling selected model: yolov5{selected_model} (expected reward={expected_rewards[selected_model]:.4f})")
        
        # 打印所有模型的预期奖励，便于调试
        for model, reward in expected_rewards.items():
            print(f"  yolov5{model}: expected_reward={reward:.4f}, "
                  f"param_norm={np.linalg.norm(sampled_params[model]):.4f}")
        
        return selected_model

    def cmab_switch_loop(self):
        """基于Thompson Sampling进行模型切换的主循环"""
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
                    # 从统计数据中获取当前模型，如果尚未设置
                    if self.current_model is None and 'cur_model_index' in stats:
                        model_index = stats['cur_model_index']
                        self.current_model = model_index
                        print(f"Initial model detected: {self.current_model}")
                    
                    # 提取上下文特征
                    context = self.extract_features(stats)
                    
                    # 如果已经有当前模型，计算奖励并更新参数
                    if self.current_model:
                        # 确保当前模型是可用模型列表中的一个
                        if self.current_model in self.available_models:
                            reward = self.calculate_reward(stats)
                            self.update_thompson_sampling(self.current_model, context, reward)
                        else:
                            print(f"Warning: Current model '{self.current_model}' not in available models list")
                            # 尝试从统计数据中更新当前模型
                            if 'cur_model_index' in stats:
                                model_index = stats['cur_model_index']
                                if isinstance(model_index, str) and model_index in self.available_models:
                                    self.current_model = model_index
                                    print(f"Current model updated to: {self.current_model}")
                    
                    # 基于Thompson Sampling选择下一个模型
                    next_model = self.select_model_thompson_sampling(context)
                    
                    # 如果选择了不同的模型，并且是有效模型，则切换
                    if next_model != self.current_model and next_model in self.available_models:
                        success = self.switch_model(next_model)
                        if not success:
                            print(f"Failed to switch to model: {next_model}")
                    else:
                        # 确保只显示已知的当前模型
                        if self.current_model in self.available_models:
                            print(f"Keeping current model: yolov5{self.current_model}")
                        else:
                            print("Current model status unknown")
                
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