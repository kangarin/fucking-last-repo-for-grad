# meta_rule_switcher.py
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

class ConservativeStrategy:
    """保守型策略：优先选择轻量模型，稳定时才升级"""
    
    def __init__(self, available_models, queue_thresholds):
        self.available_models = available_models
        self.queue_low = queue_thresholds['low']
        self.queue_high = queue_thresholds['high']
        self.queue_max = queue_thresholds['max']
        
        # 稳定计数器和阈值
        self.stability_counter = 0
        self.stability_threshold = 5  # 需要连续5个周期稳定才升级
        
    def select_model(self, stats, current_model):
        """选择模型策略"""
        current_idx = self.available_models.index(current_model)
        queue_length = stats['queue_length']
        
        # 如果队列长度超过高阈值，立即降级
        if queue_length > self.queue_high:
            self.stability_counter = 0
            # 降级到更轻量的模型
            if current_idx > 0:
                return self.available_models[current_idx - 1], True
        
        # 如果队列长度低于低阈值，增加稳定计数
        elif queue_length < self.queue_low:
            self.stability_counter += 1
            # 只有在连续多次稳定后才升级
            if self.stability_counter >= self.stability_threshold:
                self.stability_counter = 0
                # 升级到更高级的模型
                if current_idx < len(self.available_models) - 1:
                    return self.available_models[current_idx + 1], False
        
        # 其他情况重置稳定计数
        else:
            self.stability_counter = 0
            
        # 默认保持当前模型
        return current_model, False


class AggressiveStrategy:
    """激进型策略：优先选择高性能模型，只有在压力大时才降级"""
    
    def __init__(self, available_models, queue_thresholds):
        self.available_models = available_models
        self.queue_low = queue_thresholds['low']
        self.queue_high = queue_thresholds['high']
        self.queue_max = queue_thresholds['max']
        
        # 压力计数器和阈值
        self.pressure_counter = 0
        self.pressure_threshold = 2  # 只需2个周期高压力就降级
        
    def select_model(self, stats, current_model):
        """选择模型策略"""
        current_idx = self.available_models.index(current_model)
        queue_length = stats['queue_length']
        
        # 如果队列长度超过高阈值，增加压力计数
        if queue_length > self.queue_high:
            self.pressure_counter += 1
            # 在连续多次高压力后降级
            if self.pressure_counter >= self.pressure_threshold:
                self.pressure_counter = 0
                # 降级到更轻量的模型
                if current_idx > 0:
                    return self.available_models[current_idx - 1], True
        
        # 如果队列长度低于低阈值，立即尝试升级
        elif queue_length < self.queue_low:
            self.pressure_counter = 0
            # 尝试升级到更高级的模型
            if current_idx < len(self.available_models) - 1:
                return self.available_models[current_idx + 1], False
        
        # 其他情况，只有当压力持续时才增加计数
        else:
            if queue_length > (self.queue_low + self.queue_high) / 2:
                self.pressure_counter += 0.5
            else:
                self.pressure_counter = 0
            
        # 默认保持当前模型
        return current_model, False


class AdaptiveStrategy:
    """自适应策略：根据历史表现动态调整"""
    
    def __init__(self, available_models, queue_thresholds):
        self.available_models = available_models
        self.queue_low = queue_thresholds['low']
        self.queue_high = queue_thresholds['high']
        self.queue_max = queue_thresholds['max']
        
        # 历史窗口
        self.history_window = []
        self.window_size = 5
        
        # 趋势阈值
        self.trend_threshold = 0.1  # 10%变化视为趋势
        
    def select_model(self, stats, current_model):
        """选择模型策略"""
        current_idx = self.available_models.index(current_model)
        queue_length = stats['queue_length']
        
        # 更新历史窗口
        self.history_window.append(queue_length)
        if len(self.history_window) > self.window_size:
            self.history_window.pop(0)
        
        # 如果历史窗口填满，分析趋势
        if len(self.history_window) == self.window_size:
            # 计算队列长度趋势
            first_half = sum(self.history_window[:self.window_size//2]) / (self.window_size//2)
            second_half = sum(self.history_window[self.window_size//2:]) / (self.window_size//2 + self.window_size%2)
            
            # 比较前半段和后半段，判断趋势
            relative_change = (second_half - first_half) / (first_half + 1e-10)
            
            # 如果队列长度增加超过阈值，趋势向上，需要降级
            if relative_change > self.trend_threshold:
                if current_idx > 0:
                    return self.available_models[current_idx - 1], True
            
            # 如果队列长度减少超过阈值，趋势向下，可以升级
            elif relative_change < -self.trend_threshold:
                if current_idx < len(self.available_models) - 1:
                    return self.available_models[current_idx + 1], False
        
        # 紧急情况处理：队列长度已经超过高阈值
        if queue_length > self.queue_high:
            # 立即降级
            if current_idx > 0:
                return self.available_models[current_idx - 1], True
        
        # 默认保持当前模型
        return current_model, False


class MetaRuleSwitcher:
    def __init__(self):
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        
        # 获取允许的模型列表
        models_config = config.get_models_config()
        self.allowed_models = models_config['allowed_sizes']
        self.model_to_idx = {model: idx for idx, model in enumerate(self.allowed_models)}
        self.idx_to_model = {idx: model for idx, model in enumerate(self.allowed_models)}
        
        # 决策相关参数
        self.decision_interval = 2  # 决策周期(秒)
        
        # 可用模型列表
        self.available_models = ['n', 's', 'm', 'l', 'x']
        
        # 队列阈值参数
        self.queue_max_length = config.get_queue_max_length()
        self.queue_low_threshold_length = config.get_queue_low_threshold_length()
        self.queue_high_threshold_length = config.get_queue_high_threshold_length()
        
        # 初始化三种策略
        queue_thresholds = {
            'low': self.queue_low_threshold_length,
            'high': self.queue_high_threshold_length,
            'max': self.queue_max_length
        }
        
        self.strategies = {
            'conservative': ConservativeStrategy(self.available_models, queue_thresholds),
            'aggressive': AggressiveStrategy(self.available_models, queue_thresholds),
            'adaptive': AdaptiveStrategy(self.available_models, queue_thresholds)
        }
        
        # 当前使用的模型
        self.current_model = None
        
        # 元策略的Thompson Sampling参数
        self.context_dimension = 13
        self.meta_models = {}
        self.init_meta_thompson_sampling()
        
        # 记录上一次的策略选择和上下文
        self.previous_strategy = None
        self.previous_context = None
        
        # 设置Socket.IO事件处理
        self.setup_socket_events()
        
    def init_meta_thompson_sampling(self):
        """初始化元策略的Thompson Sampling参数"""
        self.lambda_reg = 1.0  # 正则化参数
        self.noise_variance = 0.1  # 噪声方差
        
        for strategy in self.strategies.keys():
            self.meta_models[strategy] = {
                'mu': np.zeros(self.context_dimension),
                'Sigma': np.eye(self.context_dimension) / self.lambda_reg,
                'precision': self.lambda_reg * np.eye(self.context_dimension),
                'precision_mean': np.zeros(self.context_dimension),
                'count': 0,
                'sum_reward': 0,
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
            response = requests.get(f"{self.http_url}/get_stats?nums=1")
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    current_stats = data[0]
                    print("Current Stats:")
                    pprint(current_stats)
                    return current_stats
            return None
        except Exception as e:
            print(f"Error getting stats: {e}")
            return None
    
    def extract_features(self, stats):
        """从统计数据中提取和归一化特征"""
        features = np.array([
            float(stats['queue_length']) / self.queue_high_threshold_length,
            float(stats['processing_latency']),
            float(stats['total_latency']),
            float(stats['target_nums']) / 10.0,
            float(stats['avg_confidence']),
            float(stats['std_confidence']),
            float(stats['avg_size']),
            float(stats['std_size']),
            float(stats['brightness']) / 255.0,
            float(stats['contrast']) / 255.0,
            float(stats['entropy']) / 10.0,
            float(stats['cur_model_accuracy']) / 100.0,
            float(self.model_to_idx.get(stats['cur_model_index'], 0))
        ])
        
        feature_names = [
            'queue_length_norm', 'processing_latency', 'target_nums_norm', 
            'avg_confidence', 'std_confidence', 'avg_size', 'std_size',
            'brightness_norm', 'contrast_norm', 'entropy_norm', 'model_accuracy_norm',
            'model_index'
        ]
        
        print("Extracted features:")
        for name, value in zip(feature_names, features):
            print(f"  {name}: {value:.4f}")
            
        return features
    
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
        """计算奖励值"""
        queue_ratio = stats['queue_length'] / self.queue_high_threshold_length
        
        # 权重计算
        w1 = max(1 - queue_ratio, 0)  # 准确率权重
        w2 = queue_ratio  # 延迟权重
        
        # 奖励计算
        reward = w1 * (stats['cur_model_accuracy']/100.0 + stats['avg_confidence']) - \
                 w2 * (stats['processing_latency'])
                 
        print(f"Calculated reward: {reward:.4f} (w1={w1:.2f}, w2={w2:.2f})")
        return reward
    
    def update_meta_thompson_sampling(self, strategy, context, reward):
        """更新元策略的Thompson Sampling参数"""
        if strategy not in self.meta_models:
            print(f"Warning: Strategy {strategy} not found in meta models")
            return
            
        model_data = self.meta_models[strategy]
        
        # 累计计数和奖励
        model_data['count'] += 1
        model_data['sum_reward'] += reward
        
        # 更新协方差和精度矩阵
        context_2d = context.reshape(-1, 1)  # 列向量
        
        # 更新精度矩阵 (X^T X)
        model_data['precision'] += context_2d @ context_2d.T
        
        # 更新精度均值 (X^T y)
        model_data['precision_mean'] += context * reward
        
        try:
            # 计算协方差矩阵 (Sigma)
            model_data['Sigma'] = np.linalg.inv(model_data['precision'])
            
            # 计算均值向量 (mu = Sigma * precision_mean)
            model_data['mu'] = model_data['Sigma'] @ model_data['precision_mean']
            
            # 更新奖励方差估计
            if model_data['count'] > 1:
                avg_reward = model_data['sum_reward'] / model_data['count']
                model_data['reward_variance'] = max(0.1, self.noise_variance)
            
            print(f"Updated Meta Thompson Sampling for strategy {strategy}:")
            print(f"  count={model_data['count']}, avg_reward={model_data['sum_reward']/model_data['count']:.4f}")
        except np.linalg.LinAlgError:
            print(f"Warning: Could not invert precision matrix for strategy {strategy}")
    
    def sample_meta_parameter(self, strategy):
        """从策略的后验分布中采样参数"""
        model_data = self.meta_models[strategy]
        
        try:
            # 从多元正态分布中采样
            L = linalg.cholesky(model_data['Sigma'], lower=True)
            standard_normal = np.random.standard_normal(self.context_dimension)
            theta_sample = model_data['mu'] + L @ standard_normal
            return theta_sample
        except Exception as e:
            print(f"Warning: Sampling error for strategy {strategy}: {e}")
            # 如果采样失败，返回均值向量
            return model_data['mu']
    
    def select_meta_strategy(self, context):
        """使用Thompson Sampling选择策略"""
        # 探索率
        exploration_rate = 0.1
        
        # 随机探索
        if np.random.random() < exploration_rate:
            selected_strategy = np.random.choice(list(self.strategies.keys()))
            print(f"EXPLORATION: Randomly selected strategy: {selected_strategy}")
            return selected_strategy
        
        # Thompson Sampling
        expected_rewards = {}
        
        # 为每个策略采样参数并计算期望奖励
        for strategy in self.strategies.keys():
            theta = self.sample_meta_parameter(strategy)
            expected_reward = np.dot(theta, context)
            expected_rewards[strategy] = expected_reward
        
        # 选择期望奖励最高的策略
        selected_strategy = max(expected_rewards, key=expected_rewards.get)
        
        print(f"Meta Thompson sampling selected strategy: {selected_strategy} (expected reward={expected_rewards[selected_strategy]:.4f})")
        
        # 打印所有策略的预期奖励
        for strategy, reward in expected_rewards.items():
            print(f"  {strategy}: expected_reward={reward:.4f}")
        
        return selected_strategy
    
    def meta_decision_loop(self):
        """元策略决策循环"""
        if not self.connect_to_server():
            print("Failed to connect to processing server")
            return
            
        print("Starting Meta Strategy switching loop...")
        print(f"Decision interval: {self.decision_interval} seconds")
        print(f"Available models: {self.available_models}")
        print(f"Strategies: {list(self.strategies.keys())}")
        
        while True:
            try:
                # 获取当前状态
                stats = self.get_current_stats()
                if not stats:
                    time.sleep(self.decision_interval)
                    continue
                
                # 获取当前模型
                if self.current_model is None and 'cur_model_index' in stats:
                    self.current_model = stats['cur_model_index']
                    print(f"Initial model detected: {self.current_model}")
                
                # 提取特征
                context = self.extract_features(stats)
                
                # 更新元策略 (如果有前一次决策)
                if self.previous_strategy and self.previous_context is not None:
                    reward = self.calculate_reward(stats)
                    self.update_meta_thompson_sampling(self.previous_strategy, self.previous_context, reward)
                
                # 使用元策略选择一个基础策略
                selected_strategy = self.select_meta_strategy(context)
                
                # 保存当前上下文和策略用于下次更新
                self.previous_context = context
                self.previous_strategy = selected_strategy
                
                # 获取各个策略的建议动作
                strategy_suggestions = {}
                for name, strategy in self.strategies.items():
                    model, is_downgrade = strategy.select_model(stats, self.current_model)
                    strategy_suggestions[name] = {
                        'model': model,
                        'is_downgrade': is_downgrade
                    }
                    print(f"{name.capitalize()} strategy suggests model: {model}" + 
                          (" (downgrade)" if is_downgrade else ""))
                
                # 执行选中策略的建议
                selected_model = strategy_suggestions[selected_strategy]['model']
                
                # 切换模型
                if selected_model != self.current_model:
                    success = self.switch_model(selected_model)
                    if success:
                        self.current_model = selected_model
                else:
                    print(f"Keeping current model: {self.current_model}")
                
                # 等待下一个决策周期
                time.sleep(self.decision_interval)
                
            except KeyboardInterrupt:
                print("\nStopping meta strategy switching...")
                self.sio.disconnect()
                break
            except Exception as e:
                print(f"Error in meta decision loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(self.decision_interval)

# 主程序
if __name__ == '__main__':
    meta_switcher = MetaRuleSwitcher()
    meta_switcher.meta_decision_loop()