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

# 这里将会实现不同的策略类
class DummyStrategy:
    """占位符策略类，将被实际的策略实现替换"""
    
    def __init__(self, available_models, queue_thresholds):
        self.available_models = available_models
        self.queue_thresholds = queue_thresholds
        
    def select_model(self, stats, current_model):
        """返回占位符模型决策"""
        return current_model

class QueueLengthStrategy(DummyStrategy):
    """基于冷却机制的自适应策略"""
    
    def __init__(self, available_models, queue_thresholds):
        self.available_models = available_models
        
        # 队列阈值 - 使用配置的高低阈值的平均值
        self.queue_threshold = (queue_thresholds['high'] + queue_thresholds['low']) // 2
        
        # 稳定性相关参数
        self.stability_counter = 0  # 连续稳定周期计数
        self.base_stability_threshold = 3  # 基础稳定阈值
        
        # 冷却机制参数
        self.cooling_factor = 1  # 冷却系数，初始为1
        self.max_cooling_factor = 5  # 最大冷却系数
        self.cooling_recovery_rate = 1  # 每次稳定周期恢复的冷却系数
    
    def select_model(self, stats, current_model):
        """基于简化规则做出模型切换决策"""
        current_idx = self.available_models.index(current_model)
        queue_length = stats['queue_length']
        
        # 更新系统稳定状态
        is_currently_stable = queue_length <= self.queue_threshold
        
        # 如果系统不稳定(队列超过阈值)，立即考虑降级
        if not is_currently_stable:
            self.stability_counter = 0  # 重置稳定计数
            
            # 如果有更轻量级的模型可用，则降级
            if current_idx > 0:
                next_model = self.available_models[current_idx - 1]
                print(f"Queue length ({queue_length}) exceeds threshold ({self.queue_threshold}). Downgrading from {current_model} to {next_model}")
                return next_model
            else:
                print(f"Queue length ({queue_length}) exceeds threshold, but already using lightest model.")
                return current_model
        
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
                if current_idx < len(self.available_models) - 1:
                    next_model = self.available_models[current_idx + 1]
                    print(f"System consistently stable ({self.stability_counter}/{current_stability_threshold}). Upgrading from {current_model} to {next_model}")
                    self.stability_counter = 0  # 重置稳定计数
                    return next_model
                else:
                    print("System stable, but already using highest model.")
                    self.stability_counter = 0  # 重置稳定计数
            
        # 保持当前模型
        return current_model
    
class LatencyEstimationStrategy(DummyStrategy):
    """基于处理时延预估的自适应策略
    
    为每个模型维护一个时延的预估，通过stats更新时延，然后选择不会导致队列积压的最高性能模型。
    """
    
    def __init__(self, available_models, queue_thresholds):
        super().__init__(available_models, queue_thresholds)
        
        # 初始化每个模型的处理时延预估（秒/帧）
        self.latency_estimates = {model: None for model in available_models}
        
        # 指数移动平均的权重
        self.alpha = 0.2
    
    def select_model(self, stats, current_model):
        """根据当前的统计数据和时延预估选择模型"""
        # 获取当前FPS和处理时延
        fps = stats['fps']
        processing_latency = stats['processing_latency']
        
        # 更新当前模型的时延预估
        if self.latency_estimates[current_model] is None:
            self.latency_estimates[current_model] = processing_latency
        else:
            # 使用指数移动平均更新时延预估
            self.latency_estimates[current_model] = (1 - self.alpha) * self.latency_estimates[current_model] + self.alpha * processing_latency
        
        # 打印当前所有模型的时延预估
        print("当前各模型时延预估:")
        for model, latency in self.latency_estimates.items():
            if latency is not None:
                print(f"  {model}: {latency:.4f}秒/帧")
            else:
                print(f"  {model}: 未测量")
        
        # 如果有未测量的模型，随机选择一个
        unmeasured_models = [model for model, latency in self.latency_estimates.items() if latency is None]
        if unmeasured_models:
            import random
            next_model = random.choice(unmeasured_models)
            print(f"选择未测量的模型 {next_model} 以获取其时延数据")
            return next_model
        
        # 所有模型都已测量，选择能满足当前FPS且性能最高的模型
        target_latency = 1.0 / fps  # 每帧目标处理时间
        
        # 找出所有不会导致队列积压的模型（时延小于等于目标时延）
        eligible_models = [model for model, latency in self.latency_estimates.items() 
                          if latency is not None and latency <= target_latency]
        
        if not eligible_models:
            # 如果没有符合条件的模型，选择时延最小的
            best_model = min(self.latency_estimates, key=lambda m: self.latency_estimates[m] or float('inf'))
            print(f"没有满足FPS要求的模型，选择时延最小的模型: {best_model}")
            return best_model
        
        # 从符合条件的模型中，选择性能最高的（假设模型列表按性能从低到高排序）
        eligible_indices = [self.available_models.index(model) for model in eligible_models]
        best_idx = max(eligible_indices)
        best_model = self.available_models[best_idx]
        
        print(f"选择能满足FPS={fps}要求的最高性能模型: {best_model}")
        return best_model

class DistributionBasedStrategy(DummyStrategy):
    """基于目标数量和大小分布的自适应策略
    
    维护目标数量和大小的统计分布，当当前状态偏离正常范围时调整模型。
    """
    
    def __init__(self, available_models, queue_thresholds):
        super().__init__(available_models, queue_thresholds)
        
        # 统计窗口大小
        self.window_size = 100
        
        # 初始化统计窗口
        self.target_nums_history = []  # 目标数量历史
        self.target_size_history = []  # 目标大小历史
        
        # 统计数据
        self.target_nums_mean = None   # 目标数量均值
        self.target_nums_std = None    # 目标数量标准差
        self.target_size_mean = None   # 目标大小均值
        self.target_size_std = None    # 目标大小标准差
        
        # 偏离标准差的阈值（多少个标准差触发切换）
        self.std_threshold = 1.0
        
        # 稳定性计数器
        self.upgrade_counter = 0
        self.downgrade_counter = 0
        self.stability_threshold = 3
    
    def _update_statistics(self, target_nums, target_size):
        """更新统计数据"""
        import numpy as np
        
        # 更新历史数据
        self.target_nums_history.append(target_nums)
        self.target_size_history.append(target_size)
        
        # 保持窗口大小
        if len(self.target_nums_history) > self.window_size:
            self.target_nums_history.pop(0)
            self.target_size_history.pop(0)
        
        # 只有当收集了足够的样本时才计算统计数据
        if len(self.target_nums_history) >= 10:
            self.target_nums_mean = np.mean(self.target_nums_history)
            self.target_nums_std = np.std(self.target_nums_history) or 1.0  # 避免除零
            
            self.target_size_mean = np.mean(self.target_size_history)
            self.target_size_std = np.std(self.target_size_history) or 1.0  # 避免除零
    
    def select_model(self, stats, current_model):
        """根据目标数量和大小的分布选择模型"""
        target_nums = stats['target_nums']
        avg_size = stats['avg_size']
        current_idx = self.available_models.index(current_model)
        
        # 更新统计数据
        self._update_statistics(target_nums, avg_size)
        
        # 如果统计数据尚未初始化，保持当前模型
        if self.target_nums_mean is None:
            print("统计数据尚未初始化，保持当前模型")
            return current_model
        
        # 计算当前状态与均值的偏差（以标准差为单位）
        nums_deviation = (target_nums - self.target_nums_mean) / self.target_nums_std
        size_deviation = (avg_size - self.target_size_mean) / self.target_size_std
        
        print(f"目标数量: {target_nums} (均值: {self.target_nums_mean:.2f}, 标准差: {self.target_nums_std:.2f}, 偏差: {nums_deviation:.2f}σ)")
        print(f"目标大小: {avg_size:.2f} (均值: {self.target_size_mean:.2f}, 标准差: {self.target_size_std:.2f}, 偏差: {size_deviation:.2f}σ)")
        
        # 检查是否需要升级模型
        need_upgrade = (
            nums_deviation > self.std_threshold or  # 目标数量超过均值+标准差
            size_deviation < -self.std_threshold    # 目标大小小于均值-标准差
        )
        
        # 检查是否需要降级模型
        need_downgrade = (
            nums_deviation < -self.std_threshold or  # 目标数量小于均值-标准差
            size_deviation > self.std_threshold      # 目标大小大于均值+标准差
        )
        
        # 决策逻辑
        if need_upgrade:
            self.upgrade_counter += 1
            self.downgrade_counter = 0
            print(f"可能需要升级: {self.upgrade_counter}/{self.stability_threshold}")
            
            if self.upgrade_counter >= self.stability_threshold:
                if current_idx < len(self.available_models) - 1:
                    next_model = self.available_models[current_idx + 1]
                    print(f"持续需要升级，切换至 {next_model}")
                    self.upgrade_counter = 0
                    return next_model
                else:
                    print("需要升级，但已经使用最高级模型")
                    self.upgrade_counter = 0
        
        elif need_downgrade:
            self.downgrade_counter += 1
            self.upgrade_counter = 0
            print(f"可能需要降级: {self.downgrade_counter}/{self.stability_threshold}")
            
            if self.downgrade_counter >= self.stability_threshold:
                if current_idx > 0:
                    next_model = self.available_models[current_idx - 1]
                    print(f"持续需要降级，切换至 {next_model}")
                    self.downgrade_counter = 0
                    return next_model
                else:
                    print("需要降级，但已经使用最轻量级模型")
                    self.downgrade_counter = 0
        
        else:
            # 状态正常，重置计数器
            self.upgrade_counter = 0
            self.downgrade_counter = 0
            print("目标数量和大小在正常范围内，保持当前模型")
        
        return current_model

class ProbabilisticStrategy(DummyStrategy):
    """基于概率的自适应策略
    
    根据队列长度的不同，以不同概率做出模型切换决策:
    1. 队列长度低于low阈值时，确定性升级
    2. 队列长度在low与high之间时，以线性增长概率降级
    3. 队列长度高于high阈值时，确定性降级
    """
    
    def __init__(self, available_models, queue_thresholds):
        super().__init__(available_models, queue_thresholds)
        
        # 定义队列阈值
        self.queue_low = queue_thresholds['low']
        self.queue_high = (queue_thresholds['high'] + queue_thresholds['low']) // 2
    
    def select_model(self, stats, current_model):
        """根据队列长度和概率做出模型切换决策"""
        import random
        
        # 获取当前队列长度和模型索引
        queue_length = stats['queue_length']
        current_idx = self.available_models.index(current_model)
        
        # 打印当前状态
        print(f"当前队列长度: {queue_length}, 模型: {current_model}")
        
        # 队列长度低于low阈值，确定性升级
        if queue_length <= self.queue_low:
            if current_idx < len(self.available_models) - 1:
                next_model = self.available_models[current_idx + 1]
                print(f"队列长度 ({queue_length}) <= 低阈值 ({self.queue_low})，确定性升级至 {next_model}")
                return next_model
            else:
                print(f"队列长度较低，但已经使用最高级模型")
                return current_model
        
        # 队列长度高于high阈值，确定性降级
        elif queue_length >= self.queue_high:
            if current_idx > 0:
                next_model = self.available_models[current_idx - 1]
                print(f"队列长度 ({queue_length}) >= 高阈值 ({self.queue_high})，确定性降级至 {next_model}")
                return next_model
            else:
                print(f"队列长度较高，但已经使用最轻量级模型")
                return current_model
        
        # 队列长度在low和high之间，概率性降级
        else:
            # 计算降级概率 (线性增长)
            # 队列长度为low时概率为0，为high时概率为1
            downgrade_prob = (queue_length - self.queue_low) / (self.queue_high - self.queue_low)
            
            # 随机决策
            rand_value = random.random()
            print(f"队列长度在阈值区间内，降级概率: {downgrade_prob:.2f}, 随机值: {rand_value:.2f}")
            
            if rand_value < downgrade_prob:
                # 触发降级
                if current_idx > 0:
                    next_model = self.available_models[current_idx - 1]
                    print(f"概率性降级触发，切换至 {next_model}")
                    return next_model
                else:
                    print("概率性降级触发，但已经使用最轻量级模型")
                    return current_model
            else:
                # 保持当前模型
                print(f"概率性降级未触发，保持当前模型 {current_model}")
                return current_model

class StagnantQueueStrategy(DummyStrategy):
    """检测持续稳定但积压的队列
    
    连续观察队列长度变化情况，如果队列长度保持稳定但始终有积压的任务，
    则在连续观察到三次队列没有显著下降后强制降级。
    """
    
    def __init__(self, available_models, queue_thresholds):
        super().__init__(available_models, queue_thresholds)
        
        # 初始化队列历史记录
        self.queue_history = []
        
        # 设置检测参数
        self.history_length = 3  # 需要记录的历史长度
        self.stagnation_threshold = 0.2  # 队列变化阈值（低于此比例视为稳定）
        self.minimum_queue_concern = queue_thresholds['low']  # 低于此阈值的队列不视为问题
        
    def select_model(self, stats, current_model):
        """根据队列稳定性做出模型切换决策"""
        current_idx = self.available_models.index(current_model)
        queue_length = stats['queue_length']
        
        # 记录当前队列长度
        self.queue_history.append(queue_length)
        
        # 仅保留最近的几次记录
        if len(self.queue_history) > self.history_length:
            self.queue_history.pop(0)
            
        # 输出队列历史
        print(f"队列历史记录: {self.queue_history}")
        
        # 如果历史记录不足，无法判断趋势
        if len(self.queue_history) < self.history_length:
            print("历史记录不足，继续观察")
            return current_model
        
        # 如果当前队列长度低于关注阈值，不需要干预
        if queue_length < self.minimum_queue_concern:
            print(f"队列长度 ({queue_length}) 低于关注阈值 ({self.minimum_queue_concern})，无需干预")
            return current_model
        
        # 计算队列变化率
        changes = []
        for i in range(1, len(self.queue_history)):
            prev = self.queue_history[i-1]
            curr = self.queue_history[i]
            if prev > 0:  # 避免除零
                change = (curr - prev) / prev
                changes.append(change)
        
        print(f"队列变化率: {[f'{c:.2f}' for c in changes]}")
        
        # 检查队列是否稳定（所有变化都低于阈值）
        is_stagnant = all(abs(change) < self.stagnation_threshold for change in changes)
        
        # 检查是否为持续积压（队列稳定且都为正值）
        if is_stagnant and queue_length > self.minimum_queue_concern:
            print(f"检测到持续稳定积压队列 ({queue_length})，连续 {self.history_length} 次观察无显著变化")
            
            # 如果有更轻量级的模型可用，则降级
            if current_idx > 0:
                next_model = self.available_models[current_idx - 1]
                print(f"强制降级模型从 {current_model} 到 {next_model}")
                # 清空历史记录，避免连续多次降级
                self.queue_history = []
                return next_model
            else:
                print("已经使用最轻量级模型，无法进一步降级")
        else:
            print("队列变化正常或无持续积压")
            
        return current_model
                
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
        self.decision_interval = 1  # 决策周期(秒)
        
        # 可用模型列表
        self.available_models = ['n', 's', 'm', 'l', 'x']
        
        # 队列阈值参数
        self.queue_max_length = config.get_queue_max_length()
        self.queue_low_threshold_length = config.get_queue_low_threshold_length()
        self.queue_high_threshold_length = config.get_queue_high_threshold_length()
        
        # 初始化策略参数
        queue_thresholds = {
            'low': self.queue_low_threshold_length,
            'high': self.queue_high_threshold_length,
            'max': self.queue_max_length
        }
        
        # 使用占位符策略
        self.strategies = {
            'strategy1': QueueLengthStrategy(self.available_models, queue_thresholds),
            'strategy2': LatencyEstimationStrategy(self.available_models, queue_thresholds),
            'strategy3': DistributionBasedStrategy(self.available_models, queue_thresholds),
            'strategy4': ProbabilisticStrategy(self.available_models, queue_thresholds),
            'strategy5': StagnantQueueStrategy(self.available_models, queue_thresholds)
        }
        
        # 当前使用的模型
        self.current_model = None
        
        # 元策略的Thompson Sampling参数
        self.context_dimension = 14
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
            float(self.model_to_idx.get(stats['cur_model_index'], 0)),
            float(stats['fps']) 
        ])
        
        feature_names = [
            'queue_length_norm', 'processing_latency', 'total_latency',
            'target_nums_norm', 
            'avg_confidence', 'std_confidence', 'avg_size', 'std_size',
            'brightness_norm', 'contrast_norm', 'entropy_norm', 'model_accuracy_norm',
            'model_index', 'fps'
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
        queue_ratio = stats['queue_length'] / self.queue_low_threshold_length
        
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
                    model = strategy.select_model(stats, self.current_model)
                    strategy_suggestions[name] = model
                    print(f"{name.capitalize()} strategy suggests model: {model}")
                
                # 执行选中策略的建议
                selected_model = strategy_suggestions[selected_strategy]
                
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