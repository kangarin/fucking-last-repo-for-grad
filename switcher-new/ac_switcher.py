import time
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from socketio import Client
from pathlib import Path
import sys
from pprint import pprint
from collections import deque, namedtuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

# Define experience tuple for storing transitions
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'log_prob', 'value'))

class LSTMActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMActorCritic, self).__init__()
        
        # 分离的LSTM层，一个用于Actor，一个用于Critic
        self.actor_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.critic_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Actor网络（策略）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络（价值函数）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 隐藏状态初始化
        self.actor_hidden = None
        self.critic_hidden = None
        
    def reset_hidden(self, batch_size=1):
        """重置LSTM的隐藏状态"""
        device = next(self.parameters()).device
        self.actor_hidden = (
            torch.zeros(1, batch_size, self.actor_lstm.hidden_size).to(device),
            torch.zeros(1, batch_size, self.actor_lstm.hidden_size).to(device)
        )
        self.critic_hidden = (
            torch.zeros(1, batch_size, self.critic_lstm.hidden_size).to(device),
            torch.zeros(1, batch_size, self.critic_lstm.hidden_size).to(device)
        )
        
    def forward(self, x, reset_hidden=False):
        """前向传播"""
        batch_size = x.size(0)
        
        # 重置隐藏状态（如果需要）
        if reset_hidden or self.actor_hidden is None or self.critic_hidden is None:
            self.reset_hidden(batch_size)
        
        # 检查隐藏状态batch_size是否与输入匹配
        if self.actor_hidden[0].size(1) != batch_size:
            self.reset_hidden(batch_size)
            
        # Actor LSTM前向传播
        actor_lstm_out, self.actor_hidden = self.actor_lstm(x, self.actor_hidden)
        actor_features = actor_lstm_out[:, -1, :]  # 取LSTM最后一个时间步的输出
        
        # Critic LSTM前向传播
        critic_lstm_out, self.critic_hidden = self.critic_lstm(x, self.critic_hidden)
        critic_features = critic_lstm_out[:, -1, :]  # 取LSTM最后一个时间步的输出
        
        # 计算动作概率和状态价值
        action_probs = self.actor(actor_features)
        state_value = self.critic(critic_features)
        
        return action_probs, state_value
    
    def act(self, state, exploration_rate=0.0):
        """根据当前状态选择动作"""
        # 确保隐藏状态与当前输入batch匹配
        self.reset_hidden(batch_size=state.size(0))
        
        # 前向传播获取动作概率和状态价值
        action_probs, state_value = self(state)
        
        # 应用探索 - 有时选择随机动作
        if np.random.random() < exploration_rate:
            action_idx = torch.randint(0, action_probs.size(-1), (1,)).item()
            log_prob = torch.log(action_probs[0, action_idx])
            return action_idx, log_prob, state_value
        
        # 创建一个分类分布并从中采样
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # 返回动作索引、对数概率和状态价值
        return action.item(), dist.log_prob(action), state_value


class LSTMActorCriticModelSwitcher:
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
        self.state_history_length = 10  # LSTM的时间步长
        
        # 可用模型列表，按性能从低到高排序
        self.available_models = ['n', 's', 'm', 'l', 'x']
        self.model_to_idx = {model: idx for idx, model in enumerate(self.available_models)}
        self.idx_to_model = {idx: model for idx, model in enumerate(self.available_models)}
        
        # 队列阈值参数
        self.queue_max_length = config.get_queue_max_length()
        self.queue_low_threshold_length = config.get_queue_low_threshold_length()
        self.queue_high_threshold_length = config.get_queue_high_threshold_length()
        
        # 当前使用的模型
        self.current_model = None
        self.previous_action = None
        self.previous_state = None
        self.previous_log_prob = None
        self.previous_value = None
        
        # 探索增强参数
        self.exploration_rate = 0.2  # 初始探索率
        self.min_exploration_rate = 0.05  # 最小探索率
        self.exploration_decay = 0.999  # 探索率衰减因子
        
        # 设置强制探索计数器
        self.force_exploration_count = 10  # 每10次决策强制探索一次
        self.exploration_counter = 0
        
        # 特征数据归一化参数
        self.feature_means = None
        self.feature_stds = None
        
        # 经验回放设置
        self.replay_buffer = deque(maxlen=1000)  # 增大缓冲区容量
        self.min_samples_before_update = 8  # 开始训练前的最小样本数
        self.batch_size = 8  # 批处理大小
        self.update_frequency = 1  # 每4个决策周期进行一次网络更新
        self.update_counter = 0  # 更新计数器
        
        # 模型网络参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 12  # 统计数据中的特征数量
        self.hidden_dim = 32  # 隐藏层维度
        self.output_dim = len(self.available_models)  # 动作空间大小
        
        # 初始化LSTM Actor-Critic网络
        self.network = LSTMActorCritic(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        
        # 使用分离的优化器
        self.actor_optimizer = optim.Adam([p for n, p in self.network.named_parameters() 
                                          if 'actor' in n or 'actor_lstm' in n], lr=0.001)
        self.critic_optimizer = optim.Adam([p for n, p in self.network.named_parameters() 
                                           if 'critic' in n or 'critic_lstm' in n], lr=0.001)
        
        # Actor-Critic超参数
        self.gamma = 0.99  # 折扣因子
        self.entropy_beta = 0.01  # 熵正则化系数
        self.critic_loss_coef = 0.5  # 价值函数损失系数
        
        # 统计信息
        self.episodes = 0
        self.training_steps = 0
        
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
            
    def get_interval_stats(self, nums=10, interval=1.0):
        """获取一系列时间间隔的系统状态，用于LSTM输入"""
        try:
            # 使用 API 接口获取间隔统计数据
            response = requests.get(f"{self.http_url}/get_interval_stats?nums={nums}&interval={interval}")
            if response.status_code == 200:
                data = response.json()
                if data:
                    # 过滤掉None值
                    valid_data = [item for item in data if item is not None]
                    if len(valid_data) < 1:
                        print("No valid interval stats received")
                        return None
                    print(f"Got {len(valid_data)} valid interval stats")
                    return valid_data
                else:
                    print("Empty interval stats received")
                    return None
            else:
                print(f"Failed to get interval stats. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting interval stats: {e}")
            return None
    
    def preprocess_stats(self, stats_list):
        """预处理统计数据用于网络输入"""
        if not stats_list:
            return None
            
        # 提取特征
        features = []
        for stats in stats_list:
            # 当前状态特征提取并归一化
            # 确保所有特征都是浮点数并且进行适当归一化
            feature = [
                float(stats['queue_length']) / self.queue_high_threshold_length,  # 队列长度的归一化值
                float(stats['processing_latency']),
                float(stats['target_nums']) / 10.0,  # 假设平均目标数量不超过10
                float(stats['avg_confidence']),  # 已经是0-1之间
                float(stats['std_confidence']),  # 已经是较小值
                float(stats['avg_size']),  # 已经是0-1之间
                float(stats['std_size']),  # 已经是较小值
                float(stats['brightness']) / 255.0,  # 亮度归一化到0-1
                float(stats['contrast']) / 255.0,  # 对比度归一化到0-1
                float(stats['entropy']) / 10.0,  # 熵通常在0-10之间
                float(stats['cur_model_accuracy']) / 100.0,  # 准确率归一化到0-1
                float(self.model_to_idx.get(stats['cur_model_index'], 0))
            ]
            features.append(feature)
            
        # 将特征转换为numpy数组
        features = np.array(features, dtype=np.float32)
        
        # # 如果是第一次运行，初始化特征归一化参数
        # if self.feature_means is None or self.feature_stds is None:
        #     self.feature_means = np.mean(features, axis=0)
        #     self.feature_stds = np.std(features, axis=0)
        #     # 防止除零错误，将标准差为0的特征设为1.0
        #     self.feature_stds[self.feature_stds == 0] = 1.0
            
        # # 特征归一化
        # normalized_features = (features - self.feature_means) / self.feature_stds

        normalized_features = features
        
        # 确保形状正确 (batch_size, sequence_length, features)
        if len(normalized_features.shape) == 2:
            normalized_features = np.expand_dims(normalized_features, axis=0)
            
        # 转换为PyTorch张量
        state_tensor = torch.FloatTensor(normalized_features).to(self.device)
        
        return state_tensor
    
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
            
    def update_network(self, batch_size=32):
        """使用经验回放缓冲区中的样本批量更新LSTM Actor-Critic网络"""
        # 如果缓冲区中的样本不足，则跳过更新
        if len(self.replay_buffer) < batch_size:
            print(f"Skip update: Not enough samples in replay buffer ({len(self.replay_buffer)}/{batch_size})")
            return
            
        # 从缓冲区中随机采样一批经验
        transitions = self.sample_from_replay_buffer(batch_size)
        
        # 验证样本有效性并过滤无效样本
        valid_transitions = []
        for t in transitions:
            if t.log_prob is not None and isinstance(t.log_prob, torch.Tensor):
                valid_transitions.append(t)
        
        if not valid_transitions:
            print("No valid transitions to update network")
            return
            
        # 提取批数据
        batch_size = len(valid_transitions)
        states = torch.cat([t.state for t in valid_transitions], dim=0)
        actions = torch.tensor([t.action for t in valid_transitions], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t.reward for t in valid_transitions], dtype=torch.float32).view(-1, 1).to(self.device)
        
        # 处理next_states (有些可能为None)
        next_states_list = []
        masks = []
        for t in valid_transitions:
            if t.next_state is not None:
                next_states_list.append(t.next_state)
                masks.append(1.0)
            else:
                # 如果next_state为None，使用零张量代替
                next_states_list.append(torch.zeros_like(t.state))
                masks.append(0.0)
        
        next_states = torch.cat(next_states_list, dim=0)
        masks = torch.tensor(masks, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # 更新Critic网络
        self.critic_optimizer.zero_grad()
        
        # 计算当前状态的价值
        self.network.reset_hidden(batch_size)
        _, current_values = self.network(states)
        
        # 计算下一个状态的价值
        self.network.reset_hidden(batch_size)
        _, next_values = self.network(next_states)
        next_values = next_values.detach()  # 停止梯度
        
        # 计算目标值和critic损失
        target_values = rewards + self.gamma * next_values * masks
        critic_loss = F.mse_loss(current_values, target_values)
        
        # 更新critic
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # 更新Actor网络
        self.actor_optimizer.zero_grad()
        
        # 重新计算策略
        self.network.reset_hidden(batch_size)
        action_probs, values = self.network(states)
        
        # 计算优势函数，使用当前网络的值
        advantages = (target_values - values).detach()
        
        # 创建一个分类分布
        dist = Categorical(action_probs)
        
        # 计算所选动作的对数概率
        action_log_probs = dist.log_prob(actions)
        
        # 计算actor损失
        actor_loss = -(action_log_probs * advantages.squeeze()).mean()
        
        # 更新actor
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # 重置LSTM隐藏状态，准备下一次预测
        self.network.reset_hidden(batch_size=1)
        
        # 更新统计信息
        self.training_steps += 1
        
        print(f"Network updated with {batch_size} samples in batch - Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
    
    def add_to_replay_buffer(self, transition):
        """将经验添加到回放缓冲区"""
        self.replay_buffer.append(transition)
    
    def sample_from_replay_buffer(self, batch_size=32):
        """从回放缓冲区采样经验
        
        Args:
            batch_size: 采样的样本数量
            
        Returns:
            list: 包含batch_size个Transition元组的列表
        """
        if len(self.replay_buffer) < batch_size:
            return []
        
        # 随机采样不放回
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        return [self.replay_buffer[i] for i in indices]
    
    def handle_emergency(self):
        """处理紧急情况，如队列长度超过最大阈值"""
        # 紧急情况下选择最轻量的模型
        selected_model = self.available_models[0]
        print(f"EMERGENCY: Queue length exceeded maximum. Selecting lightest model: yolov5{selected_model}")
        return selected_model
        
    def select_action(self, state):
        """选择动作（模型）"""
        # 确保隐藏状态与当前输入batch匹配
        self.network.reset_hidden(batch_size=state.size(0))
        
        # 检查是否是强制探索回合
        self.exploration_counter += 1
        force_exploration = self.exploration_counter >= self.force_exploration_count
        
        # 如果是强制探索回合，重置计数器并随机选择
        if force_exploration:
            self.exploration_counter = 0
            action_idx = np.random.randint(0, len(self.available_models))
            selected_model = self.idx_to_model[action_idx]
            print(f"FORCED EXPLORATION: Randomly selected model: yolov5{selected_model}")
            
            # 使用网络前向传播获取log_prob和value，但不使用其action
            action_probs, state_value = self.network(state)
            log_prob = torch.log(action_probs[0, action_idx])
            
            return action_idx, selected_model, log_prob, state_value
        
        # 使用网络策略选择动作
        action_idx, log_prob, state_value = self.network.act(state, exploration_rate=self.exploration_rate)
        
        selected_model = self.idx_to_model[action_idx]
        
        # 衰减探索率
        self.exploration_rate = max(self.exploration_rate * self.exploration_decay, 
                                    self.min_exploration_rate)
        
        print(f"Selected action: {action_idx}, Model: yolov5{selected_model}")
        return action_idx, selected_model, log_prob, state_value
    
    def save_model(self, path):
        """保存模型到文件"""
        try:
            torch.save({
                'model_state_dict': self.network.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'exploration_rate': self.exploration_rate,
                'feature_means': self.feature_means,
                'feature_stds': self.feature_stds,
                'episodes': self.episodes,
                'training_steps': self.training_steps
            }, path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, path):
        """从文件加载模型"""
        try:
            if not Path(path).exists():
                print(f"Model file not found: {path}")
                return False
                
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            
            if 'actor_optimizer_state_dict' in checkpoint and 'critic_optimizer_state_dict' in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            elif 'optimizer_state_dict' in checkpoint:
                # 向后兼容
                print("Warning: Loading from older model format with single optimizer")
                
            self.exploration_rate = checkpoint.get('exploration_rate', self.exploration_rate)
            self.feature_means = checkpoint.get('feature_means', None)
            self.feature_stds = checkpoint.get('feature_stds', None)
            self.episodes = checkpoint.get('episodes', 0)
            self.training_steps = checkpoint.get('training_steps', 0)
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def lstm_actor_critic_loop(self):
        """基于LSTM Actor-Critic进行模型切换的主循环"""
        if not self.connect_to_server():
            print("Failed to connect to processing server")
            return
            
        print(f"Starting LSTM Actor-Critic model switching loop with {self.decision_interval} second interval")
        print(f"Queue thresholds - Low: {self.queue_low_threshold_length}, High: {self.queue_high_threshold_length}, Max: {self.queue_max_length}")
        
        # 尝试加载预训练模型
        model_path = Path(project_root) / "models" / "lstm_ac_model_switcher.pt"
        self.load_model(model_path)
        
        # 主循环
        while True:
            try:
                # 1. 获取当前状态序列 (用于LSTM)
                stats_list = self.get_interval_stats(nums=self.state_history_length, interval=1.0)
                
                # 检查是否需要紧急响应 (基于最新状态)
                current_stats = self.get_current_stats()
                emergency_mode = False
                
                if current_stats:
                    # 从统计数据中获取当前模型，如果尚未设置
                    if self.current_model is None and 'cur_model_index' in current_stats:
                        self.current_model = current_stats['cur_model_index']
                        print(f"Initial model detected: yolov5{self.current_model}")
                        
                    # 检查队列紧急情况
                    if current_stats['queue_length'] >= self.queue_high_threshold_length:
                        emergency_mode = True
                        selected_model = self.handle_emergency()
                        
                        # 切换到紧急模型
                        if selected_model != self.current_model:
                            self.switch_model(selected_model)
                            self.current_model = selected_model
                            
                        # 跳过本次决策，等待队列恢复
                        print("Waiting for queue to recover...")
                        time.sleep(self.decision_interval)
                        continue
                else:
                    print("Failed to get current stats, skipping this cycle")
                    time.sleep(self.decision_interval)
                    continue
                
                # 预处理状态序列
                if stats_list:
                    current_state = self.preprocess_stats(stats_list)
                    
                    # 2. 处理上一个动作的奖励和经验（如果有）
                    if self.previous_action is not None and self.previous_state is not None:
                        # 计算奖励
                        reward = self.calculate_reward(current_stats)
                        
                        # 创建经验元组
                        transition = Transition(
                            state=self.previous_state,
                            action=self.previous_action,
                            next_state=current_state,
                            reward=reward,
                            log_prob=self.previous_log_prob,
                            value=self.previous_value
                        )
                        
                        # 添加到经验回放缓冲区
                        self.add_to_replay_buffer(transition)
                        
                        # 计数器递增，并在达到更新频率时进行网络更新
                        self.update_counter += 1
                        if self.update_counter >= self.update_frequency and len(self.replay_buffer) >= self.min_samples_before_update:
                            self.update_network(self.batch_size)
                            self.update_counter = 0
                    
                    # 3. 选择新动作
                    action_idx, selected_model, log_prob, value = self.select_action(current_state)
                    
                    # 4. 执行动作（切换模型）
                    if selected_model != self.current_model:
                        success = self.switch_model(selected_model)
                        if success:
                            self.current_model = selected_model
                        else:
                            print(f"Failed to switch to model: {selected_model}")
                    else:
                        print(f"Keeping current model: yolov5{selected_model}")
                    
                    # 5. 保存当前状态和动作以便下次使用
                    self.previous_state = current_state
                    self.previous_action = action_idx
                    self.previous_log_prob = log_prob
                    self.previous_value = value
                    
                    # 增加episode计数
                    self.episodes += 1
                    
                    # 定期保存模型
                    if self.episodes % 50 == 0:
                        self.save_model(model_path)
                else:
                    print("No valid state sequence available")
                
                # 等待下一个决策周期
                time.sleep(self.decision_interval)
                
            except KeyboardInterrupt:
                print("\nStopping model switcher...")
                # 保存模型
                self.save_model(model_path)
                self.sio.disconnect()
                break
            except Exception as e:
                print(f"Error in switching loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(self.decision_interval)

if __name__ == '__main__':
    switcher = LSTMActorCriticModelSwitcher()
    # 启动LSTM Actor-Critic切换循环
    switcher.lstm_actor_critic_loop()