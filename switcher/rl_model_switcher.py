import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import requests
from socketio import Client
from pathlib import Path
import sys
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state'])

class GlobalStateBuffer:
    def __init__(self, max_window_size=60):
        """维护一个固定大小的全局状态窗口"""
        self.max_window_size = max_window_size
        self.buffer = deque(maxlen=max_window_size)
        
    def add_state(self, state):
        """添加新状态到buffer"""
        self.buffer.append(state)
    
    def get_observation(self, window_size=10):
        """获取最近window_size个状态作为观测值"""
        if len(self.buffer) < window_size:
            return None
        return list(self.buffer)[-window_size:]
    
    def get_reward_window(self, decision_interval=5):
        """获取用于计算奖励的状态窗口"""
        if len(self.buffer) < decision_interval:
            return None
        return list(self.buffer)[-decision_interval:]

class LSTMDQN(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers=1, output_size=5):
        super(LSTMDQN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        self.input_bn = nn.BatchNorm1d(input_size)
        
        # LSTM层处理时间序列
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # 全连接层处理LSTM的输出
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(-1, self.input_size)
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        x = self.fc1(last_output)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append(Experience(state, action, reward, next_state))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_buffer, observation_window=10, decision_interval=5):
        self.state_buffer = state_buffer
        self.observation_window = observation_window
        self.decision_interval = decision_interval
        self.model_names = ['n', 's', 'm', 'l', 'x']
        self.model_to_idx = {model: idx for idx, model in enumerate(self.model_names)}
        
        # 网络参数
        self.feature_size = 7  # 每个状态的特征数
        self.hidden_size = 64
        self.output_size = len(self.model_names)
        
        # Training parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.learning_rate = 1e-4
        self.target_update = 10
        self.steps = 0
        
        # Initialize networks
        self.policy_net = LSTMDQN(self.feature_size, self.hidden_size, output_size=self.output_size).to(self.device)
        self.target_net = LSTMDQN(self.feature_size, self.hidden_size, output_size=self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(10000)
        
        self.current_model = 's'  # 默认模型

        self.queue_max_length = config.get_queue_max_length()
        self.queue_threshold_length = config.get_queue_threshold_length()
        
    def normalize_state(self, state):
        """Normalize individual state metrics to [0,1] range"""
        try:
            # 准确率归一化 (0-100 -> 0-1)
            norm_accuracy = state['accuracy'] / 100.0
            
            # 延迟
            norm_latency = state['latency']
            
            # 队列长度
            norm_queue = state['queue_length'] / 20.0  # 假设最大值为20
            
            # 置信度已经在0-1范围
            norm_confidence = state['avg_confidence']
            
            # 目标大小归一化 (假设最大值为200)
            norm_size = min(1.0, state['avg_size'] / 200.0)

            # 亮度和对比度归一化
            norm_brightness = state['brightness'] / 255.0  
            norm_contrast = min(1.0, state['contrast'] / 100.0)
            
            return {
                'accuracy': norm_accuracy,
                'latency': norm_latency,
                'queue_length': norm_queue,
                'avg_confidence': norm_confidence,
                'avg_size': norm_size,
                'brightness': norm_brightness,
                'contrast': norm_contrast
            }
        except Exception as e:
            logger.error(f"Error normalizing state: {e}")
            return None

    def state_to_tensor(self, states):
        """Convert states sequence to tensor"""
        if not states:
            return None
            
        # 确保状态序列长度正确
        if len(states) < self.observation_window:
            padding = [states[0]] * (self.observation_window - len(states))
            states = padding + states
        elif len(states) > self.observation_window:
            states = states[-self.observation_window:]
            
        # 构建序列数据
        sequence = []
        for state in states:
            norm_state = self.normalize_state(state)
            if norm_state is None:
                return None
            
            features = [
                norm_state['accuracy'],
                norm_state['latency'],
                norm_state['queue_length'],
                norm_state['avg_confidence'],
                norm_state['avg_size'],
                norm_state['brightness'],
                norm_state['contrast']
            ]
            sequence.append(features)
            
        # 转换为tensor，形状为(1, seq_len, feature_size)
        return torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
    
    def compute_reward(self, states):
            """
            动态权重的奖励函数：
            - 队列短时重视准确率和置信度
            - 队列长时重视延迟
            """
            if not states:
                return 0.0
            
            try:
                # 获取原始指标的平均值
                avg_queue = np.mean([s['queue_length'] for s in states])
                avg_accuracy = np.mean([s['accuracy'] for s in states]) / 100.0  # 归一化到0-1
                avg_confidence = np.mean([s['avg_confidence'] for s in states])
                avg_latency = np.mean([s['latency'] for s in states])
                
                # 计算队列压力系数 (0到1之间)
                queue_pressure = min(1.0, avg_queue / self.queue_max_length)
                
                # 根据队列压力动态分配权重
                # 队列压力低时: 准确率权重高，延迟权重低
                # 队列压力高时: 准确率权重低，延迟权重高
                accuracy_weight = 1.0 - queue_pressure
                latency_weight = queue_pressure
                
                # 1. 性能得分
                # 准确率和置信度得分
                accuracy_score = avg_accuracy * 0.5 + avg_confidence * 0.5
                # 延迟得分
                latency_score = np.exp(-avg_latency)
                
                # 2. 根据队列压力计算加权得分
                total_reward = (accuracy_weight * accuracy_score + 
                            latency_weight * latency_score)
                
                # 3. 额外的队列积压惩罚（当队列超过阈值时）
                queue_penalty = 0.0
                if avg_queue > self.queue_threshold_length:
                    queue_penalty = -0.2 * (avg_queue - self.queue_threshold_length)
                    total_reward += queue_penalty
                
                # 4. 额外的精度奖励（当队列很短时）
                accuracy_bonus = 0.0
                if avg_queue < self.queue_threshold_length and accuracy_score > 0.5:
                    accuracy_bonus_scale = self.queue_threshold_length - avg_queue
                    accuracy_bonus = (accuracy_score - 0.5) * accuracy_bonus_scale
                    total_reward += accuracy_bonus

                # 5. 裁剪最终奖励到合理范围
                total_reward = np.clip(total_reward, -2.0, 2.0)
                
                logger.info(f"""Reward breakdown:
                    Accuracy: {accuracy_score:.3f} (Weight: {accuracy_weight:.2f})
                    Latency: {latency_score:.3f} (Weight: {latency_weight:.2f})
                    Queue Penalty: {queue_penalty:.3f}
                    Accuracy Bonus: {accuracy_bonus:.3f}
                    Queue: {queue_pressure:.3f} (Avg Queue: {avg_queue:.1f})
                    Total Reward: {total_reward:.3f}""")
                
                return float(total_reward)
                    
            except Exception as e:
                logger.error(f"Error computing reward: {e}")
                return 0.0
    
    def select_action(self):
        """Select next model using epsilon-greedy policy"""
        observation = self.state_buffer.get_observation(self.observation_window)
        if not observation:
            return random.choice(self.model_names)
        
        state_tensor = self.state_to_tensor(observation)
        if state_tensor is None:
            return random.choice(self.model_names)
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].item()
                selected_model = self.model_names[action_idx]
        else:
            selected_model = random.choice(self.model_names)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Get reward for previous action if available
        reward_states = self.state_buffer.get_reward_window(self.decision_interval)
        if reward_states:
            reward = self.compute_reward(reward_states)
            logger.info(f"Reward: {reward:.3f}")
            if len(observation) >= self.observation_window:
                # Store experience in memory
                prev_observation = self.state_buffer.get_observation(self.observation_window)[:-1]
                prev_state_tensor = self.state_to_tensor(prev_observation)
                if prev_state_tensor is not None:
                    self.memory.push(
                        prev_state_tensor,
                        self.model_to_idx[self.current_model],
                        reward,
                        state_tensor
                    )
                    # Train the network
                    self.optimize_model()
        
        self.current_model = selected_model
        return selected_model
    
    def optimize_model(self):
        """Train the DQN"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Prepare batch tensors
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, device=self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        next_state_batch = torch.cat(batch.next_state)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + (self.gamma * next_q_values)
        
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        logger.info(f"Step: {self.steps}, Loss: {loss.item()}")
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

class ModelSwitcher:
    def __init__(self):
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        
        # 初始化状态管理和DQN代理
        self.state_buffer = GlobalStateBuffer()
        self.agent = DQNAgent(self.state_buffer)
        
        # 决策相关参数
        self.decision_interval = 5  # 决策周期(秒)
        self.last_decision_time = time.time()
        
        # 设置Socket.IO事件处理
        self.setup_socket_events()
    
    def setup_socket_events(self):
        """设置Socket.IO事件处理器"""
        @self.sio.event
        def connect():
            logger.info(f"Connected to processing server: {self.processing_server_url}")
            
        @self.sio.event
        def connect_error(data):
            logger.error(f"Connection error: {data}")
            
        @self.sio.event
        def disconnect():
            logger.info("Disconnected from processing server")
            
        @self.sio.on('model_switched')
        def on_model_switched(data):
            logger.info(f"Model successfully switched to: {data['model_name']}")
            
        @self.sio.on('error')
        def on_error(data):
            logger.error(f"Error: {data['message']}")

    def connect_to_server(self):
        """连接到处理服务器"""
        try:
            logger.info(f"Connecting to {self.processing_server_url}")
            self.sio.connect(self.processing_server_url, wait_timeout=10)
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False

    def get_current_stats(self):
        """获取当前系统状态"""
        try:
            response = requests.get(f"{self.http_url}/get_stats")
            if response.status_code == 200:
                data = response.json()
                return data.get('stats')
            else:
                logger.error(f"Failed to get stats. Status code: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
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
            logger.error(f"Error switching model: {e}")
            return False

    def adaptive_switch_loop(self):
        """使用LSTM-DQN进行自适应模型切换"""
        if not self.connect_to_server():
            logger.error("Failed to connect to processing server")
            return
            
        logger.info(f"Starting LSTM-DQN based model switching loop with {self.decision_interval} second interval")
        
        while True:
            try:
                # 获取当前状态
                current_time = time.time()
                stats = self.get_current_stats()
                
                if stats:
                    # 添加到状态缓存
                    self.state_buffer.add_state(stats)
                    
                    # 检查是否需要做出新的决策
                    if current_time - self.last_decision_time >= self.decision_interval:
                        logger.info(f"\nCurrent stats: Accuracy={stats['accuracy']:.1f} mAP, "
                                  f"Latency={stats['latency']:.3f}s, "
                                  f"Queue={stats['queue_length']}, "
                                  f"Model=yolov5{stats['model_name']}")
                        
                        # 选择新的模型
                        next_model = self.agent.select_action()
                        if next_model != stats['model_name']:
                            logger.info(f"Switching to yolov5{next_model}")
                            self.switch_model(next_model)
                        else:
                            logger.info("Keeping current model")
                            
                        self.last_decision_time = current_time
                
                # 短暂休眠避免过于频繁的查询
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                logger.info("\nStopping model switcher...")
                self.sio.disconnect()
                break
            except Exception as e:
                logger.error(f"Error in switching loop: {e}")
                time.sleep(1)

if __name__ == '__main__':
    try:
        # 设置随机种子以确保可复现性
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        switcher = ModelSwitcher()
        switcher.adaptive_switch_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)