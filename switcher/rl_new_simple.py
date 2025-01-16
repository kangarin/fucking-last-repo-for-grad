import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import time
import requests
from socketio import Client
from pathlib import Path
import sys
from datetime import datetime
import csv

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        
        # Actor网络 (输出3个动作的概率：升档、保持、降档)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, 3)
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        action_logits = self.actor(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        state_value = self.critic(shared_features)
        
        return action_probs, state_value

class SimpleAgent:
    def __init__(self, stats_update_interval=10.0):
        # 网络参数
        self.base_feature_size = 11
        self.model_levels = ['n', 's', 'm', 'l', 'x']
        self.feature_size = self.base_feature_size + len(self.model_levels)  # 基础特征 + one-hot编码
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(self.feature_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)
        
        # 训练参数
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_end = 0.1
        self.eps_decay = 0.999
        self.steps = 0
        
        # 动作空间
        self.model_levels = ['n', 's', 'm', 'l', 'x']
        self.actions = [1, 0, -1]  # 升档、保持、降档
        
        # 服务器连接
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        self.stats_update_interval = stats_update_interval
        
        # 配置参数
        self.queue_max_length = config.get_queue_max_length()
        self.queue_high_threshold = config.get_queue_high_threshold_length()
        self.queue_low_threshold = config.get_queue_low_threshold_length()
        
        # 统计和保存
        self.stats_file = Path('logs') / f'training_stats_{datetime.now():%Y%m%d_%H%M%S}.csv'
        self.stats_file.parent.mkdir(exist_ok=True)
        with open(self.stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Loss', 'Reward', 'Epsilon'])

    def normalize_state(self, state):
        """归一化状态特征并添加当前模型的one-hot编码"""
        # 基础特征归一化
        base_features = [
            state['accuracy'] / 100.0,
            state['latency'],
            state['processing_latency'],
            state['queue_length'] / self.queue_max_length,
            state['avg_confidence'],
            min(1.0, state['avg_size'] / 200.0),
            state['brightness'] / 255.0,
            min(1.0, state['contrast'] / 100.0),
            state['entropy'] / 10.0,
            state['total_targets'] / 10.0,
            state['target_fps']
        ]
        
        # 添加模型的one-hot编码
        current_model = state['model_name']
        model_idx = self.model_levels.index(current_model)
        one_hot = [1.0 if i == model_idx else 0.0 for i in range(len(self.model_levels))]
        
        # 合并所有特征
        features = base_features + one_hot
        return torch.FloatTensor(features).to(self.device)

    def compute_reward(self, state, action_value):
        """计算即时奖励"""
        queue_ratio = state['queue_length'] / self.queue_high_threshold
        w1 = max(1 - queue_ratio, 0)  # 准确率权重
        w2 = queue_ratio  # 延迟权重
        
        reward = w1 * (state['accuracy']/100.0 + state['avg_confidence']) - \
                w2 * (state['latency'] / self.queue_low_threshold)

        # 检查是否为非法动作
        current_idx = self.model_levels.index(state['model_name'])
        if (action_value == 1 and current_idx == len(self.model_levels)-1) or \
           (action_value == -1 and current_idx == 0):
            illegal_penalty = -0.5
            logger.info(f"Applied illegal action penalty: {illegal_penalty}")
            reward += illegal_penalty
                
        logger.info(f"""Reward calculation:
            Queue Length: {state['queue_length']:.1f} (Ratio: {queue_ratio:.2f})
            Weights: accuracy={w1:.2f}, latency={w2:.2f}
            Accuracy: {state['accuracy']:.1f}
            Confidence: {state['avg_confidence']:.2f}
            Latency: {state['latency']:.3f}s
            Final Reward: {reward:.3f}""")
        
        return reward

    def select_action(self, state):
        """选择动作"""
        features = self.normalize_state(state)
        
        # Epsilon-greedy策略
        if np.random.random() < self.epsilon:
            action = torch.tensor([[np.random.randint(0, 3)]], device=self.device)
            action_probs, _ = self.network(features.unsqueeze(0))
            log_prob = torch.log(action_probs[0, action[0]] + 1e-10)
        else:
            with torch.no_grad():
                action_probs, _ = self.network(features.unsqueeze(0))
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
        
        # 更新epsilon
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        
        return action, log_prob

    def update_policy(self, state, next_state, action, log_prob, reward):
        """更新策略"""
        # 准备数据
        state_tensor = self.normalize_state(state).unsqueeze(0)
        next_state_tensor = self.normalize_state(next_state).unsqueeze(0)
        reward_tensor = torch.tensor([[reward]], device=self.device)
        
        # 计算当前状态和下一状态的值
        _, current_value = self.network(state_tensor)
        with torch.no_grad():
            _, next_value = self.network(next_state_tensor)
        
        # 计算TD误差
        td_error = reward_tensor + self.gamma * next_value - current_value
        
        # 计算损失
        policy_loss = -log_prob * td_error.detach()
        value_loss = 0.5 * td_error.pow(2)
        loss = policy_loss + value_loss
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 记录统计信息
        self.steps += 1
        with open(self.stats_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.steps, loss.item(), reward, self.epsilon])
        
        logger.info(f"""
        Step: {self.steps}
        Action: {self.actions[action.item()]}
        Reward: {reward:.3f}
        Loss: {loss.item():.3f}
        Epsilon: {self.epsilon:.3f}
        """)

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
                return response.json().get('stats')
            logger.error(f"Failed to get stats. Status code: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return None

    def switch_model(self, model_name):
        """向处理服务器发送模型切换请求"""
        try:
            if not self.sio.connected and not self.connect_to_server():
                return False
            self.sio.emit('switch_model', {'model_name': model_name})
            return True
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False

    def convert_action_to_model(self, current_model, action):
        """将动作转换为目标模型"""
        current_idx = self.model_levels.index(current_model)
        action_value = self.actions[action.item()]
        
        if action_value == 1:  # 升档
            next_idx = min(current_idx + 1, len(self.model_levels)-1)
        elif action_value == -1:  # 降档
            next_idx = max(current_idx - 1, 0)
        else:  # 保持
            next_idx = current_idx
        
        return self.model_levels[next_idx]

    def run(self):
        """主循环"""
        if not self.connect_to_server():
            logger.error("Failed to connect to processing server")
            return
        
        logger.info("Starting simplified AC model switching loop...")
        
        last_state = None
        last_action = None
        last_log_prob = None
        
        while True:
            try:
                # 1. 获取当前状态
                current_state = self.get_current_stats()
                
                if current_state:
                    # 2. 如果有上一个动作的信息，更新策略
                    if last_state is not None and last_action is not None:
                        reward = self.compute_reward(current_state)
                        self.update_policy(last_state, current_state, 
                                        last_action, last_log_prob, reward)
                    
                    # 3. 选择新的动作
                    action, log_prob = self.select_action(current_state)
                    next_model = self.convert_action_to_model(current_state['model_name'], action)
                    
                    # 4. 如果需要切换模型，发送请求
                    if next_model != current_state['model_name']:
                        logger.info(f"Switching model from {current_state['model_name']} to {next_model}")
                        self.switch_model(next_model)
                    
                    # 5. 保存当前信息用于下次更新
                    last_state = current_state
                    last_action = action
                    last_log_prob = log_prob
                
                time.sleep(self.stats_update_interval)
                
            except KeyboardInterrupt:
                logger.info("\nStopping AC switcher...")
                self.sio.disconnect()
                break
            except Exception as e:
                logger.error(f"Error in switching loop: {e}")
                time.sleep(1)

if __name__ == '__main__':
    agent = SimpleAgent()
    agent.run()