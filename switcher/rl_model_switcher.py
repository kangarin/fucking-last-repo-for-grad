import time
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from socketio import Client
from collections import deque, namedtuple
import random
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

# 定义经验回放的数据结构
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class RLModelSwitcher:
    def __init__(self):
        # Socket.IO setup - 保持与原始代码一致
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        
        models_config = config.get_models_config()
        self.allowed_models = models_config['allowed_sizes']
        
        # RL parameters
        self.models_by_performance = ['n', 's', 'm', 'l', 'x']
        self.state_dim = 2  # [latency, accuracy]
        self.n_actions = len(self.models_by_performance)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN networks
        self.policy_net = DQN(self.state_dim, self.n_actions).to(self.device)
        self.target_net = DQN(self.state_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.target_update = 10
        self.decision_interval = 10  # seconds
        
        # Training stats
        self.steps_done = 0
        
        # Setup Socket.IO events
        self.setup_socket_events()
    
    def setup_socket_events(self):
        """完全保持与原始代码一致的Socket.IO事件设置"""
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
            
        @self.sio.on('error')
        def on_error(data):
            print(f"Error: {data['message']}")

    def connect_to_server(self):
        """完全保持与原始代码一致的服务器连接方法"""
        try:
            print(f"Connecting to {self.processing_server_url}")
            self.sio.connect(self.processing_server_url, wait_timeout=10)
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False

    def get_current_stats(self):
        """完全保持与原始代码一致的获取状态方法"""
        try:
            response = requests.get(f"{self.http_url}/get_stats")
            if response.status_code == 200:
                return response.json().get('stats')
            return None
        except Exception as e:
            print(f"Error getting stats: {e}")
            return None

    def switch_model(self, model_name):
        """完全保持与原始代码一致的模型切换方法"""
        try:
            if not self.sio.connected and not self.connect_to_server():
                return False
            self.sio.emit('switch_model', {'model_name': model_name})
            return True
        except Exception as e:
            print(f"Error switching model: {e}")
            return False

    def get_state(self, stats):
        """将当前统计数据转换为状态向量"""
        if not stats:
            return torch.zeros(self.state_dim, device=self.device)
        return torch.tensor([stats['latency'], stats['accuracy']], 
                          dtype=torch.float32, 
                          device=self.device)

    def get_reward(self, stats, prev_stats):
        """计算归一化的奖励值"""
        if not stats or not prev_stats:
            return torch.tensor(-1.0, device=self.device)
            
        # 计算相对变化而不是绝对变化
        latency_rel_change = (prev_stats['latency'] - stats['latency']) / prev_stats['latency']
        accuracy_rel_change = (stats['accuracy'] - prev_stats['accuracy']) / prev_stats['accuracy']
        
        # 使用 tanh 将两个指标都归一化到 [-1, 1] 范围
        norm_latency_change = np.tanh(latency_rel_change * 5)  # *5 使得变化更敏感
        norm_accuracy_change = np.tanh(accuracy_rel_change * 5)
        
        # 权重
        latency_weight = 0.4
        accuracy_weight = 0.6
        
        # 计算综合奖励
        reward = (latency_weight * norm_latency_change + 
                accuracy_weight * norm_accuracy_change)
                
        # # 添加基础惩罚项，当延迟过高时
        # if stats['latency'] > 0.5:  # 延迟阈值
        #     reward -= 0.5  # 固定惩罚
        
        return torch.tensor(reward, dtype=torch.float32, device=self.device)

    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        sample = random.random()
        if sample < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            return self.policy_net(state).max(0)[1].item()

    def optimize_model(self):
        """训练DQN网络"""
        if len(self.memory) < self.batch_size:
            return
            
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action, device=self.device)
        reward_batch = torch.stack(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def adaptive_switch_loop(self):
        """主循环-保持与原始代码相似的结构"""
        if not self.connect_to_server():
            print("Failed to connect to processing server")
            return
            
        print(f"Starting RL-based adaptive model switching loop on {self.device}")
        
        prev_stats = None
        prev_state = None
        prev_action = None
        episode_rewards = []
        
        while True:
            try:
                current_stats = self.get_current_stats()
                if current_stats:
                    print(f"\nCurrent stats: Accuracy={current_stats['accuracy']:.1f} mAP, "
                          f"Latency={current_stats['latency']:.3f}s, "
                          f"Model=yolov5{current_stats['model_name']}")
                    
                    current_state = self.get_state(current_stats)
                    
                    if prev_stats is not None:
                        reward = self.get_reward(current_stats, prev_stats)
                        episode_rewards.append(reward.item())
                        self.memory.append(Transition(prev_state, prev_action, reward, current_state))
                        self.optimize_model()
                        
                        if self.steps_done % self.target_update == 0:
                            self.target_net.load_state_dict(self.policy_net.state_dict())
                        
                        print(f"Reward: {reward:.3f}, Epsilon: {self.epsilon:.3f}, "
                              f"Avg Reward: {np.mean(episode_rewards[-100:]):.3f}")
                    
                    action = self.choose_action(current_state)
                    next_model = self.models_by_performance[action]
                    
                    if next_model != current_stats['model_name']:
                        print(f"Switching to yolov5{next_model}")
                        self.switch_model(next_model)
                    else:
                        print("Keeping current model")
                    
                    prev_stats = current_stats.copy()
                    prev_state = current_state
                    prev_action = action
                    
                    self.steps_done += 1
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay
                
                time.sleep(self.decision_interval)
                
            except KeyboardInterrupt:
                print("\nStopping RL model switcher...")
                torch.save({
                    'policy_net_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': self.epsilon,
                    'steps_done': self.steps_done
                }, 'model_switcher_checkpoint.pth')
                self.sio.disconnect()
                break
            except Exception as e:
                print(f"Error in switching loop: {e}")
                time.sleep(self.decision_interval)

if __name__ == '__main__':
    switcher = RLModelSwitcher()
    switcher.adaptive_switch_loop()