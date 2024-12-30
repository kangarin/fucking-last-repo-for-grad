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
import threading
from queue import Queue
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)

# 定义经验回放的数据结构
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class LSTMDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, num_layers=1):
        super(LSTMDQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x):
        # Ensure input has correct shape [batch_size, seq_len, input_dim]
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

class StatsCollector:
    def __init__(self, http_url, sampling_interval=1):
        self.http_url = http_url
        self.sampling_interval = sampling_interval
        self.stats_queue = deque(maxlen=10)  # 存储最近的统计数据
        self._stop_event = threading.Event()
        self.thread = None
        
    def start(self):
        """启动数据收集线程"""
        self.thread = threading.Thread(target=self._collect_loop, name="StatsCollector")
        self.thread.daemon = True  # 设为守护线程，主线程结束时自动结束
        self.thread.start()
        
    def stop(self):
        """停止数据收集"""
        self._stop_event.set()
        if self.thread:
            self.thread.join()
            
    def _collect_loop(self):
        """持续收集统计数据的循环"""
        logging.info("Stats collector started")
        while not self._stop_event.is_set():
            try:
                response = requests.get(f"{self.http_url}/get_stats")
                if response.status_code == 200:
                    stats = response.json().get('stats')
                    if stats:
                        self.stats_queue.append({
                            'timestamp': time.time(),
                            'latency': stats['latency'],
                            'accuracy': stats['accuracy'],
                            'model_name': stats.get('model_name', '')  # 添加模型名称
                        })
                        logging.debug(f"Collected stats: {stats}")
            except Exception as e:
                logging.error(f"Error collecting stats: {e}")
            
            time.sleep(self.sampling_interval)
    
    def get_recent_stats(self, window_size):
        """获取最近的统计数据序列"""
        return list(self.stats_queue)[-window_size:] if self.stats_queue else []

class RLModelSwitcher:
    def __init__(self):
        # Socket.IO setup
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        
        models_config = config.get_models_config()
        self.allowed_models = models_config['allowed_sizes']
        
        # 启动统计数据收集器
        self.stats_collector = StatsCollector(self.http_url, sampling_interval=1)
        
        # RL parameters
        self.models_by_performance = ['n', 's', 'm', 'l', 'x']
        self.window_size = 5  # 时间序列长度
        self.input_dim = 2  # [latency, accuracy]
        self.hidden_dim = 32
        self.n_actions = len(self.models_by_performance)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # LSTM-DQN networks
        self.policy_net = LSTMDQN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.n_actions
        ).to(self.device)
        
        self.target_net = LSTMDQN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.n_actions
        ).to(self.device)
        
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
        self.decision_interval = 10  # 决策间隔(秒)
        
        # Training stats
        self.steps_done = 0
        
        # Setup Socket.IO events
        self.setup_socket_events()
    
    def setup_socket_events(self):
        """设置Socket.IO事件处理"""
        @self.sio.event
        def connect():
            logging.info(f"Connected to processing server: {self.processing_server_url}")
            
        @self.sio.event
        def connect_error(data):
            logging.error(f"Connection error: {data}")
            
        @self.sio.event
        def disconnect():
            logging.info("Disconnected from processing server")
            
        @self.sio.on('model_switched')
        def on_model_switched(data):
            logging.info(f"Model successfully switched to: {data['model_name']}")
            
        @self.sio.on('error')
        def on_error(data):
            logging.error(f"Error: {data['message']}")

    def connect_to_server(self):
        """连接到服务器"""
        try:
            logging.info(f"Connecting to {self.processing_server_url}")
            self.sio.connect(self.processing_server_url, wait_timeout=10)
            return True
        except Exception as e:
            logging.error(f"Failed to connect to server: {e}")
            return False

    def switch_model(self, model_name):
        """切换模型"""
        try:
            if not self.sio.connected and not self.connect_to_server():
                return False
            self.sio.emit('switch_model', {'model_name': model_name})
            return True
        except Exception as e:
            logging.error(f"Error switching model: {e}")
            return False

    def get_state(self):
        """从收集器获取状态序列"""
        recent_stats = self.stats_collector.get_recent_stats(self.window_size)
        
        if not recent_stats:
            return torch.zeros(self.window_size, self.input_dim, device=self.device)
        
        # 准备序列数据
        sequence = []
        for stats in recent_stats:
            sequence.append([stats['latency'], stats['accuracy']])
            
        # 填充序列到固定长度
        while len(sequence) < self.window_size:
            sequence.append(sequence[-1] if sequence else [0, 0])
            
        return torch.tensor(sequence, dtype=torch.float32, device=self.device)

    def get_reward(self, current_stats, prev_stats):
        """计算奖励值"""
        if not current_stats or not prev_stats:
            return torch.tensor([[-1.0]], device=self.device)
            
        # 安全地计算相对变化
        try:
            # 防止除零，如果数值太小就用一个很小的数代替
            eps = 1e-10
            
            if abs(prev_stats['latency']) < eps:
                latency_rel_change = 0.0
            else:
                latency_rel_change = (prev_stats['latency'] - current_stats['latency']) / max(prev_stats['latency'], eps)
            
            if abs(prev_stats['accuracy']) < eps:
                accuracy_rel_change = 0.0
            else:
                accuracy_rel_change = (current_stats['accuracy'] - prev_stats['accuracy']) / max(prev_stats['accuracy'], eps)
            
            # 归一化到[-1, 1]范围
            norm_latency_change = np.tanh(latency_rel_change * 5)
            norm_accuracy_change = np.tanh(accuracy_rel_change * 5)
            
            # 权重
            latency_weight = 0.4
            accuracy_weight = 0.6
            
            reward = (latency_weight * norm_latency_change + 
                    accuracy_weight * norm_accuracy_change)
            
        except Exception as e:
            logging.error(f"Error calculating reward: {e}")
            logging.error(f"Current stats: {current_stats}")
            logging.error(f"Previous stats: {prev_stats}")
            reward = -1.0  # 发生错误时返回负奖励
                    
        return torch.tensor([[reward]], dtype=torch.float32, device=self.device)

    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            # 修改：确保状态张量维度正确
            state = state.unsqueeze(0) if len(state.size()) == 2 else state
            return self.policy_net(state).squeeze(0).max(0)[1].item()

    def optimize_model(self):
        """训练LSTM-DQN网络"""
        if len(self.memory) < self.batch_size:
            return
            
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action, device=self.device)
        reward_batch = torch.cat([r.view(1) for r in batch.reward])  # 修改：正确处理reward维度
        next_state_batch = torch.stack(batch.next_state)
        
        # 修改：确保action_batch维度正确
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # 修改：确保张量维度匹配
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        criterion = nn.SmoothL1Loss()
        # 修改：确保张量维度匹配
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def adaptive_switch_loop(self):
        """主循环"""
        if not self.connect_to_server():
            logging.error("Failed to connect to processing server")
            return
            
        logging.info(f"Starting RL-based adaptive model switching loop on {self.device}")
        
        # 启动统计数据收集器
        self.stats_collector.start()
        
        prev_stats = None
        prev_state = None
        prev_action = None
        episode_rewards = []
        
        try:
            while True:
                # 获取当前状态序列
                current_state = self.get_state()
                current_stats = self.stats_collector.get_recent_stats(1)[-1] if self.stats_collector.get_recent_stats(1) else None
                
                if current_stats:
                    logging.info(f"\nCurrent stats: Accuracy={current_stats['accuracy']:.1f} mAP, "
                              f"Latency={current_stats['latency']:.3f}s")
                    
                    if prev_stats is not None:
                        reward = self.get_reward(current_stats, prev_stats)
                        episode_rewards.append(reward.item())
                        self.memory.append(Transition(prev_state, prev_action, reward, current_state))
                        self.optimize_model()
                        
                        if self.steps_done % self.target_update == 0:
                            self.target_net.load_state_dict(self.policy_net.state_dict())
                        
                        logging.info(f"Reward: {reward.item():.3f}, Epsilon: {self.epsilon:.3f}, "  # 修改：使用item()获取标量值
                                  f"Avg Reward: {np.mean(episode_rewards[-100:]):.3f}")
                    
                    action = self.choose_action(current_state)
                    next_model = self.models_by_performance[action]
                    
                    if next_model != current_stats.get('model_name'):
                        logging.info(f"Switching to yolov5{next_model}")
                        self.switch_model(next_model)
                    else:
                        logging.info("Keeping current model")
                    
                    prev_stats = current_stats.copy()
                    prev_state = current_state
                    prev_action = action
                    
                    self.steps_done += 1
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay
                
                time.sleep(self.decision_interval)
                
        except KeyboardInterrupt:
            logging.info("\nStopping RL model switcher...")
            # 保存模型
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps_done': self.steps_done
            }, 'model_switcher_checkpoint.pth')
            # 停止数据收集
            self.stats_collector.stop()
            self.sio.disconnect()
        except Exception as e:
            logging.error(f"Error in switching loop: {e}")
            self.stats_collector.stop()
            time.sleep(self.decision_interval)

if __name__ == '__main__':
    switcher = RLModelSwitcher()
    switcher.adaptive_switch_loop()