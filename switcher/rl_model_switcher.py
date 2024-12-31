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
import copy
from threading import Lock
import traceback

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_switcher.log'),
        logging.StreamHandler()
    ]
)

# 定义经验回放的数据结构
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class MovingAverage:
    """实现移动平均来平滑指标"""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.sum = 0
        self.lock = Lock()

    def update(self, value):
        with self.lock:
            if len(self.values) == self.window_size:
                self.sum -= self.values[0]
            self.values.append(value)
            self.sum += value

    def get(self):
        with self.lock:
            if not self.values:
                return 0
            return self.sum / len(self.values)

class LSTMDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, num_layers=1):
        super(LSTMDQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # 批归一化层
        self.input_norm = nn.BatchNorm1d(input_dim)
    
    def forward(self, x):
        # 确保输入维度正确 [batch_size, seq_len, features]
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        
        # 对输入进行归一化
        batch_size, seq_len, features = x.size()
        x_reshaped = x.view(-1, features)
        x_normalized = self.input_norm(x_reshaped)
        x = x_normalized.view(batch_size, seq_len, features)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 获取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 通过全连接层
        output = self.fc(last_output)
        return output

class StatsCollector:
    def __init__(self, http_url, sampling_interval=1):
        self.http_url = http_url
        self.sampling_interval = sampling_interval
        self.stats_queue = deque(maxlen=10)
        self._stop_event = threading.Event()
        self.thread = None
        self.lock = Lock()
        self.session = requests.Session()
        self.retry_count = 3
        self.retry_delay = 1
        
    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self._stop_event.clear()
            self.thread = threading.Thread(target=self._collect_loop, name="StatsCollector")
            self.thread.daemon = True
            self.thread.start()
        
    def stop(self):
        self._stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                logging.warning("Stats collector thread did not stop gracefully")
        self.session.close()
            
    def _collect_loop(self):
        logging.info("Stats collector started")
        consecutive_errors = 0
        
        while not self._stop_event.is_set():
            try:
                for attempt in range(self.retry_count):
                    try:
                        response = self.session.get(
                            f"{self.http_url}/get_stats",
                            timeout=5
                        )
                        response.raise_for_status()
                        stats = response.json().get('stats')
                        
                        if stats:
                            with self.lock:
                                self.stats_queue.append({
                                    'timestamp': time.time(),
                                    'latency': float(stats['latency']),
                                    'accuracy': float(stats['accuracy']),
                                    'queue_length': int(stats.get('queue_length', 0)),
                                    'model_name': stats.get('model_name', '')
                                })
                            consecutive_errors = 0
                            break
                    except Exception as e:
                        if attempt < self.retry_count - 1:
                            time.sleep(self.retry_delay)
                        else:
                            raise e
                            
            except Exception as e:
                consecutive_errors += 1
                logging.error(f"Error collecting stats: {e}")
                if consecutive_errors >= 5:
                    logging.error("Too many consecutive errors, stopping collector")
                    self._stop_event.set()
                    break
            
            time.sleep(self.sampling_interval)
    
    def get_recent_stats(self, window_size):
        with self.lock:
            stats = list(self.stats_queue)[-window_size:] if self.stats_queue else []
            return copy.deepcopy(stats)

class RLModelSwitcher:
    def __init__(self):
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
        self.window_size = 5
        self.input_dim = 3  # [latency, accuracy, queue_length]
        self.hidden_dim = 64
        self.n_actions = len(self.models_by_performance)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型和优化器
        self._create_networks()
        
        # 性能指标跟踪
        self.avg_reward = MovingAverage(100)
        self.avg_loss = MovingAverage(100)
        
        # 训练参数
        self.batch_size = 16
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.target_update = 10
        self.decision_interval = 30
        self.max_gradient_norm = 1.0
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        self.min_memory_size = 16  
        
        # 训练状态
        self.steps_done = 0
        self.train_iteration = 0
        self.last_save_time = time.time()
        self.save_interval = 3600
        
        # 模型切换控制
        self.last_switch_time = 0
        self.min_switch_interval = 30
        self.switch_cooldown = 10
        self.switch_lock = Lock()
        
        # Socket.IO事件设置
        self.setup_socket_events()

    def _create_networks(self):
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
        
        try:
            checkpoint = torch.load('model_switcher_checkpoint.pth')
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
            logging.info("Loaded previous model checkpoint")
        except Exception as e:
            logging.warning(f"No previous model found or loading failed: {e}")
    
    def setup_socket_events(self):
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
        for attempt in range(3):
            try:
                logging.info(f"Connecting to {self.processing_server_url}")
                self.sio.connect(self.processing_server_url, wait_timeout=10)
                return True
            except Exception as e:
                logging.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
        return False

    def get_state(self):
        recent_stats = self.stats_collector.get_recent_stats(self.window_size)
        
        if not recent_stats:
            return torch.zeros(self.window_size, self.input_dim, device=self.device)
        
        sequence = []
        for stats in recent_stats:
            sequence.append([
                stats['latency'],
                stats['accuracy'],
                stats['queue_length']
            ])
            
        while len(sequence) < self.window_size:
            if sequence:
                sequence.append(sequence[-1])
            else:
                sequence.append([0, 0, 0])
            
        sequence = np.array(sequence, dtype=np.float32)
        if len(sequence) > 0:
            # 使用更稳定的归一化方法
            mean = np.mean(sequence, axis=0, keepdims=True)
            std = np.std(sequence, axis=0, keepdims=True) + 1e-8
            sequence = (sequence - mean) / std
            
        return torch.tensor(sequence, dtype=torch.float32, device=self.device)

    def get_reward(self, current_stats, prev_stats):
        if not current_stats or not prev_stats:
            return torch.tensor([[-0.1]], device=self.device)
            
        try:
            eps = 1e-8
            
            # 计算相对变化
            latency_change = (prev_stats['latency'] - current_stats['latency']) / (prev_stats['latency'] + eps)
            accuracy_change = (current_stats['accuracy'] - prev_stats['accuracy']) / (prev_stats['accuracy'] + eps)
            queue_change = -(current_stats['queue_length'] - prev_stats['queue_length']) / (max(prev_stats['queue_length'], 1))
            
            # 归一化并限制范围
            latency_score = np.clip(latency_change, -1, 1)
            accuracy_score = np.clip(accuracy_change, -1, 1)
            queue_score = np.clip(queue_change, -1, 1)
            
            # 加权计算奖励
            reward = (
                0.4 * latency_score +
                0.4 * accuracy_score +
                0.2 * queue_score
            )
            
            # 模型切换惩罚
            if current_stats['model_name'] != prev_stats['model_name']:
                reward = reward - 0.1
                
            reward = float(np.clip(reward, -1, 1))
            return torch.tensor([[reward]], dtype=torch.float32, device=self.device)
            
        except Exception as e:
            logging.error(f"Error calculating reward: {e}\n{traceback.format_exc()}")
            return torch.tensor([[-0.1]], device=self.device)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state = state.unsqueeze(0) if len(state.size()) == 2 else state
            q_values = self.policy_net(state)
            return q_values.squeeze(0).max(0)[1].item()

    def can_switch_model(self):
        current_time = time.time()
        with self.switch_lock:
            if current_time - self.last_switch_time < self.min_switch_interval:
                return False
            return True

    def switch_model(self, model_name):
        with self.switch_lock:
            try:
                if not self.sio.connected and not self.connect_to_server():
                    return False
                current_time = time.time()
                if current_time - self.last_switch_time < self.min_switch_interval:
                    return False
                    
                self.sio.emit('switch_model', {'model_name': model_name})
                self.last_switch_time = current_time
                time.sleep(self.switch_cooldown)
                return True
            except Exception as e:
                logging.error(f"Error switching model: {e}")
                return False

    def optimize_model(self):
        """训练LSTM-DQN网络"""
        if len(self.memory) < self.min_memory_size:
            return
            
        try:
            transitions = random.sample(self.memory, self.batch_size)
            batch = Transition(*zip(*transitions))
            
            # 将数据转移到正确的设备并确保维度正确
            state_batch = torch.stack(batch.state).to(self.device)  # [batch_size, seq_len, features]
            action_batch = torch.tensor(batch.action, device=self.device)  # [batch_size]
            reward_batch = torch.cat([r for r in batch.reward]).to(self.device)  # [batch_size]
            next_state_batch = torch.stack(batch.next_state).to(self.device)  # [batch_size, seq_len, features]
            
            # 计算当前Q值
            current_q_values = self.policy_net(state_batch)  # [batch_size, n_actions]
            state_action_values = current_q_values.gather(1, action_batch.unsqueeze(1))  # [batch_size, 1]

            # 计算下一状态的最大Q值
            with torch.no_grad():
                next_q_values = self.target_net(next_state_batch)  # [batch_size, n_actions]
                next_state_values = next_q_values.max(1)[0]  # [batch_size]
                expected_state_action_values = (next_state_values * self.gamma) + reward_batch  # [batch_size]
                expected_state_action_values = expected_state_action_values.unsqueeze(1)  # [batch_size, 1]

            # 计算损失
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values)
            
            # 优化模型
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_gradient_norm)
            self.optimizer.step()
            
            # 更新性能指标
            self.avg_loss.update(loss.item())
            
            # 定期保存模型
            current_time = time.time()
            if current_time - self.last_save_time > self.save_interval:
                self.save_model()
                self.last_save_time = current_time
            
        except Exception as e:
            logging.error(f"Error in optimize_model: {e}\n{traceback.format_exc()}")

    def save_model(self):
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),  # 保存target网络状态
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps_done': self.steps_done
            }, 'model_switcher_checkpoint.pth')
            logging.info("Model saved successfully")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

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
        
        try:
            while True:
                try:
                    # 获取当前状态序列
                    current_state = self.get_state()
                    current_stats = self.stats_collector.get_recent_stats(1)
                    if not current_stats:
                        time.sleep(1)
                        continue
                    current_stats = current_stats[-1]
                    
                    if current_stats:
                        logging.info(
                            f"\nCurrent stats: Accuracy={current_stats['accuracy']:.1f} mAP, "
                            f"Latency={current_stats['latency']:.3f}s, "
                            f"Queue={current_stats['queue_length']}"
                        )
                        
                        if prev_stats is not None:
                            reward = self.get_reward(current_stats, prev_stats)
                            self.avg_reward.update(reward.item())
                            
                            # 存储经验
                            if prev_state is not None:
                                self.memory.append(Transition(prev_state, prev_action, reward, current_state))
                            
                            # 训练模型
                            if len(self.memory) >= self.min_memory_size:
                                self.optimize_model()
                                
                                if self.steps_done % self.target_update == 0:
                                    self.target_net.load_state_dict(self.policy_net.state_dict())
                            
                            logging.info(
                                f"Reward: {reward.item():.3f}, "
                                f"Epsilon: {self.epsilon:.3f}, "
                                f"Avg Reward: {self.avg_reward.get():.3f}, "
                                f"Avg Loss: {self.avg_loss.get():.3f}"
                            )
                        
                        # 选择动作
                        if self.can_switch_model():
                            action = self.choose_action(current_state)
                            next_model = self.models_by_performance[action]
                            
                            if next_model != current_stats.get('model_name'):
                                logging.info(f"Switching to yolov5{next_model}")
                                self.switch_model(next_model)
                            else:
                                logging.info("Keeping current model")
                            
                            prev_action = action
                        
                        prev_stats = current_stats.copy()
                        prev_state = current_state
                        
                        self.steps_done += 1
                        if self.epsilon > self.epsilon_min:
                            self.epsilon *= self.epsilon_decay
                    
                    time.sleep(self.decision_interval)
                    
                except Exception as e:
                    logging.error(f"Error in main loop iteration: {e}\n{traceback.format_exc()}")
                    time.sleep(self.decision_interval)
                
        except KeyboardInterrupt:
            logging.info("\nStopping RL model switcher...")
            self.save_model()
            self.stats_collector.stop()
            self.sio.disconnect()
        except Exception as e:
            logging.error(f"Fatal error in switching loop: {e}\n{traceback.format_exc()}")
            self.stats_collector.stop()
            self.sio.disconnect()

if __name__ == '__main__':
    try:
        switcher = RLModelSwitcher()
        switcher.adaptive_switch_loop()
    except Exception as e:
        logging.error(f"Fatal error: {e}\n{traceback.format_exc()}")