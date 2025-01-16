import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import requests
from socketio import Client
from pathlib import Path
import sys
import logging
import math
from torch.nn import functional as F
import csv
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatsTracker:
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f'training_stats_{timestamp}.csv'
        
        self.loss_history = []
        self.reward_history = []
        self.print_interval = 100
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Loss', 'Reward', 'Avg_Loss_100', 'Avg_Reward_100'])
            
    def add_stats(self, step, loss, reward):
        self.loss_history.append(loss)
        self.reward_history.append(reward)
        
        if step % self.print_interval == 0:
            avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0
            avg_reward = np.mean(self.reward_history[-100:]) if self.reward_history else 0
            
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, loss, reward, avg_loss, avg_reward])
            
            logger.info(
                f"\nTraining Statistics at step {step}:\n"
                f"Average Loss (last 100): {avg_loss:.4f}\n"
                f"Average Reward (last 100): {avg_reward:.4f}\n"
            )

class GlobalStateBuffer:
    def __init__(self, max_duration=60.0):
        self.max_duration = max_duration
        self.buffer = deque()
        
    def add_state(self, state):
        state['timestamp'] = time.time()
        self.buffer.append(state)
        self._cleanup_old_states()
        
    def _cleanup_old_states(self):
        current_time = time.time()
        while self.buffer and current_time - self.buffer[0]['timestamp'] > self.max_duration:
            self.buffer.popleft()
    
    def get_reward_window(self, duration):
        current_time = time.time()
        states = [
            state for state in self.buffer 
            if current_time - state['timestamp'] <= duration
        ]
        return sorted(states, key=lambda x: x['timestamp']) if states else None

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 特征提取层
        self.input_ln = nn.LayerNorm(input_size)
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 3)
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(0.1),
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
        x = self.input_ln(x)
        shared_features = self.shared(x)
        
        action_logits = self.actor(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        state_value = self.critic(shared_features)
        
        return action_probs, state_value

class A2CAgent:
    def __init__(self, state_buffer, observation_duration=10.0, decision_duration=10.0):
        self.state_buffer = state_buffer
        self.observation_duration = observation_duration
        self.decision_duration = decision_duration
        self.model_levels = ['n', 's', 'm', 'l', 'x']
        self.actions = [1, 0, -1]  # 升档、保持、降档
        
        # 网络参数
        self.feature_size = 16  # 11 + 5(one-hot)
        self.hidden_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化网络
        self.network = ActorCritic(
            self.feature_size, 
            self.hidden_size
        ).to(self.device)
        
        # 训练参数
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.steps = 0
        
        # 探索参数
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay = 0.999
        self.epsilon = self.eps_start
        
        # 保存相关
        self.save_interval = 100
        self.model_save_dir = Path('saved_models')
        self.model_save_dir.mkdir(exist_ok=True)
        
        # 统计跟踪器
        self.stats_tracker = StatsTracker()
        
        # 保存最近一次的动作信息
        self.last_action_info = None
        
        # 获取配置
        self.queue_max_length = config.get_queue_max_length()
        self.queue_low_threshold_length = config.get_queue_low_threshold_length()
        self.queue_high_threshold_length = config.get_queue_high_threshold_length()

    def normalize_state(self, state):
        try:
            # 基础特征归一化
            base_features = {
                'accuracy': state['accuracy'] / 100.0,
                'latency': state['latency'],
                'processing_latency': state['processing_latency'],
                'queue_length': state['queue_length'] / self.queue_max_length,
                'avg_confidence': state['avg_confidence'],
                'avg_size': min(1.0, state['avg_size'] / 200.0),
                'brightness': state['brightness'] / 255.0,
                'contrast': min(1.0, state['contrast'] / 100.0),
                'entropy': state['entropy'] / 10.0,
                'total_targets': state['total_targets'] / 10.0,
                'target_fps': state['target_fps']
            }
            
            # 添加模型的one-hot编码
            current_model = state['model_name']
            model_idx = self.model_levels.index(current_model)
            one_hot = [0] * len(self.model_levels)
            one_hot[model_idx] = 1
            
            # 合并所有特征
            all_features = list(base_features.values()) + one_hot
            return torch.FloatTensor(all_features).to(self.device)
        except Exception as e:
            logger.error(f"Error normalizing state: {e}")
            return None

    def compute_reward(self, states, action_idx):
        """使用时间加权的方式计算奖励"""
        if not states or len(states) < 2:
            return 0.0
            
        try:
            # 计算时间权重
            timestamps = [s['timestamp'] for s in states]
            total_duration = timestamps[-1] - timestamps[0]
            if total_duration <= 0:
                return 0.0

            weighted_metrics = {
                'accuracy': 0,
                'latency': 0,
                'queue_length': 0,
                'avg_confidence': 0
            }
            total_weight = 0

            # 使用时间间隔作为权重计算加权平均
            for i in range(len(states) - 1):
                duration = timestamps[i+1] - timestamps[i]
                weight = duration / total_duration
                
                for metric in weighted_metrics:
                    weighted_metrics[metric] += states[i][metric] * weight
                total_weight += weight

            # 计算加权平均值
            avg_metrics = {
                metric: value / total_weight 
                for metric, value in weighted_metrics.items()
            }

            # 计算基础奖励
            queue_ratio = avg_metrics['queue_length'] / self.queue_high_threshold_length
            w1 = max(1 - queue_ratio, 0)  # 准确率权重
            w2 = queue_ratio  # 延迟权重

            reward = w1 * (avg_metrics['accuracy']/100.0 + avg_metrics['avg_confidence']) - \
                    w2 * (avg_metrics['latency'] / self.queue_low_threshold_length)

            # 检查非法动作
            current_model = states[0]['model_name']  # 使用初始状态检查
            current_idx = self.model_levels.index(current_model)
            action = self.actions[action_idx]
            
            if (action == 1 and current_idx == len(self.model_levels)-1) or \
               (action == -1 and current_idx == 0):
                illegal_penalty = -0.5
                logger.info(f"Applied illegal action penalty: {illegal_penalty}")
                reward += illegal_penalty
            
            logger.info(f"""Reward breakdown:
                Queue Length: {avg_metrics['queue_length']:.1f} (Ratio: {queue_ratio:.2f})
                Weights: accuracy={w1:.2f}, latency={w2:.2f}
                Accuracy: {avg_metrics['accuracy']:.3f}
                Latency: {avg_metrics['latency']:.3f}s
                Final Reward: {reward:.3f}
                """)
            
            return float(reward)
                
        except Exception as e:
            logger.error(f"Error computing reward: {e}")
            return 0.0
        
    def select_action(self, current_stats):
        """选择动作并转换为实际模型"""
        state = self.normalize_state(current_stats)
        if state is None:
            return random.choice(self.model_levels)
                
        # 如果有上一次的动作信息，进行策略更新
        if self.last_action_info is not None:
            self._update_policy(state)
        
        # 更新epsilon
        self.epsilon = max(self.eps_end, self.eps_start * (self.eps_decay ** self.steps))
        
        # Epsilon-greedy策略
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            action = torch.tensor([[random.randint(0, 2)]], device=self.device)
            action_probs, current_value = self.network(state.unsqueeze(0))
            log_prob = torch.log(action_probs[0, action[0]] + 1e-10)
            logger.info(f"Exploring with epsilon = {self.epsilon:.3f}")
        else:
            # 利用：使用策略网络选择动作
            with torch.no_grad():
                action_probs, current_value = self.network(state.unsqueeze(0))
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                logger.info(f"Exploiting with epsilon = {self.epsilon:.3f}")
        
        # 将动作转换为实际的模型选择
        current_model = current_stats['model_name']
        current_idx = self.model_levels.index(current_model)
        action_value = self.actions[action.item()]
        
        # 根据动作调整模型等级
        if action_value == 1:  # 升档
            next_idx = min(current_idx + 1, len(self.model_levels)-1)
        elif action_value == -1:  # 降档
            next_idx = max(current_idx - 1, 0)
        else:  # 保持
            next_idx = current_idx
        
        selected_model = self.model_levels[next_idx]
        
        # 记录当前动作信息
        self.last_action_info = {
            'state': state.unsqueeze(0),
            'action': action,
            'log_prob': log_prob,
            'value': current_value,
            'action_probs': action_probs
        }
            
        # 记录探索统计信息
        if self.steps % 100 == 0:
            logger.info(f"""
            Action Selection Stats:
            - Step: {self.steps}
            - Epsilon: {self.epsilon:.3f}
            - Mode: {"Exploration" if random.random() < self.epsilon else "Exploitation"}
            - Current Model: {current_model}
            - Selected Model: {selected_model}
            - Action Value: {action_value}
            """)
            
        return selected_model
    
    def _update_policy(self, current_state):
        """使用当前状态更新上一个动作的策略"""
        if not self.last_action_info:
            return
                
        reward_states = self.state_buffer.get_reward_window(self.decision_duration)
        if not reward_states:
            return
                
        reward = self.compute_reward(reward_states, self.last_action_info['action'].item())
        reward_tensor = torch.tensor([[reward]], device=self.device)
            
        # 计算新的状态值
        with torch.set_grad_enabled(True):
            _, next_value = self.network(current_state.unsqueeze(0))
            
            # 计算TD误差
            td_error = reward_tensor + self.gamma * next_value - self.last_action_info['value']
            
            # 计算策略损失和价值损失
            policy_loss = -self.last_action_info['log_prob'] * td_error.detach()
            value_loss = 0.5 * td_error.pow(2)
            loss = policy_loss + value_loss
            
            # 进行反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        self.steps += 1
        self.stats_tracker.add_stats(
            step=self.steps,
            loss=loss.item(),
            reward=reward
        )
            
        if self.steps % self.save_interval == 0:
            self.save_model(self.steps)
            
        logger.info(f"""
        Step: {self.steps}
        Action: {self.actions[self.last_action_info['action'].item()]}
        Reward: {reward:.3f}
        TD Error: {td_error.item():.3f}
        Loss: {loss.item():.3f}
        Epsilon: {self.epsilon:.3f}
        """)

    def save_model(self, step):
        """保存模型到文件"""
        try:
            model_path = self.model_save_dir / f'model_step_{step}.pt'
            torch.save({
                'step': step,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

class ModelSwitcher:
    def __init__(self, stats_update_interval=0.5):
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        self.stats_update_interval = stats_update_interval
        
        # 初始化状态管理和A2C代理
        self.state_buffer = GlobalStateBuffer(max_duration=60.0)
        self.agent = A2CAgent(self.state_buffer, 
                            observation_duration=10.0,
                            decision_duration=10.0)
        
        # 决策相关参数
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
        """主循环"""
        if not self.connect_to_server():
            logger.error("Failed to connect to processing server")
            return
            
        logger.info("Starting adaptive model switching loop...")
        
        while True:
            try:
                # 1. 获取当前状态
                current_stats = self.get_current_stats()
                
                if current_stats:
                    # 2. 添加到状态缓存
                    self.state_buffer.add_state(current_stats)
                    
                    # 3. 检查是否需要做出新的决策
                    current_time = time.time()
                    if current_time - self.last_decision_time >= self.agent.decision_duration:
                        logger.info(f"\nCurrent stats: Accuracy={current_stats['accuracy']:.1f} mAP, "
                                  f"Latency={current_stats['latency']:.3f}s, "
                                  f"Queue={current_stats['queue_length']}, "
                                  f"Model=yolov5{current_stats['model_name']}")
                        
                        # 4. 选择新的模型
                        next_model = self.agent.select_action(current_stats)
                        
                        # 5. 如果模型发生变化，进行切换
                        if next_model != current_stats['model_name']:
                            logger.info(f"Switching model from {current_stats['model_name']} to {next_model}")
                            self.switch_model(next_model)
                        else:
                            logger.info("Keeping current model")
                        
                        self.last_decision_time = current_time
                
                # 6. 休眠一段时间
                time.sleep(self.stats_update_interval)
                
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