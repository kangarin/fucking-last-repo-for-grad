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

# 设置日志
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
            
            if len(self.loss_history) > 100:
                self.loss_history = self.loss_history[-100:]
                self.reward_history = self.reward_history[-100:]

class GlobalStateBuffer:
    def __init__(self, max_window_size=60):
        self.max_window_size = max_window_size
        self.buffer = deque(maxlen=max_window_size)
        
    def add_state(self, state):
        self.buffer.append(state)
    
    def get_observation(self, window_size=10):
        if len(self.buffer) < window_size:
            return None
        return list(self.buffer)[-window_size:]
    
    def get_reward_window(self, decision_interval=5):
        if len(self.buffer) < decision_interval:
            return None
        return list(self.buffer)[-decision_interval:]

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions, lstm_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 特征提取层
        self.input_ln = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, 
                           batch_first=True, dropout=0.1)
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_actions)
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        x = self.input_ln(x)
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]  # 取最后一个时间步
        
        # Actor: 输出动作概率分布
        action_probs = F.softmax(self.actor(features), dim=-1)
        
        # Critic: 输出状态值
        state_value = self.critic(features)
        
        return action_probs, state_value

class A2CAgent:
    def __init__(self, state_buffer, observation_window=10, decision_interval=5):
        self.state_buffer = state_buffer
        self.observation_window = observation_window
        self.decision_interval = decision_interval
        self.model_names = ['n', 's', 'm', 'l', 'x']
        
        # 网络参数
        self.feature_size = 11
        self.hidden_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化网络
        self.network = ActorCritic(
            self.feature_size, 
            self.hidden_size,
            len(self.model_names)
        ).to(self.device)
        
        # 训练参数
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.steps = 0
        
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
        self.queue_threshold_length = config.get_queue_threshold_length()

    def normalize_state(self, state):
        """归一化状态数据"""
        try:
            return {
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
        except Exception as e:
            logger.error(f"Error normalizing state: {e}")
            return None

    def state_to_tensor(self, states):
        """将状态转换为tensor"""
        if not states:
            return None
            
        if len(states) < self.observation_window:
            padding = [states[0]] * (self.observation_window - len(states))
            states = padding + states
        elif len(states) > self.observation_window:
            states = states[-self.observation_window:]
            
        sequence = []
        for state in states:
            norm_state = self.normalize_state(state)
            if norm_state is None:
                return None
            
            features = [
                norm_state['accuracy'],
                norm_state['latency'],
                norm_state['processing_latency'],
                norm_state['queue_length'],
                norm_state['avg_confidence'],
                norm_state['avg_size'],
                norm_state['brightness'],
                norm_state['contrast'],
                norm_state['entropy'],
                norm_state['total_targets'],
                norm_state['target_fps']
            ]
            sequence.append(features)
            
        return torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

    def compute_reward(self, states):
        """计算奖励"""
        if not states:
            return 0.0
        
        try:
            # 获取当前状态的关键指标
            avg_queue = np.mean([s['queue_length'] for s in states])
            avg_accuracy = np.mean([s['accuracy'] for s in states]) / 100.0
            avg_confidence = np.mean([s['avg_confidence'] for s in states])
            avg_latency = np.mean([s['latency'] for s in states])
            avg_processing_latency = np.mean([s['processing_latency'] for s in states])
            
            # 计算队列压力
            queue_pressure = avg_queue / self.queue_threshold_length
            
            # 结合准确率和置信度
            performance_score = 2.0 * max(avg_accuracy + avg_confidence - 0.5, 0.0)
            
            # 延迟分数
            latency_score = 1.0 - avg_latency
            processing_latency_score = - avg_processing_latency
            
            # 根据队列压力调整奖励计算
            if queue_pressure < 1.0:
                reward = performance_score * 0.9 + latency_score * 0.1
            else:
                reward = processing_latency_score * 10 * (avg_queue - self.queue_threshold_length) / self.queue_threshold_length
            
            # 裁剪奖励
            reward = max(min(reward, 3.0), -3.0)
            
            logger.info(f"""Reward breakdown:
                Queue Length: {avg_queue:.1f} (Pressure: {queue_pressure:.2f})
                Performance Score: {performance_score:.3f} (Accuracy: {avg_accuracy:.3f}, Confidence: {avg_confidence:.3f})
                Latency Score: {latency_score:.3f} (Latency: {avg_latency:.3f}s)
                Processing Latency Score: {processing_latency_score:.3f} (Processing Latency: {avg_processing_latency:.3f}s)
                Final Reward: {reward:.3f}
                """)
            
            return float(reward)
                
        except Exception as e:
            logger.error(f"Error computing reward: {e}")
            return 0.0

    def save_model(self, step):
        """保存模型"""
        try:
            save_path = self.model_save_dir / f'a2c_model_step_{step}.pth'
            torch.save({
                'step': step,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, save_path)
            logger.info(f"Model saved successfully at step {step}: {save_path}")
        except Exception as e:
            logger.error(f"Error saving model at step {step}: {e}")

    def select_action(self, current_stats):
        """选择动作并可能更新策略"""
        # 1. 获取当前状态
        current_observation = self.state_buffer.get_observation(self.observation_window)
        if not current_observation:
            return random.choice(self.model_names)
                
        current_state = self.state_to_tensor(current_observation)
        if current_state is None:
            return random.choice(self.model_names)
                
        # 2. 如果有上一次的动作信息，进行策略更新
        if self.last_action_info is not None:
            self._update_policy(current_state)
                
        # 3. 选择新动作
        # 注意：这里我们不使用 with torch.no_grad()，因为我们需要梯度
        action_probs, current_value = self.network(current_state)
                
        # 对动作采样时使用 no_grad
        with torch.no_grad():
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        selected_model = self.model_names[action.item()]
            
        # 4. 保存当前动作信息供下次更新使用
        self.last_action_info = {
            'state': current_state,
            'action': action,
            'log_prob': log_prob,
            'value': current_value,
            'action_probs': action_probs  # 保存动作概率分布
        }
            
        return selected_model

    def _update_policy(self, current_state):
        """使用当前状态更新上一个动作的策略"""
        # 确保我们有上一次的动作信息
        if not self.last_action_info:
            return
                
        # 1. 计算上一个动作的奖励
        reward_states = self.state_buffer.get_reward_window(self.decision_interval)
        if not reward_states:
            return
                
        reward = self.compute_reward(reward_states)
        reward_tensor = torch.tensor([[reward]], device=self.device)
            
        # 2. 获取当前状态的值估计
        _, next_value = self.network(current_state)
            
        # 3. 计算TD误差 (使用 detach 以避免梯度从 next_value 传播)
        td_error = reward_tensor + self.gamma * next_value.detach() - self.last_action_info['value']
            
        # 4. 计算 Actor (策略) 损失
        # 使用保存的 log_prob
        policy_loss = -self.last_action_info['log_prob'] * td_error.detach()
            
        # 5. 计算 Critic (值) 损失
        value_loss = 0.5 * td_error.pow(2)
            
        # 6. 计算总损失
        loss = policy_loss + value_loss
            
        # 7. 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
            
        # 8. 记录统计信息
        self.steps += 1
        self.stats_tracker.add_stats(
            step=self.steps,
            loss=loss.item(),
            reward=reward
        )
            
        # 9. 定期保存模型
        if self.steps % self.save_interval == 0:
            self.save_model(self.steps)
            
        logger.info(f"""
        Step: {self.steps}
        Previous Model: {self.model_names[self.last_action_info['action'].item()]}
        Reward: {reward:.3f}
        TD Error: {td_error.item():.3f}
        Loss: {loss.item():.3f}
        """)
        
class ModelSwitcher:
    def __init__(self, stats_update_interval=1.0):
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        self.stats_update_interval = stats_update_interval
        
        # 初始化状态管理和A2C代理
        self.state_buffer = GlobalStateBuffer()
        self.agent = A2CAgent(self.state_buffer)
        
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
                    if current_time - self.last_decision_time >= self.decision_interval:
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