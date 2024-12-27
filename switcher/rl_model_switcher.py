import time
import requests
import numpy as np
from socketio import Client
from collections import deque
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

class RLModelSwitcher:
    def __init__(self):
        # Socket.IO setup
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        
        models_config = config.get_models_config()
        self.allowed_models = models_config['allowed_sizes']
        
        # RL parameters
        self.models_by_performance = ['n', 's', 'm', 'l', 'x']
        self.n_states = 10  # 将延迟分成10个区间
        self.n_actions = len(self.models_by_performance)
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.decision_interval = 10  # 决策周期(秒)
        
        # Initialize Q-table: states × actions
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        # Experience replay
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Metrics history
        self.latency_history = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        
        # Setup Socket.IO events
        self.setup_socket_events()
    
    def setup_socket_events(self):
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
        try:
            print(f"Connecting to {self.processing_server_url}")
            self.sio.connect(self.processing_server_url, wait_timeout=10)
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False

    def get_current_stats(self):
        try:
            response = requests.get(f"{self.http_url}/get_stats")
            if response.status_code == 200:
                return response.json().get('stats')
            return None
        except Exception as e:
            print(f"Error getting stats: {e}")
            return None

    def switch_model(self, model_name):
        try:
            if not self.sio.connected and not self.connect_to_server():
                return False
            self.sio.emit('switch_model', {'model_name': model_name})
            return True
        except Exception as e:
            print(f"Error switching model: {e}")
            return False

    def get_state(self, stats):
        """将当前统计数据转换为状态"""
        if not stats:
            return 0
            
        latency = stats['latency']
        # 将延迟映射到状态空间 [0, n_states-1]
        state = int(min(latency * 10, self.n_states - 1))
        return state

    def get_reward(self, stats, prev_stats):
        """计算奖励"""
        if not stats or not prev_stats:
            return -1
            
        # 奖励计算考虑准确率和延迟的平衡
        accuracy_weight = 0.7
        latency_weight = 0.3
        
        # 准确率变化
        accuracy_change = stats['accuracy'] - prev_stats['accuracy']
        
        # 延迟变化（负值表示延迟降低）
        latency_change = stats['latency'] - prev_stats['latency']
        
        # 归一化延迟变化
        norm_latency_change = -np.tanh(latency_change)  # 将延迟减少映射为正奖励
        
        # 综合奖励
        reward = (accuracy_weight * accuracy_change + 
                 latency_weight * norm_latency_change)
                 
        return reward

    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        """更新Q表"""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-learning更新公式
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.discount_factor * next_max)
        
        self.q_table[state, action] = new_value

    def decay_epsilon(self):
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def adaptive_switch_loop(self):
        if not self.connect_to_server():
            print("Failed to connect to processing server")
            return
            
        print("Starting RL-based adaptive model switching loop")
        
        prev_stats = None
        prev_state = None
        prev_action = None
        
        while True:
            try:
                current_stats = self.get_current_stats()
                if not current_stats:
                    time.sleep(self.decision_interval)
                    continue
                    
                print(f"\nCurrent stats: Accuracy={current_stats['accuracy']:.1f} mAP, "
                      f"Latency={current_stats['latency']:.3f}s, "
                      f"Model=yolov5{current_stats['model_name']}")
                
                # 获取当前状态
                current_state = self.get_state(current_stats)
                
                if prev_stats is not None:
                    # 计算奖励并更新Q表
                    reward = self.get_reward(current_stats, prev_stats)
                    self.update_q_table(prev_state, prev_action, reward, current_state)
                    print(f"Reward: {reward:.3f}, Epsilon: {self.epsilon:.3f}")
                
                # 选择动作
                action = self.choose_action(current_state)
                next_model = self.models_by_performance[action]
                
                if next_model != current_stats['model_name']:
                    print(f"Switching to yolov5{next_model}")
                    self.switch_model(next_model)
                else:
                    print("Keeping current model")
                
                # 更新历史记录
                prev_stats = current_stats
                prev_state = current_state
                prev_action = action
                
                # 衰减探索率
                self.decay_epsilon()
                
                time.sleep(self.decision_interval)
                
            except KeyboardInterrupt:
                print("\nStopping RL model switcher...")
                self.sio.disconnect()
                break
            except Exception as e:
                print(f"Error in switching loop: {e}")
                time.sleep(self.decision_interval)

if __name__ == '__main__':
    switcher = RLModelSwitcher()
    # 启动强化学习自适应切换循环
    switcher.adaptive_switch_loop()