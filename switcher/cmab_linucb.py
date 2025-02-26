import numpy as np
import logging
import time
import requests
from socketio import Client
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CMABSwitcher:
    def __init__(self, stats_update_interval=5.0):
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        self.stats_update_interval = stats_update_interval
        
        # CMAB参数
        self.feature_dim = 11  # 特征维度
        self.model_levels = ['n', 's', 'm', 'l', 'x']
        self.alpha = 1.0  # UCB exploration parameter
        self.decay_factor = 0.95  # 衰减因子
        
        # 为每个arm初始化参数
        self.A = {arm: np.eye(self.feature_dim) for arm in self.model_levels}
        self.b = {arm: np.zeros((self.feature_dim, 1)) for arm in self.model_levels}
        self.theta = {arm: np.zeros((self.feature_dim, 1)) for arm in self.model_levels}
        
        # 获取配置
        self.queue_max_length = config.get_queue_max_length()
        self.queue_high_threshold = config.get_queue_high_threshold_length()
        self.queue_low_threshold = config.get_queue_low_threshold_length()

        # 设置Socket事件处理器
        self.setup_socket_events()

    def normalize_state(self, state):
        """归一化状态特征"""
        features = [
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
        return np.array(features).reshape(-1, 1)

    def compute_reward(self, stats):
        """计算即时奖励"""
        queue_ratio = stats['queue_length'] / self.queue_high_threshold
        w1 = max(1 - queue_ratio, 0)  # 准确率权重
        w2 = queue_ratio  # 延迟权重
        
        reward = w1 * (stats['accuracy']/100.0 + stats['avg_confidence']) - \
                w2 * (stats['processing_latency'])
                
        logger.info(f"""Reward calculation:
            Queue Length: {stats['queue_length']:.1f} (Ratio: {queue_ratio:.2f})
            Weights: accuracy={w1:.2f}, latency={w2:.2f}
            Accuracy: {stats['accuracy']:.1f}
            Confidence: {stats['avg_confidence']:.2f}
            Latency: {stats['latency']:.3f}s
            Final Reward: {reward:.3f}""")
        
        return reward

    def select_model(self, features):
        """使用LinUCB选择最佳模型"""
        ucb_scores = {}
        
        for arm in self.model_levels:
            # 计算均值和不确定性
            mu = self.theta[arm].T @ features
            sigma = np.sqrt(features.T @ np.linalg.inv(self.A[arm]) @ features)
            ucb = float(mu + self.alpha * sigma)
            ucb_scores[arm] = ucb
            
            logger.debug(f"Model {arm}: mu={float(mu):.3f}, sigma={float(sigma):.3f}, UCB={ucb:.3f}")
        
        best_model = max(ucb_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"UCB scores: {ucb_scores}")
        return best_model

    def update_model(self, arm, features, reward):
        """更新模型参数，加入衰减机制"""
        # 应用衰减
        self.A[arm] = self.decay_factor * self.A[arm] + features @ features.T
        self.b[arm] = self.decay_factor * self.b[arm] + reward * features
        self.theta[arm] = np.linalg.solve(self.A[arm], self.b[arm])
        
        logger.debug(f"""Updated model {arm} parameters: 
            theta={self.theta[arm].flatten()}
            decay_factor={self.decay_factor}""")

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

    def run(self):
        """主循环"""
        if not self.connect_to_server():
            logger.error("Failed to connect to processing server")
            return
            
        logger.info("Starting CMAB model switching loop...")
        
        last_model = None
        last_features = None
        
        while True:
            try:
                # 1. 获取当前状态
                current_stats = self.get_current_stats()
                
                if current_stats:
                    # 2. 提取特征
                    features = self.normalize_state(current_stats)
                    
                    # 3. 如果有上一个动作的信息，更新模型
                    if last_model is not None and last_features is not None:
                        reward = self.compute_reward(current_stats)
                        self.update_model(last_model, last_features, reward)
                    
                    # 4. 选择新的模型
                    next_model = self.select_model(features)
                    
                    # 5. 如果需要切换模型，发送请求
                    if next_model != current_stats['model_name']:
                        logger.info(f"Switching model from {current_stats['model_name']} to {next_model}")
                        self.switch_model(next_model)
                    
                    # 6. 保存当前信息用于下次更新
                    last_model = next_model
                    last_features = features
                
                time.sleep(self.stats_update_interval)
                
            except KeyboardInterrupt:
                logger.info("\nStopping CMAB switcher...")
                self.sio.disconnect()
                break
            except Exception as e:
                logger.error(f"Error in switching loop: {e}")
                time.sleep(1)

if __name__ == '__main__':
    switcher = CMABSwitcher()
    switcher.run()