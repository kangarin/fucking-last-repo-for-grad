import numpy as np
import logging
import time
import requests
from socketio import Client
from pathlib import Path
import sys
from scipy.stats import multivariate_normal

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThompsonSwitcher:
    def __init__(self, stats_update_interval=5.0):
        self.sio = Client()
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        self.stats_update_interval = stats_update_interval
        
        # Thompson Sampling参数
        self.feature_dim = 11  # 特征维度
        self.model_levels = ['n', 's', 'm', 'l', 'x']
        self.v = 1.0  # 先验方差
        
        # 为每个arm初始化参数
        self.B = {arm: np.eye(self.feature_dim) for arm in self.model_levels}  # 精度矩阵
        self.mu = {arm: np.zeros((self.feature_dim, 1)) for arm in self.model_levels}  # 后验均值
        self.f = {arm: np.zeros((self.feature_dim, 1)) for arm in self.model_levels}  # 累积特征奖励
        self.observations = {arm: 0 for arm in self.model_levels}  # 每个arm的观测次数
        
        # 获取配置
        self.queue_max_length = config.get_queue_max_length()
        self.queue_high_threshold = config.get_queue_high_threshold_length()
        self.queue_low_threshold = config.get_queue_low_threshold_length()

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
                w2 * (stats['latency'] / self.queue_low_threshold)
                
        logger.info(f"""Reward calculation:
            Queue Length: {stats['queue_length']:.1f} (Ratio: {queue_ratio:.2f})
            Weights: accuracy={w1:.2f}, latency={w2:.2f}
            Accuracy: {stats['accuracy']:.1f}
            Confidence: {stats['avg_confidence']:.2f}
            Latency: {stats['latency']:.3f}s
            Final Reward: {reward:.3f}""")
        
        return reward

    def sample_theta(self, arm):
        """从后验分布中采样参数"""
        try:
            # 计算后验协方差
            cov = np.linalg.inv(self.B[arm])
            # 从多元正态分布中采样
            theta_sample = multivariate_normal.rvs(
                mean=self.mu[arm].flatten(),
                cov=cov
            ).reshape(-1, 1)
            return theta_sample
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，返回当前均值
            logger.warning(f"Sampling failed for arm {arm}, using mean")
            return self.mu[arm]

    def select_model(self, features):
        """使用Thompson Sampling选择最佳模型"""
        expected_rewards = {}
        
        for arm in self.model_levels:
            # 从后验分布中采样参数
            theta = self.sample_theta(arm)
            # 计算预期奖励
            reward = float(theta.T @ features)
            expected_rewards[arm] = reward
            
            logger.debug(f"Model {arm}: sampled reward={reward:.3f}")
        
        best_model = max(expected_rewards.items(), key=lambda x: x[1])[0]
        logger.info(f"Expected rewards: {expected_rewards}")
        return best_model

    def update_model(self, arm, features, reward):
        """更新模型参数"""
        # 更新精度矩阵
        self.B[arm] += features @ features.T
        # 更新累积特征奖励
        self.f[arm] += reward * features
        # 更新后验均值
        self.mu[arm] = np.linalg.solve(self.B[arm], self.f[arm])
        # 更新观测次数
        self.observations[arm] += 1
        
        logger.debug(f"Updated model {arm} parameters: mu={self.mu[arm].flatten()}")

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
            
        logger.info("Starting Thompson Sampling model switching loop...")
        
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
                logger.info("\nStopping Thompson Sampling switcher...")
                self.sio.disconnect()
                break
            except Exception as e:
                logger.error(f"Error in switching loop: {e}")
                time.sleep(1)

if __name__ == '__main__':
    switcher = ThompsonSwitcher()
    switcher.run()