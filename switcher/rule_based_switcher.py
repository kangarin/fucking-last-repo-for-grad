import time
import logging
import requests
from socketio import Client
from collections import defaultdict
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleBasedSwitcher:
    def __init__(self, stats_update_interval=1.0, decision_interval=5):
        self.sio = Client()
        # 从config获取服务器配置
        processing_config = config.get_edge_processing_config()
        self.processing_server_url = processing_config['url']
        self.http_url = self.processing_server_url
        self.stats_update_interval = stats_update_interval
        
        # 存储每个模型的性能数据
        self.model_latencies = {
            'n': None,  # YOLOv5n
            's': None,  # YOLOv5s
            'm': None,  # YOLOv5m
            'l': None,  # YOLOv5l
            'x': None   # YOLOv5x
        }
        
        # 初始化时延统计
        self.latency_stats = defaultdict(list)
        self.calibration_samples = 50  # 每个模型收集多少样本计算平均值
        self.is_calibrated = False
        
        # EWMA参数
        self.alpha = 0.1  # 指数加权移动平均的权重因子
        
        # 决策参数
        self.decision_interval = decision_interval
        self.last_decision_time = time.time()
    
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

    def calibrate_models(self):
        """测量所有模型的基准性能"""
        logger.info("Starting model calibration...")
        
        for model_name in self.model_latencies.keys():
            logger.info(f"Calibrating model YOLOv5{model_name}...")
            
            # 切换到当前要测试的模型
            self.switch_model(model_name)
            time.sleep(2)  # 等待模型加载完成
            
            # 收集性能数据
            samples_collected = 0
            while samples_collected < self.calibration_samples:
                stats = self.get_current_stats()
                if stats and stats['processing_latency'] > 0:
                    self.latency_stats[model_name].append(stats['processing_latency'])
                    samples_collected += 1
                time.sleep(0.1)
            
            # 计算平均推理时延
            avg_latency = sum(self.latency_stats[model_name]) / len(self.latency_stats[model_name])
            self.model_latencies[model_name] = avg_latency
            
            logger.info(f"Model YOLOv5{model_name} initial average latency: {avg_latency:.3f}s")
        
        self.is_calibrated = True
        logger.info("Model calibration completed.")
        logger.info(f"Initial latencies: {dict(self.model_latencies)}")

    def update_latency(self, model_name, new_latency):
        """使用指数加权移动平均更新模型的推理时延"""
        if self.model_latencies[model_name] is None:
            self.model_latencies[model_name] = new_latency
        else:
            # 使用EWMA更新时延估计
            current_latency = self.model_latencies[model_name]
            updated_latency = (1 - self.alpha) * current_latency + self.alpha * new_latency
            self.model_latencies[model_name] = updated_latency
            logger.debug(f"Updated YOLOv5{model_name} latency: {updated_latency:.3f}s")

    def select_best_model(self, target_fps):
        """根据目标FPS选择最合适的模型"""
        if not self.is_calibrated:
            return 'n'  # 如果没有校准数据，默认使用最轻量级的模型
            
        target_latency = 1.0 / target_fps
        
        # 找到所有延迟小于目标延迟的模型
        valid_models = [(name, latency) for name, latency in self.model_latencies.items() 
                       if latency is not None and latency < target_latency]
        
        if not valid_models:
            # 如果没有满足条件的模型，返回延迟最小的模型
            return min(self.model_latencies.items(), 
                      key=lambda x: float('inf') if x[1] is None else x[1])[0]
        
        # 在满足条件的模型中选择延迟最接近目标的
        return max(valid_models, key=lambda x: x[1])[0]

    def adaptive_switch_loop(self):
        """主循环"""
        if not self.connect_to_server():
            logger.error("Failed to connect to processing server")
            return
            
        logger.info("Starting adaptive model switching loop...")
        
        # 首先进行模型校准
        if not self.is_calibrated:
            self.calibrate_models()
        
        while True:
            try:
                # 获取当前状态
                current_stats = self.get_current_stats()
                
                if current_stats:
                    target_fps = current_stats['target_fps']
                    current_model_name = current_stats['model_name']
                    current_latency = current_stats['processing_latency']
                    
                    # 更新对应模型的时延估计
                    self.update_latency(current_model_name, current_latency)
                    
                    # 检查是否需要做出新的决策
                    current_time = time.time()
                    if current_time - self.last_decision_time >= self.decision_interval:
                        logger.info(f"\nCurrent stats: "
                                  f"Model=YOLOv5{current_model_name}, "
                                  f"Latency={current_latency:.3f}s, "
                                  f"EWMA Latency={self.model_latencies[current_model_name]:.3f}s")
                        
                        next_model = self.select_best_model(target_fps)
                        
                        # 如果最佳模型与当前模型不同，进行切换
                        if next_model != current_model_name:
                            logger.info(f"Target FPS: {target_fps}, switching model from "
                                      f"YOLOv5{current_model_name} to YOLOv5{next_model}")
                            self.switch_model(next_model)
                        
                        self.last_decision_time = current_time
                
                # 休眠一段时间
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
        switcher = RuleBasedSwitcher()
        switcher.adaptive_switch_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)