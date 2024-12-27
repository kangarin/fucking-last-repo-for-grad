import torch
import torchvision.models as models
import numpy as np
import threading
import time
import random
from typing import List

class GPULoadGenerator:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
            
        self.model_configs = {
            'resnet18': models.resnet18,
            'squeezenet': models.squeezenet1_0,
            'mobilenet': models.mobilenet_v2
        }
        self.batch_sizes = [1, 2, 4, 8, 16]
        self.current_model = None
        
    def _setup_new_load(self):
        """设置新的负载配置"""
        if self.current_model is not None:
            del self.current_model
            torch.cuda.empty_cache()
            
        model_name = random.choice(list(self.model_configs.keys()))
        batch_size = random.choice(self.batch_sizes)
        print(f"\nNew GPU load: model={model_name}, batch={batch_size}")
        
        self.current_model = self.model_configs[model_name](pretrained=True).cuda().eval()
        return batch_size
        
    def generate_load(self):
        """持续生成GPU负载"""
        try:
            while True:
                # 设置新的负载配置
                batch_size = self._setup_new_load()
                inputs = torch.randn(batch_size, 3, 224, 224).cuda()
                
                # 运行30-60秒
                end_time = time.time() + random.randint(30, 60)
                while time.time() < end_time:
                    with torch.no_grad():
                        _ = self.current_model(inputs)
                        torch.cuda.synchronize()
                    time.sleep(0.001)
                    
        except Exception as e:
            print(f"GPU error: {e}")

class CPULoadGenerator:
    def __init__(self):
        self.sizes = [256, 512, 1024, 2048]  # 不同矩阵大小
        self.current_size = None
        
    def _setup_new_load(self):
        """设置新的负载配置"""
        self.current_size = random.choice(self.sizes)
        print(f"\nNew CPU load: matrix_size={self.current_size}")
        return np.random.rand(self.current_size, self.current_size)
        
    def generate_load(self):
        """持续生成CPU负载"""
        try:
            while True:
                # 设置新的负载配置
                matrix = self._setup_new_load()
                
                # 运行30-60秒
                end_time = time.time() + random.randint(30, 60)
                while time.time() < end_time:
                    _ = np.dot(matrix, matrix)
                    time.sleep(0.001)
                    
        except Exception as e:
            print(f"CPU error: {e}")

class MemoryLoadGenerator:
    def __init__(self):
        self.sizes = [64, 128, 256, 512]  # MB
        self.data_blocks: List[np.ndarray] = []
        self.max_blocks = 5
        
    def _setup_new_load(self):
        """设置新的负载配置"""
        size_mb = random.choice(self.sizes)
        print(f"\nNew Memory load: block_size={size_mb}MB, max_blocks={self.max_blocks}")
        return int(size_mb * 1024 * 1024 / 8)
        
    def generate_load(self):
        """持续生成内存负载"""
        try:
            while True:
                # 清空之前的数据块
                self.data_blocks.clear()
                array_size = self._setup_new_load()
                
                # 运行30-60秒
                end_time = time.time() + random.randint(30, 60)
                while time.time() < end_time:
                    data = np.random.rand(array_size)
                    self.data_blocks.append(data)
                    
                    if len(self.data_blocks) > self.max_blocks:
                        self.data_blocks.pop(0)
                        
                    time.sleep(1)
                    
        except Exception as e:
            print(f"Memory error: {e}")

def main():
    try:
        print("Starting load generators (press Ctrl+C to stop)...")
        
        # 创建生成器实例
        gpu_gen = GPULoadGenerator() if torch.cuda.is_available() else None
        cpu_gen = CPULoadGenerator()
        mem_gen = MemoryLoadGenerator()
        
        # 启动负载生成线程
        threads = [
            # threading.Thread(target=gpu_gen.generate_load, daemon=True),
            threading.Thread(target=cpu_gen.generate_load, daemon=True),
            threading.Thread(target=mem_gen.generate_load, daemon=True)
        ]
        if gpu_gen:
            threads.append(threading.Thread(target=gpu_gen.generate_load, daemon=True))
        
        for t in threads:
            t.start()
            
        # 主线程等待
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping load generators...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()