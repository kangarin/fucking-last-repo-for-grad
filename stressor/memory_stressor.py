import psutil
import time
import logging
from typing import Dict, Any, List
import ctypes
import threading
import sys

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class DirectMemoryStressor:
    def __init__(self):
        self.stop_flag = threading.Event()
        self.allocated_memory: List[Any] = []
        self.allocation_lock = threading.Lock()
        self.current_allocated = 0
        
    def _allocate_chunk(self, size_bytes: int) -> bool:
        """分配一块内存并返回是否成功"""
        try:
            # 使用 ctypes 分配内存
            arr = (ctypes.c_char * size_bytes)()
            # 每 4KB 写入一次数据，防止内存优化
            for i in range(0, size_bytes, 4096):
                arr[i] = ctypes.c_char(i % 256)
            
            with self.allocation_lock:
                self.allocated_memory.append(arr)
                self.current_allocated += size_bytes
            return True
        except Exception as e:
            logging.error(f"Memory allocation failed: {e}")
            return False
    
    def get_current_allocated_percent(self) -> float:
        """获取当前已分配的内存百分比"""
        total_memory = psutil.virtual_memory().total
        with self.allocation_lock:
            return (self.current_allocated / total_memory) * 100
    
    def stress_memory(self,
                     target_percent: float,
                     duration: int,
                     chunk_size_mb: int = 256,  # 默认每次分配 256MB
                     monitor: bool = True) -> Dict[str, Any]:
        """
        直接进行内存压力测试
        
        Args:
            target_percent: 目标内存使用率 (0-100)
            duration: 持续时间(秒)
            chunk_size_mb: 每次分配的内存块大小(MB)
            monitor: 是否监控并打印状态
        """
        if not 0 <= target_percent <= 100:
            raise ValueError("target_percent must be between 0 and 100")
        
        # 重置状态
        self.stop_flag.clear()
        self.allocated_memory.clear()
        self.current_allocated = 0
        
        # 计算目标内存大小
        total_memory = psutil.virtual_memory().total
        target_bytes = int(total_memory * target_percent / 100)
        chunk_bytes = chunk_size_mb * 1024 * 1024
        
        # 收集监控数据
        stats = {
            'target_percent': target_percent,
            'duration': duration,
            'memory_usages': [],
            'timestamps': [],
            'allocation_success': False
        }
        
        # 监控线程
        monitor_stop = threading.Event()
        monitor_start = threading.Event()
        
        def monitor_thread():
            while not monitor_start.is_set():
                time.sleep(0.1)  # 等待内存分配完成
                
            start_time = time.time()
            while not monitor_stop.is_set() and time.time() - start_time < duration:
                memory_info = psutil.virtual_memory()
                current_usage = memory_info.percent
                current_time = time.time() - start_time
                
                stats['memory_usages'].append(current_usage)
                stats['timestamps'].append(current_time)
                
                allocated_percent = self.get_current_allocated_percent()
                
                logging.info(
                    f"Time: {current_time:.1f}s, "
                    f"Memory Usage: {current_usage:.1f}% "
                    f"(Target: {target_percent}%), "
                    f"Allocated: {allocated_percent:.1f}%, "
                    f"Available: {memory_info.available / 1024 / 1024:.0f}MB"
                )
                
                time.sleep(1)
        
        # 启动监控线程
        if monitor:
            monitor_thread = threading.Thread(target=monitor_thread)
            monitor_thread.start()
        
        # 分配内存
        try:
            logging.info(
                f"Starting direct memory stress test: {target_percent}% "
                f"({target_bytes / 1024 / 1024:.0f}MB) "
                f"for {duration}s with {chunk_size_mb}MB chunks"
            )
            
            while self.current_allocated < target_bytes and not self.stop_flag.is_set():
                remaining = target_bytes - self.current_allocated
                chunk = min(chunk_bytes, remaining)
                
                if not self._allocate_chunk(chunk):
                    logging.warning(
                        f"Failed to allocate memory at "
                        f"{self.current_allocated / 1024 / 1024:.0f}MB"
                    )
                    break
                
                # 短暂暂停，让监控线程有机会运行
                time.sleep(0.1)
            
            stats['allocation_success'] = self.current_allocated >= target_bytes
            
            # 标记内存分配完成，开始计时
            monitor_start.set()
            
            if not self.stop_flag.is_set():
                # 等待指定时间
                time.sleep(duration)
            
        except KeyboardInterrupt:
            logging.info("Stress test interrupted by user")
        except Exception as e:
            logging.error(f"Error during stress test: {e}")
        finally:
            # 停止监控并清理
            monitor_stop.set()
            if monitor:
                monitor_thread.join()
            self.stop()
        
        # 添加统计信息
        if stats['memory_usages']:
            stats.update({
                'avg_usage': sum(stats['memory_usages']) / len(stats['memory_usages']),
                'max_usage': max(stats['memory_usages']),
                'min_usage': min(stats['memory_usages']),
                'usage_variance': sum((x - target_percent) ** 2 for x in stats['memory_usages']) / len(stats['memory_usages']),
                'final_allocated_mb': self.current_allocated / 1024 / 1024
            })
        
        return stats
    
    def stop(self):
        """停止压测并释放内存"""
        self.stop_flag.set()
        with self.allocation_lock:
            self.allocated_memory.clear()
            self.current_allocated = 0
        logging.info("Memory released")


# 使用示例
if __name__ == "__main__":
    stressor = DirectMemoryStressor()
    
    # 测试不同内存负载水平
    test_loads = [30, 50, 70]
    for load in test_loads:
        print(f"\nTesting {load}% memory usage...")
        stats = stressor.stress_memory(
            target_percent=load,
            duration=10,
            chunk_size_mb=256,  # 使用较小的内存块
            monitor=True
        )
        print(f"\nTest statistics for {load}% target memory usage:")
        print(f"Average usage: {stats.get('avg_usage', 0):.1f}%")
        print(f"Max usage: {stats.get('max_usage', 0):.1f}%")
        print(f"Allocation success: {stats.get('allocation_success', False)}")
        print(f"Final allocated: {stats.get('final_allocated_mb', 0):.0f}MB")