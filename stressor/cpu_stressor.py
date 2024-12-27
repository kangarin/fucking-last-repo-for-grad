import subprocess
import sys
import time
import psutil
import signal
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

class CPUStressor:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self._check_stress_ng()
    
    def _check_stress_ng(self):
        """检查 stress-ng 是否已安装"""
        try:
            subprocess.run(['stress-ng', '--version'], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise RuntimeError(
                "stress-ng not found. Please install it first.\n"
                "Ubuntu/Debian: sudo apt-get install stress-ng\n"
                "CentOS/RHEL: sudo yum install stress-ng"
            )

    def stress_cpu(self, 
                  load_percent: float, 
                  duration: int,
                  cpu_cores: Optional[int] = None,
                  monitor: bool = True) -> Dict[str, Any]:
        """
        对 CPU 进行压力测试
        
        Args:
            load_percent: CPU 负载百分比 (0-100)
            duration: 持续时间(秒)
            cpu_cores: 要使用的 CPU 核心数。如果为 None，使用所有核心
            monitor: 是否监控并打印状态
        
        Returns:
            包含测试统计信息的字典
        """
        if not 0 <= load_percent <= 100:
            raise ValueError("load_percent must be between 0 and 100")
            
        # 如果没指定核心数，使用所有核心
        if cpu_cores is None:
            cpu_cores = psutil.cpu_count()

        # 准备 stress-ng 命令
        # 检测操作系统
        is_mac = 'darwin' in sys.platform.lower()
        
        cmd = [
            'stress-ng',
            '--cpu', str(cpu_cores),
            '--cpu-load', str(load_percent),
            '--timeout', str(duration),
            '--metrics',
        ]
        
        # Mac 和 Linux 使用不同的参数
        if is_mac:
            cmd.append('--aggressive')  # Mac 下使用激进模式
        else:
            # Linux 下使用更精确的控制
            cmd.extend([
                '--cpu-method', 'all',  # 使用所有可用的 CPU 压测方法
                '--cpu-load-slice', '50',  # 更细粒度的负载控制
                '--taskset', '0-{}'.format(cpu_cores - 1)  # 绑定到特定核心
            ])
        
        logging.info(f"Starting CPU stress test: {load_percent}% load for {duration}s on {cpu_cores} cores")
        
        # 启动压测进程
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 收集监控数据
        stats = {
            'target_load': load_percent,
            'duration': duration,
            'cpu_cores': cpu_cores,
            'loads': [],
            'timestamps': []
        }
        
        # 监控循环
        start_time = time.time()
        try:
            while self.process.poll() is None:
                if monitor:
                    current_load = psutil.cpu_percent(interval=1)
                    current_time = time.time() - start_time
                    stats['loads'].append(current_load)
                    stats['timestamps'].append(current_time)
                    
                    logging.info(
                        f"Time: {current_time:.1f}s, "
                        f"CPU Load: {current_load:.1f}% "
                        f"(Target: {load_percent}%)"
                    )
        except KeyboardInterrupt:
            self.stop()
            logging.info("Stress test interrupted by user")
        
        # 等待进程完成并获取输出
        stdout, stderr = self.process.communicate()
        
        # 添加统计信息
        if stats['loads']:
            stats.update({
                'avg_load': sum(stats['loads']) / len(stats['loads']),
                'max_load': max(stats['loads']),
                'min_load': min(stats['loads']),
                'load_variance': sum((x - load_percent) ** 2 for x in stats['loads']) / len(stats['loads'])
            })
        
        return stats
    
    def stop(self):
        """停止当前压测"""
        if self.process:
            self.process.send_signal(signal.SIGTERM)
            self.process.wait()
            self.process = None
            logging.info("Stress test stopped")

# 使用示例
if __name__ == "__main__":
    stressor = CPUStressor()
    
    # 测试不同负载水平
    test_loads = [30, 50, 70]
    for load in test_loads:
        print(f"\nTesting {load}% CPU load...")
        stats = stressor.stress_cpu(
            load_percent=load,
            duration=10,
            cpu_cores=None,  # 使用所有核心
            monitor=True
        )
        print(f"\nTest statistics for {load}% target load:")
        print(f"Average load: {stats.get('avg_load', 0):.1f}%")
        print(f"Load variance: {stats.get('load_variance', 0):.2f}")