import psutil
import torch
import threading
import time
from typing import Dict, Optional

class ResourceMonitor:
    """
    监控系统资源使用情况，包括CPU、内存和GPU/XPU使用率
    可以在后台线程中运行，定期记录资源使用情况到TensorBoard
    """
    def __init__(self, logger=None, interval: int = 10):
        """
        初始化资源监控器
        
        Args:
            logger: 日志对象（需要支持log_scalar方法）
            interval: 监控间隔（秒）
        """
        self.logger = logger
        self.interval = interval
        self.running = False
        self.monitor_thread = None
        self.steps = 0
        
    def start(self):
        """启动监控线程"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True  # 守护线程，主线程结束时自动结束
        self.monitor_thread.start()
        print(f"资源监控已启动，每{self.interval}秒记录一次")
        
    def stop(self):
        """停止监控线程"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
    def _monitor_resources(self):
        """监控资源的线程函数"""
        while self.running:
            # 获取当前资源使用情况
            metrics = self.get_resource_metrics()
            
            # 记录到日志（如果有日志器）
            if self.logger:
                self.steps += 1
                for key, value in metrics.items():
                    self.logger.log_scalar(f"resources/{key}", value, self.steps)
            
            # 等待下一个间隔
            time.sleep(self.interval)
    
    def get_resource_metrics(self) -> Dict[str, float]:
        """
        获取当前系统资源使用情况
        
        Returns:
            包含资源指标的字典
        """
        metrics = {}
        
        # CPU使用率
        metrics['cpu_percent'] = psutil.cpu_percent()
        
        # 内存使用率
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_used_gb'] = memory.used / (1024 * 1024 * 1024)
        metrics['memory_available_gb'] = memory.available / (1024 * 1024 * 1024)
        
        # XPU使用情况（如果可用）
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            try:
                # 当前XPU内存使用
                metrics['xpu_memory_allocated_mb'] = torch.xpu.memory_allocated() / (1024 * 1024)
                metrics['xpu_memory_reserved_mb'] = torch.xpu.memory_reserved() / (1024 * 1024)
                
                # XPU使用率（需要系统支持）
                try:
                    import intel_extension_for_pytorch as ipex
                    metrics['xpu_utilization'] = ipex.xpu.utilization()
                except:
                    pass
            except:
                pass
                
        # CUDA使用情况（如果可用）
        if torch.cuda.is_available():
            try:
                # 当前CUDA内存使用
                metrics['cuda_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                metrics['cuda_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
                
                # GPU使用率（需要系统支持）
                try:
                    metrics['cuda_utilization'] = torch.cuda.utilization()
                except:
                    pass
            except:
                pass
                
        return metrics
