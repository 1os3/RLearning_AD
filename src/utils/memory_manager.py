import gc
import torch
import psutil
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Union, Tuple

class MemoryManager:
    """
    内存和GPU显存管理器，负责监控和优化内存使用，防止内存泄漏和OOM错误。
    提供周期性垃圾回收、内存使用统计和紧急清理等功能。
    """
    
    def __init__(self, 
                 device: torch.device = None,
                 check_interval: int = 1000, 
                 emergency_threshold: float = 0.9,
                 clear_cuda_cache: bool = True,
                 debug_mode: bool = False):
        """
        初始化内存管理器
        
        Args:
            device: PyTorch设备（CPU/CUDA/XPU）
            check_interval: 检查间隔（训练步数）
            emergency_threshold: 紧急清理阈值（使用比例）
            clear_cuda_cache: 是否清理CUDA缓存
            debug_mode: 调试模式（打印详细信息）
        """
        self.device = device if device is not None else torch.device("cpu")
        self.check_interval = check_interval
        self.emergency_threshold = emergency_threshold
        self.clear_cuda_cache = clear_cuda_cache
        self.debug_mode = debug_mode
        self.last_check_time = time.time()
        self.last_check_step = 0
        
        # 仅在使用GPU时启用CUDA相关功能
        self.use_cuda = self.device.type == 'cuda' and torch.cuda.is_available()
        self.use_xpu = hasattr(torch, 'xpu') and self.device.type == 'xpu' and torch.xpu.is_available() if hasattr(torch, 'xpu') else False
        
        self.stats_history = {
            'system_ram': [],
            'gpu_memory': [],
            'timestamps': []
        }
        
        if self.debug_mode:
            print(f"内存管理器已初始化: 设备={self.device}, 检查间隔={self.check_interval}步, "
                  f"紧急阈值={self.emergency_threshold*100}%")
    
    def collect_garbage(self, force: bool = False) -> None:
        """
        执行垃圾回收
        
        Args:
            force: 是否强制执行完整回收
        """
        # Python对象垃圾回收
        gc.collect()
        
        # GPU内存释放
        if self.use_cuda and self.clear_cuda_cache:
            torch.cuda.empty_cache()
            if force:
                # 更彻底的显存回收（适用于紧急情况）
                torch.cuda.synchronize()
        
        # XPU内存释放（如果可用）
        if self.use_xpu and self.clear_cuda_cache and hasattr(torch.xpu, 'empty_cache'):
            torch.xpu.empty_cache()
            if force:
                # 如果XPU支持同步操作
                if hasattr(torch.xpu, 'synchronize'):
                    torch.xpu.synchronize()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        获取当前内存使用状态
        
        Returns:
            包含系统和GPU内存使用情况的字典
        """
        stats = {}
        
        # 系统内存
        mem = psutil.virtual_memory()
        stats['system_ram_used'] = mem.used / (1024**3)  # GB
        stats['system_ram_total'] = mem.total / (1024**3)  # GB
        stats['system_ram_percent'] = mem.percent / 100.0
        
        # GPU内存（如果可用）
        if self.use_cuda:
            try:
                stats['gpu_used'] = torch.cuda.memory_allocated() / (1024**3)  # GB
                stats['gpu_reserved'] = torch.cuda.memory_reserved() / (1024**3)  # GB
                if torch.cuda.max_memory_allocated() > 0:
                    stats['gpu_percent'] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                else:
                    stats['gpu_percent'] = 0
            except (RuntimeError, AttributeError) as e:
                stats['gpu_error'] = str(e)
        
        # 收集XPU统计信息（如果可用）
        if self.use_xpu:
            try:
                # 首先尝试使用PyTorch API获取基础信息
                if hasattr(torch.xpu, 'memory_allocated'):
                    stats['xpu_allocated'] = torch.xpu.memory_allocated() / (1024**3)  # GB
                if hasattr(torch.xpu, 'memory_reserved'):
                    stats['xpu_reserved'] = torch.xpu.memory_reserved() / (1024**3)  # GB
                
                # 然后尝试使用系统级命令获取真实使用情况
                try:
                    import subprocess
                    import re
                    
                    # 通过Intel工具获取XPU内存使用情况
                    result = subprocess.run(['xpu-smi', '-m'], capture_output=True, text=True, check=False)
                    if result.returncode == 0:
                        # 解析输出以获取内存使用统计
                        output = result.stdout
                        memory_match = re.search(r'Used\s+:\s+(\d+)\s+MiB', output)
                        if memory_match:
                            stats['xpu_used'] = float(memory_match.group(1)) / 1024  # 转换为GB
                        
                        total_match = re.search(r'Total\s+:\s+(\d+)\s+MiB', output)
                        if total_match and memory_match:
                            total = float(total_match.group(1))
                            used = float(memory_match.group(1))
                            stats['xpu_percent'] = used / total if total > 0 else 0
                            stats['xpu_total'] = total / 1024  # 转换为GB
                    else:
                        # 如果命令失败，回退到PyTorch提供的数据
                        if 'xpu_allocated' in stats and hasattr(torch.xpu, 'max_memory_allocated') and torch.xpu.max_memory_allocated() > 0:
                            stats['xpu_used'] = stats['xpu_allocated']
                            stats['xpu_percent'] = stats['xpu_allocated'] / (torch.xpu.max_memory_allocated() / (1024**3))
                        else:
                            stats['xpu_used'] = stats.get('xpu_allocated', 0)  # 默认使用分配的内存
                            stats['xpu_percent'] = 0
                except (ImportError, FileNotFoundError):
                    # 如果无法使用系统命令，回退到PyTorch提供的数据
                    if 'xpu_allocated' in stats:
                        stats['xpu_used'] = stats['xpu_allocated']
                        if hasattr(torch.xpu, 'max_memory_allocated') and torch.xpu.max_memory_allocated() > 0:
                            stats['xpu_percent'] = stats['xpu_used'] / (torch.xpu.max_memory_allocated() / (1024**3))
                        else:
                            stats['xpu_percent'] = 0
            except (RuntimeError, AttributeError) as e:
                stats['xpu_error'] = str(e)
                
        return stats
    
    def log_memory_stats(self, logger=None, step: int = 0) -> Dict[str, float]:
        """
        记录内存使用情况
        
        Args:
            logger: 可选的Logger实例
            step: 当前训练步数
            
        Returns:
            内存统计信息
        """
        stats = self.get_memory_stats()
        
        # 更新历史记录
        self.stats_history['system_ram'].append(stats.get('system_ram_percent', 0))
        if self.use_cuda:
            self.stats_history['gpu_memory'].append(stats.get('gpu_percent', 0))
        elif self.use_xpu:
            self.stats_history['gpu_memory'].append(stats.get('xpu_percent', 0))
        else:
            self.stats_history['gpu_memory'].append(0)
        self.stats_history['timestamps'].append(time.time())
        
        # 保持历史记录的合理大小
        if len(self.stats_history['timestamps']) > 1000:
            for key in self.stats_history:
                self.stats_history[key] = self.stats_history[key][-1000:]
        
        # 使用Logger记录
        if logger:
            logger.log_scalar("memory/system_ram_percent", stats.get('system_ram_percent', 0), step)
            if self.use_cuda:
                logger.log_scalar("memory/gpu_used_gb", stats.get('gpu_used', 0), step)
                logger.log_scalar("memory/gpu_percent", stats.get('gpu_percent', 0), step)
            elif self.use_xpu:
                logger.log_scalar("memory/xpu_used_gb", stats.get('xpu_used', 0), step)
                logger.log_scalar("memory/xpu_percent", stats.get('xpu_percent', 0), step)
        
        # 如果处于调试模式，打印统计信息
        if self.debug_mode:
            print(f"======= 内存状态 (步数: {step}) =======")
            print(f"系统内存: {stats['system_ram_used']:.2f}GB / {stats['system_ram_total']:.2f}GB "
                  f"({stats['system_ram_percent']*100:.1f}%)")
            
            if self.use_cuda:
                print(f"CUDA显存: {stats.get('gpu_used', 0):.2f}GB / {stats.get('gpu_reserved', 0):.2f}GB "
                      f"({stats.get('gpu_percent', 0)*100:.1f}%)")
            elif self.use_xpu:
                print(f"XPU显存: {stats.get('xpu_used', 0):.2f}GB / {stats.get('xpu_reserved', 0):.2f}GB "
                      f"({stats.get('xpu_percent', 0)*100:.1f}%)")
            print("=====================================")
        
        return stats
    
    def check_and_collect(self, step: int, logger=None, force: bool = False) -> Tuple[bool, Dict[str, float]]:
        """
        检查是否需要执行内存回收
        
        Args:
            step: 当前训练步数
            logger: 可选的Logger实例
            force: 是否强制执行
            
        Returns:
            (是否执行了回收, 内存统计信息)
        """
        # 检查是否达到检查间隔
        if not force and (step - self.last_check_step) < self.check_interval:
            return False, {}
        
        self.last_check_step = step
        self.last_check_time = time.time()
        
        # 获取内存状态
        stats = self.log_memory_stats(logger, step)
        
        # 确定是否超过阈值
        system_threshold_exceeded = stats.get('system_ram_percent', 0) > self.emergency_threshold
        
        gpu_threshold_exceeded = False
        if self.use_cuda:
            gpu_threshold_exceeded = stats.get('gpu_percent', 0) > self.emergency_threshold
        elif self.use_xpu:
            gpu_threshold_exceeded = stats.get('xpu_percent', 0) > self.emergency_threshold
        
        # 执行回收
        if force or system_threshold_exceeded or gpu_threshold_exceeded:
            if self.debug_mode:
                print(f"执行内存回收！原因: {'强制' if force else ''}"
                      f"{'系统内存过高' if system_threshold_exceeded else ''}"
                      f"{'GPU显存过高' if gpu_threshold_exceeded else ''}")
            
            self.collect_garbage(force=force or system_threshold_exceeded or gpu_threshold_exceeded)
            
            # 在紧急情况下减少历史记录大小
            if system_threshold_exceeded or gpu_threshold_exceeded:
                for key in self.stats_history:
                    if len(self.stats_history[key]) > 100:
                        self.stats_history[key] = self.stats_history[key][-100:]
            
            return True, stats
        
        return False, stats
    
    def emergency_cleanup(self, logger=None, step: int = 0) -> Dict[str, float]:
        """
        紧急内存清理，用于OOM前的最后手段
        
        Args:
            logger: 可选的Logger实例
            step: 当前训练步数
            
        Returns:
            内存统计信息
        """
        print("===== 执行紧急内存清理 =====")
        
        # 首先执行常规垃圾回收
        self.collect_garbage(force=True)
        
        # 重要！清理成批梯度和保存的中间值
        torch.cuda.empty_cache() if self.use_cuda else None
        if self.use_xpu and hasattr(torch.xpu, 'empty_cache'):
            torch.xpu.empty_cache()
        
        # 删除历史统计数据
        for key in self.stats_history:
            self.stats_history[key] = self.stats_history[key][-10:] if len(self.stats_history[key]) > 10 else []
        
        # 多次执行GC
        for _ in range(3):
            gc.collect()
        
        # 记录清理后的状态
        stats = self.log_memory_stats(logger, step)
        print(f"紧急清理完成，当前系统内存: {stats.get('system_ram_percent', 0)*100:.1f}%, "
              f"{'CUDA' if self.use_cuda else 'XPU' if self.use_xpu else 'CPU'}内存: "
              f"{stats.get('gpu_percent' if self.use_cuda else 'xpu_percent' if self.use_xpu else 'cpu_percent', 0)*100:.1f}%")
        
        return stats
