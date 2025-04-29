import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

class Logger:
    """
    Training logger for recording metrics and statistics.
    Supports multiple output formats including CSV, JSON, and TensorBoard.
    """
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize containers for metrics
        self.metrics = {}
        self.csv_logs = {}
        
        # 创建TensorBoard writer (不再检查是否可用，强制启用)
        self.tb_writer = None
        # 不考虑use_tensorboard参数，强制启用
        self.use_tensorboard = True
        
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(log_dir, 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(tb_dir)
        print(f"强制启用TensorBoard日志，存储到 {tb_dir}")
        
        # Log experiment start time
        self.start_time = time.time()
        self.log_metadata()
    
    def log_metadata(self):
        """Log basic experiment metadata."""
        metadata = {
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'log_dir': self.log_dir,
            'tensorboard': self.use_tensorboard,
        }
        
        with open(os.path.join(self.log_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a scalar value.
        
        Args:
            name: Metric name
            value: Scalar value
            step: Training step or episode number
        """
        try:
            # Log to TensorBoard
            if self.use_tensorboard and self.tb_writer is not None:
                self.tb_writer.add_scalar(name, value, step)
            
            # Store in memory for CSV logging
            namespace = name.split('/')[0] if '/' in name else 'default'
            metric_name = name.split('/')[-1] if '/' in name else name
            
            # 初始化namespace如果需要
            if namespace not in self.csv_logs:
                self.csv_logs[namespace] = {'step': []}
            
            # 初始化metric如果需要
            if metric_name not in self.csv_logs[namespace]:
                self.csv_logs[namespace][metric_name] = []
            
            # 添加新的step
            curr_step = step if step is not None else len(self.csv_logs[namespace]['step'])
            
            # 如果是新的step
            if curr_step not in self.csv_logs[namespace]['step']:
                # 添加新的step到列表
                self.csv_logs[namespace]['step'].append(curr_step)
                # 确保所有指标的数组长度一致 
                for key in self.csv_logs[namespace]:
                    if len(self.csv_logs[namespace][key]) < len(self.csv_logs[namespace]['step']):
                        # 填充NaN直到数组长度一致
                        self.csv_logs[namespace][key].extend([np.nan] * 
                                               (len(self.csv_logs[namespace]['step']) - len(self.csv_logs[namespace][key])))
            
            # 找到当前step的索引
            try:
                idx = self.csv_logs[namespace]['step'].index(curr_step)
                # 更新值
                if len(self.csv_logs[namespace][metric_name]) <= idx:
                    # 确保数组足够长
                    self.csv_logs[namespace][metric_name].extend([np.nan] * (idx + 1 - len(self.csv_logs[namespace][metric_name])))
                self.csv_logs[namespace][metric_name][idx] = value
            except ValueError:
                # 如果没有找到step，将其添加到列表末尾
                self.csv_logs[namespace]['step'].append(curr_step)
                self.csv_logs[namespace][metric_name].append(value)
            
            # 每10次调用保存一次CSV，减少IO操作
            if sum(len(data['step']) for data in self.csv_logs.values()) % 10 == 0:
                self._save_csvs()
        except Exception as e:
            # 捕获并打印错误，但不中断训练
            print(f"警告: 日志记录出错 ({name}, {value}, {step}): {e}")
    
    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None):
        """
        Log a histogram of values.
        
        Args:
            name: Metric name
            values: Array of values to create histogram from
            step: Training step or episode number
        """
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_histogram(name, values, step)
    
    def log_image(self, name: str, image: np.ndarray, step: Optional[int] = None):
        """
        Log an image.
        
        Args:
            name: Image name
            image: Image array (H, W, C) with values in [0, 255]
            step: Training step or episode number
        """
        if self.use_tensorboard and self.tb_writer is not None:
            # TensorBoard expects (C, H, W) with values in [0, 1]
            import torch
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
            self.tb_writer.add_image(name, img_tensor, step)
    
    def log_video(self, name: str, video: np.ndarray, step: Optional[int] = None, fps: int = 30):
        """
        Log a video.
        
        Args:
            name: Video name
            video: Video array (T, H, W, C) with values in [0, 255]
            step: Training step or episode number
            fps: Frames per second
        """
        if self.use_tensorboard and self.tb_writer is not None:
            # TensorBoard expects (T, C, H, W) with values in [0, 1]
            import torch
            if video.dtype != np.uint8:
                video = (video * 255).astype(np.uint8)
            
            # Convert to tensor and normalize
            video_tensor = torch.from_numpy(video.transpose(0, 3, 1, 2)).float() / 255.0
            
            self.tb_writer.add_video(name, video_tensor.unsqueeze(0), step, fps=fps)
    
    def log_hyperparams(self, hparams: Dict):
        """
        Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        # Save hyperparameters to JSON
        with open(os.path.join(self.log_dir, 'hyperparams.json'), 'w') as f:
            json.dump(hparams, f, indent=4)
        
        # Log to TensorBoard if enabled
        if self.use_tensorboard and self.tb_writer is not None:
            from torch.utils.tensorboard.summary import hparams
            self.tb_writer.add_hparams(hparams, {})
    
    def log_model_graph(self, model, input_tensor):
        """
        Log model graph for visualization.
        
        Args:
            model: PyTorch model
            input_tensor: Example input tensor for tracing
        """
        if self.use_tensorboard and self.tb_writer is not None:
            try:
                self.tb_writer.add_graph(model, input_tensor)
            except Exception as e:
                print(f"Error adding model graph to TensorBoard: {e}")
    
    def _save_csvs(self):
        """Save metrics to CSV files."""
        for namespace, data in self.csv_logs.items():
            try:
                # 防止数组长度不一致错误
                # 首先确定最大长度
                max_length = max([len(arr) for arr in data.values()])
                
                # 填充所有数组到同样的长度
                normalized_data = {}
                for key, values in data.items():
                    if len(values) < max_length:
                        # 用NaN填充使数组长度相同
                        normalized_data[key] = values + [np.nan] * (max_length - len(values))
                    else:
                        normalized_data[key] = values
                
                # 创建DataFrame并保存
                df = pd.DataFrame(normalized_data)
                df.to_csv(os.path.join(self.log_dir, f'{namespace}_metrics.csv'), index=False)
            except Exception as e:
                print(f"警告: 保存{namespace}CSV数据时出错: {e}")
    
    def close(self):
        """Close logger and write all pending data."""
        # Save all CSVs
        self._save_csvs()
        
        # Close TensorBoard writer
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.close()
        
        # Log experiment end time
        metadata = {
            'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration_seconds': time.time() - self.start_time,
        }
        
        # Update metadata file
        try:
            with open(os.path.join(self.log_dir, 'metadata.json'), 'r') as f:
                existing_metadata = json.load(f)
            existing_metadata.update(metadata)
            
            with open(os.path.join(self.log_dir, 'metadata.json'), 'w') as f:
                json.dump(existing_metadata, f, indent=4)
        except:
            # If reading fails, just write the end metadata
            with open(os.path.join(self.log_dir, 'metadata_end.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
    
    def __del__(self):
        """Ensure logger is closed properly."""
        try:
            self.close()
        except:
            pass
