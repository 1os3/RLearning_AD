import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import time
import os
import json
from collections import deque, defaultdict
import csv
import datetime
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    # Fallback if tensorboard not available
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
        def add_scalar(self, *args, **kwargs):
            pass
        def add_scalars(self, *args, **kwargs):
            pass
        def add_histogram(self, *args, **kwargs):
            pass
        def add_image(self, *args, **kwargs):
            pass
        def close(self):
            pass

class MetricsManager:
    """
    Comprehensive metrics tracking and analysis for reinforcement learning.
    
    Tracks, computes, and visualizes various performance metrics during training
    and evaluation, providing insights into agent behavior and learning progress.
    """
    def __init__(self, config: Dict = None):
        """
        Initialize metrics manager with configuration.
        
        Args:
            config: Configuration dictionary with metrics parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.defaults = {
            'moving_average_window': 100,  # Window size for moving averages
            'evaluation_metrics': [        # Metrics to track during evaluation
                'episode_return',
                'episode_length',
                'success_rate',
                'distance_to_goal',
                'average_velocity',
                'energy_efficiency',
                'smoothness',
                'stability',
                'collision_rate'
            ],
            'training_metrics': [          # Metrics to track during training
                'episode_return',
                'episode_length',
                'actor_loss',
                'critic_loss',
                'entropy',
                'q_value',
                'target_q_value',
                'learning_rate',
                'alpha',
                'auxiliary_losses'
            ],
            'percentiles': [25, 50, 75],   # Percentiles to compute for metrics
            'record_frequency': 1,         # Record metrics every N episodes
            'log_to_csv': True,            # Whether to log metrics to CSV
            'log_to_tensorboard': True,    # Whether to log to TensorBoard
            'visualization_dpi': 150,      # DPI for saved visualizations
            'tensorboard_flush_secs': 10   # Flush TensorBoard logs every N seconds
        }
        
        # Combine default and user configuration
        self.metrics_config = {**self.defaults, **self.config}
        
        # Initialize metric storage
        self.reset()
        
        # Initialize CSV loggers
        self.csv_loggers = {'training': None, 'evaluation': None}
        self.csv_files = {'training': None, 'evaluation': None}
        
        # Initialize TensorBoard writer
        self.tb_writer = None
        if self.metrics_config['log_to_tensorboard']:
            self._init_tensorboard()
        
        # Initialize paths
        self.output_dir = self.config.get('output_dir', 'results')
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        try:
            if self.metrics_config['log_to_csv']:
                os.makedirs(os.path.join(self.output_dir, 'csv'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'stats'), exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create output directories: {e}")
    
    def _init_tensorboard(self):
        """
        Initialize TensorBoard writer.
        """
        try:
            log_dir = os.path.join(self.config.get('output_dir', 'results'), 'tensorboard', self.timestamp)
            self.tb_writer = SummaryWriter(
                log_dir=log_dir,
                flush_secs=self.metrics_config['tensorboard_flush_secs']
            )
            print(f"TensorBoard logs will be saved to {log_dir}")
        except Exception as e:
            print(f"Warning: Could not initialize TensorBoard: {e}")
            self.tb_writer = None
    
    def reset(self):
        """
        Reset all metrics tracking.
        """
        # Storage for all episodes
        self.all_episodes = {
            'training': defaultdict(list),
            'evaluation': defaultdict(list)
        }
        
        # Moving averages
        self.moving_averages = {
            'training': defaultdict(lambda: deque(maxlen=self.metrics_config['moving_average_window'])),
            'evaluation': defaultdict(lambda: deque(maxlen=self.metrics_config['moving_average_window']))
        }
        
        # Current episode metrics
        self.current_episode = {
            'training': defaultdict(list),
            'evaluation': defaultdict(list)
        }
        
        # Summary statistics
        self.summary_stats = {
            'training': {},
            'evaluation': {},
            'auxiliary': {}
        }
        
        # Episode counter
        self.episode_count = {
            'training': 0,
            'evaluation': 0
        }
        
        # Step counter
        self.total_steps = {
            'training': 0,
            'evaluation': 0
        }
        
        # Auxiliary task specific metrics
        self.auxiliary_metrics = defaultdict(list)
        
        # Timestamp for this session
        self.session_start_time = time.time()
    
    def start_episode(self, mode: str = 'training'):
        """
        Start recording metrics for a new episode.
        
        Args:
            mode: Either 'training' or 'evaluation'
        """
        # Clear current episode metrics
        self.current_episode[mode] = defaultdict(list)
        
        # Record start time
        self.current_episode[mode]['start_time'] = time.time()
    
    def add_step_metric(self, metric_name: str, value: float, mode: str = 'training'):
        """
        Add a metric value for the current step.
        
        Args:
            metric_name: Name of the metric
            value: Value to record
            mode: Either 'training' or 'evaluation'
        """
        self.current_episode[mode][metric_name].append(value)
    
    def add_multi_step_metrics(self, metrics: Dict[str, float], mode: str = 'training'):
        """
        Add multiple metric values for the current step.
        
        Args:
            metrics: Dictionary of metrics {name: value}
            mode: Either 'training' or 'evaluation'
        """
        for name, value in metrics.items():
            self.add_step_metric(name, value, mode)
    
    def end_episode(self, episode_metrics: Dict[str, float] = None, mode: str = 'training'):
        """
        End the current episode and compute episode-level metrics.
        
        Args:
            episode_metrics: Additional episode-level metrics
            mode: Either 'training' or 'evaluation'
        """
        # Compute episode duration
        if 'start_time' in self.current_episode[mode]:
            duration = time.time() - self.current_episode[mode]['start_time']
            self.add_episode_metric('episode_duration', duration, mode)
        
        # Compute step-based episode metrics
        for metric_name, values in self.current_episode[mode].items():
            if isinstance(values, list) and len(values) > 0 and metric_name != 'start_time':
                # Compute average for the episode
                avg_value = np.mean(values)
                self.add_episode_metric(f'{metric_name}_mean', avg_value, mode)
                
                # Compute min/max for the episode
                min_value = np.min(values)
                max_value = np.max(values)
                self.add_episode_metric(f'{metric_name}_min', min_value, mode)
                self.add_episode_metric(f'{metric_name}_max', max_value, mode)
                
                # Compute standard deviation for the episode
                if len(values) > 1:
                    std_value = np.std(values)
                    self.add_episode_metric(f'{metric_name}_std', std_value, mode)
        
        # Add additional episode metrics
        if episode_metrics:
            for name, value in episode_metrics.items():
                self.add_episode_metric(name, value, mode)
        
        # Increment episode counter
        self.episode_count[mode] += 1
        
        # Add step count to total steps
        if 'action' in self.current_episode[mode]:
            steps = len(self.current_episode[mode]['action'])
            self.total_steps[mode] += steps
            self.add_episode_metric('episode_length', steps, mode)
        
        # Log to CSV if enabled
        if self.metrics_config['log_to_csv'] and self.episode_count[mode] % self.metrics_config['record_frequency'] == 0:
            self._log_episode_to_csv(mode)
        
        # Log to TensorBoard if enabled
        if self.metrics_config['log_to_tensorboard'] and self.tb_writer is not None and self.episode_count[mode] % self.metrics_config['record_frequency'] == 0:
            self._log_episode_to_tensorboard(mode)
    
    def add_episode_metric(self, metric_name: str, value: float, mode: str = 'training'):
        """
        Add an episode-level metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to record
            mode: Either 'training' or 'evaluation'
        """
        # Add to all episodes
        self.all_episodes[mode][metric_name].append(value)
        
        # Add to moving average
        self.moving_averages[mode][metric_name].append(value)
    
    def add_auxiliary_metric(self, task_name: str, metric_name: str, value: float):
        """
        Add a metric specific to an auxiliary task.
        
        Args:
            task_name: Name of the auxiliary task
            metric_name: Name of the metric
            value: Value to record
        """
        self.auxiliary_metrics[f'{task_name}_{metric_name}'].append(value)
    
    def get_moving_average(self, metric_name: str, mode: str = 'training') -> float:
        """
        Get the moving average of a metric.
        
        Args:
            metric_name: Name of the metric
            mode: Either 'training' or 'evaluation'
            
        Returns:
            Moving average value or NaN if no data
        """
        values = self.moving_averages[mode].get(metric_name, [])
        if len(values) > 0:
            return np.mean(values)
        return float('nan')
    
    def get_episode_statistics(self, metric_name: str, mode: str = 'training', percentiles: List[int] = None) -> Dict[str, float]:
        """
        Get statistics for a metric across all episodes.
        
        Args:
            metric_name: Name of the metric
            mode: Either 'training' or 'evaluation'
            percentiles: List of percentiles to compute
            
        Returns:
            Dictionary of statistics
        """
        if percentiles is None:
            percentiles = self.metrics_config['percentiles']
        
        values = self.all_episodes[mode].get(metric_name, [])
        if len(values) == 0:
            return {}
        
        stats = {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 else 0,
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values),
            'latest': values[-1]
        }
        
        # Add percentiles
        for p in percentiles:
            stats[f'p{p}'] = np.percentile(values, p)
        
        return stats
    
    def compute_success_rate(self, window: int = None, mode: str = 'evaluation') -> float:
        """
        Compute the success rate over recent episodes.
        
        Args:
            window: Number of recent episodes to consider
            mode: Either 'training' or 'evaluation'
            
        Returns:
            Success rate (0-1)
        """
        successes = self.all_episodes[mode].get('success', [])
        if not successes:
            return 0.0
        
        if window is not None and window < len(successes):
            successes = successes[-window:]
        
        return sum(successes) / len(successes)
    
    def compute_summary_statistics(self, mode: str = 'evaluation'):
        """
        Compute summary statistics for all metrics.
        
        Args:
            mode: Either 'training' or 'evaluation'
        """
        metrics_list = self.metrics_config[f'{mode}_metrics']
        
        # Add derived metrics
        if mode == 'evaluation' and 'success' in self.all_episodes[mode]:
            if 'success_rate' not in metrics_list:
                metrics_list.append('success_rate')
        
        # Compute statistics for each metric
        for metric in metrics_list:
            # Check if metric exists in data
            if metric in self.all_episodes[mode] or f'{metric}_mean' in self.all_episodes[mode]:
                # Use _mean version if available
                metric_name = f'{metric}_mean' if f'{metric}_mean' in self.all_episodes[mode] else metric
                self.summary_stats[mode][metric] = self.get_episode_statistics(metric_name, mode)
        
        # Add auxiliary metrics if available
        for metric_name in self.auxiliary_metrics.keys():
            if len(self.auxiliary_metrics[metric_name]) > 0:
                values = self.auxiliary_metrics[metric_name]
                self.summary_stats['auxiliary'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values) if len(values) > 1 else 0,
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values),
                    'latest': values[-1]
                }
        
        return self.summary_stats[mode]
    
    def _log_episode_to_tensorboard(self, mode: str = 'training'):
        """
        Log episode metrics to TensorBoard.
        
        Args:
            mode: Either 'training' or 'evaluation'
        """
        if self.tb_writer is None:
            return
        
        # Current global step (total environment interactions)
        global_step = self.total_steps[mode]
        
        # Get metrics for this episode
        for metric in self.metrics_config[f'{mode}_metrics']:
            # Try to get the mean version if it exists
            if f'{metric}_mean' in self.all_episodes[mode] and len(self.all_episodes[mode][f'{metric}_mean']) > 0:
                value = self.all_episodes[mode][f'{metric}_mean'][-1]
                self.tb_writer.add_scalar(f'{mode}/{metric}', value, global_step)
            elif metric in self.all_episodes[mode] and len(self.all_episodes[mode][metric]) > 0:
                value = self.all_episodes[mode][metric][-1]
                self.tb_writer.add_scalar(f'{mode}/{metric}', value, global_step)
        
        # Log moving averages
        for metric in self.metrics_config[f'{mode}_metrics']:
            if metric in self.moving_averages[mode] and len(self.moving_averages[mode][metric]) > 0:
                value = self.get_moving_average(metric, mode)
                self.tb_writer.add_scalar(f'{mode}/{metric}_avg', value, global_step)
        
        # Log auxiliary metrics
        for metric_name, values in self.auxiliary_metrics.items():
            if len(values) > 0:
                self.tb_writer.add_scalar(f'auxiliary/{metric_name}', values[-1], global_step)
    
    def _log_episode_to_csv(self, mode: str = 'training'):
        """
        Log episode metrics to CSV.
        
        Args:
            mode: Either 'training' or 'evaluation'
        """
        # Initialize CSV logger if not already done
        if self.csv_loggers[mode] is None:
            csv_path = os.path.join(self.output_dir, 'csv', f'{mode}_{self.timestamp}.csv')
            self.csv_files[mode] = open(csv_path, 'w', newline='')
            self.csv_loggers[mode] = csv.DictWriter(self.csv_files[mode], 
                                                    fieldnames=['episode', 'step'] + self.metrics_config[f'{mode}_metrics'])
            self.csv_loggers[mode].writeheader()
        
        # Prepare row data
        row = {
            'episode': self.episode_count[mode],
            'step': self.total_steps[mode]
        }
        
        # Add metrics data
        for metric in self.metrics_config[f'{mode}_metrics']:
            # Try to get the mean version if it exists
            if f'{metric}_mean' in self.all_episodes[mode] and len(self.all_episodes[mode][f'{metric}_mean']) > 0:
                row[metric] = self.all_episodes[mode][f'{metric}_mean'][-1]
            elif metric in self.all_episodes[mode] and len(self.all_episodes[mode][metric]) > 0:
                row[metric] = self.all_episodes[mode][metric][-1]
            else:
                row[metric] = float('nan')
        
        # Write row
        self.csv_loggers[mode].writerow(row)
        self.csv_files[mode].flush()
    
    def export_data(self, mode: str = 'all', format: str = 'csv'):
        """
        Export metrics data to various formats.
        
        Args:
            mode: Which data to export, one of 'all', 'training', 'evaluation', 'auxiliary'
            format: Format to export, one of 'csv', 'json', 'pandas'
            
        Returns:
            Exported data in the requested format
        """
        # Prepare data to export
        if mode == 'all':
            modes = ['training', 'evaluation', 'auxiliary']
        else:
            modes = [mode]
        
        data = {}
        for m in modes:
            if m == 'auxiliary':
                data[m] = dict(self.auxiliary_metrics)
            else:
                data[m] = dict(self.all_episodes[m])
        
        # Convert to the requested format
        if format == 'json':
            # Convert to JSON-serializable format
            for m in data.keys():
                for k, v in data[m].items():
                    if isinstance(v, np.ndarray):
                        data[m][k] = v.tolist()
                    elif isinstance(v, deque):
                        data[m][k] = list(v)
            
            # Export timestamp and metadata
            data['metadata'] = {
                'timestamp': self.timestamp,
                'session_duration': time.time() - self.session_start_time,
                'episode_count': dict(self.episode_count),
                'total_steps': dict(self.total_steps)
            }
            
            # Return JSON string
            return json.dumps(data, indent=2)
        
        elif format == 'pandas':
            # Convert to pandas DataFrame
            dfs = {}
            for m in data.keys():
                # Skip if no data
                if not data[m]:
                    continue
                
                # Try to convert to DataFrame
                try:
                    df = pd.DataFrame(data[m])
                    dfs[m] = df
                except Exception as e:
                    print(f"Warning: Could not convert {m} data to DataFrame: {e}")
            
            return dfs
        
        elif format == 'csv':
            # Export to CSV files
            results = {}
            for m in data.keys():
                # Skip if no data
                if not data[m]:
                    continue
                
                # Try to convert to DataFrame and export to CSV
                try:
                    df = pd.DataFrame(data[m])
                    csv_path = os.path.join(self.output_dir, 'csv', f'{m}_export_{self.timestamp}.csv')
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    df.to_csv(csv_path, index=False)
                    results[m] = csv_path
                    print(f"Exported {m} data to {csv_path}")
                except Exception as e:
                    print(f"Warning: Could not export {m} data to CSV: {e}")
            
            return results
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def generate_report(self, mode: str = 'evaluation', save_path: str = None):
        """
        Generate a comprehensive performance report.
        
        Args:
            mode: Which mode to generate report for, one of 'training', 'evaluation'
            save_path: Path to save the report, if None uses default path
            
        Returns:
            Report as a string
        """
        # Compute summary statistics if not already done
        self.compute_summary_statistics(mode)
        
        # Start building the report
        report = []
        report.append(f"# Performance Report: {mode.capitalize()} Mode")
        report.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Session started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.session_start_time))}")
        report.append(f"Session duration: {time.time() - self.session_start_time:.1f} seconds")
        report.append(f"Episodes: {self.episode_count[mode]}")
        report.append(f"Total steps: {self.total_steps[mode]}")
        report.append("")
        
        # Add summary statistics for each metric
        report.append("## Summary Statistics")
        report.append("|  Metric  |   Mean   |   Std   |   Min   |   Max   |  Median  |   Latest  |")
        report.append("|----------|---------|---------|---------|---------|----------|----------|")
        
        for metric in self.metrics_config[f'{mode}_metrics']:
            if metric in self.summary_stats[mode]:
                stats = self.summary_stats[mode][metric]
                if not stats:
                    continue
                
                # Format values
                mean = f"{stats.get('mean', float('nan')):.4f}"
                std = f"{stats.get('std', float('nan')):.4f}"
                min_val = f"{stats.get('min', float('nan')):.4f}"
                max_val = f"{stats.get('max', float('nan')):.4f}"
                median = f"{stats.get('p50', float('nan')):.4f}"
                latest = f"{stats.get('latest', float('nan')):.4f}"
                
                report.append(f"| {metric} | {mean} | {std} | {min_val} | {max_val} | {median} | {latest} |")
        
        report.append("")
        
        # Add success rate if available for evaluation mode
        if mode == 'evaluation' and 'success' in self.all_episodes[mode]:
            success_rate = self.compute_success_rate(mode=mode)
            report.append(f"## Success Rate: {success_rate:.2%}")
            report.append("")
        
        # Add auxiliary tasks if available
        if self.auxiliary_metrics and mode == 'training':
            report.append("## Auxiliary Tasks")
            report.append("|  Task  |   Mean Loss   |   Min Loss   |   Max Loss   |   Latest Loss  |")
            report.append("|--------|--------------|-------------|-------------|---------------|")
            
            for metric_name, values in self.auxiliary_metrics.items():
                if len(values) == 0:
                    continue
                
                mean = np.mean(values)
                min_val = np.min(values)
                max_val = np.max(values)
                latest = values[-1]
                
                report.append(f"| {metric_name} | {mean:.6f} | {min_val:.6f} | {max_val:.6f} | {latest:.6f} |")
            
            report.append("")
        
        # Combine the report
        report_text = "\n".join(report)
        
        # Save if requested
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'stats', f'{mode}_report_{self.timestamp}.md')
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Saved performance report to {save_path}")
        except Exception as e:
            print(f"Warning: Could not save report to {save_path}: {e}")
        
        return report_text
    
    def save_metrics_data(self, filepath: str = None):
        """
        Save all metrics data to a file.
        
        Args:
            filepath: Path to save the data, if None uses default path
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'stats', f'metrics_data_{self.timestamp}.json')
        
        try:
            # Export data as JSON
            data_json = self.export_data(mode='all', format='json')
            
            # Save to file
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(data_json)
            
            print(f"Saved metrics data to {filepath}")
            return filepath
        except Exception as e:
            print(f"Warning: Could not save metrics data to {filepath}: {e}")
            return None
    
    def load_metrics_data(self, filepath: str):
        """
        Load metrics data from a file.
        
        Args:
            filepath: Path to load the data from
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore metrics data
            if 'training' in data:
                self.all_episodes['training'] = defaultdict(list, data['training'])
            if 'evaluation' in data:
                self.all_episodes['evaluation'] = defaultdict(list, data['evaluation'])
            if 'auxiliary' in data:
                self.auxiliary_metrics = defaultdict(list, data['auxiliary'])
            
            # Restore metadata
            if 'metadata' in data:
                if 'episode_count' in data['metadata']:
                    self.episode_count = data['metadata']['episode_count']
                if 'total_steps' in data['metadata']:
                    self.total_steps = data['metadata']['total_steps']
                if 'timestamp' in data['metadata']:
                    self.timestamp = data['metadata']['timestamp']
            
            # Rebuild moving averages
            window = self.metrics_config['moving_average_window']
            for mode in ['training', 'evaluation']:
                for metric, values in self.all_episodes[mode].items():
                    self.moving_averages[mode][metric] = deque(values[-window:], maxlen=window)
            
            print(f"Loaded metrics data from {filepath}")
            return True
        except Exception as e:
            print(f"Warning: Could not load metrics data from {filepath}: {e}")
            return False
    
    def close(self):
        """
        Close all resources.
        """
        # Close CSV files
        for mode in ['training', 'evaluation']:
            if self.csv_files[mode] is not None:
                try:
                    self.csv_files[mode].close()
                except Exception:
                    pass
        
        # Close TensorBoard writer
        if self.tb_writer is not None:
            try:
                self.tb_writer.close()
            except Exception:
                pass
    
    def plot_metrics(self, metrics: List[str] = None, mode: str = 'training', save_path: str = None, show: bool = True):
        """
        Plot metrics over episodes.
        
        Args:
            metrics: List of metrics to plot, if None uses default metrics for the mode
            mode: Either 'training' or 'evaluation'
            save_path: Path to save the plot, if None uses default path
            show: Whether to show the plot
        """
        if metrics is None:
            metrics = self.metrics_config[f'{mode}_metrics']
        
        # Set up the figure
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
        if n_metrics == 1:
            axes = [axes]  # Make it iterable
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Try to get the mean version if it exists
            if f'{metric}_mean' in self.all_episodes[mode] and len(self.all_episodes[mode][f'{metric}_mean']) > 0:
                values = self.all_episodes[mode][f'{metric}_mean']
                y_label = f'{metric} (mean)'
            elif metric in self.all_episodes[mode] and len(self.all_episodes[mode][metric]) > 0:
                values = self.all_episodes[mode][metric]
                y_label = metric
            else:
                continue  # Skip if no data
            
            # Get episode numbers
            episodes = list(range(1, len(values) + 1))
            
            # Plot raw values
            ax.plot(episodes, values, 'b-', alpha=0.3, label='Raw')
            
            # Calculate moving average
            window = min(len(values), self.metrics_config['moving_average_window'])
            if window > 1:
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                # Plot moving average
                ax.plot(episodes[window-1:], moving_avg, 'r-', label=f'Moving Avg (n={window})')
            
            # Set labels and title
            ax.set_ylabel(y_label)
            ax.set_title(f'{metric} over Episodes ({mode})')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
        
        # Set x-label for the bottom plot
        axes[-1].set_xlabel('Episode')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path is None and not show:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, 'plots', f'{mode}_metrics_{timestamp}.png')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=self.metrics_config['visualization_dpi'])
            print(f"Saved metrics plot to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()


class MetricsTracker:
    """
    A convenient wrapper around MetricsManager that provides a simplified interface
    for tracking metrics during training and inference.
    
    This class acts as a facade to the more complex MetricsManager, providing
    simple methods for common operations like updating metrics and generating summaries.
    """
    def __init__(self, config: Dict, output_dir: str, mode: str = 'training'):
        """
        Initialize a metrics tracker.
        
        Args:
            config: Configuration dictionary containing metrics settings
            output_dir: Directory to save metrics outputs
            mode: Mode of operation ('training' or 'inference')
        """
        self.mode = mode
        self.output_dir = output_dir
        
        # Get metrics configuration from config
        metrics_config = {}
        if 'metrics' in config:
            metrics_config = config['metrics']
        
        # Initialize the metrics manager
        self.manager = MetricsManager(config)
        self.manager.configure(metrics_config)
        self.manager.set_output_dir(output_dir)
        
        # Start tracking
        self.manager.reset()
        self.manager.start_episode(mode=self.mode)
        self.current_step = 0
    
    def update(self, metrics_dict: Dict[str, Any]):
        """
        Update metrics with new values.
        
        Args:
            metrics_dict: Dictionary of metric names and values
        """
        self.current_step += 1
        
        # Update each metric
        for name, value in metrics_dict.items():
            # Convert numpy arrays or lists to scalar values if needed
            if isinstance(value, (np.ndarray, list)) and len(value) == 1:
                value = value[0]
            
            # Update the metric
            self.manager.update_metric(name, value, step=self.current_step, mode=self.mode)
    
    def end_episode(self, info: Dict[str, Any] = None):
        """
        End the current episode and compute statistics.
        
        Args:
            info: Additional episode information
        """
        self.manager.end_episode(mode=self.mode, info=info)
        self.manager.start_episode(mode=self.mode)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get a summary of all tracked metrics.
        
        Returns:
            Dictionary of metric statistics
        """
        return self.manager.compute_summary_statistics(mode=self.mode)
    
    def export_csv(self, filepath: str = None):
        """
        Export metrics to CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, f"{self.mode}_metrics.csv")
        return self.manager.export_data(mode=self.mode, format='csv')
    
    def generate_report(self):
        """
        Generate a comprehensive report of metrics.
        """
        self.manager.generate_report(mode=self.mode, save_path=self.output_dir)
    
    def close(self):
        """
        Close the metrics tracker and release resources.
        """
        try:
            if hasattr(self.manager, 'tb_writer') and self.manager.tb_writer is not None:
                self.manager.tb_writer.close()
            
            # Save final metrics
            self.manager.save_metrics(f"{self.mode}_final_metrics.json")
            
            print(f"Metrics tracker closed. Data saved to {self.output_dir}")
        except Exception as e:
            print(f"Error while closing metrics tracker: {e}")