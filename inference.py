import os
import time
import argparse
import yaml
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

from src.environment.airsim_env import AirSimDroneEnv as AirSimUAVEnvironment
from src.models.policy_module.shared_trunk import ActorCriticNetwork
from src.algorithms.sac import SAC
from src.utils.visualizer import Visualizer
from src.utils.metrics import MetricsTracker
from src.utils.domain_random import DomainRandomizer

def parse_args():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description="Run UAV RL Controller in inference mode")
    parser.add_argument("--config", type=str, default="config/inference.yaml", 
                        help="Path to config file (default: config/inference.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint directory (if not specified, will use the path in config)")
    parser.add_argument("--version", type=str, default="best", 
                        help="Checkpoint version to load (best, latest or timestamp)")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (auto, cuda, xpu, cpu, overrides config file)")
    parser.add_argument("--render", action="store_true", 
                        help="Render visualization (overrides config setting)")
    parser.add_argument("--record", action="store_true", 
                        help="Record video of inference (overrides config setting)")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save recordings and logs (overrides config file)")
    parser.add_argument("--episodes", type=int, default=None, 
                        help="Number of episodes to run (overrides config file)")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize internal network activations (overrides config setting)")
    parser.add_argument("--deterministic", action="store_true", 
                        help="Use deterministic actions (mean of policy, overrides config setting)")
    parser.add_argument("--show_attention", action="store_true", 
                        help="Visualize attention maps if available (overrides config setting)")
    parser.add_argument("--airsim_client_timeout", type=float, default=None,
                        help="AirSim client connection timeout in seconds (overrides config file)")
    
    return parser.parse_args()

def load_config_with_inheritance(config_path):
    """Load YAML configuration with inheritance support.
    
    Supports _inherit field in YAML files to inherit from another config file.
    Child config values override parent config values.
    
    Args:
        config_path: Path to the main config file
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        FileNotFoundError: 如果配置文件或父配置文件不存在
        yaml.YAMLError: 如果YAML格式无效
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                config = yaml.safe_load(f)
                if config is None:  # 空文件或格式错误
                    print(f"警告：配置文件 {config_path} 为空或格式错误，使用空字典")
                    config = {}
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"配置文件 {config_path} YAML格式错误: {str(e)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件 {config_path} 不存在，请检查路径")
    
    # Process inheritance if specified
    if '_inherit' in config:
        try:
            parent_path = os.path.join(os.path.dirname(config_path), config['_inherit'])
            try:
                with open(parent_path, 'r', encoding='utf-8') as f:
                    try:
                        parent_config = yaml.safe_load(f)
                        if parent_config is None:  # 空文件或格式错误
                            print(f"警告：父配置文件 {parent_path} 为空或格式错误，使用空字典")
                            parent_config = {}
                    except yaml.YAMLError as e:
                        raise yaml.YAMLError(f"父配置文件 {parent_path} YAML格式错误: {str(e)}")
            except FileNotFoundError:
                raise FileNotFoundError(f"父配置文件 {parent_path} 不存在，请检查路径或_inherit值 '{config['_inherit']}'")
            
            # Remove inheritance key from config
            inherit_key = config.pop('_inherit')
            
            # Deep merge the configs (parent serves as base, child overrides)
            merged_config = deep_merge_configs(parent_config, config)
            print(f"已加载配置 {config_path} (继承自 {inherit_key})")
            return merged_config
        except Exception as e:
            print(f"处理配置继承时出错: {str(e)}")
            raise
    else:
        print(f"已加载配置 {config_path}")
        return config

def deep_merge_configs(base, override):
    """Deep merge two configuration dictionaries.
    
    Values from override dict will take precedence over base dict.
    
    Args:
        base: Base configuration (parent)
        override: Override configuration (child)
        
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_configs(result[key], value)
        else:
            # Override or add values
            result[key] = value
    
    return result

def select_device(device_arg):
    """Select appropriate device based on arguments and availability."""
    if device_arg == "auto":
        # Try to use XPU first (Intel GPU), then CUDA, then CPU
        if torch.xpu.is_available():
            return torch.device("xpu:0")
        elif torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    elif device_arg == "xpu":
        if torch.xpu.is_available():
            return torch.device("xpu:0")
        else:
            print("Warning: XPU requested but not available. Using CPU instead.")
            return torch.device("cpu")
    elif device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            print("Warning: CUDA requested but not available. Using CPU instead.")
            return torch.device("cpu")
    else:
        return torch.device("cpu")

def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()
    
    # Load config with inheritance support
    config = load_config_with_inheritance(args.config)
    
    # 确保所有必需的配置节存在
    if 'inference' not in config:
        config['inference'] = {}
    if 'visualization' not in config:
        config['visualization'] = {}
    if 'general' not in config:
        config['general'] = {}
    if 'environment' not in config:
        config['environment'] = {}
    if 'airsim' not in config.get('environment', {}):
        config['environment']['airsim'] = {}
    if 'metrics' not in config:
        config['metrics'] = {}
        
    # Override inference settings if provided by user
    if args.render:
        config['visualization']['enabled'] = True
        config['visualization']['render_mode'] = 'human'
    
    if args.record:
        config['visualization']['save_video'] = True
    
    if args.deterministic:
        config['inference']['deterministic'] = True
    
    if args.visualize:
        config['visualization']['enabled'] = True
    
    if args.show_attention:
        config['visualization']['show_attention'] = True
    
    if args.episodes is not None:
        config['inference']['n_episodes'] = args.episodes
    
    if args.airsim_client_timeout is not None:
        # 确保环境和airsim配置节存在
        if 'environment' not in config:
            config['environment'] = {}
        if 'airsim' not in config['environment']:
            config['environment']['airsim'] = {}
        config['environment']['airsim']['timeout_ms'] = int(args.airsim_client_timeout * 1000)
    
    if args.device is not None:
        # 确保general配置节存在
        if 'general' not in config:
            config['general'] = {}
        config['general']['device'] = args.device
    
    if args.output_dir is not None:
        output_dir_base = args.output_dir
    else:
        output_dir_base = config.get('general', {}).get('log_dir', 'output')
    
    # 处理检查点路径
    checkpoint_path = None
    if args.checkpoint is not None:
        # 优先使用命令行指定的检查点
        checkpoint_path = args.checkpoint
        print(f"将使用命令行指定的检查点路径: {checkpoint_path}")
    elif 'model_path' in config['inference'] and config['inference']['model_path']:
        # 如果配置文件中有非空模型路径，使用它
        checkpoint_path = config['inference']['model_path']
        print(f"将使用配置文件中的模型路径: {checkpoint_path}")
    else:
        # 尝试使用默认检查点目录
        default_checkpoint_dir = config.get('general', {}).get('checkpoint_dir', 'checkpoints')
        if os.path.exists(default_checkpoint_dir):
            checkpoint_path = default_checkpoint_dir
            print(f"未指定模型路径，将使用默认检查点目录: {default_checkpoint_dir}")
        else:
            print(f"警告: 默认检查点目录 {default_checkpoint_dir} 不存在，将使用未训练的模型")
    
    # 处理检查点版本
    if args.version is not None:
        # 优先使用命令行指定的版本
        checkpoint_version = args.version
    elif 'checkpoint_type' in config['inference']:
        # 如果配置文件中有版本，使用它
        checkpoint_version = config['inference']['checkpoint_type']
    else:
        # 否则使用默认值 'best'
        checkpoint_version = 'best'
    
    print(f"将使用检查点版本: {checkpoint_version}")
    
    # Setup device
    device_str = config.get('general', {}).get('device', 'auto')
    device = select_device(device_str)
    print(f"Using device: {device}")
    
    # Setup output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir_base, timestamp)
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(output_dir, exist_ok=True)
    if config.get('visualization', {}).get('save_video', False):
        os.makedirs(video_dir, exist_ok=True)
    
    # Save inference config
    with open(os.path.join(output_dir, 'inference_config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # 创建环境配置
    env_config = config['environment'].copy()
    env_config['render'] = config.get('visualization', {}).get('enabled', False)
    
    # 域随机化设置
    domain_randomizer = None
    if config.get('domain_randomization', {}).get('enabled', False):
        try:
            # 对推理模式设置域随机化
            domain_random_config = config.get('domain_randomization', {})
            
            # 在推理时使用固定难度级别，不使用课程学习
            domain_random_config['use_curriculum_learning'] = False
            domain_random_config['fixed_difficulty'] = config.get('inference', {}).get('randomization_difficulty', 0.5)
            
            print(f"\n初始化域随机化: 难度={domain_random_config['fixed_difficulty']}")
            domain_randomizer = DomainRandomizer(domain_random_config)
        except Exception as e:
            print(f"\n警告: 初始化域随机化时出错: {str(e)}")
            print("将继续使用无域随机化的环境")
            domain_randomizer = None
    
    # 创建环境，并传入域随机化器
    env = AirSimUAVEnvironment(
        env_config, 
        evaluation=True, 
        domain_randomizer=domain_randomizer
    )
    
    # Create agent
    agent = SAC(config, device)
    
    # Load checkpoint (if provided)
    try:
        if checkpoint_path:
            agent.load(checkpoint_path, checkpoint_version)
            print(f"\n成功加载模型: {checkpoint_path} (版本: {checkpoint_version})")
        else:
            print("\n警告: 未提供有效的模型路径，将使用未训练的模型进行推理。")
            print("若需使用训练好的模型，请通过 --checkpoint 参数指定，或在配置文件中设置 inference.model_path")
    except Exception as e:
        print(f"\n错误: 加载模型失败: {str(e)}")
        print("将尝试使用未训练的模型继续...")  # 继续使用未初始化的模型
    
    # Set agent to eval mode
    agent.ac_network.eval()
    
    # Setup visualizer if requested
    visualizer = None
    if config.get('visualization', {}).get('enabled', False):
        visualizer = Visualizer(output_dir)
    
    # Setup video writer if recording
    video_writer = None
    if config.get('visualization', {}).get('save_video', False):
        try:
            # Get frame size from the environment
            test_frame = env.get_rgb_image()
            if test_frame is not None and test_frame.ndim >= 2:
                frame_height, frame_width = test_frame.shape[:2]
                video_path = os.path.join(video_dir, f"inference_{timestamp}.mp4")
                video_fps = config.get('visualization', {}).get('video_fps', 30)
                video_codec = config.get('visualization', {}).get('video_codec', 'mp4v')
                video_writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*video_codec),
                    video_fps,
                    (frame_width, frame_height)
                )
                print(f"Recording video to {video_path} ({video_fps} FPS)")
            else:
                print("Warning: Could not get valid frame from environment, video recording disabled")
        except Exception as e:
            print(f"Error setting up video recording: {e}")
            video_writer = None
    
    # Setup metrics tracking
    metrics_tracker = MetricsTracker(config, output_dir, 'inference')
    
    # Run inference for specified number of episodes
    n_episodes = config.get('inference', {}).get('n_episodes', 10)
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Reset environment
        state = env.reset()
        
        print(f"Starting episode {episode + 1}/{n_episodes}")
        
        # Episode loop
        while not done:
            # Select action from policy (deterministic if specified)
            deterministic = config.get('inference', {}).get('deterministic', False)
            action = agent.select_action(state, deterministic=deterministic)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Update episode stats
            episode_reward += reward
            episode_length += 1
            
            # Track metrics
            metrics_tracker.update({
                'reward': reward,
                'episode_length': episode_length,
                'cumulative_reward': episode_reward,
                'action_values': action.tolist(),
                **info  # Include all environment info in metrics
            })
            
            # Visualize if requested
            if visualizer is not None and config.get('visualization', {}).get('enabled', False):
                visualizer.update(
                    state=state,
                    action=action,
                    agent=agent,
                    step=episode_length,
                    episode=episode,
                    show_attention=config.get('visualization', {}).get('show_attention', False)
                )
            
            # Record video frame if requested
            if video_writer is not None and config.get('visualization', {}).get('save_video', False):
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    # OpenCV uses BGR format
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
            
            # Move to next state
            state = next_state
            
            # Brief pause for visualization
            if config.get('visualization', {}).get('enabled', False):
                render_delay = config.get('visualization', {}).get('render_delay', 0.01)
                time.sleep(render_delay)
        
        # Episode finished
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1} finished: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Get final metrics from the metrics tracker
    metrics_summary = metrics_tracker.get_summary()
    
    # Calculate and print statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print("\nInference complete!")
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.2f} ± {std_length:.2f}")
    
    # Generate detailed metrics report
    if config.get('inference', {}).get('generate_report', True):
        metrics_tracker.generate_report()
    
    # Export metrics in different formats
    if config.get('inference', {}).get('export_csv', True):
        metrics_tracker.export_csv(os.path.join(output_dir, "metrics.csv"))
    
    # Save detailed results dictionary
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'std_length': std_length,
        'timestamp': timestamp,
        'checkpoint': checkpoint_path,
        'checkpoint_version': checkpoint_version,
        'n_episodes': n_episodes,
        'deterministic': config.get('inference', {}).get('deterministic', False),
        'config': config,
        'metrics_summary': metrics_summary
    }
    
    # Save results to file
    np.save(os.path.join(output_dir, "results.npy"), results)
    
    # Also save as JSON for easier inspection
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in results.items() if k != 'config'}
        json.dump(json_results, f, indent=2)
    
    # 正确清理所有资源，确保即使出错也会执行清理
    try:
        # 打印最终的性能汇总
        print("\n====== 推理性能汇总 ======")
        print(f"\u8fd0行回合数: {n_episodes}")
        print(f"\u5e73均回报: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"\u5e73均回合长度: {mean_length:.2f} ± {std_length:.2f}")
        
        if metrics_summary:
            print("\n\u5173键指标详情:")
            for key, value in metrics_summary.items():
                if isinstance(value, dict) and 'mean' in value:
                    print(f"  {key}: {value['mean']:.4f} ± {value.get('std', 0):.4f}")
        
        # 先关闭视频写入器
        if video_writer is not None:
            try:
                video_writer.release()
                print("\n已关闭视频写入器")
            except Exception as e:
                print(f"\n关闭视频写入器时出错: {str(e)}")
                
        # 完成指标跟踪和导出报告
        try:
            if config.get('inference', {}).get('export_csv', True):
                metrics_tracker.export_csv()
                print("\n已导出指标数据为CSV格式")
            
            if config.get('inference', {}).get('generate_report', True):
                metrics_tracker.generate_report()
                print("\n已生成性能分析报告")
            
            # 关闭指标跟踪器
            metrics_tracker.close()
            print("\n已关闭指标跟踪器")
        except Exception as e:
            print(f"\n指标处理过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
    finally:
        # 确保环境始终被关闭
        try:
            env.close()
            print("\n已关闭环境连接")
        except Exception as e:
            print(f"\n关闭环境时出错: {str(e)}")
        
        try:
            # 释放 PyTorch 资源
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
            if hasattr(torch, 'xpu'):
                torch.xpu.empty_cache()
            
            # 尝试释放其他资源
            import gc
            gc.collect()
        except Exception as e:
            print(f"\n释放资源时出错: {str(e)}")
        
    print(f"\n结果已保存到 {output_dir}")
    print("\n====== 推理完成 ======")

if __name__ == "__main__":
    main()
