import os
import time
import argparse
import yaml
import numpy as np
import torch
import random
import gc
from pathlib import Path
from datetime import datetime

from src.environment.airsim_env import AirSimDroneEnv as AirSimUAVEnvironment
from src.models.policy_module.shared_trunk import ActorCriticNetwork
from src.algorithms.sac import SAC
from src.utils.logger import Logger
from src.utils.evaluator import Evaluator
from src.utils.resource_monitor import ResourceMonitor
from src.utils.memory_manager import MemoryManager

def set_seed(seed=None):
    """Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed to use. If None, uses 42 as default.
    """
    # 如果未指定种子，使用默认值
    if seed is None:
        seed = 42
        print(f"警告: 未指定随机种子，使用默认值 {seed}")
    
    # 设置所有随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 对XPU设备也做相同处理（如果可用）
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)
    
    print(f"已设置所有随机种子为: {seed}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train UAV RL Controller")
    parser.add_argument("--config", type=str, default="config/training.yaml", 
                        help="Path to config file (default: config/training.yaml)")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed (overrides config file)")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (auto, cuda, xpu, cpu, overrides config file)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint_version", type=str, default="latest", 
                        help="Checkpoint version to load (latest or timestamp)")
    parser.add_argument("--eval_only", action="store_true", 
                        help="Run evaluation only (no training)")
    parser.add_argument("--log_dir", type=str, default=None, 
                        help="Directory to save logs (overrides config file)")
    parser.add_argument("--save_dir", type=str, default=None, 
                        help="Directory to save checkpoints (overrides config file)")
    parser.add_argument("--save_freq", type=int, default=None, 
                        help="Save checkpoint every N steps (overrides config file)")
    parser.add_argument("--eval_freq", type=int, default=None, 
                        help="Run evaluation every N steps (overrides config file)")
    parser.add_argument("--eval_episodes", type=int, default=None, 
                        help="Number of episodes for evaluation")
    parser.add_argument("--total_steps", type=int, default=10000000, 
                        help="Total number of training steps")
    parser.add_argument("--airsim_client_timeout", type=float, default=60.0,
                        help="AirSim client connection timeout in seconds")
    
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
            print("XPU requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    elif device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            print("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    else:
        return torch.device("cpu")

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config with inheritance support
    config = load_config_with_inheritance(args.config)
    
    # Update config with command line arguments (only if values are provided)
    if args.total_steps is not None:
        config['training']['total_steps'] = args.total_steps
    if args.seed is not None:
        config['general']['seed'] = args.seed
    if args.save_freq is not None:
        config['training']['save_interval'] = args.save_freq
    if args.eval_freq is not None:
        config['training']['eval_interval'] = args.eval_freq
    if args.eval_episodes is not None:
        config['training']['eval_episodes'] = args.eval_episodes
    if args.airsim_client_timeout is not None:
        config['environment']['airsim']['timeout_ms'] = int(args.airsim_client_timeout * 1000)
    if args.device is not None:
        config['general']['device'] = args.device
    if args.log_dir is not None:
        config['general']['log_dir'] = args.log_dir
    if args.save_dir is not None:
        config['general']['checkpoint_dir'] = args.save_dir
    
    # Setup device
    # 当命令行参数未指定时，使用配置文件中的设备设置
    device_setting = args.device if args.device is not None else config['general'].get('device', 'auto')
    device = select_device(device_setting)
    print(f"Using device: {device}")
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 使用配置文件中的默认值，如果命令行参数为None
    log_dir_base = args.log_dir if args.log_dir is not None else config['general'].get('log_dir', 'logs')
    save_dir_base = args.save_dir if args.save_dir is not None else config['general'].get('checkpoint_dir', 'checkpoints')
    
    # 确保目录路径存在
    if not os.path.exists(log_dir_base):
        os.makedirs(log_dir_base, exist_ok=True)
        print(f"创建日志目录: {log_dir_base}")
    
    if not os.path.exists(save_dir_base):
        os.makedirs(save_dir_base, exist_ok=True)
        print(f"创建检查点目录: {save_dir_base}")
    
    # 创建带时间戳的子目录
    log_dir = os.path.join(log_dir_base, timestamp)
    save_dir = os.path.join(save_dir_base, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"训练日志存储到: {log_dir}")
    print(f"模型检查点存储到: {save_dir}")
    
    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Setup logger
    logger = Logger(log_dir)
    
    # 资源监控已禁用
    # resource_monitor = ResourceMonitor(logger=logger, interval=5)
    # resource_monitor.start()
    
    # 初始化内存管理器
    memory_check_interval = config.get('memory_management', {}).get('check_interval', 1000)
    memory_threshold = config.get('memory_management', {}).get('emergency_threshold', 0.85)
    memory_manager = MemoryManager(
        device=device,
        check_interval=memory_check_interval,
        emergency_threshold=memory_threshold,
        clear_cuda_cache=True,
        debug_mode=config.get('debug', False)
    )
    print(f"内存管理器已初始化: 检查间隔={memory_check_interval}步, 阈值={memory_threshold*100:.1f}%")
    
    # Create environment
    env = AirSimUAVEnvironment(config['environment'])
    eval_env = AirSimUAVEnvironment(config['environment'], evaluation=True)
    
    # Setup evaluator
    evaluator = Evaluator(
        env=eval_env,
        num_episodes=args.eval_episodes,
        render=config['evaluation'].get('render', False),
        record=config['evaluation'].get('record', False),
        record_dir=os.path.join(log_dir, 'videos')
    )
    
    # Create agent
    agent = SAC(config, device)
    
    # Load checkpoint if specified
    if args.checkpoint is not None:
        agent.load(args.checkpoint, args.checkpoint_version)
        print(f"Loaded checkpoint from {args.checkpoint} (version {args.checkpoint_version})")
    
    # Evaluation only mode
    if args.eval_only:
        print("Running evaluation only...")
        eval_results = evaluator.evaluate(agent)
        for key, value in eval_results.items():
            print(f"{key}: {value}")
        return
    
    # 训练循环变量
    total_steps = config['training'].get('total_steps', config['training'].get('num_steps', 1000000))
    save_freq = config['training'].get('save_freq', config['training'].get('save_interval', 10000))
    eval_freq = config['training'].get('eval_freq', config['training'].get('eval_interval', 5000))
    initial_step = agent.total_steps  # For resuming from checkpoint
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    episode_count = agent.episodes
    
    # Reset environment - Gymnasium API返回(observation, info)元组
    state, _ = env.reset()
    
    # Main training loop
    print(f"Starting training for {total_steps} steps...")
    for step in range(initial_step, total_steps):
        # Select action from policy
        if step < config['training'].get('start_steps', 1000):
            # Random actions for initial exploration
            action = env.action_space.sample()
        else:
            # Sample action from policy
            action = agent.select_action(state)
        
        # Take step in environment
        # 新版Gymnasium API返回5个值：(observation, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, info = env.step(action)
        # 将terminated和truncated合并为done标志
        done = terminated or truncated
        
        # Store transition in replay buffer
        # Convert to tensors - 注意处理字典类垏observation
        # 检查state是否为字典类垏（环境返回的复合观测）
        if isinstance(state, dict):
            # 处理字典类垏观测 - 将各组件转换为tensor
            state_dict = {}
            for key, value in state.items():
                if key == 'image':
                    # 图像数据需要转换为float并规范化
                    if value is not None:
                        state_dict[key] = torch.FloatTensor(value).to(device) / 255.0  # 规范化到[0,1]
                    else:
                        state_dict[key] = None
                else:
                    # 其他类垏的数据（state、target等）
                    state_dict[key] = torch.FloatTensor(value).to(device)
            state_tensor = state_dict
        else:
            # 如果不是字典，直接转换（兼容性处理）
            state_tensor = torch.FloatTensor(state).to(device)
        
        # 同样处理next_state
        if isinstance(next_state, dict):
            next_state_dict = {}
            for key, value in next_state.items():
                if key == 'image':
                    # 图像数据需要转换为float并规范化
                    if value is not None:
                        next_state_dict[key] = torch.FloatTensor(value).to(device) / 255.0
                    else:
                        next_state_dict[key] = None
                else:
                    next_state_dict[key] = torch.FloatTensor(value).to(device)
            next_state_tensor = next_state_dict
        else:
            next_state_tensor = torch.FloatTensor(next_state).to(device)
        
        # 其他数据转换
        action_tensor = torch.FloatTensor(action).to(device)
        reward_tensor = torch.FloatTensor([reward]).to(device)
        done_tensor = torch.FloatTensor([float(done)]).to(device)
        
        # Add to buffer
        agent.replay_buffer.push(
            state_tensor, 
            action_tensor, 
            reward_tensor, 
            next_state_tensor, 
            done_tensor
        )
        
        # Update current episode stats
        current_episode_reward += reward
        current_episode_length += 1
        
        # Update agent
        if step >= config['training'].get('update_after', 1000):
            update_metrics = agent.train(step)
            
            # 日志更新指标并打印详细训练信息
            if update_metrics:
                # 打印训练步骤详细信息
                print(f"Step {step}/{total_steps} (进度: {(step/total_steps)*100:.2f}%) | ", end="")
                metric_strs = []
                for key, value in update_metrics.items():
                    logger.log_scalar(f"train/{key}", value, step)
                    metric_strs.append(f"{key}: {value:.4f}")
                print(f"Loss: {' | '.join(metric_strs)} | 模型正在学习...")
                
                # 定期执行内存管理
                cleaned, mem_stats = memory_manager.check_and_collect(step, logger)
                if cleaned:
                    # 记录内存清理信息
                    logger.log_scalar("memory/cleanup_triggered", 1.0, step)
                    if mem_stats.get('system_ram_percent', 0) > 0:
                        print(f"⚠️ 内存清理: 系统内存使用率 {mem_stats.get('system_ram_percent', 0)*100:.1f}%, " +
                              f"{'CUDA' if mem_stats.get('gpu_used', 0) > 0 else 'XPU' if mem_stats.get('xpu_used', 0) > 0 else ''}" +
                              f"显存使用 {mem_stats.get('gpu_used', mem_stats.get('xpu_used', 0)):.2f}GB")
        
        # Handle episode termination
        if done:
            # Reset environment - Gymnasium API返回(observation, info)元组
            state, _ = env.reset()
            
            # Log episode results
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            logger.log_scalar("train/episode_reward", current_episode_reward, step)
            logger.log_scalar("train/episode_length", current_episode_length, step)
            
            # Reset episode stats
            current_episode_reward = 0
            current_episode_length = 0
            episode_count += 1
            agent.episodes = episode_count
            
            # Print episode summary
            if episode_count % 10 == 0:
                mean_reward = np.mean(episode_rewards[-10:])
                mean_length = np.mean(episode_lengths[-10:])
                print(f"Episode {episode_count}, Step {step}/{total_steps}, "
                      f"Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.2f}")
        else:
            # Continue to next step
            state = next_state
        
        # Save checkpoint
        if (step + 1) % save_freq == 0:
            # 在保存模型前执行内存回收，避免不必要的临时对象占用空间
            memory_manager.check_and_collect(step, logger, force=True)
            
            try:
                agent.save(save_dir)
                print(f"Saved checkpoint at step {step + 1}")
            except Exception as e:
                print(f"⚠️ 保存模型时出错: {e}")
                # 出现错误时紧急清理内存
                memory_manager.emergency_cleanup(logger, step)
        
        # Run evaluation
        if (step + 1) % eval_freq == 0:
            # 在评估前执行内存回收，确保有足够空间执行评估
            memory_manager.check_and_collect(step, logger, force=True)
            
            try:
                print(f"Running evaluation at step {step + 1}...")
                eval_results = evaluator.evaluate(agent)
                
                # Log evaluation results
                for key, value in eval_results.items():
                    logger.log_scalar(f"eval/{key}", value, step)
            except Exception as e:
                print(f"⚠️ 评估过程中出错: {e}")
                # 出现错误时紧急清理内存
                memory_manager.emergency_cleanup(logger, step)
            
            print(f"Evaluation at step {step + 1}: Mean Reward: {eval_results['mean_reward']:.2f}")
    
    # 资源监控已禁用
    # resource_monitor.stop()
    
    # 训练结束前进行一次全面内存清理
    print("训练完成，执行最终内存清理...")
    memory_manager.emergency_cleanup(logger, step)
    
    # 记录最终内存使用情况
    final_mem_stats = memory_manager.get_memory_stats()
    logger.log_scalar("memory/final_system_ram", final_mem_stats.get('system_ram_percent', 0), step)
    if memory_manager.use_cuda:
        logger.log_scalar("memory/final_gpu_used_gb", final_mem_stats.get('gpu_used', 0), step)
    elif memory_manager.use_xpu:
        logger.log_scalar("memory/final_xpu_used_gb", final_mem_stats.get('xpu_used', 0), step)
    
    # Final save and evaluation
    try:
        print("保存最终模型...")
        agent.save(save_dir)
        
        print("执行最终评估...")
        eval_results = evaluator.evaluate(agent)
        
        # Print final summary
        print("========== 训练完成 ==========")
        print(f"最终评估结果: 平均奖励 {eval_results['mean_reward']:.2f}")
        print(f"模型已保存至: {save_dir}")
        print(f"日志已保存至: {log_dir}")
        
        # 记录训练总结
        logger.log_scalar("final/mean_reward", eval_results['mean_reward'], step)
        logger.log_scalar("final/episode_count", episode_count, step)
        logger.log_scalar("final/training_steps", step, step)
        logger.log_text("final/training_summary", f"训练完成，共执行{step}步，{episode_count}回合，最终平均奖励{eval_results['mean_reward']:.2f}")
    except Exception as e:
        print(f"⚠️ 最终保存或评估过程中出错: {e}")
    finally:
        # 确保资源释放
        print("释放环境资源...")
        try:
            env.close()
        except Exception as e:
            print(f"关闭训练环境出错: {e}")
            
        try:
            eval_env.close() 
        except Exception as e:
            print(f"关闭评估环境出错: {e}")
            
        # 最终确保所有GPU/内存资源释放
        memory_manager.collect_garbage(force=True)
        print("内存资源已释放")

if __name__ == "__main__":
    main()
