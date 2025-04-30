import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Union, Any

from src.models.policy_module.actor import Actor, SharedTrunkActor
from src.models.policy_module.critic import TwinCritic, SharedTrunkCritic
from src.models.policy_module.shared_trunk import SharedTrunk, ActorCriticNetwork
from src.algorithms.replay_buffer import PrioritizedReplayBuffer, Transition
from src.algorithms.auxiliary import ContrastivePredictiveCoding, FrameReconstruction, PoseRegression

class SAC:
    """
    Soft Actor-Critic implementation with twin critics and automatic entropy tuning.
    Implements the framework's "Soft Actor-Critic (SAC) 变体 + 双Critic 降低 overestimation" feature.
    
    Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
               Learning with a Stochastic Actor" (https://arxiv.org/abs/1801.01290)
    """
    def __init__(self, config: dict, device: torch.device):
        """
        Initialize SAC algorithm with full support for multiple auxiliary tasks.
        
        Args:
            config: Configuration dictionary with algorithm parameters
            device: Device to run the algorithm on (CPU/GPU/XPU)
        """
        self.config = config
        self.device = device
        
        # Extract SAC hyperparameters
        self.gamma = config.get('gamma', 0.99)  # Discount factor
        self.tau = config.get('tau', 0.005)     # Target network update rate
        self.alpha_lr = config.get('alpha_lr', 3e-4)  # Entropy temperature learning rate
        self.actor_lr = config.get('actor_lr', 3e-4)  # Actor learning rate
        self.critic_lr = config.get('critic_lr', 3e-4)  # Critic learning rate
        self.buffer_size = config.get('buffer_size', 1_000_000)  # Replay buffer size
        self.batch_size = config.get('batch_size', 256)  # Batch size for updates
        self.init_alpha = config.get('init_alpha', 0.2)  # Initial entropy coefficient
        self.target_update_interval = config.get('target_update_interval', 1)  # Freq of target updates
        self.use_automatic_entropy_tuning = config.get('use_automatic_entropy_tuning', True)  # Auto adjust alpha
        self.use_prioritized_replay = config.get('use_prioritized_replay', True)  # Use prioritized replay buffer
        self.update_after = config.get('update_after', 1000)  # Start updating after this many steps
        self.update_every = config.get('update_every', 50)  # Update freq (env steps per gradient step)
        self.use_auxiliary_tasks = config.get('use_auxiliary_tasks', True)  # Use auxiliary tasks
        self.auxiliary_weight = config.get('auxiliary_weight', 0.5)  # Weight for auxiliary losses
        self.per_alpha = config.get('per_alpha', 0.6)  # Prioritized replay alpha (how much prioritization to use)
        self.per_beta_start = config.get('per_beta_start', 0.4)  # Initial importance sampling weight
        self.per_beta_frames = config.get('per_beta_frames', 100_000)  # Frames over which to anneal beta
        
        # Initialize SAC networks and get their config
        model_config = self.config.get('model', {})
        policy_config = model_config.get('policy_module', {})
        self.action_dim = self.config.get('environment', {}).get('action_space', {}).get('action_dim', 4)
        
        # Create the Actor-Critic networks
        self.ac_network = ActorCriticNetwork(model_config).to(device)
        self.actor = self.ac_network.actor
        self.critic = self.ac_network.critic
        
        # Target critic network - 创建方式和参数需与主critic保持一致
        # 使用与主网络相同维度参数的trunk，确保维度匹配
        target_trunk = SharedTrunk(policy_config).to(device)
        target_trunk.trunk_dim = self.ac_network.trunk.trunk_dim  # 确保维度一致
        
        # 使用统一的critic配置，确保维度一致
        self.target_critic = SharedTrunkCritic(target_trunk, 
                                              policy_config.get('critic', {})).to(device)
        # 硬拷贝参数
        self.hard_update(self.target_critic, self.critic)
        
        # Freeze target critic
        for p in self.target_critic.parameters():
            p.requires_grad = False
            
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # Automatic entropy tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(self.init_alpha).to(device)
            
        # Replay buffer
        if self.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size=self.buffer_size, 
                alpha=self.per_alpha,
                device=device
            )
        else:
            # Fall back to standard replay buffer
            from src.algorithms.replay_buffer import ReplayBuffer
            self.replay_buffer = ReplayBuffer(
                buffer_size=self.buffer_size,
                device=device
            )
            
        # Beta schedule for prioritized replay
        if self.use_prioritized_replay:
            self.beta_schedule = lambda frame_idx: min(1.0, self.per_beta_start + frame_idx * 
                                                  (1.0 - self.per_beta_start) / self.per_beta_frames)
        
        # Auxiliary tasks if enabled
        self.auxiliary_tasks = []
        if self.use_auxiliary_tasks:
            auxiliary_config = self.config.get('auxiliary_tasks', {})
            
            # Contrastive Predictive Coding
            if auxiliary_config.get('use_contrastive', True):
                self.auxiliary_tasks.append(
                    ContrastivePredictiveCoding(config=auxiliary_config.get('contrastive', {})).to(device)
                )
                
            # Frame Reconstruction
            if auxiliary_config.get('use_reconstruction', True):
                self.auxiliary_tasks.append(
                    FrameReconstruction(config=auxiliary_config.get('reconstruction', {})).to(device)
                )
                
            # Pose Regression
            if auxiliary_config.get('use_pose_regression', True):
                self.auxiliary_tasks.append(
                    PoseRegression(config=auxiliary_config.get('pose_regression', {})).to(device)
                )
                
            # Create optimizers for auxiliary tasks
            self.auxiliary_optimizers = []
            for task in self.auxiliary_tasks:
                self.auxiliary_optimizers.append(
                    torch.optim.Adam(task.parameters(), lr=auxiliary_config.get('lr', 3e-4))
                )
                
        # Training metrics
        self.total_steps = 0
        self.episodes = 0
        self.updates = 0
        
    def select_action(self, state, deterministic: bool = False) -> np.ndarray:
        """
        Select action from policy given state.
        
        Args:
            state: Current state/observation (can be dict or array)
            deterministic: If True, use deterministic actions (mean), useful for evaluation
            
        Returns:
            Selected action as numpy array
        """
        with torch.no_grad():
            # 处理字典类垏的observation
            if isinstance(state, dict):
                # 将字典类垏的观测转换为tensor字典
                processed_state = {}
                for key, value in state.items():
                    if key == 'image':
                        # 图像数据需要规范化
                        if value is not None:
                            # 确保是浮点类型并规范化到[0,1]
                            processed_state[key] = torch.FloatTensor(value).to(self.device) / 255.0
                        else:
                            processed_state[key] = None
                    else:
                        # 其他类垏数据
                        processed_state[key] = torch.FloatTensor(value).to(self.device)
                
                # 对于字典类垏观测，我们需要提取实际用于策略的部分
                # 这里我们假设 Actor 需要 'state' 和 'target' 部分
                # 具体逻辑应该根据模型的实际要求调整
                if 'state' in processed_state and 'target' in processed_state:
                    state_vector = torch.cat([processed_state['state'], processed_state['target']], dim=-1)
                elif 'state' in processed_state:
                    state_vector = processed_state['state']
                else:
                    # 如果缺少必要的组件，返回随机动作
                    print("警告: 观测缺少必要组件，使用随机动作")
                    return np.random.uniform(-1, 1, self.action_dim)
            else:
                # 如果不是字典，假设是原始数组，直接转换
                if not isinstance(state, torch.Tensor):
                    state_vector = torch.FloatTensor(state).to(self.device)
                else:
                    state_vector = state
                
            # 确保数据有正确的形状
            if state_vector.dim() == 1:  # 如果是向量（没有batch维度）
                state_vector = state_vector.unsqueeze(0)  # 添加batch维度
                
            # Sample action from policy - 使用处理后的state_vector而非原始state
            action, _, _ = self.actor.sample(state_vector, deterministic=deterministic)
            
        return action.detach().cpu().numpy()[0]  # Return as numpy array
    
    def update_parameters(self, current_step: int) -> Dict[str, float]:
        """
        Update networks parameters with SAC update rule.
        
        Args:
            current_step: Current step in training (for beta annealing)
            
        Returns:
            Dictionary with training metrics/losses
        """
        # Check if we should update on this step
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        metrics = {}
        
        # Compute beta for importance sampling
        if self.use_prioritized_replay:
            beta = self.beta_schedule(current_step)
        else:
            beta = 1.0  # Not used if not using PER
        
        # Sample batch from replay buffer
        if self.use_prioritized_replay:
            batch, weights, indices = self.replay_buffer.sample(self.batch_size, beta)
        else:
            batch = self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size, device=self.device)  # All weights equal
            indices = None  # No indices for standard replay buffer
            
        # Transition unpacking
        state, action, reward, next_state, done = batch
        
        # ------- Update Critics ------- #
        # Compute target Q-value
        with torch.no_grad():
            # Sample action from policy for next state
            next_action, _, next_log_prob = self.actor.sample(next_state)
            
            # Target Q-value = r + gamma * min(Q1', Q2') - alpha * log_prob
            next_q1, next_q2 = self.target_critic(next_state, next_action)
            next_q = torch.min(next_q1, next_q2)
            next_q = next_q - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * next_q
        
        # Current Q estimates
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss1 = F.mse_loss(current_q1, target_q, reduction='none')
        critic_loss2 = F.mse_loss(current_q2, target_q, reduction='none')
        
        # Apply importance sampling weights if using PER
        critic_loss1 = (critic_loss1 * weights).mean()
        critic_loss2 = (critic_loss2 * weights).mean()
        critic_loss = critic_loss1 + critic_loss2
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update priorities in PER if used
        if self.use_prioritized_replay and indices is not None:
            with torch.no_grad():
                td_errors = torch.abs(current_q1 - target_q).detach().cpu().numpy()
                self.replay_buffer.update_priorities(indices, td_errors)
        
        # ------- Update Actor ------- #
        # Compute actor loss
        current_action, _, log_prob = self.actor.sample(state)
        q1, q2 = self.critic(state, current_action)
        min_q = torch.min(q1, q2)
        
        # Actor loss = alpha * log_prob - Q(s,a)
        actor_loss = (self.alpha * log_prob - min_q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ------- Update entropy coefficient (alpha) ------- #
        if self.use_automatic_entropy_tuning:
            # Compute alpha loss
            alpha_loss = -1.0 * (self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            # Update alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Update alpha value
            self.alpha = self.log_alpha.exp()
            metrics['alpha'] = self.alpha.item()
            metrics['alpha_loss'] = alpha_loss.item()
        
        # ------- Update Target Networks ------- #
        if self.updates % self.target_update_interval == 0:
            self.soft_update(self.target_critic, self.critic, self.tau)
            
        # ------- Update Auxiliary Tasks (if enabled) ------- #
        auxiliary_losses = []
        if self.use_auxiliary_tasks and len(self.auxiliary_tasks) > 0:
            for i, (task, optimizer) in enumerate(zip(self.auxiliary_tasks, self.auxiliary_optimizers)):
                # Forward pass
                aux_loss = task(state, action, next_state)
                
                # Backward and optimize
                optimizer.zero_grad()
                aux_loss.backward()
                optimizer.step()
                
                auxiliary_losses.append(aux_loss.item())
                metrics[f'aux_loss_{i}'] = aux_loss.item()
                
            metrics['aux_loss_total'] = sum(auxiliary_losses)
        
        # Increment update counter
        self.updates += 1
        
        # Log metrics
        metrics.update({
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q1_value': current_q1.mean().item(),
            'q2_value': current_q2.mean().item(),
            'target_q': target_q.mean().item(),
            'entropy': -log_prob.mean().item()
        })
        
        return metrics
    
    def train(self, current_step: int, training_enabled: bool = True) -> Dict[str, float]:
        """
        Training method to be called from environment loop.
        
        Args:
            current_step: Current step in training
            training_enabled: Whether to enable training updates
            
        Returns:
            Dictionary with training metrics if updated, empty dict otherwise
        """
        self.total_steps = current_step
        
        if not training_enabled:
            return {}
        
        # Only update if enough steps have been taken and the update interval is met
        if current_step >= self.update_after and current_step % self.update_every == 0:
            metrics = self.update_parameters(current_step)
            return metrics
            
        return {}
    
    def save(self, directory: str) -> None:
        """
        Save model to specified directory.
        
        Args:
            directory: Directory to save model to
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Create timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save all networks
        torch.save(self.ac_network.state_dict(), os.path.join(directory, f'ac_network_{timestamp}.pt'))
        torch.save(self.target_critic.state_dict(), os.path.join(directory, f'target_critic_{timestamp}.pt'))
        
        # Save optimizers
        torch.save({
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.use_automatic_entropy_tuning else None
        }, os.path.join(directory, f'optimizers_{timestamp}.pt'))
        
        # Save training state
        torch.save({
            'alpha': self.alpha,
            'log_alpha': self.log_alpha if self.use_automatic_entropy_tuning else None,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'updates': self.updates
        }, os.path.join(directory, f'training_state_{timestamp}.pt'))
        
        # Save auxiliary tasks if enabled
        if self.use_auxiliary_tasks and len(self.auxiliary_tasks) > 0:
            for i, task in enumerate(self.auxiliary_tasks):
                torch.save(task.state_dict(), os.path.join(directory, f'auxiliary_task_{i}_{timestamp}.pt'))
        
        # Also save the latest version without timestamp for easier loading
        torch.save(self.ac_network.state_dict(), os.path.join(directory, 'ac_network_latest.pt'))
        torch.save(self.target_critic.state_dict(), os.path.join(directory, 'target_critic_latest.pt'))
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'updates': self.updates,
            'config': self.config
        }
        
        torch.save(metadata, os.path.join(directory, f'metadata_{timestamp}.pt'))
        torch.save(metadata, os.path.join(directory, 'metadata_latest.pt'))
        
        print(f"Model saved to {directory} with timestamp {timestamp}")
    
    def load(self, directory: str, version: str = 'latest') -> None:
        """
        Load model from specified directory with version.
        
        Args:
            directory: Directory to load model from
            version: Version to load ('latest' or specific timestamp)
        """
        # Get file name pattern
        if version == 'latest':
            network_path = os.path.join(directory, 'ac_network_latest.pt')
            target_critic_path = os.path.join(directory, 'target_critic_latest.pt')
            metadata_path = os.path.join(directory, 'metadata_latest.pt')
            optimizers_path = os.path.join(directory, 'optimizers_latest.pt')
            training_state_path = os.path.join(directory, 'training_state_latest.pt')
        else:
            # Use specific timestamp
            network_path = os.path.join(directory, f'ac_network_{version}.pt')
            target_critic_path = os.path.join(directory, f'target_critic_{version}.pt')
            metadata_path = os.path.join(directory, f'metadata_{version}.pt')
            optimizers_path = os.path.join(directory, f'optimizers_{version}.pt')
            training_state_path = os.path.join(directory, f'training_state_{version}.pt')
        
        # Check if files exist
        if not os.path.exists(network_path) or not os.path.exists(target_critic_path):
            raise FileNotFoundError(f"Model files not found in {directory} with version {version}")
        
        # Load networks
        self.ac_network.load_state_dict(torch.load(network_path, map_location=self.device))
        self.target_critic.load_state_dict(torch.load(target_critic_path, map_location=self.device))
        
        # Update references
        self.actor = self.ac_network.actor
        self.critic = self.ac_network.critic
        
        # Load training state if exists
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)
            self.alpha = training_state['alpha']
            if self.use_automatic_entropy_tuning and 'log_alpha' in training_state:
                self.log_alpha = training_state['log_alpha']
            self.total_steps = training_state.get('total_steps', 0)
            self.episodes = training_state.get('episodes', 0)
            self.updates = training_state.get('updates', 0)
        
        # Load optimizers if exists
        if os.path.exists(optimizers_path):
            optimizer_dict = torch.load(optimizers_path, map_location=self.device)
            self.actor_optimizer.load_state_dict(optimizer_dict['actor_optimizer'])
            self.critic_optimizer.load_state_dict(optimizer_dict['critic_optimizer'])
            if self.use_automatic_entropy_tuning and 'alpha_optimizer' in optimizer_dict:
                self.alpha_optimizer.load_state_dict(optimizer_dict['alpha_optimizer'])
        
        # Load auxiliary tasks if enabled
        if self.use_auxiliary_tasks and len(self.auxiliary_tasks) > 0:
            for i, task in enumerate(self.auxiliary_tasks):
                aux_path = os.path.join(directory, f'auxiliary_task_{i}_{version}.pt')
                if os.path.exists(aux_path):
                    task.load_state_dict(torch.load(aux_path, map_location=self.device))
        
        print(f"Model loaded from {directory} with version {version}")
        print(f"Loaded model at step {self.total_steps}, episode {self.episodes}")
    
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        """
        Soft update of target network parameters.
        θ_target = τ * θ_source + (1 - τ) * θ_target
        
        Args:
            target: Target network with parameters to be updated
            source: Source network with parameters to copy from
            tau: Interpolation parameter
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )
    
    def hard_update(self, target: nn.Module, source: nn.Module) -> None:
        """
        Hard update of target network parameters.
        θ_target = θ_source
        
        Args:
            target: Target network with parameters to be updated
            source: Source network with parameters to copy from
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)