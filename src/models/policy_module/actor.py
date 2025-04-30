import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import os
from typing import Dict, List, Optional, Tuple, Union

class Actor(nn.Module):
    """
    Actor network for continuous action spaces.
    Implements framework's "Actor Head (策略网络): MLP → 连续动作输出" feature.
    Outputs mean and log standard deviation for diagonal Gaussian policy.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Actor network.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # 定义默认参数
        self.trunk_dim = 512  # 默认值
        
        # 如果没有提供配置，从文件加载
        if config is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "config", "default.yaml"
            )
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            # 从完整配置中提取trunk_dim和其他参数
            if 'model' in full_config and 'policy_module' in full_config['model']:
                self.trunk_dim = full_config['model']['policy_module'].get('trunk_dim', 512)
                
            # 提取actor节点
            if 'model' in full_config and 'policy_module' in full_config['model'] and 'actor' in full_config['model']['policy_module']:
                config = full_config['model']['policy_module']['actor']
            else:
                config = {}
                print("\u8b66告: 未找到完整的Actor配置节点，使用默认值")
        
        # 从提供的完整配置中获取trunk_dim
        elif isinstance(config, dict) and 'trunk_dim' in config:
            self.trunk_dim = config.get('trunk_dim', 512)
        
        # 打印配置信息便于调试
        print(f"Actor 参数: trunk_dim={self.trunk_dim}")
        self.hidden_dims = config.get('hidden_dims', [256, 128])
        self.activation_name = config.get('activation', 'relu')
        self.log_std_min = config.get('log_std_min', -10)
        self.log_std_max = config.get('log_std_max', 2)
        
        # 设置默认的动作空间维度
        self.action_dim = 4  # 默认为4维动作：[vx, vy, vz, yaw_rate]
        
        # 如果配置中有action_dim，则使用配置的值
        if isinstance(config, dict) and 'action_dim' in config:
            self.action_dim = config.get('action_dim', 4)
            
        print(f"Actor 动作空间维度: {self.action_dim}")
        
        # Set activation function
        if self.activation_name == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif self.activation_name == 'tanh':
            self.activation = nn.Tanh()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
        
        # Build MLP layers
        layers = []
        prev_dim = self.trunk_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean = nn.Linear(self.hidden_dims[-1], self.action_dim)
        self.log_std = nn.Linear(self.hidden_dims[-1], self.action_dim)
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Actor network.
        
        Args:
            x: Input tensor of state features (batch_size, fusion_dim) or dict of tensors
            
        Returns:
            Tuple of:
                - Mean action of shape (batch_size, action_dim)
                - Log standard deviation of shape (batch_size, action_dim)
        """
        # 处理字典类型的输入
        if isinstance(x, dict):
            # 处理字典类型的输入
            if 'state' in x and 'target' in x:
                # 将状态和目标向量连接
                features = torch.cat([x['state'], x['target']], dim=-1)
            elif 'state' in x:
                # 仅使用状态向量
                features = x['state']
            else:
                raise ValueError("Actor需要'state'组件来生成动作")
            trunk_input = features
        else:
            # 标准Tensor输入，直接使用
            trunk_input = x
            
        # Process through shared trunk
        x = self.trunk(trunk_input)
        
        # Compute mean and log_std
        mean = self.mean(x)
        log_std = self.log_std(x)
        
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, x, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            x: Input data, can be tensor or dictionary 
            deterministic: If True, return the mean action instead of sampling
            
        Returns:
            Tuple of:
                - Sampled action of shape (batch_size, action_dim)
                - Mean action of shape (batch_size, action_dim)
                - Log probability of the sampled action
        """
        # 直接调用forward方法处理输入(包括字典类型)
        # forward方法会处理字典类型输入并调用trunk网络
        # 这样我们只需要在一个地方维护字典处理逻辑
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        
        # Create normal distribution
        normal = torch.distributions.Normal(mean, std)
        
        # Sample or use mean
        if deterministic:
            action = mean
        else:
            # Reparameterization trick
            x_t = normal.rsample()
            action = torch.tanh(x_t)  # Constrain to [-1, 1]
        
        # Calculate log probability (accounting for tanh transformation)
        if deterministic:
            log_prob = None
        else:
            # Calculate log probability of x_t under the Gaussian
            log_prob = normal.log_prob(x_t)
            
            # Apply correction for tanh transformation
            # log(det|jacobian|) = log(1 - tanh²(x)) = log(1 - action²)
            log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Get mean action with tanh applied
        tanh_mean = torch.tanh(mean)
        
        return action, tanh_mean, log_prob


class SharedTrunkActor(nn.Module):
    """
    Actor network with shared trunk between policy and value networks.
    This implements the framework's design where Actor/Critic share a common trunk.
    """
    def __init__(self, trunk_network: nn.Module, actor_config: Optional[Dict] = None):
        """
        Initialize Actor with shared trunk.
        
        Args:
            trunk_network: Shared trunk network
            actor_config: Configuration dictionary for actor head
        """
        super().__init__()
        
        self.trunk = trunk_network
        self.actor = Actor(actor_config)
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Actor network with shared trunk.
        
        Args:
            x: Input tensor or dictionary to the trunk network
            
        Returns:
            Tuple of:
                - Mean action of shape (batch_size, action_dim)
                - Log standard deviation of shape (batch_size, action_dim)
        """
        # 处理字典类型的输入
        if isinstance(x, dict):
            # 处理字典类型的输入
            if 'state' in x and 'target' in x:
                # 将状态和目标向量连接
                features = torch.cat([x['state'], x['target']], dim=-1)
            elif 'state' in x:
                # 仅使用状态向量
                features = x['state']
            else:
                raise ValueError("Actor需要'state'组件来生成动作")
            trunk_input = features
        else:
            # 标准Tensor输入，直接使用
            trunk_input = x
            
        # Pass through shared trunk
        trunk_features = self.trunk(trunk_input)
        
        # Pass through actor head
        mean, log_std = self.actor(trunk_features)
        
        return mean, log_std
    
    def sample(self, x, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy with shared trunk.
        
        Args:
            x: Input tensor or dictionary to the trunk network
            deterministic: If True, return the mean action instead of sampling
            
        Returns:
            Tuple of:
                - Sampled action of shape (batch_size, action_dim)
                - Mean action of shape (batch_size, action_dim)
                - Log probability of the sampled action
        """
        # 处理字典类型的输入
        if isinstance(x, dict):
            # 处理字典类型的输入
            if 'state' in x and 'target' in x:
                # 将状态和目标向量连接
                features = torch.cat([x['state'], x['target']], dim=-1)
            elif 'state' in x:
                # 仅使用状态向量
                features = x['state']
            else:
                raise ValueError("Actor需要'state'组件来生成动作")
            trunk_input = features
        else:
            # 标准Tensor输入，直接使用
            trunk_input = x
            
        # Pass through shared trunk
        trunk_features = self.trunk(trunk_input)
        
        # Sample from actor head
        action, mean, log_prob = self.actor.sample(trunk_features, deterministic)
        
        return action, mean, log_prob