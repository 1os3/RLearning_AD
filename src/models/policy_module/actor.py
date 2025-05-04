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
        
        # 使用更简洁的配置加载方式
        if config is None:
            # 如果没有提供配置，加载默认配置文件
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "config", "default.yaml"
            )
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            # 正确从模型配置中提取策略模块配置
            model_config = full_config.get('model', {})
            policy_config = model_config.get('policy_module', {})
            actor_config = policy_config.get('actor', {})
            
            # 获取trunk_dim从策略模块级别
            self.trunk_dim = policy_config.get('trunk_dim', 768)
            
            # 使用actor专用配置
            config = actor_config
            
            print(f"Actor从配置文件加载参数")
            print(f"  - trunk_dim: {self.trunk_dim}")
        elif isinstance(config, dict):
            # 如果提供了外部配置，优先使用外部配置中的trunk_dim
            self.trunk_dim = config.get('trunk_dim', 768)  # 默认值为768
        
        print(f"Actor 参数: trunk_dim={self.trunk_dim}")
        # 从配置中获取其他参数
        self.action_dim = config.get('action_dim', 4)  # 默认为4维动作：[vx, vy, vz, yaw_rate]
        self.hidden_dims = config.get('hidden_dims', [768, 512, 384, 256])  # 默认值与配置文件一致
        self.log_std_min = config.get('log_std_min', -10)  # 最小对数标准差
        self.log_std_max = config.get('log_std_max', 2)     # 最大对数标准差
        self.activation_name = config.get('activation', 'relu')  # 激活函数
        self.use_layernorm = config.get('use_layernorm', True)  # 是否使用层归一化
        
        # 输出配置信息
        print(f"  - 动作空间维度: {self.action_dim}")
        print(f"  - 隐藏层维度: {self.hidden_dims}")
        print(f"  - 激活函数: {self.activation_name}")
        print(f"  - log_std范围: [{self.log_std_min}, {self.log_std_max}]")
        print(f"  - 使用层归一化: {self.use_layernorm}")
            
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
        
        # 构建更型施网络
        layers = []
        prev_dim = self.trunk_dim
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
                
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        # 序列模型结构
        self.trunk = nn.Sequential(*layers)
        
        # 均值头部网络
        self.mean = nn.Linear(prev_dim, self.action_dim)
        
        # 对数标准差头部网络
        self.log_std = nn.Linear(prev_dim, self.action_dim)
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute action distribution parameters.
        
        Args:
            x: Input tensor from trunk network
            
        Returns:
            Tuple of (mean, std) tensors for Gaussian distribution
        """
        # 处理字典类型输入
        if isinstance(x, dict):
            # 处理字典类型观测的逻辑
            if 'state' in x and 'target' in x:
                # 如果同时提供了state和target，将它们连接
                state_features = torch.cat([x['state'], x['target']], dim=-1)
                x = state_features
            elif 'state' in x:
                x = x['state']
            else:
                raise ValueError("Actor需要观测字典中的'state'组件")
        
        # 通过主干网络前向传播
        features = self.trunk(x)
        
        # 计算均值和对数标准差
        mean = self.mean(features)
        log_std = self.log_std(features)
        
        # 限制log_std范围以提高数值稳定性
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
        
        # 获取trunk_network的输出维度
        if hasattr(trunk_network, 'trunk_dim'):
            trunk_dim = trunk_network.trunk_dim
        else:
            trunk_dim = 768  # 默认与配置文件一致
        
        # 确保将trunk_dim添加到actor_config中
        if actor_config is None:
            actor_config = {}
        actor_config_copy = actor_config.copy()  # 创建副本避免修改原始配置
        actor_config_copy['trunk_dim'] = trunk_dim
        
        # 创建Actor实例并传入正确的trunk_dim
        self.actor = Actor(actor_config_copy)
    
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