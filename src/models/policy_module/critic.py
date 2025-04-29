import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
from typing import Dict, List, Optional, Tuple, Union

class Critic(nn.Module):
    """
    Critic network for estimating Q-values of state-action pairs.
    Implements the framework's "Critic Head: MLP → Q值" feature.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Critic network.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # 使用更健壮的配置加载方式
        default_config = {}
        
        # 如果未提供配置，尝试加载默认配置
        if config is None:
            try:
                config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                    "config", "default.yaml"
                )
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                
                # 安全地提取配置
                if 'model' in loaded_config and 'policy_module' in loaded_config['model'] \
                   and 'critic' in loaded_config['model']['policy_module']:
                    default_config = loaded_config['model']['policy_module']['critic']
                    print("TwinCritic: 使用默认配置文件中的参数")
            except Exception as e:
                print(f"TwinCritic 警告: 无法加载配置文件 - {e}")
                print("使用内置默认配置")
            
            # 使用默认配置进行初始化
            config = default_config
        
        # 设置默认参数
        self.trunk_dim = 512
        self.hidden_dims = [256, 128]
        self.activation_name = 'relu'
        self.action_dim = 4  # [vx, vy, vz, yaw_rate]
        
        # 从配置中读取参数（如果提供）
        if config is None:
            # 如果未提供配置，使用默认配置文件
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "config", "default.yaml"
            )
            try:
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                
                # 安全地从完整配置中提取参数
                if 'model' in full_config and 'policy_module' in full_config['model']:
                    policy_config = full_config['model']['policy_module']
                    self.trunk_dim = policy_config.get('trunk_dim', self.trunk_dim)
                
                if 'model' in full_config and 'policy_module' in full_config['model'] and 'critic' in full_config['model']['policy_module']:
                    critic_config = full_config['model']['policy_module']['critic']
                    self.hidden_dims = critic_config.get('hidden_dims', self.hidden_dims)
                    self.activation_name = critic_config.get('activation', self.activation_name)
                
                if 'environment' in full_config and 'action_space' in full_config['environment']:
                    self.action_dim = full_config['environment']['action_space'].get('action_dim', self.action_dim)
            except Exception as e:
                print(f"警告: 无法加载 Critic 配置文件: {e}")
                print("使用默认参数...")
        
        # 如果传入配置字典，从字典获取参数
        elif isinstance(config, dict):
            self.trunk_dim = config.get('trunk_dim', self.trunk_dim)
            self.hidden_dims = config.get('hidden_dims', self.hidden_dims)
            self.activation_name = config.get('activation', self.activation_name)
            self.action_dim = config.get('action_dim', self.action_dim)
        
        # 打印参数信息便于调试
        print(f"Critic 参数: trunk_dim={self.trunk_dim}, "
              f"action_dim={self.action_dim}, hidden_dims={self.hidden_dims}")
        
        # Set activation function
        if self.activation_name == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif self.activation_name == 'tanh':
            self.activation = nn.Tanh()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
        
        # Build critic network layers
        layers = []
        prev_dim = self.trunk_dim + self.action_dim  # Concat state and action
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        # Output layer for Q-value
        layers.append(nn.Linear(self.hidden_dims[-1], 1))
        
        self.q_network = nn.Sequential(*layers)
    
    def forward(self, state_features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Critic network. Estimates Q-value for state-action pair.
        
        Args:
            state_features: State features from trunk network (batch_size, trunk_dim)
            action: Action tensor (batch_size, action_dim)
            
        Returns:
            Q-value estimates (batch_size, 1)
        """
        # Concatenate state features and action
        x = torch.cat([state_features, action], dim=1)
        
        # Forward pass through Q-network
        q_value = self.q_network(x)
        
        return q_value


class TwinCritic(nn.Module):
    """
    Double critic architecture to reduce overestimation bias in Q-learning.
    Implements the framework's "双Q网络降低过估计" feature.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Twin Critic networks.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # 使用更健壮的配置加载方式
        default_config = {}
        
        # 如果未提供配置，尝试加载默认配置
        if config is None:
            try:
                config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                    "config", "default.yaml"
                )
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                
                # 安全地提取配置
                if 'model' in loaded_config and 'policy_module' in loaded_config['model'] \
                   and 'critic' in loaded_config['model']['policy_module']:
                    default_config = loaded_config['model']['policy_module']['critic']
                    print("TwinCritic: 使用默认配置文件中的参数")
            except Exception as e:
                print(f"TwinCritic 警告: 无法加载配置文件 - {e}")
                print("使用内置默认配置")
            
            # 使用默认配置进行初始化
            config = default_config
        
        # Initialize two Q-networks with same architecture but different parameters
        self.q1 = Critic(config)
        self.q2 = Critic(config)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both critic networks.
        
        Args:
            state: State features tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
            
        Returns:
            Tuple of (q1_values, q2_values), each of shape (batch_size, 1)
        """
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        
        return q1, q2
    
    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Return minimum of the two Q-values to reduce overestimation bias.
        
        Args:
            state: State features tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
            
        Returns:
            Minimum Q-value (batch_size, 1)
        """
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class SharedTrunkCritic(nn.Module):
    """
    Twin Critic network with shared trunk from Actor-Critic architecture.
    This implements the framework's design where Actor/Critic share a common trunk.
    """
    def __init__(self, trunk_network: nn.Module, critic_config: Optional[Dict] = None):
        """
        Initialize Critics with shared trunk.
        
        Args:
            trunk_network: Shared trunk network
            critic_config: Configuration dictionary for critic heads
        """
        super().__init__()
        
        self.trunk = trunk_network
        self.critics = TwinCritic(critic_config)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both critic networks with shared trunk.
        
        Args:
            state: Raw state input tensor
            action: Action tensor
            
        Returns:
            Tuple of (q1_values, q2_values), each of shape (batch_size, 1)
        """
        # Extract features using shared trunk
        state_features = self.trunk(state)
        
        # Forward through both critic networks
        q1, q2 = self.critics(state_features, action)
        
        return q1, q2
    
    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get minimum Q-value from both critics to reduce overestimation.
        
        Args:
            state: Raw state input tensor
            action: Action tensor
            
        Returns:
            Minimum Q-value (batch_size, 1)
        """
        # Extract features using shared trunk
        state_features = self.trunk(state)
        
        # Get minimum Q-value
        min_q = self.critics.min_q(state_features, action)
        
        return min_q
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get Q1 value (useful for some SAC implementations).
        
        Args:
            state: Raw state input tensor
            action: Action tensor
            
        Returns:
            Q1 value (batch_size, 1)
        """
        state_features = self.trunk(state)
        q1 = self.critics.q1(state_features, action)
        return q1