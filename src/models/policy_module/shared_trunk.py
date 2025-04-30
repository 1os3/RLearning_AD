import torch
import torch.nn as nn
import yaml
import os
from typing import Dict, List, Optional, Tuple, Union, Any

class SharedTrunk(nn.Module):
    """
    Shared trunk network for Actor-Critic architecture.
    This implements the framework's "Actor/Critic共用前端trunk，分支后独立优化" feature.
    Takes fusion module output and produces features for Actor and Critic heads.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize shared trunk network.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Load config from file if not provided
        if config is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "config", "default.yaml"
            )
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            config = full_config['model']['policy_module']
        
        # Get parameters from config
        # 自动适配输入维度，如果配置了fusion_dim则使用，否则使用较小的输入维度
        # 防止维度不匹配错误
        self.fusion_dim = config.get('fusion_dim', 15)  # 默认值设为15而不是512
        self.trunk_dim = config.get('trunk_dim', 768)   # 更新为新的默认值768
        self.hidden_dims = config.get('trunk_hidden_dims', [256, 512, 768])  # 增大隐藏层维度
        self.activation_name = config.get('activation', 'relu')
        self.use_layernorm = config.get('use_layernorm', True)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        print(f"SharedTrunk 输入维度: {self.fusion_dim}, 输出维度: {self.trunk_dim}")
        
        # Set activation function
        if self.activation_name == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif self.activation_name == 'tanh':
            self.activation = nn.Tanh()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif self.activation_name == 'silu':
            self.activation = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
        
        # Build trunk network layers
        layers = []
        prev_dim = self.fusion_dim
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
                
            layers.append(self.activation)
            
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
                
            prev_dim = hidden_dim
        
        # Final output layer to produce trunk features
        layers.append(nn.Linear(prev_dim, self.trunk_dim))
        
        if self.use_layernorm:
            layers.append(nn.LayerNorm(self.trunk_dim))
            
        layers.append(self.activation)
        
        self.trunk = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared trunk network.
        
        Args:
            x: Input tensor from fusion module (batch_size, fusion_dim)
            
        Returns:
            Trunk features for policy and value heads (batch_size, trunk_dim)
        """
        return self.trunk(x)


class ActorCriticNetwork(nn.Module):
    """
    Complete Actor-Critic network combining shared trunk, actor, and critic.
    Implements the full architecture in the framework with shared features.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the full Actor-Critic architecture.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Load config from file if not provided
        if config is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "config", "default.yaml"
            )
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            config = full_config['model']
        
        # Get references to different module configs
        policy_config = config.get('policy_module', {})
        actor_config = policy_config.get('actor', {})
        critic_config = policy_config.get('critic', {})
        
        # Import actor and critic classes here to avoid circular imports
        from src.models.policy_module.actor import SharedTrunkActor
        from src.models.policy_module.critic import SharedTrunkCritic
        
        # Initialize shared trunk
        self.trunk = SharedTrunk(policy_config)
        
        # Initialize actor and critic networks with shared trunk
        self.actor = SharedTrunkActor(self.trunk, actor_config)
        self.critic = SharedTrunkCritic(self.trunk, critic_config)
        
        # These flags help during training to selectively update parameters
        self.actor_only = False  # Flag to only update actor in training
        self.critic_only = False  # Flag to only update critic in training
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through entire Actor-Critic network.
        Note: This method is mainly for debugging; during training we typically
        call actor and critic separately.
        
        Args:
            x: Input tensor from fusion module
            
        Returns:
            Tuple of (action_mean, action_log_std, q1_value) - the main outputs
            from actor and critic networks
        """
        # Get action parameters from actor
        action_mean, action_log_std = self.actor(x)
        
        # Sample action for critic evaluation
        action = torch.tanh(action_mean)  # Deterministic action for simplicity
        
        # Get Q-value for the action
        q1, _ = self.critic(x, action)
        
        return action_mean, action_log_std, q1
    
    def get_trainable_parameters(self) -> Dict[str, List[torch.nn.Parameter]]:
        """
        Get trainable parameters organized by component for separate optimizers.
        
        Returns:
            Dictionary mapping component names to their parameters
        """
        params = {
            'trunk': list(self.trunk.parameters()),
            'actor': list(self.actor.actor.parameters()),  # Just the actor head
            'critic': list(self.critic.critics.parameters())  # Just the critic heads
        }
        return params
    
    def set_actor_only(self, actor_only: bool = True) -> None:
        """Set the network to only update actor parameters."""
        self.actor_only = actor_only
        self.critic_only = False
    
    def set_critic_only(self, critic_only: bool = True) -> None:
        """Set the network to only update critic parameters."""
        self.critic_only = critic_only
        self.actor_only = False
    
    def reset_update_flags(self) -> None:
        """Reset selective update flags, allowing all parameters to be updated."""
        self.actor_only = False
        self.critic_only = False
