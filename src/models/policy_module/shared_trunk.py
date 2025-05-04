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
        # 自动适配输入维度，第一次调用时动态设置
        # 默认值18对应启用所有状态变量时的状态+目标维度
        self.fusion_dim = config.get('fusion_dim', 18)
        # 在首次前向传递中动态适应真实输入维度（见forward方法）
        self.first_forward = True
        
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
        
        # 将trunk网络构建逐出到独立方法
        self._build_trunk()
        
    def _build_trunk(self):
        """构建主干网络，可在首次前向传递时动态调整输入维度"""
        # Build trunk network
        layers = []
        dims = [self.fusion_dim] + self.hidden_dims + [self.trunk_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # No normalization or activation after the final layer
            if i < len(dims) - 2:
                if self.use_layernorm:
                    layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(self.activation)
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
            
        layers.append(self.activation)
        
        self.trunk = nn.Sequential(*layers)
    
    def forward(self, x) -> torch.Tensor:
        """
        Forward pass through shared trunk network.
        
        Args:
            x: Input tensor or dictionary with state/target from fusion module
            
        Returns:
            Trunk features for policy and value heads (batch_size, trunk_dim)
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
                raise ValueError("SharedTrunk需要'state'组件")
            trunk_input = features
        else:
            # 标准Tensor输入，直接使用
            trunk_input = x
        
        # 如果是第一次前向传递，检查实际输入维度并定义网络
        if hasattr(self, 'first_forward') and self.first_forward and trunk_input is not None:
            actual_dim = trunk_input.shape[-1]  # 获取实际输入维度
            
            if actual_dim != self.fusion_dim:
                print(f"\n检测到输入维度与配置不符，已自动调整: {self.fusion_dim} -> {actual_dim}")
                self.fusion_dim = actual_dim
                # 重新创建Trunk网络
                self._build_trunk()
            
            self.first_forward = False
            print(f"SharedTrunk 输入维度: {self.fusion_dim}, 输出维度: {self.trunk_dim}")
        
        return self.trunk(trunk_input)


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
