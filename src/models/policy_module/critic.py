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
    def __init__(self, config: Optional[Dict] = None, shared_trunk: Optional[nn.Module] = None):
        """
        Initialize Critic network.
        
        Args:
            config: Configuration dictionary
            shared_trunk: Optional shared trunk network module
        """
        super().__init__()
        
        # 初始化默认属性值
        self.trunk_dim = 768  # 默认特征输入维度
        self.action_dim = 4   # 默认动作空间维度
        self.hidden_dims = [768, 512, 384, 256]  # 默认隐藏层维度
        self.activation_name = 'relu'  # 默认激活函数
        self.use_layernorm = True  # 默认使用层归一化
        self.num_critics = 2  # 双Q网络(用于减少过估计偏差)
        
        # 简化的配置加载逻辑
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
            critic_config = policy_config.get('critic', {})
            
            # 获取trunk_dim从策略模块级别
            self.trunk_dim = policy_config.get('trunk_dim', self.trunk_dim)
            
            # 使用critic专用配置
            config = critic_config
            
            print(f"Critic从配置文件加载参数")
            print(f"  - trunk_dim: {self.trunk_dim}")
        elif isinstance(config, dict):
            # 如果提供了字典配置，获取trunk_dim
            self.trunk_dim = config.get('trunk_dim', self.trunk_dim)
        
        # 如果提供了配置，用配置更新属性
        if isinstance(config, dict):
            self.action_dim = config.get('action_dim', self.action_dim)
            self.hidden_dims = config.get('hidden_dims', self.hidden_dims)
            self.activation_name = config.get('activation', self.activation_name)
            self.use_layernorm = config.get('use_layernorm', self.use_layernorm)
            self.num_critics = config.get('num_critics', self.num_critics)
        
        # 输出配置信息
        print(f"  - 动作空间维度: {self.action_dim}")
        print(f"  - 隐藏层维度: {self.hidden_dims}")
        print(f"  - 激活函数: {self.activation_name}")
        print(f"  - 使用层归一化: {self.use_layernorm}")
        print(f"  - Q网络数量: {self.num_critics}")
        
        # 设置激活函数
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
        
        # Build Critic Networks
        if shared_trunk is not None and isinstance(shared_trunk, nn.Module):
            # 如果提供了共享trunk，使用它
            self.trunk = shared_trunk
            self.shared_trunk = True
            input_dim = shared_trunk.trunk_dim
            print(f"Critic使用共享trunk网络，输入维度={input_dim}")
        else:
            # 创建自己的trunk
            self.shared_trunk = False
            # 不需要创建自己的trunk，直接使用输入维度
            input_dim = self.trunk_dim
            print(f"Critic创建独立Q网络，输入维度={input_dim}")
        
        # 创建多个Q网络头
        self.q_networks = nn.ModuleList()
        
        # 初始化Q网络列表，但不创建多余的网络
        
        # 为每个Q网络创建更复杂的结构
        for _ in range(self.num_critics):
            q_layers = []
            prev_dim = input_dim + self.action_dim
            
            # 为每个Q网络创建隐藏层
            for i, hidden_dim in enumerate(self.hidden_dims):
                q_layers.append(nn.Linear(prev_dim, hidden_dim))
                
                if self.use_layernorm:
                    q_layers.append(nn.LayerNorm(hidden_dim))
                
                q_layers.append(self.activation)
                prev_dim = hidden_dim
            
            # 最终输出层
            q_layers.append(nn.Linear(prev_dim, 1))
            
            # 添加到Q网络列表
            self.q_networks.append(nn.Sequential(*q_layers))
    
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
        q_value = self.q_networks[0](x)
        
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
        
        # 确保config是字典类型
        if config is None:
            config = {}
        
        # 确保我们不会修改原始配置
        config_copy = config.copy() if isinstance(config, dict) else {}
        
        # 初始化两个Q网络，使用相同的架构但不同的参数
        self.q1 = Critic(config_copy, None)  # 显式传入None作为shared_trunk
        self.q2 = Critic(config_copy, None)  # 显式传入None作为shared_trunk
    
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
        # 添加shared_trunk属性表示使用共享trunk
        self.shared_trunk = True  # 设置为True因为这个类始终使用共享trunk
        
        # 获取trunk_network的输出维度
        if hasattr(trunk_network, 'trunk_dim'):
            trunk_dim = trunk_network.trunk_dim
        else:
            trunk_dim = 768  # 默认与配置文件一致
        
        # 确保将trunk_dim添加到critic_config中
        if critic_config is None:
            critic_config = {}
        critic_config_copy = critic_config.copy()  # 创建副本避免修改原始配置
        critic_config_copy['trunk_dim'] = trunk_dim
        
        # 创建TwinCritic实例并传入正确的trunk_dim
        self.critics = TwinCritic(critic_config_copy)
    
    def forward(self, state, action=None) -> List[torch.Tensor]:
        """
        Forward pass for critic network.
        
        Args:
            state: State tensor or dict
            action: Action tensor
            
        Returns:
            List of Q values from each critic network
        """
        # 处理字典形式输入
        if isinstance(state, dict):
            # 处理字典观测
            if 'state' in state and 'target' in state:
                # 将状态和目标连接
                state_features = torch.cat([state['state'], state['target']], dim=-1)
            elif 'state' in state:
                state_features = state['state']
            else:
                raise ValueError("Critic需要观测字典中的'state'组件")
        else:
            state_features = state
            
        # 使用主干网络处理
        if self.shared_trunk:
            # 如果使用共享trunk，将整个输入传递给trunk
            trunk_output = self.trunk(state)
        else:
            # 如果使用自己的trunk，处理状态特征
            trunk_output = self.trunk(state_features)
        
        if action is None:
            # 如果没有提供动作，返回trunk输出
            return trunk_output
        
        # 将trunk输出和动作连接
        sa_features = torch.cat([trunk_output, action], dim=1)
        
        # 从双Q网络获取两个Q值
        q1 = self.critics.q1.q_networks[0](sa_features)
        q2 = self.critics.q1.q_networks[1](sa_features) if len(self.critics.q1.q_networks) > 1 else q1
        
        return q1, q2
    
    def min_q(self, state, action: torch.Tensor) -> torch.Tensor:
        """
        Get minimum Q-value from both critics to reduce overestimation.
        
        Args:
            state: Raw state input tensor or dictionary of tensors
            action: Action tensor
            
        Returns:
            Minimum Q-value (batch_size, 1)
        """
        # 从两个Q网络中获取Q值
        q1, q2 = self.forward(state, action)
        
        # 返回最小值
        return torch.min(q1, q2)
    
    def q1(self, state, action: torch.Tensor) -> torch.Tensor:
        """
        Get Q1 value (useful for some SAC implementations).
        
        Args:
            state: Raw state input tensor or dictionary
            action: Action tensor
            
        Returns:
            Q1 value (batch_size, 1)
        """
        # 从forward方法获取第一个Q值
        q1, _ = self.forward(state, action)
        
        return q1