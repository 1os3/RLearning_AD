import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

class ContrastivePredictiveCoding(nn.Module):
    """
    Contrastive Predictive Coding (CPC) auxiliary task.
    Implements the framework's "视觉对比预测 (Contrastive Predictive Coding)" feature.
    
    CPC predicts future latent representations in a self-supervised manner,
    which helps the agent learn useful state representations.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CPC network.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Load config from file if not provided
        if config is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config", "default.yaml"
            )
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            # 正确读取配置路径algorithm.auxiliary.contrastive
            algorithm_config = full_config.get('algorithm', {})
            auxiliary_config = algorithm_config.get('auxiliary', {})
            config = auxiliary_config.get('contrastive', {})
            
            print(f"ContrastivePredictiveCoding 从配置路径加载: {'algorithm.auxiliary.contrastive' if 'contrastive' in auxiliary_config else '默认值'}")
        
        # Get parameters from config
        if isinstance(config, dict):
            self.embedding_dim = config.get('embedding_dim', 256)
            self.hidden_dim = config.get('hidden_dim', 256)
            self.output_dim = config.get('output_dim', 256)
            self.num_negative_samples = config.get('num_negative_samples', 10)
            self.temperature = config.get('temperature', 0.1)
            self.use_gru = config.get('use_gru', True)
        else:
            # Default values if config is not provided or is not a dictionary
            # 与配置文件保持一致的默认值
            self.embedding_dim = 384
            self.hidden_dim = 384
            self.output_dim = 512
            self.num_negative_samples = 32
            self.temperature = 0.1
            self.use_gru = True
        
        # 创建初始编码器，使用默认维度，但允许在首次前向传递时动态更新
        self.state_dim = 18  # 默认值，实际会在首次前向传递时动态调整
        self.first_forward = True  # 标记是否是首次前向传递
        print(f"CPC辅助任务使用初始输入维度: {self.state_dim}, 输出维度: {self.output_dim}。将在首次前向传递时动态调整")
        
        # 首先创建默认编码器，使优化器有参数可用
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # Context network (for encoding sequence context)
        if self.use_gru:
            self.context_net = nn.GRU(
                input_size=self.output_dim, 
                hidden_size=self.hidden_dim,
                batch_first=True
            )
        else:
            self.context_net = nn.Sequential(
                nn.Linear(self.output_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        
        # Prediction network (predicts future latent representations)
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, state, action: torch.Tensor, next_state) -> torch.Tensor:
        """
        Forward pass of Contrastive Predictive Coding.
        
        Args:
            state: Current state tensor or dictionary of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)
            next_state: Next state tensor or dictionary of shape (batch_size, state_dim)
            
        Returns:
            CPC loss
        """
        # 处理字典类型输入
        if isinstance(state, dict):
            # 与模型保持一致的特征提取逻辑
            if 'state' in state and 'target' in state:
                state_features = torch.cat([state['state'], state['target']], dim=-1)
            elif 'state' in state:
                state_features = state['state']
            else:
                raise ValueError("CPC辅助任务需要'state'组件")
        else:
            state_features = state
            
        # 处理next_state字典
        if isinstance(next_state, dict):
            if 'state' in next_state and 'target' in next_state:
                next_state_features = torch.cat([next_state['state'], next_state['target']], dim=-1)
            elif 'state' in next_state:
                next_state_features = next_state['state']
            else:
                raise ValueError("CPC辅助任务需要'state'组件")
        else:
            next_state_features = next_state
            
        batch_size = state_features.shape[0]
        
        # 如果是首次前向传递，动态创建encoder
        if self.first_forward:
            actual_dim = state_features.shape[-1]  # 获取实际输入维度
            if self.state_dim != actual_dim:
                self.state_dim = actual_dim
                print(f"\nCPC辅助任务检测到输入维度为{actual_dim}。动态创建编码器网络")
            
            # 构建encoder网络
            self.encoder = nn.Sequential(
                nn.Linear(self.state_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.output_dim)
            ).to(state_features.device)  # 确保网络在正确的设备上
            self.first_forward = False
            print(f"CPC辅助任务编码器已创建，输入维度: {self.state_dim}, 输出维度: {self.output_dim}, 设备: {state_features.device}")
        
        # Encode current and next state
        z_t = self.encoder(state_features)  # Current state embedding
        z_tp1 = self.encoder(next_state_features)  # Next state embedding (target)
        
        # Get context representation
        if self.use_gru:
            # For GRU, reshape to sequence of length 1
            z_t_reshaped = z_t.unsqueeze(1)  # Shape: (batch_size, 1, output_dim)
            _, context = self.context_net(z_t_reshaped)  # GRU returns (output, h_n)
            context = context.squeeze(0)  # Shape: (batch_size, hidden_dim)
        else:
            context = self.context_net(z_t)  # Shape: (batch_size, hidden_dim)
        
        # Predict next state embedding
        pred_z_tp1 = self.predictor(context)  # Shape: (batch_size, output_dim)
        
        # Calculate positive logits (how well we predict the actual next state)
        positive_logits = torch.sum(pred_z_tp1 * z_tp1, dim=1) / self.temperature
        
        # Generate negative samples (other batch elements serve as negatives)
        # This creates a matrix where each row has the current prediction compared to all other targets
        all_targets = z_tp1.unsqueeze(0).expand(batch_size, batch_size, -1)  # Shape: (batch_size, batch_size, output_dim)
        all_preds = pred_z_tp1.unsqueeze(1).expand(batch_size, batch_size, -1)  # Shape: (batch_size, batch_size, output_dim)
        
        # Calculate similarity between predictions and all targets
        logits = torch.sum(all_preds * all_targets, dim=2) / self.temperature  # Shape: (batch_size, batch_size)
        
        # Create labels (diagonal elements are positives, all others are negatives)
        # 从Tensor中获取设备，而不是从字典中获取
        device = z_t.device  # 从Encoder输出中获取设备
        labels = torch.arange(batch_size, device=device)
        
        # InfoNCE loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class FrameReconstruction(nn.Module):
    """
    Frame Reconstruction auxiliary task.
    Implements the framework's "未来帧重建 (Reconstruction Loss)" feature.
    
    This task attempts to reconstruct the next observation from the current observation and action,
    encouraging the agent to learn a better representation of the environment dynamics.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Frame Reconstruction network.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Load config from file if not provided
        if config is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config", "default.yaml"
            )
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            # 正确读取配置路径algorithm.auxiliary.reconstruction
            algorithm_config = full_config.get('algorithm', {})
            auxiliary_config = algorithm_config.get('auxiliary', {})
            config = auxiliary_config.get('reconstruction', {})
            
            print(f"FrameReconstruction 从配置路径加载: {'algorithm.auxiliary.reconstruction' if 'reconstruction' in auxiliary_config else '默认值'}")
        
        # Get parameters from config
        if isinstance(config, dict):
            # 使用实际的状态维度而不是预定义的512
            self.state_dim = config.get('state_dim', 15)  # 实际特征维度是9维状态+6维目标=15维
            self.action_dim = config.get('action_dim', 4)  # Action dimension
            # 更新为与配置文件一致的隐藏层维度
            self.hidden_dims = config.get('hidden_dims', [128, 256, 384])
            self.output_dim = config.get('output_dim', 15)  # 输出维度也应与状态维度一致
            self.use_l2_loss = config.get('use_l2_loss', True)  # Use L2 or L1 loss
        else:
            # Default values if config is not provided or is not a dictionary
            self.state_dim = 15  # 实际特征维度是9维状态+6维目标=15维
            self.action_dim = 4
            # 更新为与配置文件一致的隐藏层维度
            self.hidden_dims = [128, 256, 384]
            self.output_dim = 15
            self.use_l2_loss = True
            
        print(f"FrameReconstruction辅助任务输入维度: {self.state_dim + self.action_dim}, 输出维度: {self.output_dim}")
        
        # 创建初始网络，使用默认维度，但允许在首次前向传递时动态更新
        self.first_forward = True  # 标记是否是首次前向传递
        print(f"FrameReconstruction辅助任务使用初始输入维度: {self.state_dim}, 输出维度: {self.output_dim}。将在首次前向传递时动态调整")
        
        # 首先创建默认网络，使优化器有参数可用
        layers = []
        in_dim = self.state_dim + self.action_dim  # 连接状态和动作
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(in_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, action: torch.Tensor, next_state) -> torch.Tensor:
        """
        Forward pass of Frame Reconstruction.
        
        Args:
            state: Current state tensor or dictionary (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
            next_state: Next state tensor or dictionary (batch_size, state_dim)
            
        Returns:
            Reconstruction loss
        """
        # 处理字典类型输入
        if isinstance(state, dict):
            # 与其他模块保持一致的特征提取逻辑
            if 'state' in state and 'target' in state:
                state_features = torch.cat([state['state'], state['target']], dim=-1)
            elif 'state' in state:
                state_features = state['state']
            else:
                raise ValueError("FrameReconstruction辅助任务需要'state'组件")
        else:
            state_features = state
            
        # 处理next_state字典
        if isinstance(next_state, dict):
            if 'state' in next_state and 'target' in next_state:
                next_state_features = torch.cat([next_state['state'], next_state['target']], dim=-1)
            elif 'state' in next_state:
                next_state_features = next_state['state']
            else:
                raise ValueError("FrameReconstruction需要'state'组件")
        else:
            next_state_features = next_state
        
        # 如果是首次前向传递，动态创建network
        if self.first_forward:
            actual_dim = state_features.shape[-1]  # 获取实际输入维度
            if self.state_dim != actual_dim or self.output_dim != actual_dim:
                self.state_dim = actual_dim
                self.output_dim = actual_dim  # 确保输出维度与输入相同
                print(f"\nFrameReconstruction辅助任务检测到输入维度为{actual_dim}。动态创建网络")
            
            # 构建网络
            layers = []
            in_dim = self.state_dim + self.action_dim  # 连接状态和动作
            
            for hidden_dim in self.hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                in_dim = hidden_dim
            
            # 输出层
            layers.append(nn.Linear(in_dim, self.output_dim))
            
            self.network = nn.Sequential(*layers).to(state_features.device)  # 确保网络在正确的设备上
            self.first_forward = False
            print(f"FrameReconstruction辅助任务网络已创建，输入维度: {self.state_dim + self.action_dim}, 输出维度: {self.output_dim}, 设备: {state_features.device}")
        
        # 连接状态和动作
        sa = torch.cat([state_features, action], dim=1)
        
        # 预测下一个状态
        pred_next_state = self.network(sa)
        
        # 计算重建损失，但添加值裁剪和缩放以防止损失过大
        # 首先裁剪预测值和目标值，防止极端值
        pred_next_state_clipped = torch.clamp(pred_next_state, -10.0, 10.0)
        next_state_features_clipped = torch.clamp(next_state_features, -10.0, 10.0)
        
        if self.use_l2_loss:
            # 计算MSE损失但缩小权重
            raw_loss = F.mse_loss(pred_next_state_clipped, next_state_features_clipped, reduction='none')
            # 应用额外的损失缩放因子(0.01)来降低此任务的权重
            loss = raw_loss.mean() * 0.01
            
            # 输出调试信息
            if torch.rand(1).item() < 0.01:  # 1%概率打印
                with torch.no_grad():
                    unclamped_loss = F.mse_loss(pred_next_state, next_state_features)
                    print(f"FrameReconstruction - 原始损失: {unclamped_loss.item():.4f}, 裁剪后损失: {loss.item():.4f}")
        else:
            # 计算L1损失但缩小权重
            raw_loss = F.l1_loss(pred_next_state_clipped, next_state_features_clipped, reduction='none')
            loss = raw_loss.mean() * 0.01
        
        return loss


class PoseRegression(nn.Module):
    """
    Pose Regression auxiliary task.
    Implements the framework's "姿态预测辅助 (Auxiliary Regression)" feature.
    
    This task predicts the next pose (position, orientation, velocity, etc.) from 
    the current state and action, aiding in learning better state representations.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Pose Regression network.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Load config from file if not provided
        if config is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config", "default.yaml"
            )
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            # 正确读取配置路径algorithm.auxiliary.pose_regression
            algorithm_config = full_config.get('algorithm', {})
            auxiliary_config = algorithm_config.get('auxiliary', {})
            config = auxiliary_config.get('pose_regression', {})
            
            print(f"PoseRegression 从配置路径加载: {'algorithm.auxiliary.pose_regression' if 'pose_regression' in auxiliary_config else '默认值'}")
        
        # Get parameters from config
        if isinstance(config, dict):
            # 使用实际的状态维度而不是预定义的512
            self.state_dim = config.get('state_dim', 15)  # 实际特征维度是9维状态+6维目标=15维
            self.action_dim = config.get('action_dim', 4)  # Action dimension
            self.pose_dim = config.get('pose_dim', 9)  # Pose dimension (position, orientation, velocity)
            # 更新为与配置文件一致的隐藏层维度
            self.hidden_dims = config.get('hidden_dims', [128, 256, 384])
            self.normalize_targets = config.get('normalize_targets', True)  # Normalize regression targets
        else:
            # Default values if config is not provided or is not a dictionary
            self.state_dim = 15  # 实际特征维度是9维状态+6维目标=15维
            self.action_dim = 4
            self.pose_dim = 9
            # 更新为与配置文件一致的隐藏层维度
            self.hidden_dims = [128, 256, 384]
            self.normalize_targets = True
            
        print(f"PoseRegression辅助任务输入维度: {self.state_dim + self.action_dim}, 输出维度: {self.pose_dim}")
        
        # Running statistics for target normalization
        self.register_buffer('running_mean', torch.zeros(self.pose_dim))
        self.register_buffer('running_var', torch.ones(self.pose_dim))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        
        # 创建初始网络，使用默认维度，但允许在首次前向传递时动态更新
        self.first_forward = True  # 标记是否是首次前向传递
        print(f"PoseRegression辅助任务使用初始输入维度: {self.state_dim}, 输出维度: {self.pose_dim}。将在首次前向传阒时动态调整")
        
        # 首先创建默认网络，使优化器有参数可用
        layers = []
        in_dim = self.state_dim + self.action_dim  # 连接状态和动作
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(in_dim, self.pose_dim))
        
        self.network = nn.Sequential(*layers)
    
    def update_normalization_stats(self, targets: torch.Tensor) -> None:
        """
        Update running mean and variance statistics for target normalization.
        
        Args:
            targets: Batch of target poses
        """
        if not self.normalize_targets:
            return
            
        batch_mean = targets.mean(dim=0)
        batch_var = targets.var(dim=0, unbiased=False)
        batch_size = targets.shape[0]
        
        # Update running statistics using Welford's online algorithm
        new_count = self.count + batch_size
        delta = batch_mean - self.running_mean
        self.running_mean = self.running_mean + delta * (batch_size / max(new_count, 1))
        
        # Update running variance
        m_a = self.running_var * self.count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta ** 2 * self.count * batch_size / max(new_count, 1)
        self.running_var = M2 / max(new_count, 1)
        
        # Update count
        self.count = new_count
    
    def normalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Normalize pose using running statistics.
        
        Args:
            pose: Pose tensor to normalize
            
        Returns:
            Normalized pose tensor
        """
        if not self.normalize_targets or self.count == 0:
            return pose
            
        return (pose - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
    
    def denormalize_pose(self, normalized_pose: torch.Tensor) -> torch.Tensor:
        """
        Denormalize pose using running statistics.
        
        Args:
            normalized_pose: Normalized pose tensor
            
        Returns:
            Denormalized pose tensor
        """
        if not self.normalize_targets or self.count == 0:
            return normalized_pose
            
        return normalized_pose * torch.sqrt(self.running_var + 1e-8) + self.running_mean
    
    def extract_pose_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Extract pose information from state tensor.
        
        Note: This method assumes that the pose information is embedded in the state tensor
        in a specific format. The actual implementation will depend on how poses are
        represented in your specific environment and state representation.
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            Pose tensor (batch_size, pose_dim)
        """
        # This is a placeholder implementation
        # In a real implementation, you would extract the actual pose information from the state
        # For example, if the first pose_dim elements of the state contain pose information:
        if self.pose_dim <= state.shape[1]:
            return state[:, :self.pose_dim]
        else:
            # If the state doesn't contain enough dimensions, return zeros
            return torch.zeros(state.shape[0], self.pose_dim, device=state.device)
    
    def forward(self, state, action: torch.Tensor, next_state) -> torch.Tensor:
        """
        Forward pass of Pose Regression.
        
        Args:
            state: Current state tensor or dictionary (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
            next_state: Next state tensor or dictionary (batch_size, state_dim)
            
        Returns:
            Pose regression loss
        """
        # 处理字典类型输入
        if isinstance(state, dict):
            # 与其他模块保持一致的特征提取逻辑
            if 'state' in state and 'target' in state:
                state_features = torch.cat([state['state'], state['target']], dim=-1)
            elif 'state' in state:
                state_features = state['state']
            else:
                raise ValueError("PoseRegression辅助任务需要'state'组件")
        else:
            state_features = state
            
        # 处理next_state字典
        if isinstance(next_state, dict):
            if 'state' in next_state and 'target' in next_state:
                next_state_features = torch.cat([next_state['state'], next_state['target']], dim=-1)
            if 'state' in next_state and next_state['state'] is not None:
                # 检查下一状态形状是否合理
                next_state_shape = next_state['state'].shape
                if len(next_state_shape) < 2:
                    raise ValueError(f"PoseRegression需要形状为(B,D)的下一状态，但得到: {next_state_shape}")
            elif 'state' in next_state:
                next_state_features = next_state['state']
            else:
                raise ValueError("PoseRegression需要'state'组件")
        else:
            next_state_features = next_state
            
        # 如果是首次前向传递，动态创建network
        if self.first_forward:
            actual_dim = state_features.shape[-1]  # 获取实际输入维度
            if self.state_dim != actual_dim:
                self.state_dim = actual_dim
                print(f"\nPoseRegression辅助任务检测到输入维度为{actual_dim}。动态创建网络")
            
            # 构建网络
            layers = []
            in_dim = self.state_dim + self.action_dim  # 连接状态和动作
            
            for hidden_dim in self.hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                in_dim = hidden_dim
            
            # 输出层
            layers.append(nn.Linear(in_dim, self.pose_dim))
            
            self.network = nn.Sequential(*layers).to(state_features.device)  # 确保网络在正确的设备上
            self.first_forward = False
            print(f"PoseRegression辅助任务网络已创建，输入维度: {self.state_dim + self.action_dim}, 输出维度: {self.pose_dim}, 设备: {state_features.device}")
            
            # 确保 running_mean 和 running_var 也在同一个设备上
            self.running_mean = self.running_mean.to(state_features.device)
            self.running_var = self.running_var.to(state_features.device)
            self.count = self.count.to(state_features.device)
        
        # Extract target pose from next state features
        target_pose = self.extract_pose_from_state(next_state_features)
        
        # Update normalization statistics
        if self.training and self.normalize_targets:
            self.update_normalization_stats(target_pose)
        
        # Normalize target pose
        normalized_target_pose = self.normalize_pose(target_pose)
        
        # Concatenate state features and action
        x = torch.cat([state_features, action], dim=1)
        
        # Forward pass through network to predict next pose
        predicted_normalized_pose = self.network(x)
        
        # Compute weighted MSE loss with higher weights on position and orientation
        # This weighting can be adjusted based on the importance of different pose components
        loss = F.mse_loss(predicted_normalized_pose, normalized_target_pose)
        
        return loss