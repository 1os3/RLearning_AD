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
            config = full_config['auxiliary_tasks']['contrastive']
        
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
            self.embedding_dim = 256
            self.hidden_dim = 256
            self.output_dim = 256
            self.num_negative_samples = 10
            self.temperature = 0.1
            self.use_gru = True
        
        # Define encoder network (converts state to latent representation)
        # We'll use a simple MLP for this
        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
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
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Contrastive Predictive Coding.
        
        Args:
            state: Current state tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)
            next_state: Next state tensor of shape (batch_size, state_dim)
            
        Returns:
            CPC loss
        """
        batch_size = state.shape[0]
        
        # Encode current and next state
        z_t = self.encoder(state)  # Current state embedding
        z_tp1 = self.encoder(next_state)  # Next state embedding (target)
        
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
        labels = torch.arange(batch_size, device=state.device)
        
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
            config = full_config['auxiliary_tasks']['reconstruction']
        
        # Get parameters from config
        if isinstance(config, dict):
            self.state_dim = config.get('state_dim', 512)  # Latent state dimension
            self.action_dim = config.get('action_dim', 4)  # Action dimension
            self.hidden_dims = config.get('hidden_dims', [256, 256])
            self.output_dim = config.get('output_dim', 512)  # Output dimension (reconstructed next state)
            self.use_l2_loss = config.get('use_l2_loss', True)  # Use L2 or L1 loss
        else:
            # Default values if config is not provided or is not a dictionary
            self.state_dim = 512
            self.action_dim = 4
            self.hidden_dims = [256, 256]
            self.output_dim = 512
            self.use_l2_loss = True
        
        # Build network layers
        layers = []
        input_dim = self.state_dim + self.action_dim  # Concatenated state and action
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim
        
        # Output layer for reconstructed next state
        layers.append(nn.Linear(input_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Frame Reconstruction.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
            next_state: Next state tensor (batch_size, state_dim)
            
        Returns:
            Reconstruction loss
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Forward pass through reconstruction network
        predicted_next_state = self.network(x)
        
        # Compute reconstruction loss
        if self.use_l2_loss:
            # MSE loss
            loss = F.mse_loss(predicted_next_state, next_state)
        else:
            # L1 loss (more robust to outliers)
            loss = F.l1_loss(predicted_next_state, next_state)
        
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
            config = full_config['auxiliary_tasks']['pose_regression']
        
        # Get parameters from config
        if isinstance(config, dict):
            self.state_dim = config.get('state_dim', 512)  # State dimension
            self.action_dim = config.get('action_dim', 4)  # Action dimension
            self.pose_dim = config.get('pose_dim', 9)  # Pose dimension (position, orientation, velocity)
            self.hidden_dims = config.get('hidden_dims', [256, 128])
            self.normalize_targets = config.get('normalize_targets', True)  # Normalize regression targets
        else:
            # Default values if config is not provided or is not a dictionary
            self.state_dim = 512
            self.action_dim = 4
            self.pose_dim = 9
            self.hidden_dims = [256, 128]
            self.normalize_targets = True
        
        # Running statistics for target normalization
        self.register_buffer('running_mean', torch.zeros(self.pose_dim))
        self.register_buffer('running_var', torch.ones(self.pose_dim))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        
        # Build regression network
        layers = []
        input_dim = self.state_dim + self.action_dim  # Concatenated state and action
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim
        
        # Output layer for pose prediction
        layers.append(nn.Linear(input_dim, self.pose_dim))
        
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
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Pose Regression.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
            next_state: Next state tensor (batch_size, state_dim)
            
        Returns:
            Pose regression loss
        """
        # Extract target pose from next state
        target_pose = self.extract_pose_from_state(next_state)
        
        # Update normalization statistics
        if self.training and self.normalize_targets:
            self.update_normalization_stats(target_pose)
        
        # Normalize target pose
        normalized_target_pose = self.normalize_pose(target_pose)
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Forward pass through network to predict next pose
        predicted_normalized_pose = self.network(x)
        
        # Compute weighted MSE loss with higher weights on position and orientation
        # This weighting can be adjusted based on the importance of different pose components
        loss = F.mse_loss(predicted_normalized_pose, normalized_target_pose)
        
        return loss