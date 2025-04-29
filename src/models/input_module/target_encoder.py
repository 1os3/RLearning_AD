import torch
import torch.nn as nn
import yaml
import os
import math
from typing import Dict, List, Optional, Tuple

class TargetEncoder(nn.Module):
    """
    Encodes target point information (distance and direction to target).
    Implements the framework's "目标距离+方向 (r, θ) → 1 层 MLP → embedding token"
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize target encoder.
        
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
            config = full_config['model']['input_module']['target_encoder']
        
        # Get parameters from config
        self.input_dim = config.get('input_dim', 3)  # Distance (1) + Direction (2: horizontal angle, vertical angle)
        self.hidden_dims = config.get('hidden_dims', [32])  # Only one hidden layer as per framework
        self.output_dim = config.get('output_dim', 128)
        
        # Build MLP layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        layers.append(nn.LayerNorm(self.output_dim))  # Normalize the output embedding
        
        self.mlp = nn.Sequential(*layers)
        
        # Distance normalization parameters
        self.max_distance = 100.0  # Maximum expected distance in meters
        
    def forward(self, target_info: torch.Tensor) -> torch.Tensor:
        """
        Encode target information into a fixed-dimensional embedding.
        
        Args:
            target_info: Tensor of shape (batch_size, input_dim) containing:
                        - Distance to target (scalar)
                        - Direction to target (azimuth angle)
                        - Direction to target (elevation angle)
                    
        Returns:
            Encoded target embedding of shape (batch_size, output_dim)
        """
        # Ensure input has the correct shape
        if target_info.dim() == 1:
            target_info = target_info.unsqueeze(0)  # Add batch dimension if missing
            
        # Verify input dimension
        if target_info.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {target_info.shape[-1]}")
        
        # Apply normalization to the distance component
        normalized_input = target_info.clone()
        normalized_input[:, 0] = torch.clamp(normalized_input[:, 0] / self.max_distance, 0.0, 1.0)
        
        # Apply MLP
        embedding = self.mlp(normalized_input)
        
        return embedding
    
    def compute_target_info(self, current_position: torch.Tensor, target_position: torch.Tensor, 
                          current_orientation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute target information (distance and direction) based on current and target positions.
        
        Args:
            current_position: Tensor of shape (batch_size, 3) containing current [x, y, z]
            target_position: Tensor of shape (batch_size, 3) containing target [x, y, z]
            current_orientation: Optional tensor of shape (batch_size, 3) containing [roll, pitch, yaw]
                              If provided, direction will be relative to current orientation
                              
        Returns:
            Tensor of shape (batch_size, input_dim) containing target information
        """
        # Ensure input has the correct shape
        if current_position.dim() == 1:
            current_position = current_position.unsqueeze(0)
        if target_position.dim() == 1:
            target_position = target_position.unsqueeze(0)
        
        batch_size = current_position.shape[0]
        device = current_position.device
        
        # Calculate vector to target
        vector_to_target = target_position - current_position
        
        # Calculate distance to target
        distance_to_target = torch.norm(vector_to_target, dim=1, keepdim=True)
        
        # Calculate direction angles
        horizontal_distance = torch.norm(vector_to_target[:, :2], dim=1, keepdim=True) + 1e-8
        
        # Calculate azimuth angle (xy-plane)
        azimuth = torch.atan2(vector_to_target[:, 1:2], vector_to_target[:, 0:1])
        
        # Calculate elevation angle (vertical)
        elevation = torch.atan2(vector_to_target[:, 2:3], horizontal_distance)
        
        # If orientation is provided, adjust azimuth to be relative to current orientation
        if current_orientation is not None:
            if current_orientation.dim() == 1:
                current_orientation = current_orientation.unsqueeze(0)
            
            # Extract yaw (we only need this for azimuth adjustment)
            yaw = current_orientation[:, 2:3]
            
            # Adjust azimuth to be relative to current orientation
            azimuth = azimuth - yaw
            
            # Normalize to [-pi, pi]
            azimuth = torch.atan2(torch.sin(azimuth), torch.cos(azimuth))
        
        # Concatenate distance and direction information
        target_info = torch.cat([distance_to_target, azimuth, elevation], dim=1)
        
        return target_info
    
    def extract_features(self, state_dict: Dict) -> torch.Tensor:
        """
        Extract target features from a state dictionary (useful when working with environment states).
        
        Args:
            state_dict: Dictionary containing state information from the environment
                The keys should include:
                - 'current_position': [x, y, z]
                - 'target_position': [x, y, z]
                - (Optional) 'orientation': [roll, pitch, yaw]
                
        Returns:
            Tensor containing target information (distance, azimuth, elevation)
        """
        # Extract positions from state dict
        if 'current_position' not in state_dict or 'target_position' not in state_dict:
            raise ValueError("State dict must contain 'current_position' and 'target_position'")
        
        current_position = torch.tensor(state_dict['current_position'], dtype=torch.float32)
        target_position = torch.tensor(state_dict['target_position'], dtype=torch.float32)
        
        # Extract orientation if available
        current_orientation = None
        if 'orientation' in state_dict:
            current_orientation = torch.tensor(state_dict['orientation'], dtype=torch.float32)
        
        # Compute target information
        target_info = self.compute_target_info(current_position, target_position, current_orientation)
        
        return target_info