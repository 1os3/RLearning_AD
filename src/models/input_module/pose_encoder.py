import torch
import torch.nn as nn
import yaml
import os
from typing import Dict, List, Optional, Tuple

class PoseEncoder(nn.Module):
    """
    Encodes drone pose information (velocity, acceleration, orientation).
    Implements the framework's "姿态向量 (v, a, roll, pitch, yaw) → 2 层 MLP + LayerNorm → embedding token"
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pose encoder.
        
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
            config = full_config['model']['input_module']['pose_encoder']
        
        # Get parameters from config
        self.input_dim = config.get('input_dim', 12)  # v(3), a(3), angles(3), angular_v(3)
        self.hidden_dims = config.get('hidden_dims', [64, 128])
        self.output_dim = config.get('output_dim', 256)
        
        # Build MLP layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        layers.append(nn.LayerNorm(self.output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, pose_vector: torch.Tensor) -> torch.Tensor:
        """
        Encode pose information into a fixed-dimensional embedding.
        
        Args:
            pose_vector: Tensor of shape (batch_size, input_dim) containing pose information
                    This typically includes velocity, acceleration, orientation angles
                    
        Returns:
            Encoded pose embedding of shape (batch_size, output_dim)
        """
        # Ensure input has the correct shape
        if pose_vector.dim() == 1:
            pose_vector = pose_vector.unsqueeze(0)  # Add batch dimension if missing
            
        # Verify input dimension
        if pose_vector.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {pose_vector.shape[-1]}")
        
        # Apply MLP with layer normalization
        embedding = self.mlp(pose_vector)
        
        return embedding
    
    def extract_features(self, state_dict: Dict) -> torch.Tensor:
        """
        Extract pose features from a state dictionary (useful when working with environment states).
        
        Args:
            state_dict: Dictionary containing state information from the environment
                The keys should include some or all of:
                - 'position': [x, y, z]
                - 'velocity': [vx, vy, vz]
                - 'acceleration': [ax, ay, az]
                - 'orientation': [roll, pitch, yaw]
                - 'angular_velocity': [ang_vx, ang_vy, ang_vz]
                
        Returns:
            Tensor containing concatenated pose features
        """
        features = []
        
        # Extract available features and handle missing ones with zeros
        if 'velocity' in state_dict:
            features.append(torch.tensor(state_dict['velocity'], dtype=torch.float32))
        else:
            features.append(torch.zeros(3, dtype=torch.float32))
            
        if 'acceleration' in state_dict:
            features.append(torch.tensor(state_dict['acceleration'], dtype=torch.float32))
        else:
            features.append(torch.zeros(3, dtype=torch.float32))
            
        if 'orientation' in state_dict:
            features.append(torch.tensor(state_dict['orientation'], dtype=torch.float32))
        else:
            features.append(torch.zeros(3, dtype=torch.float32))
            
        if 'angular_velocity' in state_dict:
            features.append(torch.tensor(state_dict['angular_velocity'], dtype=torch.float32))
        else:
            features.append(torch.zeros(3, dtype=torch.float32))
        
        # Concatenate all features
        pose_vector = torch.cat(features)
        
        # Ensure we have the expected number of features
        if pose_vector.shape[0] != self.input_dim:
            # Pad with zeros if necessary
            if pose_vector.shape[0] < self.input_dim:
                padding = torch.zeros(self.input_dim - pose_vector.shape[0], dtype=torch.float32)
                pose_vector = torch.cat([pose_vector, padding])
            else:
                # Truncate if we have too many features
                pose_vector = pose_vector[:self.input_dim]
        
        return pose_vector.unsqueeze(0)  # Add batch dimension