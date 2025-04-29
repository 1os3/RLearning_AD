import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
import os
from typing import Dict, List, Optional, Tuple, Union

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module with standard scaled dot-product attention.
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-head attention module.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, query_len, dim)
            key: Key tensor of shape (batch_size, key_len, dim)
            value: Value tensor of shape (batch_size, key_len, dim)
            mask: Optional attention mask of shape (batch_size, query_len, key_len)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape (batch_size, query_len, dim)
            Optionally, attention weights of shape (batch_size, num_heads, query_len, key_len)
        """
        batch_size, query_len, _ = query.size()
        key_len = key.size(1)
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute scaled dot-product attention
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-1, -2))  # (batch_size, num_heads, query_len, key_len)
        
        if mask is not None:
            if mask.dim() == 3:  # (batch_size, query_len, key_len)
                mask = mask.unsqueeze(1)  # Add head dimension
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        output = torch.matmul(attn, v)  # (batch_size, num_heads, query_len, head_dim)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, self.dim)
        
        # Project output
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn
        else:
            return output

class CrossModalAttentionBlock(nn.Module):
    """
    Cross-modal attention block for fusing information from different modalities.
    Implements the framework's "Cross-modal Self-Attention：图像 tokens 作为 Key/Value，姿态／目标 tokens 作为 Query"
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize cross-modal attention block.
        
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
            config = full_config['model']['fusion_module']['cross_attention']
        
        # Get parameters from config
        self.hidden_dim = config.get('hidden_dim', 512)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        self.num_layers = config.get('num_layers', 4)
        
        # Multi-head attention layers
        self.cross_attention_layers = nn.ModuleList([
            MultiHeadAttention(self.hidden_dim, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Layer normalization and feed-forward networks
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.norm3_layers = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
                nn.GELU(),
                nn.Linear(4 * self.hidden_dim, self.hidden_dim),
                nn.Dropout(self.dropout)
            ) for _ in range(self.num_layers)
        ])
        
        # Projection layers for each modality to ensure consistent dimensions
        self.image_projection = nn.Linear(512, self.hidden_dim)  # Assuming frame_processor outputs 512-dim
        self.pose_projection = nn.Linear(256, self.hidden_dim)   # Assuming pose_encoder outputs 256-dim
        self.target_projection = nn.Linear(128, self.hidden_dim) # Assuming target_encoder outputs 128-dim
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        
    def forward(self, 
                image_features: torch.Tensor, 
                pose_features: torch.Tensor, 
                target_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-modal attention block.
        
        Args:
            image_features: Image features from frame_processor (batch_size, image_dim)
            pose_features: Pose features from pose_encoder (batch_size, pose_dim)
            target_features: Target features from target_encoder (batch_size, target_dim)
            
        Returns:
            Tuple of:
                - Fused features (batch_size, hidden_dim)
                - Attention weights of the last layer
        """
        batch_size = image_features.size(0)
        
        # Project all modalities to the same dimension
        image_feats = self.image_projection(image_features).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        pose_feats = self.pose_projection(pose_features).unsqueeze(1)    # (batch_size, 1, hidden_dim)
        target_feats = self.target_projection(target_features).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Concatenate pose and target features as queries
        queries = torch.cat([pose_feats, target_feats], dim=1)  # (batch_size, 2, hidden_dim)
        
        # Use image features as keys and values
        keys_values = image_feats  # (batch_size, 1, hidden_dim)
        
        # Process through multiple layers of cross-attention
        x = queries
        last_attention = None
        
        for i in range(self.num_layers):
            # Self-attention among queries (pose + target)
            residual = x
            x = self.norm1_layers[i](x)
            x = x + self.cross_attention_layers[i](x, x, x)  # Self-attention
            x = x + residual
            
            # Cross-attention with image features
            residual = x
            x = self.norm2_layers[i](x)
            
            if i == self.num_layers - 1:  # Last layer, save attention weights
                x_cross, attn = self.cross_attention_layers[i](x, keys_values, keys_values, return_attention=True)
                last_attention = attn
                x = x + x_cross
            else:
                x = x + self.cross_attention_layers[i](x, keys_values, keys_values)
            
            x = x + residual
            
            # Feed-forward network
            residual = x
            x = self.norm3_layers[i](x)
            x = x + self.ffn_layers[i](x)
            x = x + residual
        
        # Final normalization
        x = self.final_norm(x)  # (batch_size, 2, hidden_dim)
        
        # Combine pose and target features
        fused_features = torch.mean(x, dim=1)  # (batch_size, hidden_dim)
        
        return fused_features, last_attention

class CrossModalTransformer(nn.Module):
    """
    Full cross-modal transformer model that combines image, pose, and target information.
    Includes multiple cross-modal attention blocks and gated fusion.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize cross-modal transformer.
        
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
            config = full_config['model']['fusion_module']['cross_attention']
        
        # Get parameters from config
        self.hidden_dim = config.get('hidden_dim', 512)
        
        # Cross-modal attention block
        self.cross_modal_block = CrossModalAttentionBlock(config)
        
        # Gated fusion mechanism
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )
        
        # Skip connection projection
        self.skip_projection = nn.Sequential(
            nn.Linear(512 + 256 + 128, self.hidden_dim),  # Sum of original dimensions
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, 
                image_features: torch.Tensor, 
                pose_features: torch.Tensor, 
                target_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of cross-modal transformer.
        
        Args:
            image_features: Image features from frame_processor (batch_size, image_dim)
            pose_features: Pose features from pose_encoder (batch_size, pose_dim)
            target_features: Target features from target_encoder (batch_size, target_dim)
            
        Returns:
            Fused features (batch_size, hidden_dim)
        """
        # Skip connection: direct concatenation of original features
        skip_features = torch.cat([image_features, pose_features, target_features], dim=1)
        skip_features = self.skip_projection(skip_features)  # (batch_size, hidden_dim)
        
        # Cross-modal attention
        attended_features, _ = self.cross_modal_block(image_features, pose_features, target_features)
        
        # Gated fusion of skip connection and attended features
        concat_features = torch.cat([skip_features, attended_features], dim=1)  # (batch_size, hidden_dim*2)
        gate_values = self.gate(concat_features)  # (batch_size, hidden_dim)
        
        fused_features = gate_values * attended_features + (1 - gate_values) * skip_features
        
        # Final projection
        output = self.output_projection(fused_features)  # (batch_size, hidden_dim)
        
        return output