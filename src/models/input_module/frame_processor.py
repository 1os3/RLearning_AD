import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional, Union
from torch.nn.modules.utils import _pair


class FactorizedDepthwiseSpatiotemporalConv(nn.Module):
    """
    Factorized 3D Depthwise convolution for efficient spatiotemporal processing.
    This implements the factorization described in the technical framework:
    (Spatial₂D + Temporal₁D) factorization.
    
    First applies a 2D spatial convolution to each frame, then applies a 1D
    temporal convolution across frames.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 spatial_kernel_size: Union[int, Tuple[int, int]], 
                 temporal_kernel_size: int,
                 stride: Union[int, Tuple[int, int, int]] = 1, 
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 bias: bool = True):
        """
        Initialize factorized depthwise spatiotemporal convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            spatial_kernel_size: Kernel size for spatial convolution (height, width)
            temporal_kernel_size: Kernel size for temporal convolution
            stride: Stride for convolution operations
            padding: Padding for convolution operations
            bias: Whether to include bias parameters
        """
        super().__init__()
        
        # Convert parameters to tuples if they are ints
        spatial_kernel_size = _pair(spatial_kernel_size)
        
        if isinstance(stride, int):
            self.spatial_stride = (stride, stride)
            self.temporal_stride = stride
        else:
            assert len(stride) == 3, "Stride should be an int or a 3-tuple"
            self.spatial_stride = (stride[1], stride[2])
            self.temporal_stride = stride[0]
        
        if isinstance(padding, int):
            self.spatial_padding = (padding, padding)
            self.temporal_padding = padding
        else:
            assert len(padding) == 3, "Padding should be an int or a 3-tuple"
            self.spatial_padding = (padding[1], padding[2])
            self.temporal_padding = padding[0]
        
        # Spatial convolution (depthwise)
        self.spatial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # Same as input for depthwise
            kernel_size=spatial_kernel_size,
            stride=self.spatial_stride,
            padding=self.spatial_padding,
            groups=in_channels,  # Makes it depthwise
            bias=bias
        )
        
        # Temporal convolution (depthwise)
        self.temporal_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,  # Same as input for depthwise
            kernel_size=(temporal_kernel_size, 1, 1),  # Only convolve in temporal dimension
            stride=(self.temporal_stride, 1, 1),
            padding=(self.temporal_padding, 0, 0),
            groups=in_channels,  # Makes it depthwise
            bias=bias
        )
        
        # Pointwise convolution to mix channels (1x1 conv)
        self.pointwise_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the factorized spatiotemporal convolution.
        
        Args:
            x: Input tensor of shape (batch_size, channels, temporal_dim, height, width)
            
        Returns:
            Output tensor after spatiotemporal convolution
        """
        batch_size, channels, time_dim, height, width = x.size()
        
        # Reshape for spatial convolution: merge batch and time
        x_spatial = x.transpose(1, 2).contiguous().view(-1, channels, height, width)
        
        # Apply spatial convolution
        x_spatial = self.spatial_conv(x_spatial)
        
        # Reshape back to 5D
        _, channels, height, width = x_spatial.size()
        x_spatial = x_spatial.view(batch_size, time_dim, channels, height, width).transpose(1, 2).contiguous()
        
        # Apply temporal convolution
        x_temporal = self.temporal_conv(x_spatial)
        
        # Apply pointwise convolution to mix channels
        x_out = self.pointwise_conv(x_temporal)
        
        return x_out


class DynamicPatchExtractor(nn.Module):
    """
    Dynamic patch extraction module that adjusts patch size and count based on scene complexity.
    This implements the "动态Patch提取：依据场景复杂度自动调整Patch大小与数量" feature from the framework.
    """
    def __init__(self, 
                 in_channels: int,
                 base_patch_size: int = 16,
                 min_patch_size: int = 8,
                 max_patch_size: int = 32,
                 complexity_channels: int = 32):
        """
        Initialize dynamic patch extractor.
        
        Args:
            in_channels: Number of input channels
            base_patch_size: Default patch size
            min_patch_size: Minimum possible patch size
            max_patch_size: Maximum possible patch size
            complexity_channels: Number of channels in complexity estimation network
        """
        super().__init__()
        
        self.base_patch_size = base_patch_size
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        
        # Small network to estimate scene complexity
        self.complexity_estimator = nn.Sequential(
            nn.Conv2d(in_channels, complexity_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(complexity_channels, 1, kernel_size=1),
            nn.Sigmoid()  # Normalize complexity score between 0 and 1
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        """
        Dynamically extract patches based on estimated scene complexity.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of:
                - Original image for skip connection
                - List of extracted patches
                - Actual patch size used
        """
        batch_size, channels, height, width = x.size()
        
        # Estimate scene complexity (higher score = more complex scene)
        complexity = self.complexity_estimator(x).view(batch_size, 1)
        
        # Adjust patch size based on complexity (smaller patches for complex scenes)
        # Scale between min and max patch size
        patch_size = (
            self.max_patch_size - 
            (self.max_patch_size - self.min_patch_size) * complexity
        ).int().item()
        
        # Ensure patch_size is even for easier processing
        patch_size = max(self.min_patch_size, patch_size - (patch_size % 2))
        
        # Calculate number of patches
        n_patches_h = math.ceil(height / patch_size)
        n_patches_w = math.ceil(width / patch_size)
        
        # Extract patches
        patches = []
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch boundaries
                top = i * patch_size
                left = j * patch_size
                bottom = min(top + patch_size, height)
                right = min(left + patch_size, width)
                
                # Extract patch
                patch = x[:, :, top:bottom, left:right]
                
                # Resize to standard size if needed
                if patch.size(2) != patch_size or patch.size(3) != patch_size:
                    patch = F.interpolate(
                        patch, 
                        size=(patch_size, patch_size), 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                patches.append(patch)
        
        return x, patches, patch_size


class LightweightAttention(nn.Module):
    """
    Lightweight attention mechanism based on Linformer/Performer approach.
    This implements the "轻量化注意力（Linformer／Performer）减少计算与内存" feature.
    """
    def __init__(self, 
                 dim: int, 
                 num_heads: int = 8, 
                 dropout: float = 0.1, 
                 low_rank_dim: Optional[int] = None,
                 use_performer: bool = False):
        """
        Initialize lightweight attention module.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            low_rank_dim: Dimension for low-rank projection (if None, uses sqrt of sequence length)
            use_performer: Whether to use Performer attention instead of Linformer
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_performer = use_performer
        self.low_rank_dim = low_rank_dim
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # For Performer attention (random features approximation)
        if use_performer:
            self.feature_map_size = self.head_dim
            self.register_buffer("random_matrix", torch.randn(self.feature_map_size, self.head_dim) * 0.4)
    
    def _linformer_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention using Linformer's approach (low-rank approximation).
        
        Args:
            q, k, v: Query, key, value tensors
            mask: Optional attention mask
            
        Returns:
            Output after attention
        """
        batch_size, seq_len, _ = q.size()
        
        # Determine low-rank dimension
        if self.low_rank_dim is None:
            low_rank_dim = max(4, int(math.sqrt(seq_len)))
        else:
            low_rank_dim = self.low_rank_dim
        
        # Linear projection of keys and values for dimension reduction
        k_projection = nn.Linear(seq_len, low_rank_dim, device=q.device)
        v_projection = nn.Linear(seq_len, low_rank_dim, device=q.device)
        
        # Project keys and values to lower dimension
        k_projected = k_projection(k.transpose(1, 2)).transpose(1, 2)  # (batch, low_rank_dim, head_dim)
        v_projected = v_projection(v.transpose(1, 2)).transpose(1, 2)  # (batch, low_rank_dim, head_dim)
        
        # Scale query
        q = q / math.sqrt(self.head_dim)
        
        # Compute attention scores
        attn = torch.bmm(q, k_projected.transpose(1, 2))  # (batch, seq_len, low_rank_dim)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        output = torch.bmm(attn, v_projected)  # (batch, seq_len, head_dim)
        
        return output
    
    def _performer_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention using Performer's approach (random feature approximation).
        
        Args:
            q, k, v: Query, key, value tensors
            mask: Optional attention mask
            
        Returns:
            Output after attention
        """
        # Apply random feature map approximation
        def feature_map(x):
            # Explicit feature map approximation
            x = x / math.sqrt(self.head_dim)
            projection = torch.einsum('bhnd,dm->bhnm', x, self.random_matrix)
            return torch.exp(projection - torch.norm(x, dim=-1, keepdim=True).square() / 2)
        
        # Apply feature maps
        q_prime = feature_map(q)  # (batch, heads, seq_len, feature_map_size)
        k_prime = feature_map(k)  # (batch, heads, seq_len, feature_map_size)
        
        if mask is not None:
            k_prime = k_prime * mask.unsqueeze(-1)
        
        # Compute normalization factor
        kv = torch.einsum('bhnd,bhne->bhde', k_prime, v)
        z = 1.0 / (torch.einsum('bhnd->bhn', q_prime) + 1e-8).unsqueeze(-1)
        
        # Compute output
        output = torch.einsum('bhnd,bhde,bhn->bhne', q_prime, kv, z)
        
        return output
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of lightweight attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor after attention
        """
        batch_size, seq_len, _ = x.size()
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use appropriate attention mechanism
        if self.use_performer:
            attn_output = self._performer_attention(q, k, v, mask)
        else:
            attn_output = self._linformer_attention(
                q.reshape(batch_size * self.num_heads, seq_len, self.head_dim),
                k.reshape(batch_size * self.num_heads, seq_len, self.head_dim),
                v.reshape(batch_size * self.num_heads, seq_len, self.head_dim),
                mask
            )
            attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # Combine heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)
        
        return output


class FrameProcessor(nn.Module):
    """
    Process multiple high-resolution frames with factorized 3D convolutions
    and dynamic patch extraction.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize frame processor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Load config from file if not provided
        if config is None:
            import os
            import yaml
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "config", "default.yaml"
            )
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            config = full_config['model']['input_module']['image_processor']
        
        # Get parameters from config
        self.backbone = config.get('backbone', 'efficientnet')
        self.pretrained = config.get('pretrained', True)
        self.feature_dim = config.get('feature_dim', 512)
        self.patch_size = config.get('patch_size', 16)
        
        # Backbone CNN
        if self.backbone == 'efficientnet':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT if self.pretrained else None
            backbone = efficientnet_b0(weights=weights)
            self.backbone_channels = 1280  # EfficientNet-B0 feature channels
            # Remove classifier head
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        elif self.backbone == 'resnet':
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if self.pretrained else None
            backbone = resnet18(weights=weights)
            self.backbone_channels = 512  # ResNet18 feature channels
            # Remove classifier head
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        elif self.backbone == 'mobilenet':
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.DEFAULT if self.pretrained else None
            backbone = mobilenet_v2(weights=weights)
            self.backbone_channels = 1280  # MobileNetV2 feature channels
            # Remove classifier head
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        # Dynamic patch extraction
        self.dynamic_patch_extractor = DynamicPatchExtractor(
            in_channels=3,  # RGB input
            base_patch_size=self.patch_size,
            min_patch_size=8,
            max_patch_size=32
        )
        
        # Factorized 3D Depthwise convolution for spatiotemporal processing
        self.spatiotemporal_conv = FactorizedDepthwiseSpatiotemporalConv(
            in_channels=self.backbone_channels,
            out_channels=self.feature_dim,
            spatial_kernel_size=3,
            temporal_kernel_size=3,
            padding=(1, 1, 1)
        )
        
        # Lightweight attention for aggregating information
        self.attention = LightweightAttention(
            dim=self.feature_dim,
            num_heads=8,
            dropout=0.1,
            use_performer=True
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process multiple frames with efficient spatiotemporal convolutions.
        
        Args:
            x: Input tensor of shape (batch_size, C*frames, H, W)
                where C is the number of channels per frame (typically 3 for RGB)
                and frames is the number of stacked frames.
                
        Returns:
            Processed frame features of shape (batch_size, feature_dim)
        """
        batch_size, channels, height, width = x.size()
        frame_channels = 3  # RGB channels per frame
        num_frames = channels // frame_channels
        
        # Reshape input to separate frames
        x = x.view(batch_size, num_frames, frame_channels, height, width)
        
        # Process each frame with the backbone
        frame_features = []
        for i in range(num_frames):
            frame = x[:, i]  # (batch_size, frame_channels, height, width)
            
            # Apply dynamic patch extraction (returns original, patches, patch_size)
            _, patches, _ = self.dynamic_patch_extractor(frame)
            
            # Process each patch with the feature extractor
            patch_features = []
            for patch in patches:
                # Forward through backbone
                features = self.feature_extractor(patch)  # (batch_size, backbone_channels, patch_h, patch_w)
                # Global average pooling
                features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
                patch_features.append(features)
            
            # Average patch features for this frame
            frame_feature = torch.stack(patch_features, dim=1).mean(dim=1)  # (batch_size, backbone_channels)
            frame_features.append(frame_feature)
        
        # Stack frame features (batch_size, num_frames, backbone_channels)
        stacked_features = torch.stack(frame_features, dim=1)
        
        # Reshape for spatiotemporal convolution
        # From (batch_size, num_frames, backbone_channels) to (batch_size, backbone_channels, num_frames, 1, 1)
        reshaped_features = stacked_features.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        
        # Apply spatiotemporal convolution
        spatiotemporal_features = self.spatiotemporal_conv(reshaped_features)
        
        # Reshape to (batch_size, num_frames, feature_dim)
        spatiotemporal_features = spatiotemporal_features.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        
        # Apply lightweight attention to aggregate temporal information
        attended_features = self.attention(spatiotemporal_features)
        
        # Global pooling across time dimension
        pooled_features = attended_features.mean(dim=1)  # (batch_size, feature_dim)
        
        # Final projection
        output = self.output_projection(pooled_features)
        
        return output