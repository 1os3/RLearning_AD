import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
import os
from typing import Dict, List, Optional, Tuple, Union
from torch.nn.modules.utils import _pair, _triple


class FactorizedDepthwise3DConv(nn.Module):
    """
    Factorized 3D Depthwise convolution for efficient spatiotemporal processing.
    Implements the framework's "3D-Depthwise 时空分离卷积块" feature.
    
    The factorization splits a 3D convolution into a 2D spatial convolution
    followed by a 1D temporal convolution, significantly reducing parameters
    and computation.
    """
    def __init__(self, 
                 channels: int,
                 spatial_kernel_size: Union[int, Tuple[int, int]] = 3,
                 temporal_kernel_size: int = 3,
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 output_channels: Optional[int] = None,
                 bias: bool = True):
        """
        Initialize factorized 3D Depthwise convolution.
        
        Args:
            channels: Number of input/output channels (unless output_channels is specified)
            spatial_kernel_size: Kernel size for spatial dimensions (height, width)
            temporal_kernel_size: Kernel size for temporal dimension
            stride: Stride for convolution
            padding: Padding for convolution
            output_channels: Optional number of output channels (for channel expansion/reduction)
            bias: Whether to include bias parameters
        """
        super().__init__()
        
        # Get parameters
        self.channels = channels
        self.output_channels = output_channels if output_channels is not None else channels
        
        # Convert parameters to tuples if they're ints
        spatial_kernel_size = _pair(spatial_kernel_size)
        
        if isinstance(stride, int):
            spatial_stride = (stride, stride)
            temporal_stride = stride
        else:
            assert len(stride) == 3, "Stride should be an int or a 3-tuple"
            spatial_stride = (stride[1], stride[2])
            temporal_stride = stride[0]
        
        if isinstance(padding, int):
            spatial_padding = (padding, padding)
            temporal_padding = padding
        else:
            assert len(padding) == 3, "Padding should be an int or a 3-tuple"
            spatial_padding = (padding[1], padding[2])
            temporal_padding = padding[0]
        
        # Spatial depthwise convolution (2D)
        self.spatial_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=spatial_kernel_size,
            stride=spatial_stride,
            padding=spatial_padding,
            groups=channels,  # Makes it depthwise
            bias=bias
        )
        
        # Temporal depthwise convolution (1D applied to 3D)
        self.temporal_conv = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(temporal_kernel_size, 1, 1),  # Only convolve in temporal dimension
            stride=(temporal_stride, 1, 1),
            padding=(temporal_padding, 0, 0),
            groups=channels,  # Makes it depthwise
            bias=bias
        )
        
        # Pointwise convolution to mix channels (1x1x1 conv)
        self.pointwise_conv = nn.Conv3d(
            in_channels=channels,
            out_channels=self.output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        
        # Normalization and activation
        self.norm = nn.GroupNorm(num_groups=min(4, self.output_channels), num_channels=self.output_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of factorized 3D Depthwise convolution.
        
        Args:
            x: Input tensor of shape (batch_size, channels, temporal_dim, height, width)
            
        Returns:
            Output tensor after convolution
        """
        batch_size, channels, time_dim, height, width = x.size()
        
        # Apply spatial convolution to each time slice
        spatial_out = []
        for t in range(time_dim):
            spatial_frame = self.spatial_conv(x[:, :, t])  # (batch_size, channels, height', width')
            spatial_out.append(spatial_frame)
        
        # Stack frames back together
        x_spatial = torch.stack(spatial_out, dim=2)  # (batch_size, channels, time_dim, height', width')
        
        # Apply temporal convolution
        x_temporal = self.temporal_conv(x_spatial)
        
        # Apply pointwise convolution
        x_out = self.pointwise_conv(x_temporal)
        
        # Apply normalization and activation
        x_out = self.norm(x_out)
        x_out = self.activation(x_out)
        
        return x_out


class ResidualDepthwiseBlock(nn.Module):
    """
    Residual block with factorized 3D Depthwise convolutions.
    Includes a skip connection for better gradient flow.
    """
    def __init__(self, channels: int, spatial_kernel_size: int = 3, temporal_kernel_size: int = 3):
        """
        Initialize residual depthwise block.
        
        Args:
            channels: Number of input/output channels
            spatial_kernel_size: Kernel size for spatial dimensions
            temporal_kernel_size: Kernel size for temporal dimension
        """
        super().__init__()
        
        # First factorized convolution
        self.conv1 = FactorizedDepthwise3DConv(
            channels=channels,
            spatial_kernel_size=spatial_kernel_size,
            temporal_kernel_size=temporal_kernel_size,
            padding=(temporal_kernel_size//2, spatial_kernel_size//2, spatial_kernel_size//2)
        )
        
        # Second factorized convolution
        self.conv2 = FactorizedDepthwise3DConv(
            channels=channels,
            spatial_kernel_size=spatial_kernel_size,
            temporal_kernel_size=temporal_kernel_size,
            padding=(temporal_kernel_size//2, spatial_kernel_size//2, spatial_kernel_size//2)
        )
        
        # Layer normalization for skip connection
        self.norm = nn.GroupNorm(num_groups=min(4, channels), num_channels=channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of residual depthwise block.
        
        Args:
            x: Input tensor of shape (batch_size, channels, temporal_dim, height, width)
            
        Returns:
            Output tensor after residual connection
        """
        # Save input for skip connection
        identity = x
        
        # Apply convolutions
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Add skip connection and normalize
        out = out + identity
        out = self.norm(out)
        
        return out


class SpatioTemporalFeatureExtractor(nn.Module):
    """
    Feature extractor using factorized 3D depthwise convolutions.
    Processes spatiotemporal data efficiently for drone navigation.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize spatiotemporal feature extractor.
        
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
            config = full_config['model']['fusion_module']['depthwise_conv']
        
        # Get parameters from config
        self.channels = config.get('channels', 512)
        self.spatial_kernel_size = config.get('spatial_kernel_size', 3)
        self.temporal_kernel_size = config.get('temporal_kernel_size', 3)
        
        # Initial projection to desired channel dimension
        self.input_projection = nn.Conv3d(
            in_channels=3,  # Assuming RGB input
            out_channels=self.channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )
        
        # Factorized 3D depthwise convolution blocks
        self.blocks = nn.ModuleList([
            ResidualDepthwiseBlock(
                channels=self.channels,
                spatial_kernel_size=self.spatial_kernel_size,
                temporal_kernel_size=self.temporal_kernel_size
            ) for _ in range(3)  # Stack 3 residual blocks
        ])
        
        # Global average pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.output_projection = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatiotemporal feature extractor.
        
        Args:
            x: Input tensor of shape (batch_size, channels, temporal_dim, height, width)
                or (batch_size, temporal_dim, channels, height, width)
            
        Returns:
            Extracted features of shape (batch_size, channels)
        """
        # Ensure input is in the correct format (batch, channels, time, height, width)
        if x.dim() == 5 and x.size(1) > x.size(2):
            # Input is (batch, time, channels, height, width), transpose to correct format
            x = x.permute(0, 2, 1, 3, 4)
        
        # Apply initial projection
        x = self.input_projection(x)
        x = F.relu(x, inplace=True)
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Global pooling and final projection
        x = self.global_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)  # (batch_size, channels)
        x = self.output_projection(x)
        
        return x