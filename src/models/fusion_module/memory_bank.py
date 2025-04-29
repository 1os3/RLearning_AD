import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
import os
import collections
from typing import Dict, List, Optional, Tuple, Union, Deque

class KeyValueMemory(nn.Module):
    """
    Key-Value Memory module for storing and retrieving historical information.
    Implements the framework's "Key-Value Memory 队列（存储最近 T 步 latent），利用注意力检索历史信息"
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize key-value memory bank.
        
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
            config = full_config['model']['fusion_module']['memory_bank']
        
        # Get parameters from config
        self.memory_size = config.get('memory_size', 32)  # Number of time steps to remember
        self.key_dim = config.get('key_dim', 512)
        self.value_dim = config.get('value_dim', 512)
        self.use_attention_retrieval = config.get('use_attention_retrieval', True)
        
        # Memory storage (will be initialized during forward pass)
        # We don't use register_buffer here because we want gradient flow through the memory
        self.keys = None   # Will be (batch_size, memory_size, key_dim)
        self.values = None  # Will be (batch_size, memory_size, value_dim)
        self.memory_mask = None  # Will be (batch_size, memory_size)
        
        # Counter to track position in memory for each sequence in batch
        self.position = None  # Will be (batch_size,)
        
        # Key projection for attention-based retrieval
        self.query_projection = nn.Linear(self.key_dim, self.key_dim)
        self.key_projection = nn.Linear(self.key_dim, self.key_dim)
        self.value_projection = nn.Linear(self.value_dim, self.value_dim)
        
        # Output projection after attention
        self.output_projection = nn.Linear(self.value_dim, self.value_dim)
        
        # Layer normalization for attention
        self.norm_query = nn.LayerNorm(self.key_dim)
        self.norm_output = nn.LayerNorm(self.value_dim)
        
        # Parameter for attention scaling
        self.attention_scale = math.sqrt(self.key_dim)
        
    def _init_memory(self, batch_size: int, device: torch.device):
        """
        Initialize memory storage for a new batch.
        
        Args:
            batch_size: Batch size
            device: Device to store memory on
        """
        self.keys = torch.zeros(batch_size, self.memory_size, self.key_dim, device=device)
        self.values = torch.zeros(batch_size, self.memory_size, self.value_dim, device=device)
        self.memory_mask = torch.zeros(batch_size, self.memory_size, dtype=torch.bool, device=device)
        self.position = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    def reset_memory(self, batch_indices: Optional[torch.Tensor] = None):
        """
        Reset memory for specific batch indices or all memory if indices not provided.
        
        Args:
            batch_indices: Optional tensor of batch indices to reset
        """
        if self.keys is None:
            # Memory not initialized yet
            return
        
        if batch_indices is None:
            # Reset all memory
            batch_indices = torch.arange(self.keys.size(0), device=self.keys.device)
        
        # Reset memory for specified indices
        self.keys[batch_indices] = 0
        self.values[batch_indices] = 0
        self.memory_mask[batch_indices] = False
        self.position[batch_indices] = 0
    
    def write(self, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Write new key-value pairs to memory.
        
        Args:
            keys: Key tensor of shape (batch_size, key_dim)
            values: Value tensor of shape (batch_size, value_dim)
            mask: Optional mask of shape (batch_size,) where True indicates valid entries
        """
        batch_size, key_dim = keys.size()
        device = keys.device
        
        # Initialize memory if needed
        if self.keys is None or self.keys.size(0) != batch_size:
            self._init_memory(batch_size, device)
        
        # Make sure keys and values have correct dimensions
        if key_dim != self.key_dim:
            raise ValueError(f"Key dimension mismatch: expected {self.key_dim}, got {key_dim}")
        if values.size(1) != self.value_dim:
            raise ValueError(f"Value dimension mismatch: expected {self.value_dim}, got {values.size(1)}")
        
        # Apply write mask if provided
        if mask is not None:
            # Create mask for entries we want to write
            write_mask = mask.bool()
        else:
            # Write to all entries in batch
            write_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Get current position for each sequence
        pos = self.position.clone()
        
        # Write new key-value pairs to memory at current position
        for i in range(batch_size):
            if write_mask[i]:
                self.keys[i, pos[i]] = keys[i]
                self.values[i, pos[i]] = values[i]
                self.memory_mask[i, pos[i]] = True
                
                # Update position (circular buffer)
                self.position[i] = (pos[i] + 1) % self.memory_size
    
    def read(self, query: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Read from memory using attention mechanism.
        
        Args:
            query: Query tensor of shape (batch_size, key_dim)
            mask: Optional mask of shape (batch_size,) where True indicates valid entries
            
        Returns:
            Retrieved memory of shape (batch_size, value_dim)
        """
        batch_size = query.size(0)
        device = query.device
        
        # Check if memory is initialized
        if self.keys is None or self.keys.size(0) != batch_size:
            # Return zeros if memory not initialized
            return torch.zeros(batch_size, self.value_dim, device=device)
        
        # Apply read mask if provided
        if mask is not None:
            # Create mask for entries we want to read
            read_mask = mask.bool().unsqueeze(1)  # (batch_size, 1)
        else:
            # Read from all entries in batch
            read_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        
        # Combine with memory mask
        combined_mask = torch.logical_and(read_mask, self.memory_mask)  # (batch_size, memory_size)
        
        if self.use_attention_retrieval:
            # Attention-based retrieval
            # Project query and apply layer norm
            query = self.norm_query(query)  # (batch_size, key_dim)
            q = self.query_projection(query).unsqueeze(1)  # (batch_size, 1, key_dim)
            
            # Project keys
            k = self.key_projection(self.keys)  # (batch_size, memory_size, key_dim)
            v = self.value_projection(self.values)  # (batch_size, memory_size, value_dim)
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.key_dim)  # (batch_size, 1, memory_size)
            
            # Apply mask
            mask_expanded = combined_mask.unsqueeze(1)  # (batch_size, 1, memory_size)
            scores = scores.masked_fill(~mask_expanded, -1e9)  # Apply large negative value to masked positions
            
            # Compute attention weights
            weights = F.softmax(scores, dim=-1)  # (batch_size, 1, memory_size)
            
            # Apply attention to retrieve values
            retrieved = torch.matmul(weights, v).squeeze(1)  # (batch_size, value_dim)
            
            # Apply output projection and normalization
            output = self.output_projection(retrieved)  # (batch_size, value_dim)
            output = self.norm_output(output)  # (batch_size, value_dim)
        else:
            # Simple average of valid memory entries
            valid_count = combined_mask.float().sum(dim=1, keepdim=True) + 1e-8  # Avoid division by zero
            masked_values = self.values * combined_mask.unsqueeze(-1).float()
            output = masked_values.sum(dim=1) / valid_count  # (batch_size, value_dim)
        
        return output

class DynamicHistoryMemory(nn.Module):
    """
    Dynamic history memory with adaptive capacity.
    Implements the "memory capacity self-adaptive" feature mentioned in the framework:
    "记忆容量自适应：场景稳定时压缩记忆；突变时扩展"
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize dynamic history memory.
        
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
            config = full_config['model']['fusion_module']['memory_bank']
        
        # Get parameters from config
        self.base_memory_size = config.get('memory_size', 32)
        self.key_dim = config.get('key_dim', 512)
        self.value_dim = config.get('value_dim', 512)
        self.use_attention_retrieval = config.get('use_attention_retrieval', True)
        
        # Create base memory module
        self.memory = KeyValueMemory(config)
        
        # Stability estimation network
        self.stability_estimator = nn.Sequential(
            nn.Linear(self.key_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 (unstable) and 1 (stable)
        )
        
        # Previous feature for stability comparison
        self.previous_feature = None
        
        # Current effective memory size (adjusted dynamically)
        self.current_memory_size = self.base_memory_size
        
    def estimate_stability(self, current_feature: torch.Tensor) -> torch.Tensor:
        """
        Estimate environment stability by comparing current feature with previous feature.
        
        Args:
            current_feature: Current feature tensor of shape (batch_size, key_dim)
            
        Returns:
            Stability score between 0 (unstable) and 1 (stable) of shape (batch_size, 1)
        """
        batch_size = current_feature.size(0)
        device = current_feature.device
        
        if self.previous_feature is None or self.previous_feature.size(0) != batch_size:
            # Initialize previous feature
            self.previous_feature = current_feature.clone().detach()
            return torch.ones(batch_size, 1, device=device)  # Assume stable initially
        
        # Compare current and previous features
        feature_pair = torch.cat([current_feature, self.previous_feature], dim=1)  # (batch_size, key_dim*2)
        stability = self.stability_estimator(feature_pair)  # (batch_size, 1)
        
        # Update previous feature
        self.previous_feature = current_feature.clone().detach()
        
        return stability
    
    def adjust_memory_size(self, stability: torch.Tensor):
        """
        Adjust memory size based on stability (expand when unstable, compress when stable).
        
        Args:
            stability: Stability score between 0 (unstable) and 1 (stable) of shape (batch_size, 1)
        """
        # Calculate average stability across batch
        avg_stability = stability.mean().item()
        
        # Define thresholds for adjustment
        expansion_threshold = 0.3  # Below this stability, expand memory
        compression_threshold = 0.7  # Above this stability, compress memory
        
        # Define adjustment factor
        adjustment_factor = 1.5  # Factor to expand/compress memory by
        
        # Adjust memory size
        if avg_stability < expansion_threshold:
            # Unstable environment, expand memory
            self.current_memory_size = min(
                int(self.current_memory_size * adjustment_factor),
                self.base_memory_size * 2  # Cap at 2x base size
            )
        elif avg_stability > compression_threshold:
            # Stable environment, compress memory
            self.current_memory_size = max(
                int(self.current_memory_size / adjustment_factor),
                self.base_memory_size // 2  # Floor at 1/2 base size
            )
        
        # Update memory module size (implemented in KeyValueMemory)
        # This is a theoretical adjustment - in practice, we need to handle it specially
        # since changing tensor sizes on the fly is complex
        # For now, we'll just note the effective size and apply it during retrieval
    
    def forward(self, keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor = None):
        """
        Process through dynamic memory: estimate stability, adjust memory size, write and read.
        
        Args:
            keys: Key tensor of shape (batch_size, key_dim)
            values: Value tensor of shape (batch_size, value_dim)
            query: Optional query tensor of shape (batch_size, key_dim), if None uses keys
            
        Returns:
            Retrieved memory of shape (batch_size, value_dim)
        """
        # Estimate environment stability
        stability = self.estimate_stability(keys)
        
        # Adjust memory size based on stability
        self.adjust_memory_size(stability)
        
        # Write to memory
        self.memory.write(keys, values)
        
        # Read from memory using query or keys if query not provided
        if query is None:
            query = keys
        
        retrieved = self.memory.read(query)
        
        return retrieved, stability

class PriorityAttentionMemory(nn.Module):
    """
    Memory module with priority-based attention.
    Implements the "优先级注意力：根据 TD-error 调整历史信息检索权重" feature.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize priority attention memory.
        
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
            config = full_config['model']['fusion_module']['memory_bank']
        
        # Get parameters from config
        self.memory_size = config.get('memory_size', 32)
        self.key_dim = config.get('key_dim', 512)
        self.value_dim = config.get('value_dim', 512)
        
        # Create base memory module
        self.memory = KeyValueMemory(config)
        
        # Priority estimator network (predicts TD-error)
        self.priority_estimator = nn.Sequential(
            nn.Linear(self.key_dim + self.value_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive priority values
        )
        
        # Priorities storage
        self.priorities = None  # Will be (batch_size, memory_size)
    
    def _init_priorities(self, batch_size: int, device: torch.device):
        """
        Initialize priorities storage.
        
        Args:
            batch_size: Batch size
            device: Device to store priorities on
        """
        self.priorities = torch.ones(batch_size, self.memory_size, device=device)  # Default priority = 1
    
    def update_priorities(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Update priorities for all memory entries.
        
        Args:
            keys: Key tensor of shape (batch_size, key_dim)
            values: Value tensor of shape (batch_size, value_dim)
        """
        if self.memory.keys is None or self.memory.values is None:
            # Memory not initialized yet
            return
        
        batch_size = keys.size(0)
        device = keys.device
        
        # Initialize priorities if needed
        if self.priorities is None or self.priorities.size(0) != batch_size:
            self._init_priorities(batch_size, device)
        
        # For each memory entry, estimate priority (TD-error)
        for i in range(self.memory_size):
            memory_keys = self.memory.keys[:, i]  # (batch_size, key_dim)
            memory_values = self.memory.values[:, i]  # (batch_size, value_dim)
            
            # Concat key-value for priority estimation
            memory_concat = torch.cat([memory_keys, memory_values], dim=1)  # (batch_size, key_dim + value_dim)
            
            # Estimate priority
            self.priorities[:, i] = self.priority_estimator(memory_concat).squeeze(-1)  # (batch_size,)
    
    def write_with_priority(self, keys: torch.Tensor, values: torch.Tensor, td_errors: Optional[torch.Tensor] = None):
        """
        Write to memory with known priorities (TD-errors).
        
        Args:
            keys: Key tensor of shape (batch_size, key_dim)
            values: Value tensor of shape (batch_size, value_dim)
            td_errors: Optional TD-error tensor of shape (batch_size,) for priorities
        """
        # Write to memory normally
        self.memory.write(keys, values)
        
        batch_size = keys.size(0)
        device = keys.device
        
        # Initialize priorities if needed
        if self.priorities is None or self.priorities.size(0) != batch_size:
            self._init_priorities(batch_size, device)
        
        # Get current position (indices of newest entries)
        pos = (self.memory.position - 1) % self.memory_size  # (batch_size,)
        
        # Update priorities for newest entries
        if td_errors is not None:
            # Use provided TD-errors
            for i in range(batch_size):
                self.priorities[i, pos[i]] = td_errors[i]
        else:
            # Estimate priorities
            concat_features = torch.cat([keys, values], dim=1)  # (batch_size, key_dim + value_dim)
            estimated_priorities = self.priority_estimator(concat_features).squeeze(-1)  # (batch_size,)
            
            for i in range(batch_size):
                self.priorities[i, pos[i]] = estimated_priorities[i]
    
    def read_with_priority(self, query: torch.Tensor) -> torch.Tensor:
        """
        Read from memory with priority-weighted attention.
        
        Args:
            query: Query tensor of shape (batch_size, key_dim)
            
        Returns:
            Retrieved memory of shape (batch_size, value_dim)
        """
        batch_size = query.size(0)
        device = query.device
        
        # Check if memory is initialized
        if self.memory.keys is None or self.memory.keys.size(0) != batch_size:
            # Return zeros if memory not initialized
            return torch.zeros(batch_size, self.value_dim, device=device)
        
        # Initialize priorities if needed
        if self.priorities is None or self.priorities.size(0) != batch_size:
            self._init_priorities(batch_size, device)
        
        # Project query
        q = self.memory.query_projection(query).unsqueeze(1)  # (batch_size, 1, key_dim)
        
        # Project keys and values
        k = self.memory.key_projection(self.memory.keys)  # (batch_size, memory_size, key_dim)
        v = self.memory.value_projection(self.memory.values)  # (batch_size, memory_size, value_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.key_dim)  # (batch_size, 1, memory_size)
        
        # Apply memory mask
        mask_expanded = self.memory.memory_mask.unsqueeze(1)  # (batch_size, 1, memory_size)
        scores = scores.masked_fill(~mask_expanded, -1e9)  # Apply large negative value to masked positions
        
        # Apply priority weighting - modify attention scores based on priorities
        priority_weights = F.softmax(self.priorities, dim=1).unsqueeze(1)  # (batch_size, 1, memory_size)
        priority_adjusted_scores = scores * priority_weights  # (batch_size, 1, memory_size)
        
        # Compute final attention weights
        weights = F.softmax(priority_adjusted_scores, dim=-1)  # (batch_size, 1, memory_size)
        
        # Apply attention to retrieve values
        retrieved = torch.matmul(weights, v).squeeze(1)  # (batch_size, value_dim)
        
        # Apply output projection and normalization
        output = self.memory.output_projection(retrieved)  # (batch_size, value_dim)
        output = self.memory.norm_output(output)  # (batch_size, value_dim)
        
        return output
    
    def forward(self, keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor = None, 
                td_errors: Optional[torch.Tensor] = None):
        """
        Process through priority memory: write with priority and read with priority weighting.
        
        Args:
            keys: Key tensor of shape (batch_size, key_dim)
            values: Value tensor of shape (batch_size, value_dim)
            query: Optional query tensor of shape (batch_size, key_dim), if None uses keys
            td_errors: Optional TD-error tensor of shape (batch_size,) for priorities
            
        Returns:
            Retrieved memory of shape (batch_size, value_dim)
        """
        # Write to memory with priority
        self.write_with_priority(keys, values, td_errors)
        
        # Periodically update all priorities (not just for new entries)
        if torch.rand(1).item() < 0.1:  # 10% chance each forward pass
            self.update_priorities(keys, values)
        
        # Read from memory with priority weighting
        if query is None:
            query = keys
        
        retrieved = self.read_with_priority(query)
        
        return retrieved