import numpy as np
import torch
import random
import gc
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple
from collections import deque
from src.utils.tensor_utils import safe_stack, safe_process_batch

class Transition(NamedTuple):
    """
    Named tuple to store transitions in replay buffer.
    Using NamedTuple for better memory efficiency and type safety.
    
    Note: state and next_state can be either tensors or dictionaries of tensors
    """
    state: Any  # 可以是torch.Tensor或Dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: Any  # 可以是torch.Tensor或Dict[str, torch.Tensor]
    done: torch.Tensor

class ReplayBuffer:
    """
    Standard (uniform) replay buffer for off-policy RL algorithms.
    Implements simple FIFO storage with random batch sampling.
    """
    def __init__(self, buffer_size: int, device: torch.device):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum size of buffer
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.device = device
        
        # Empty buffer
        self.buffer = []
        self.position = 0
    
    def __len__(self) -> int:
        """
        Return current buffer size.
        """
        return len(self.buffer)
        
    def push(self, 
             state, 
             action: torch.Tensor, 
             reward: torch.Tensor, 
             next_state, 
             done: torch.Tensor) -> None:
        """
        Store transition in buffer.
        
        Args:
            state: Current state (tensor or dictionary of tensors)
            action: Action tensor
            reward: Reward tensor
            next_state: Next state (tensor or dictionary of tensors)
            done: Done flag tensor
        """
        # 处理状态可能是字典类型的情况
        if isinstance(state, dict):
            # 处理字典类垏的状态
            processed_state = {}
            for key, value in state.items():
                if value is not None:
                    processed_state[key] = value.to(self.device)
                else:
                    processed_state[key] = None
        else:
            # 如果已经是张量，直接使用
            processed_state = state.to(self.device)
            
        # 同样处理next_state
        if isinstance(next_state, dict):
            # 处理字典类垏的next_state
            processed_next_state = {}
            for key, value in next_state.items():
                if value is not None:
                    processed_next_state[key] = value.to(self.device)
                else:
                    processed_next_state[key] = None
        else:
            # 如果已经是张量，直接使用
            processed_next_state = next_state.to(self.device)
        
        # Create transition tuple
        transition = Transition(
            state=processed_state,
            action=action.to(self.device),
            reward=reward.to(self.device),
            next_state=processed_next_state,
            done=done.to(self.device)
        )
        
        # Add to buffer using circular buffer pattern
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, batch_size: int) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Sample random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions as tuple (state, action, reward, next_state, done)
            Note: state and next_state may be dictionaries of tensors if observations are structured
        """
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        
        # 处理可能是字典类垏的状态数据
        if batch and isinstance(batch[0].state, dict):
            # 如果状态是字典类垏，我们需要分别处理每个键
            states = {}
            for key in batch[0].state.keys():
                if batch[0].state[key] is not None:
                    # 如果该组件不是None，则堆叠所有批次中该键的张量
                    try:
                        states[key] = torch.stack([t.state[key] for t in batch if t.state[key] is not None])
                    except:
                        # 如果堆叠失败，可能是因为有些数据形状不一致，跳过该组件
                        print(f"警告: 无法堆叠{key}组件，已跳过")
                        states[key] = None
                else:
                    states[key] = None
        else:
            # 如果状态是张量，直接堆叠
            states = torch.stack([t.state for t in batch])
        
        # 同样处理next_state
        if batch and isinstance(batch[0].next_state, dict):
            next_states = {}
            for key in batch[0].next_state.keys():
                if batch[0].next_state[key] is not None:
                    try:
                        next_states[key] = torch.stack([t.next_state[key] for t in batch if t.next_state[key] is not None])
                    except:
                        print(f"警告: 无法堆叠next_state的{key}组件，已跳过")
                        next_states[key] = None
                else:
                    next_states[key] = None
        else:
            next_states = torch.stack([t.next_state for t in batch])
        
        # 其他元素正常堆叠
        actions = torch.stack([t.action for t in batch])
        rewards = torch.stack([t.reward for t in batch])
        dones = torch.stack([t.done for t in batch])
        
        return states, actions, rewards, next_states, dones
    
    def clear(self) -> None:
        """
        Clear buffer.
        """
        self.buffer = []
        self.position = 0
        

class SumTree:
    """
    SumTree data structure for efficient priority-based sampling.
    Used by PrioritizedReplayBuffer for O(log n) updates and sampling.
    """
    def __init__(self, capacity: int):
        """
        Initialize SumTree with given capacity.
        
        Args:
            capacity: Maximum number of elements in the tree
        """
        self.capacity = capacity  # Number of leaf nodes
        self.tree = np.zeros(2 * capacity - 1)  # Total nodes in binary tree
        self.data = [None] * capacity  # Data stored at leaf nodes
        self.size = 0  # Current size
        self.position = 0  # Current write position (circular)
    
    def __len__(self) -> int:
        """
        Return current size of data (number of items).
        """
        return self.size
    
    def total_priority(self) -> float:
        """
        Return sum of all priorities in the tree (root node value).
        
        Returns:
            Total priority
        """
        return self.tree[0]
    
    def update(self, idx: int, priority: float) -> None:
        """
        Update priority of item at given data index.
        
        Args:
            idx: Index of data in leaf nodes
            priority: New priority value
        """
        # Convert to tree index
        tree_idx = idx + self.capacity - 1
        
        # Update tree
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change up through tree
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def add(self, priority: float, data: Any) -> None:
        """
        Add new item with given priority.
        
        Args:
            priority: Priority value
            data: Data to store
        """
        # Get leaf node index
        idx = self.position
        
        # Update data and size
        self.data[idx] = data
        self.size = min(self.size + 1, self.capacity)
        
        # Update tree with new priority
        self.update(idx, priority)
        
        # Update position for circular buffer
        self.position = (self.position + 1) % self.capacity
    
    def get(self, s: float) -> Tuple[int, int, Any]:
        """
        Find item based on cumulative priority (binary search).
        
        Args:
            s: Cumulative priority to search for (0 <= s <= total_priority)
            
        Returns:
            Tuple of (tree_idx, data_idx, data)
        """
        idx = 0  # Start at root
        
        while idx < self.capacity - 1:  # While not at leaf nodes
            left = 2 * idx + 1
            right = left + 1
            
            # Go left or right based on value
            if s <= self.tree[left] or right >= len(self.tree):
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        
        # Convert to data index
        data_idx = idx - (self.capacity - 1)
        
        return idx, data_idx, self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.
    Implements the framework's "Off-policy 分布式采样 + 优先经验回放 (PER)" feature.
    
    Based on "Prioritized Experience Replay" (Schaul et al., 2015)
    https://arxiv.org/abs/1511.05952
    
    Uses SumTree for efficient priority-based sampling in O(log n) time.
    """
    def __init__(self, buffer_size: int, alpha: float = 0.6, device: torch.device = None):
        """
        Initialize PER buffer.
        
        Args:
            buffer_size: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.alpha = alpha  # Priority exponent
        self.device = device if device is not None else torch.device('cpu')
        self.max_priority = 1.0  # Max priority for new transitions
        
        # Initialize SumTree for efficient priority sampling
        self.tree = SumTree(buffer_size)
    
    def __len__(self) -> int:
        """
        Return current buffer size.
        """
        return len(self.tree)
        
    def push(self, 
             state, 
             action: torch.Tensor, 
             reward: torch.Tensor, 
             next_state, 
             done: torch.Tensor) -> None:
        """
        Store transition in buffer with maximum priority.
        
        Args:
            state: Current state (tensor or dictionary of tensors)
            action: Action tensor
            reward: Reward tensor
            next_state: Next state (tensor or dictionary of tensors)
            done: Done flag tensor
        """
        # 处理状态可能是字典类垏的情况
        if isinstance(state, dict):
            # 处理字典类垏的状态
            processed_state = {}
            for key, value in state.items():
                if value is not None:
                    processed_state[key] = value.to(self.device)
                else:
                    processed_state[key] = None
        else:
            # 如果已经是张量，直接使用
            processed_state = state.to(self.device)
            
        # 同样处理next_state
        if isinstance(next_state, dict):
            # 处理字典类垏的next_state
            processed_next_state = {}
            for key, value in next_state.items():
                if value is not None:
                    processed_next_state[key] = value.to(self.device)
                else:
                    processed_next_state[key] = None
        else:
            # 如果已经是张量，直接使用
            processed_next_state = next_state.to(self.device)
        
        # 处理其他张量
        action_device = action.to(self.device)
        reward_device = reward.to(self.device)
        done_device = done.to(self.device)
        
        # Create transition tuple
        transition = Transition(processed_state, action_device, reward_device, processed_next_state, done_device)
        
        # Calculate priority
        priority = self.max_priority ** self.alpha
        
        # Add to buffer
        self.tree.add(priority, transition)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[Tuple, torch.Tensor, List[int]]:
        """
        Sample batch based on priorities.
        
        Args:
            batch_size: Number of transitions to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            
        Returns:
            Tuple of:
                - Batch of transitions as tuple of tensors (state, action, reward, next_state, done)
                - Importance sampling weights
                - Indices of sampled transitions
        """
        batch_size = min(batch_size, len(self.tree))
        
        # Lists to store sampled items
        batch = []
        indices = []
        priorities = []
        
        # Calculate segment size for even sampling
        segment = self.tree.total_priority() / batch_size
        
        # Sample from each segment
        for i in range(batch_size):
            # Sample uniformly from segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            # Get sample from tree
            tree_idx, data_idx, transition = self.tree.get(s)
            
            # Store results
            batch.append(transition)
            indices.append(data_idx)
            priorities.append(self.tree.tree[tree_idx])
        
        # Calculate importance sampling weights
        # Convert priorities to probabilities
        sampling_probs = np.array(priorities) / self.tree.total_priority()
        
        # Calculate weights
        weights = (len(self.tree) * sampling_probs) ** -beta
        
        # 归一化权重
        weights = weights / weights.max()
        try:
            weights = torch.FloatTensor(weights).to(self.device)
        except Exception as e:
            print(f"警告: 权重转换到设备时出错: {str(e)}，尝试备用方法")
            # 尝试更安全的方法
            weights = torch.tensor(weights, dtype=torch.float32, device='cpu').to(self.device)
        
        # 处理可能是字典类垏的状态数据
        if batch and isinstance(batch[0].state, dict):
            # 如果状态是字典类垏，我们需要分别处理每个键
            states = {}
            for key in batch[0].state.keys():
                if batch[0].state[key] is not None:
                    # 如果该组件不是None，则堆叠所有批次中该键的张量
                    try:
                        states[key] = torch.stack([t.state[key] for t in batch if t.state[key] is not None])
                    except:
                        # 如果堆叠失败，可能是因为有些数据形状不一致，跳过该组件
                        print(f"警告: 无法堆叠{key}组件，已跳过")
                        states[key] = None
                else:
                    states[key] = None
        else:
            # 如果状态是张量，直接堆叠
            states = torch.stack([t.state for t in batch])
        
        # 使用安全堆叠工具处理整个批次
        try:
            # 收集调试信息，快速排查问题
            debug_mode = self.device.type in ['xpu', 'meta', 'hpu'] or len(self.buffer) > 1000
            
            # 尝试直接使用安全处理函数
            states, actions, rewards, next_states, dones = safe_process_batch(
                batch, device=self.device, debug=debug_mode
            )
            
            # 清理可能的内存碎片
            if self.device.type == 'xpu' and hasattr(torch, 'xpu') and hasattr(torch.xpu, 'empty_cache'):
                # 每100次采样执行一次缓存清理
                if hasattr(self, '_sample_count'):
                    self._sample_count += 1
                    if self._sample_count % 100 == 0:
                        torch.xpu.empty_cache()
                else:
                    self._sample_count = 1
            
        except Exception as e:
            # 如果出错，记录错误并尝试更保守的方法
            print(f"警告: 批次处理出错: {str(e)}。尝试备用方法...")
            
            try:
                # 如果出错，尝试将全部数据移动到CPU上处理
                cpu_batch = []
                for t in batch:
                    if isinstance(t.state, dict):
                        cpu_state = {k: (v.cpu() if v is not None else None) for k, v in t.state.items()}
                    else:
                        cpu_state = t.state.cpu()
                        
                    if isinstance(t.next_state, dict):
                        cpu_next_state = {k: (v.cpu() if v is not None else None) for k, v in t.next_state.items()}
                    else:
                        cpu_next_state = t.next_state.cpu()
                        
                    cpu_batch.append(Transition(
                        state=cpu_state,
                        action=t.action.cpu(),
                        reward=t.reward.cpu(),
                        next_state=cpu_next_state,
                        done=t.done.cpu()
                    ))
                    
                # 在CPU上处理后移回原设备
                states, actions, rewards, next_states, dones = safe_process_batch(
                    cpu_batch, device=torch.device('cpu'), debug=True
                )
                # 移回原设备
                if self.device.type != 'cpu':
                    if isinstance(states, dict):
                        states = {k: (v.to(self.device) if v is not None else None) for k, v in states.items()}
                    else:
                        states = states.to(self.device)
                        
                    if isinstance(next_states, dict):
                        next_states = {k: (v.to(self.device) if v is not None else None) for k, v in next_states.items()}
                    else:
                        next_states = next_states.to(self.device)
                        
                    actions = actions.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                
                # 清理临时变量
                del cpu_batch
                gc.collect()
                
            except Exception as final_e:
                print(f"错误: 所有堆叠方法都失败了！{str(final_e)}")
                # 紧急情况下返回空批次
                if self.device.type == 'cpu':
                    empty_tensor = torch.zeros(1, device='cpu')
                else:
                    empty_tensor = torch.zeros(1, device=self.device)
                    
                # 创建空引用返回值
                empty_dict = {}
                return (empty_dict, empty_tensor, empty_tensor, empty_dict, empty_tensor), weights, indices
        
        # 正常返回结果
        return (states, actions, rewards, next_states, dones), weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Update priorities of sampled transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            # Add small constant for stability and apply alpha exponent
            priority = (priority + 1e-5) ** self.alpha
            
            # Update priority in tree
            self.tree.update(idx, priority)
            
            # Update max priority for new transitions
            self.max_priority = max(self.max_priority, priority)
    
    def clear(self) -> None:
        """
        Clear buffer.
        """
        self.tree = SumTree(self.buffer_size)
        self.max_priority = 1.0