import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import gc

def safe_stack(tensors: List[torch.Tensor], 
               dim: int = 0, 
               device: Optional[torch.device] = None,
               debug: bool = False) -> torch.Tensor:
    """
    安全地堆叠张量列表，处理设备不一致和XPU特定问题。
    
    Args:
        tensors: 要堆叠的张量列表
        dim: 堆叠的维度
        device: 目标设备，如果为None则使用第一个张量的设备
        debug: 是否打印调试信息
        
    Returns:
        堆叠后的张量
    """
    if not tensors:
        raise ValueError("无法堆叠空列表")
    
    # 检查所有张量是否都是torch.Tensor类型
    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"第{i}个元素不是张量，而是{type(t)}")
    
    # 设备处理逻辑
    if device is None:
        device = tensors[0].device
    
    # 确保所有张量都在同一设备上
    tensors_on_device = []
    for t in tensors:
        if t.device != device:
            tensors_on_device.append(t.to(device))
        else:
            tensors_on_device.append(t)
    
    # 检查所有张量形状是否兼容
    shapes = [t.shape for t in tensors_on_device]
    if debug and len(set(str(s) for s in shapes)) > 1:
        print(f"警告：堆叠的张量形状不一致: {shapes}")
    
    # 尝试按照类别使用不同的方法堆叠
    try:
        # XPU设备特殊处理
        if device.type in ('xpu', 'meta') and hasattr(torch, 'xpu'):
            try:
                # 首先确保原始堆叠是可行的
                result = torch.stack(tensors_on_device, dim=dim)
                return result
            except RuntimeError as e:
                if "UR_RESULT_ERROR_UNKNOWN" in str(e) or "Native API failed" in str(e):
                    # XPU特殊处理：尝试先拷贝到CPU，堆叠后再传回XPU
                    if debug:
                        print("XPU堆叠失败，尝试CPU堆叠后再传回XPU...")
                    cpu_tensors = [t.cpu() for t in tensors_on_device]
                    result = torch.stack(cpu_tensors, dim=dim)
                    # 清理临时张量
                    del cpu_tensors
                    gc.collect()
                    return result.to(device)
                else:
                    raise e
        else:
            # 标准堆叠
            return torch.stack(tensors_on_device, dim=dim)
            
    except Exception as e:
        # 如果出现其他错误，尝试更保守的方法
        try:
            if debug:
                print(f"标准堆叠失败，尝试替代方法... 错误: {str(e)}")
            
            # 获取共同的形状
            if all(len(t.shape) == len(tensors_on_device[0].shape) for t in tensors_on_device):
                # 形状维度一致，可以cat后reshape
                if dim == 0:  # 常见情况处理
                    concat_dim = 0
                    result_shape = (len(tensors_on_device),) + tensors_on_device[0].shape
                    result = torch.cat([t.unsqueeze(0) for t in tensors_on_device], dim=concat_dim)
                    return result
                else:
                    # 其他维度的堆叠，换成先cat再reshape
                    base_shape = list(tensors_on_device[0].shape)
                    target_shape = base_shape.copy()
                    target_shape.insert(dim, len(tensors_on_device))
                    
                    # 调整维度顺序以便cat
                    permute_dims = list(range(len(base_shape) + 1))
                    permute_dims.insert(0, permute_dims.pop(dim))
                    
                    # 执行操作
                    result = torch.cat([t.unsqueeze(dim) for t in tensors_on_device], dim=dim)
                    return result.reshape(target_shape)
            else:
                # 最后尝试
                if debug:
                    print("张量维度不一致，尝试最后的方法...")
                result = torch.stack([t.to(device) for t in tensors], dim=dim)
                return result
                
        except Exception as final_e:
            # 所有方法都失败了，提供详细的错误报告
            shapes_str = ", ".join(str(t.shape) for t in tensors)
            devices_str = ", ".join(str(t.device) for t in tensors)
            dtypes_str = ", ".join(str(t.dtype) for t in tensors)
            
            error_msg = (f"堆叠失败: 原始错误 '{str(e)}', 备选方法错误 '{str(final_e)}'\n"
                         f"形状: [{shapes_str}]\n设备: [{devices_str}]\n类型: [{dtypes_str}]")
            
            if debug:
                print(error_msg)
            
            raise RuntimeError(error_msg) from final_e


def safe_process_batch(batch: List[Any], device: torch.device = None, debug: bool = False) -> Tuple:
    """
    安全处理批次数据，解决XPU/CUDA堆叠问题
    
    Args:
        batch: 经验回放样本批次
        device: 目标设备
        debug: 是否打印调试信息
        
    Returns:
        处理后的元组 (states, actions, rewards, next_states, dones)
    """
    if not batch:
        raise ValueError("批次不能为空")
    
    # 当device为None时，尝试从样本中获取设备信息
    if device is None:
        if hasattr(batch[0].action, 'device'):
            device = batch[0].action.device
        else:
            device = torch.device('cpu')  # 默认设备
    
    # 处理状态可能是字典的情况
    if hasattr(batch[0], 'state') and isinstance(batch[0].state, dict):
        states = {}
        for key in batch[0].state.keys():
            if batch[0].state[key] is not None:
                try:
                    # 使用安全堆叠函数
                    valid_tensors = [t.state[key] for t in batch if t.state[key] is not None]
                    if valid_tensors:
                        states[key] = safe_stack(valid_tensors, dim=0, device=device, debug=debug)
                    else:
                        states[key] = None
                except Exception as e:
                    if debug:
                        print(f"警告: 无法堆叠state的{key}组件, 错误: {str(e)}")
                    states[key] = None
            else:
                states[key] = None
    else:
        # 如果状态是张量，直接堆叠
        states = safe_stack([t.state for t in batch], dim=0, device=device, debug=debug)
    
    # 同样处理next_state
    if hasattr(batch[0], 'next_state') and isinstance(batch[0].next_state, dict):
        next_states = {}
        for key in batch[0].next_state.keys():
            if batch[0].next_state[key] is not None:
                try:
                    valid_tensors = [t.next_state[key] for t in batch if t.next_state[key] is not None]
                    if valid_tensors:
                        next_states[key] = safe_stack(valid_tensors, dim=0, device=device, debug=debug)
                    else:
                        next_states[key] = None
                except Exception as e:
                    if debug:
                        print(f"警告: 无法堆叠next_state的{key}组件, 错误: {str(e)}")
                    next_states[key] = None
            else:
                next_states[key] = None
    else:
        # 如果next_state是张量，直接堆叠
        next_states = safe_stack([t.next_state for t in batch], dim=0, device=device, debug=debug)
    
    # 处理其他元素
    try:
        actions = safe_stack([t.action for t in batch], dim=0, device=device, debug=debug)
    except Exception as e:
        if debug:
            print(f"动作堆叠出错: {str(e)}，尝试逐个移动到目标设备并重试")
        # 尝试解决设备不一致问题
        actions = safe_stack([t.action.to(device) for t in batch], dim=0, device=device, debug=debug)
    
    try:
        rewards = safe_stack([t.reward for t in batch], dim=0, device=device, debug=debug)
    except Exception as e:
        if debug:
            print(f"奖励堆叠出错: {str(e)}，尝试逐个移动到目标设备并重试")
        rewards = safe_stack([t.reward.to(device) for t in batch], dim=0, device=device, debug=debug)
    
    try:
        dones = safe_stack([t.done for t in batch], dim=0, device=device, debug=debug)
    except Exception as e:
        if debug:
            print(f"完成标志堆叠出错: {str(e)}，尝试逐个移动到目标设备并重试")
        dones = safe_stack([t.done.to(device) for t in batch], dim=0, device=device, debug=debug)
    
    return states, actions, rewards, next_states, dones
