import torch
import sys

print(f"PyTorch版本: {torch.__version__}")
print(f"Python版本: {sys.version}")

# 检查是否有XPU属性
has_xpu_attr = hasattr(torch, 'xpu')
print(f"torch是否有xpu属性: {has_xpu_attr}")

# 如果有xpu属性，检查是否可用
if has_xpu_attr:
    try:
        print(f"XPU是否可用: {torch.xpu.is_available()}")
        print(f"XPU设备数量: {torch.xpu.device_count() if torch.xpu.is_available() else 0}")
        if torch.xpu.is_available():
            print(f"XPU设备名称: {torch.xpu.get_device_name(0)}")
    except Exception as e:
        print(f"检查XPU时出错: {e}")
else:
    print("PyTorch未编译XPU支持")

# 检查CUDA是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")

# 列出所有可用设备
print("\n可用的设备:")
available_devices = ["cpu"]
if has_xpu_attr and torch.xpu.is_available():
    available_devices.append("xpu")
if torch.cuda.is_available():
    available_devices.append("cuda")
print(available_devices)

# 检查Intel OneAPI环境变量
import os
print("\nIntel XPU相关环境变量:")
xpu_env_vars = [var for var in os.environ if "XPU" in var or "SYCL" in var or "ONEAPI" in var or "INTEL" in var]
for var in xpu_env_vars:
    print(f"{var}: {os.environ.get(var)}")
