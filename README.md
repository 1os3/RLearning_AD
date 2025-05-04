# UAV 自动驾驶强化学习系统

本项目基于PyTorch和AirSim实现了一个完整的无人机自动驾驶强化学习系统，支持多模态输入处理、复合观测空间、SAC强化学习算法、优先经验回放、XPU加速及高级断点续训功能。

## 系统架构

```
┌───────────────────────────────────────────────┐  
│  Input Module                                 │  
│ ├─ 多帧图像处理（高分辨率RGB N帧）            │  
│ ├─ 姿态/速度/加速度/角度编码器               │  
│ └─ 目标点（输入距离目标点距离和目标点相对方向）│  
└───────────────┬───────────────────────────────┘  
                │  
┌───────────────▼───────────────────────────────┐  
│  多模态融合 & 历史记忆模块                     │  
│ ├─ 3D-Depthwise 时空分离卷积块                 │  
│ ├─ Cross-modal Transformer 自注意力块         │  
│ ├─ 历史记忆库（Key-Value Memory + 注意力检索）│  
│ └─ 动态 Token Pruning / 量化推理               │  
└───────────────┬───────────────────────────────┘  
                │  
┌───────────────▼──────────┐   ┌───────────────┐  
│  Actor Head (策略网络)     │   │  Critic Head  │  
│  └─ MLP → 连续动作输出    │   │  └─ MLP → Q值 │  
└──────────────────────────┘   └───────────────┘  
```

## 已实现的主要特性

- **多模态输入处理**：支持同时处理图像、状态向量和目标信息
- **复合观测空间**：支持字典类型的观测空间，灵活组合不同模态的输入
- **SAC强化学习算法**：基于Soft Actor-Critic实现，支持连续动作空间
- **优先经验回放**：使用SumTree实现基于TD-error优先级的经验回放
- **XPU/GPU加速**：支持Intel XPU及GPU加速，自动设备选择
- **断点续训**：支持训练状态、模型参数、回放缓冲区的完整保存与恢复
- **资源监控**：实时监控CPU、内存及XPU/GPU使用情况，并记录到TensorBoard
- **TensorBoard可视化**：强制启用TensorBoard记录，支持训练指标和资源使用的可视化
- **详细训练输出**：每步迭代提供详细的进度和损失信息
- **配置继承与多路径访问**：健壮的配置系统，支持配置继承和多路径访问

## 环境要求

- Python 3.8+
- AirSim 1.8.1+
- PyTorch 2.6.0+ (原生支持XPU，无需IPEX)
- Windows 10/11 (支持AirSim)
- Gymnasium 0.28.0+ (使用新版API)
- TensorBoard 2.15.0+ (用于训练可视化)

## 安装指南

1. **创建并激活虚拟环境**

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **安装AirSim**

按照[AirSim官方文档](https://microsoft.github.io/AirSim/build_windows/)安装并配置AirSim环境。

4. **验证XPU支持**（可选）

```bash
python check_xpu.py
```

## 使用方法

### 训练模型

最简单的训练方式是使用提供的批处理脚本：

```bash
# 使用默认配置进行训练
.\run_train.bat
```

或者直接运行Python脚本：

```bash
# 使用默认配置进行训练
python train.py

# 使用自定义配置文件
python train.py --config config/custom.yaml

# 从检查点继续训练
python train.py --checkpoint path/to/checkpoint

# 指定设备
python train.py --device xpu  # 使用Intel XPU
python train.py --device cuda  # 使用NVIDIA GPU
python train.py --device cpu  # 使用CPU
```

### 查看训练日志

训练过程中可以使用TensorBoard查看实时训练曲线和资源使用情况：

```bash
# 启动TensorBoard
.\.venv\Scripts\tensorboard --logdir logs/training
```

然后在浏览器中访问 http://localhost:6006 查看训练进度。

## 开发者注意事项

- **多模态观测处理**：系统支持字典类型的观测空间，各组件（Actor、Critic、ReplayBuffer等）已适配
- **XPU支持**：PyTorch 2.6.0+原生支持XPU，使用自动设备选择逻辑选择最佳可用设备
- **AirSim连接**：确保AirSim已启动并配置正确，检查连接端口和超时设置
- **TensorBoard集成**：系统强制启用TensorBoard记录，包括训练指标和资源使用情况
- **配置健壮性**：所有配置项均支持多路径访问，可在不同层级定义相同配置以增强兼容性

## 问题排查

**Q: 训练脚本使用CPU而不是XPU**  
**A:** 检查设备选择逻辑，确保`config/training.yaml`中的`general.device`设置为"xpu"，运行`check_xpu.py`验证XPU可用

**Q: AirSim连接失败**  
**A:** 确保AirSim正在运行，检查连接端口和IP地址，增加连接超时时间

**Q: 字典观测空间处理错误**  
**A:** 检查所有涉及观测处理的代码，确保正确处理字典类型的状态数据

**Q: 优先经验回放类型错误**  
**A:** 确保PrioritizedReplayBuffer.push和sample方法正确处理字典类型的状态

**Q: TensorBoard未显示数据**  
**A:** 检查日志路径，确认`logs/training`目录下有事件文件，重启TensorBoard服务
