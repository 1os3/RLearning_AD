# UAV 自动驾驶强化学习系统

本项目基于PyTorch和AirSim实现了一个完整的无人机自动驾驶强化学习系统，支持高分辨率多帧图像处理、多模态融合、历史记忆、Actor-Critic策略网络、多种辅助任务和XPU加速。

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

## 主要特性

- **高分辨率多帧图像处理**：使用因式化3D Depthwise时空分离卷积高效处理视觉输入
- **多模态融合**：结合图像、姿态和目标信息，通过Transformer自注意力机制进行融合
- **历史记忆**：基于Key-Value Memory实现高效历史信息检索与利用
- **Actor-Critic网络**：基于SAC算法实现，支持连续动作空间和高斯策略
- **多种辅助任务**：实现对比预测编码、帧重建和姿态回归等辅助学习任务
- **优先经验回放**：使用SumTree实现基于TD-error的优先采样
- **域随机化**：随机化光照、风速、场景纹理等环境参数，提升泛化能力
- **断点续训**：完整的检查点管理系统，支持训练状态的保存与恢复
- **推理与评估**：独立的推理与评估模块，支持可视化与性能统计
- **XPU加速**：支持Intel XPU及NVIDIA GPU加速，充分利用硬件性能

## 环境要求

- Python 3.8+
- AirSim 1.8.1+
- PyTorch 2.6.0+ (XPU或CUDA版本)
- Windows 10/11 (支持AirSim)

## 安装指南

1. **安装依赖**

```bash
pip install -r requirements.txt
```

2. **安装AirSim**

按照[AirSim官方文档](https://microsoft.github.io/AirSim/build_windows/)安装并配置AirSim环境。

3. **配置XPU支持**（可选）

参考requirements.txt中的说明使用Intel XPU专用索引安装PyTorch。

## 项目结构

```
├── config/                 # 配置文件目录
│   └── default.yaml        # 默认配置参数
├── src/                    # 源代码目录
│   ├── algorithms/         # 强化学习算法
│   │   ├── auxiliary.py    # 辅助任务实现
│   │   ├── replay_buffer.py # 经验回放缓冲区
│   │   └── sac.py          # SAC算法实现
│   ├── environment/        # 环境适配层
│   │   ├── airsim_env.py   # AirSim环境适配器
│   │   └── reward.py       # 奖励函数实现
│   ├── models/             # 神经网络模型
│   │   ├── input_module/   # 输入处理模块
│   │   ├── fusion_module/  # 多模态融合模块
│   │   └── policy_module/  # 策略与价值网络
│   └── utils/              # 工具函数
│       ├── checkpoint.py   # 检查点管理
│       ├── domain_random.py # 域随机化
│       ├── evaluator.py    # 评估工具
│       ├── logger.py       # 日志记录
│       ├── metrics.py      # 性能指标统计
│       └── visualizer.py   # 可视化工具
├── train.py                # 训练入口脚本
├── inference.py            # 推理入口脚本
└── requirements.txt        # 项目依赖
```

## 使用方法

### 训练模型

```bash
# 使用默认配置进行训练
python train.py

# 使用自定义配置文件
python train.py --config path/to/config.yaml

# 从检查点继续训练
python train.py --resume path/to/checkpoint.pth

# 使用XPU加速训练
python train.py --device xpu
```

### 推理测试

```bash
# 使用训练好的模型进行推理
python inference.py --model path/to/model.pth

# 录制推理视频
python inference.py --model path/to/model.pth --record

# 设置推理回合数
python inference.py --model path/to/model.pth --episodes 10
```

### 评估模型

评估功能集成在train.py和inference.py中：

```bash
# 在训练期间进行定期评估
python train.py --eval_interval 10

# 单独进行评估
python inference.py --model path/to/model.pth --eval
```

## 配置参数

系统的主要配置参数在`config/default.yaml`中定义，以下是关键配置项：

### 环境配置

```yaml
environment:
  name: AirSimUAVEnvironment
  sim_mode: multirotor
  vehicle_name: PX4
  image_shape: [3, 384, 512]  # 图像分辨率 [C, H, W]
  num_frames: 4               # 帧堆叠数量
  max_steps: 1000             # 每回合最大步数
  action_space: continuous    # 动作空间类型
  observation_space: image    # 观察空间类型
```

### 模型配置

```yaml
model:
  frame_processor:
    backbone: resnet18        # 视觉主干网络
    pretrained: true          # 是否使用预训练权重
    use_patch_embedding: true # 是否使用Patch embedding
  attention:
    num_heads: 8              # 注意力头数量
    dim_feedforward: 512      # 前馈网络维度
    dropout: 0.1              # Dropout率
  memory:
    enabled: true             # 是否启用记忆模块
    capacity: 64              # 记忆容量
    key_dim: 256              # 键向量维度
    value_dim: 256            # 值向量维度
```

### 算法配置

```yaml
algorithm:
  name: SAC                   # 算法名称
  gamma: 0.99                 # 折扣因子
  tau: 0.005                  # 目标网络软更新比例
  lr_actor: 3.0e-4            # Actor学习率
  lr_critic: 3.0e-4           # Critic学习率
  alpha_lr: 3.0e-4            # 熵系数学习率
  buffer_size: 100000         # 回放缓冲区大小
  batch_size: 256             # 批次大小
  learning_starts: 1000       # 开始学习前的步数
  target_entropy_scale: 0.98  # 目标熵缩放
  use_automatic_entropy_tuning: true  # 自动熵调整
```

### 辅助任务配置

```yaml
auxiliary_tasks:
  enabled: true               # 是否启用辅助任务
  contrastive_prediction:
    enabled: true             # 对比预测编码
    weight: 0.1               # 损失权重
  frame_reconstruction:
    enabled: true             # 帧重建任务
    weight: 0.05              # 损失权重
  pose_regression:
    enabled: true             # 姿态回归任务
    weight: 0.05              # 损失权重
```

### 奖励配置

```yaml
reward:
  distance_weight: 1.0        # 距离奖励权重
  orientation_weight: 0.5     # 朝向奖励权重
  velocity_weight: 0.3        # 速度奖励权重
  stability_weight: 0.5       # 稳定性奖励权重
  energy_weight: 0.2          # 能耗奖励权重
  progress_weight: 0.8        # 进展奖励权重
  collision_penalty: 10.0     # 碰撞惩罚
  success_reward: 20.0        # 成功奖励
```

### 训练配置

```yaml
training:
  total_timesteps: 1000000    # 总训练步数
  eval_interval: 10000        # 评估间隔
  checkpoint_interval: 50000  # 检查点保存间隔
  max_episode_steps: 1000     # 每回合最大步数
  num_eval_episodes: 5        # 评估回合数
```

## 性能指标

系统提供了全面的性能指标记录与分析功能：

- **训练指标**：回合回报、回合长度、Actor/Critic损失、熵值、Q值等
- **评估指标**：成功率、距离目标、平均速度、能效比、稳定性、碰撞率等
- **可视化**：支持TensorBoard、图表生成和行为可视化
- **导出格式**：CSV、JSON、Pandas DataFrame等多种格式

## 扩展与定制

### 添加新的环境

1. 创建一个继承自 `src/environment/airsim_env.py` 的新环境类
2. 实现 `reset()`, `step()`, `get_observation()` 等必要方法
3. 在配置文件中指定新环境类名

### 添加新的奖励组件

1. 在 `src/environment/reward.py` 中定义新的奖励函数
2. 在 `RewardFunction.compute_reward()` 方法中集成新奖励
3. 在配置文件中添加相应的权重配置

### 添加新的辅助任务

1. 在 `src/algorithms/auxiliary.py` 中定义新的辅助任务网络
2. 在 `AuxiliaryTasks` 类的 `forward()` 方法中集成新任务
3. 在配置文件中添加新任务的启用标志与权重

## 开发者注意事项

- **AirSim连接**：确保AirSim已启动并配置正确，API版本匹配
- **XPU支持**：PyTorch 2.6.0+原生支持XPU，无需IPEX库
- **内存使用**：高分辨率多帧堆叠可能占用大量内存，根据硬件调整配置
- **断点续训**：检查点包含模型、优化器、回放缓冲区和训练状态，支持完整恢复
- **域随机化**：根据训练进度动态调整随机化强度，避免初期过度干扰

## 问题排查

**Q: AirSim连接失败**
A: 确保AirSim正在运行，检查API版本匹配，网络连接无问题

**Q: 训练不稳定**
A: 尝试调整学习率、熵系数，增加batch_size，使用优先经验回放

**Q: 内存消耗过大**
A: 降低图像分辨率，减少帧堆叠数量，调整回放缓冲区大小

**Q: XPU加速无效**
A: 确保已正确安装PyTorch XPU版本，且设备支持XPU加速

## 参考文献

- Soft Actor-Critic: [Off-Policy Maximum Entropy Deep RL with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- Contrastive Predictive Coding: [Representation Learning with CPC](https://arxiv.org/abs/1807.03748)
- AirSim: [AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles](https://arxiv.org/abs/1705.05065)

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件