import os
import time
import math
import numpy as np
import gym
from gym import spaces
import airsim
import yaml
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any

class AirSimDroneEnv(gym.Env):
    """
    AirSim Drone Environment for Reinforcement Learning.
    This environment interfaces with AirSim simulator for drone control.
    It handles:
    - Multi-frame image observations
    - Pose/velocity/acceleration information
    - Target-based navigation
    - Proper reward calculation
    - Termination conditions
    """
    
    def __init__(self, config=None, evaluation=False):
        """
        Initialize AirSim drone environment.
        
        Args:
            config: Path to configuration YAML file or configuration dictionary
            evaluation: Whether this environment is used for evaluation
        """
        # 存储一些已知的障碍物区域位置，避免生成目标点在障碍物附近
        # 格式：[(min_x, min_y, min_z, max_x, max_y, max_z)]
        self.obstacle_regions = []
        
        # Load config
        if config is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                       "config", "default.yaml")
            self.config = self._load_config(config_path)
        elif isinstance(config, str):
            # If config is a string, treat it as a path to YAML file
            self.config = self._load_config(config)
        elif isinstance(config, dict):
            # If config is a dictionary, use it directly
            self.config = config
        else:
            raise TypeError(f"Expected config to be str, dict or None, got {type(config)}")
            
        # Flag for evaluation mode
        self.evaluation = evaluation
        
        # 安全访问环境配置
        env_config = self.config.get('environment', {})
        image_config = env_config.get('image', {})
        action_config = env_config.get('action_space', {})
        term_config = env_config.get('termination', {})
        
        # 图像设置 - 必须在初始化AirSim客户端之前设置，因为客户端初始化依赖这些属性
        self.frame_stack = image_config.get('frame_stack', 4)
        self.use_rgb = image_config.get('use_rgb', True)
        self.use_depth = image_config.get('use_depth', False)
        self.keep_original_resolution = image_config.get('keep_original_resolution', True)
        
        # 初始化图像请求
        self.image_requests = []
        
        # 初始化AirSim客户端
        self.client = self._setup_client()
        
        # 动作空间设置
        self.continuous = action_config.get('continuous', True)
        self.action_dim = action_config.get('action_dim', 4)
        self.max_velocity = np.array(action_config.get('max_velocity', [5.0, 5.0, 5.0]))
        self.max_yaw_rate = action_config.get('max_yaw_rate', 1.0)
        
        # 终止条件
        self.max_steps = term_config.get('max_steps', 1000)
        self.success_distance = term_config.get('success_distance_threshold', 2.0)
        
        # 使用普通字符串，避免f-string中的Unicode转义序列
        print("环境配置总结:")
        print(f"- 图像帧堆叠: {self.frame_stack}")
        print(f"- 动作空间: {'连续' if self.continuous else '离散'}, 维度: {self.action_dim}")
        print(f"- 最大速度: {self.max_velocity}, 最大偏航角速率: {self.max_yaw_rate}")
        print(f"- 最大步数: {self.max_steps}, 成功距离阈值: {self.success_distance}m")
        
        # Frame buffer for stacking
        self.frame_buffer = None
        
        # Target 
        self.target_position = None
        
        # State tracking
        self.step_count = 0
        self.collision_info = None
        self.done = False
        
        # Define action and observation spaces
        self._setup_spaces()
        
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def _setup_client(self) -> airsim.MultirotorClient:
        """
        Set up the AirSim client with proper configuration.
        
        Returns:
            Configured AirSim client
        """
        # 安全地访问配置
        env_config = self.config.get('environment', {})
        airsim_config = env_config.get('airsim', {})
        
        # 获取连接参数，提供默认值
        ip = airsim_config.get('ip', '127.0.0.1')
        port = airsim_config.get('port', 41451)
        timeout_ms = airsim_config.get('timeout_ms', 10000)
        
        print(f"连接到 AirSim: {ip}:{port}，超时={timeout_ms/1000}秒")
        
        try:
            # Connect to AirSim（针对Windows环境优化）
            client = airsim.MultirotorClient(
                ip=ip,
                port=port,
                timeout_value=timeout_ms / 1000
            )
            
            # 确认连接
            client.confirmConnection()
            client.enableApiControl(True)
            client.armDisarm(True)
            
            # 重置到初始状态
            client.reset()
            print("已成功连接并初始化 AirSim 客户端")
        except Exception as e:
            print(f"AirSim 连接错误: {str(e)}")
            print("请确保 AirSim 模拟器已运行并正确配置")
            raise
        
        # 配置图像请求
        # 注意: self.image_requests 已在构造函数中初始化为空列表
        if self.use_rgb:
            self.image_requests.append(airsim.ImageRequest(
                "0", airsim.ImageType.Scene, False, False
            ))
        if self.use_depth:
            self.image_requests.append(airsim.ImageRequest(
                "0", airsim.ImageType.DepthPlanar, True, False
            ))
            
        # 打印图像请求配置
        if len(self.image_requests) > 0:
            print(f"已配置 {len(self.image_requests)} 个图像请求: "
                  f"RGB={self.use_rgb}, Depth={self.use_depth}")
        else:
            print("警告: 未配置任何图像请求, 请检查配置")
        
        return client
    
    def _setup_spaces(self):
        """
        Define action and observation spaces based on config.
        """
        # Action space (continuous or discrete)
        if self.continuous:
            # Continuous actions: [vx, vy, vz, yaw_rate]
            # Each normalized between -1 and 1, will be scaled by max values
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
            )
        else:
            raise NotImplementedError("Discrete action space not implemented yet")
        
        # Observation space - will be a Dict space with multiple components
        observation_spaces = {}
        
        # Image space - We don't know exact dimensions yet since we need to get an image first
        # Will be updated in reset()
        observation_spaces['image'] = spaces.Box(
            low=0, high=255, shape=(84, 84, 3 * self.frame_stack), dtype=np.uint8
        )
        
        # State space - position, velocity, acceleration, orientation
        state_dim = 0
        
        # 安全访问状态空间配置
        state_space_config = self.config.get('environment', {}).get('state_space', {})
        
        # 安全访问各个状态标志，默认为False
        if state_space_config.get('use_position', True):
            state_dim += 3  # x, y, z
        if state_space_config.get('use_velocity', True):
            state_dim += 3  # vx, vy, vz
        if state_space_config.get('use_acceleration', False):
            state_dim += 3  # ax, ay, az
        if state_space_config.get('use_orientation', True):
            state_dim += 3  # roll, pitch, yaw
        if state_space_config.get('use_angular_velocity', False):
            state_dim += 3  # angular velocities
            
        # 确保状态空间至少有一个维度
        state_dim = max(state_dim, 1)
            
        observation_spaces['state'] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Target space - relative distance and direction to target
        observation_spaces['target'] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        
        # Combine all spaces into a Dict space
        self.observation_space = spaces.Dict(observation_spaces)
    
    def _is_in_obstacle_region(self, position: np.ndarray) -> bool:
        """
        检查位置是否在障碍物区域内。
        
        Args:
            position: 要检查的位置坐标
            
        Returns:
            如果在障碍物区域内则返回true
        """
        for obstacle in self.obstacle_regions:
            min_x, min_y, min_z, max_x, max_y, max_z = obstacle
            if (min_x <= position[0] <= max_x and 
                min_y <= position[1] <= max_y and 
                min_z <= position[2] <= max_z):
                return True
        return False
    
    def _check_obstacle_nearby(self, position: np.ndarray, radius: float = 5.0) -> bool:
        """
        检测指定位置附近是否有障碍物。
        使用以下方法：
        1. 首先检查静态障碍物区域
        2. 尝试使用距离探测（如果AirSim支持）
        3. 根据上次碰撞位置评估安全性
        
        Args:
            position: 要检查的位置
            radius: 检测半径，默认5m
            
        Returns:
            如果有障碍物在位置附近返回true
        """
        # 1. 首先检查静态障碍物区域
        if self._is_in_obstacle_region(position):
            return True
            
        # 2. 检查与已知碰撞位置的距离
        try:
            # 如果我们有之前的碰撞信息，使用它来评估安全性
            if hasattr(self, 'last_collision_position') and self.last_collision_position is not None:
                collision_pos = np.array(self.last_collision_position)
                dist_to_collision = np.linalg.norm(position - collision_pos)
                if dist_to_collision < radius * 1.5:  # 使用更大的安全边界
                    return True
                    
            # 3. 检查当前位置是否安全（不在碰撞状态）
            # 获取当前碰撞信息
            collision_info = self.client.simGetCollisionInfo()
            if collision_info.has_collided:
                # 记录这个位置作为不安全区域
                collision_pos = collision_info.position
                self.last_collision_position = [collision_pos.x_val, collision_pos.y_val, collision_pos.z_val]
                # 当前有碰撞，这个区域不安全
                return True
                
            # 4. 使用LiDAR API尝试检测障碍物（如果可用）
            # 注意：这可能在某些AirSim版本中不可用
            try:
                lidar_data = self.client.getLidarData()
                if len(lidar_data.point_cloud) > 3:
                    # 点云数据是一个1D数组，每3个值代表一个点的(x,y,z)
                    points = np.array(lidar_data.point_cloud).reshape(-1, 3)
                    
                    # 将位置转换为AirSim坐标系
                    drone_pos = np.array([lidar_data.pose.position.x_val, 
                                      lidar_data.pose.position.y_val, 
                                      lidar_data.pose.position.z_val])
                    
                    # 计算每个点到目标位置的距离
                    dist_to_target = np.linalg.norm(points - position, axis=1)
                    
                    # 如果任何一个点距离目标太近，认为有障碍物
                    if np.any(dist_to_target < radius):
                        return True
            except Exception as lidar_error:
                # LiDAR API可能不可用，忽略错误
                pass
            
            # 没有检测到障碍物
            return False
            
        except Exception as e:
            # 如果任何探测失败，打印警告，保守地假设区域安全
            print(f"警告: 执行障碍物检测时出错: {e}")
            return False  # 假设安全，继续执行
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, Any], Dict]:
        """
        Reset environment state.
        
        Args:
            seed: Random seed for reproducibility
            options: Optional configuration dictionary
            
        Returns:
            Initial observation and info dictionary
        """
        # 在重置时清除初始高度记录，确保每次重置均使用新的初始高度
        if hasattr(self, '_initial_height'):
            delattr(self, '_initial_height')
            
        super().reset(seed=seed)  # Reset parent (Gymnasium env)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset AirSim
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Wait for drone to stabilize
        self.client.takeoffAsync().join()
        time.sleep(1.0)
        
        # Reset internal state
        self.step_count = 0
        self.collision_info = None
        self.done = False
        
        # 从配置中获取目标点采样参数
        target_sampling_config = {}
        
        # 尝试不同的配置路径
        if 'environment' in self.config and 'target_sampling' in self.config['environment']:
            target_sampling_config = self.config['environment']['target_sampling']
        elif 'target_sampling' in self.config:
            target_sampling_config = self.config['target_sampling']
        
        # 安全获取参数，提供默认值
        xy_min = target_sampling_config.get('xy_min', -50.0)
        xy_max = target_sampling_config.get('xy_max', 50.0)
        z_min = target_sampling_config.get('z_min', -20.0)  # AirSim中负值表示向上
        z_max = target_sampling_config.get('z_max', -50.0)  # 高空目标
        min_distance = target_sampling_config.get('min_distance', 20.0)
        max_distance = target_sampling_config.get('max_distance', 80.0)
        
        # 获取当前位置
        drone_state = self.client.getMultirotorState()
        current_position = drone_state.kinematics_estimated.position
        initial_position = np.array([current_position.x_val, current_position.y_val, current_position.z_val])
        
        # 生成目标点，确保与初始点保持合适距离
        max_attempts = 50  # 防止无限循环
        attempt = 0
        valid_target = False
        
        while not valid_target and attempt < max_attempts:
            # 生成XY平面坐标
            x = np.random.uniform(xy_min, xy_max)
            y = np.random.uniform(xy_min, xy_max)
            
            # 生成Z坐标（高度）
            # 注意：AirSim中负值表示向上，所以z_min通常大于z_max
            z = np.random.uniform(min(z_min, z_max), max(z_min, z_max))
            
            # 拼接目标点
            target_pos = np.array([x, y, z])
            
            # 计算与初始点的距离
            distance_to_initial = np.linalg.norm(target_pos - initial_position)
            
            # 验证距离是否在合适范围内，并检查是否距离障碍物足够远
            if min_distance <= distance_to_initial <= max_distance:
                # 检查目标点附近是否有障碍物
                obstacle_nearby = self._check_obstacle_nearby(target_pos, radius=6.0)  # 使用6m的安全距离，更保守
                
                if not obstacle_nearby:
                    valid_target = True
                    self.target_position = target_pos
                else:
                    print(f"  目标点 [{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}] 附近存在障碍物，重新采样")
            
            attempt += 1
        
        # 如果多次尝试后仍找不到合适目标，使用备选方案
        if not valid_target:
            print("警告: 无法在合适范围内生成目标点，使用备选方案")
            
            # 尝试不同的方向，找到一个无障碍物的方向
            for _ in range(8):  # 尝试8个不同方向
                # 生成随机方向
                direction = np.random.uniform(-1, 1, size=3)
                direction = direction / np.linalg.norm(direction)
                test_target = initial_position + direction * min_distance * 1.5
                # 确保高度在合理范围内
                test_target[2] = np.clip(test_target[2], min(z_min, z_max), max(z_min, z_max))
                
                # 检查这个方向是否距障碍物足够远
                if not self._check_obstacle_nearby(test_target, radius=4.0):
                    self.target_position = test_target
                    valid_target = True
                    print(f"  找到合适的无障碍物方向，使用备选目标点")
                    break
            
            # 如果仍然找不到无障碍物的方向，使用高空目标
            if not valid_target:
                print(f"  所有方向都有障碍物，使用更高高度的目标点")
                # 生成一个基本目标点，但设置更高的高度
                direction = np.random.uniform(-1, 1, size=2)  # 只生成XY平面方向
                direction = np.append(direction, 0)  # 添加Z=0
                direction = direction / np.linalg.norm(direction)
                self.target_position = initial_position + direction * min_distance * 1.2
                # 更高的高度，超出正常范围但仍如正常可达
                higher_altitude = min(z_min, z_max) * 1.5  # 更高的高度，注意负值表示向上
                self.target_position[2] = higher_altitude
        
        # 打印目标信息
        distance_to_target = np.linalg.norm(self.target_position - initial_position)
        print(f"生成目标点: [{self.target_position[0]:.1f}, {self.target_position[1]:.1f}, {self.target_position[2]:.1f}], "
              f"与起始点距离: {distance_to_target:.1f}m")
        
        # 转换目标位置为标准Python类型，避免AirSim接口调用时出现序列化问题
        self.target_position = [float(x) for x in self.target_position]
        
        # Get initial observation
        observation = self._get_observation()
        
        # Update observation space image shape if necessary based on actual image dimensions
        if isinstance(observation['image'], np.ndarray):
            image_shape = observation['image'].shape
            if self.observation_space.spaces['image'].shape != image_shape:
                self.observation_space.spaces['image'] = spaces.Box(
                    low=0, high=255, shape=image_shape, dtype=np.uint8
                )
        
        return observation, {}
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute [vx, vy, vz, yaw_rate] normalized between -1 and 1
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        
        # Scale normalized actions to actual values
        scaled_action = self._scale_action(action)
        
        # Execute action in AirSim
        self._execute_action(scaled_action)
        
        # Check for collision
        self.collision_info = self.client.simGetCollisionInfo()
        collision = self.collision_info.has_collided
        
        # Get current state
        observation = self._get_observation()
        
        # Calculate reward
        reward, reward_info = self._compute_reward(scaled_action, collision)
        
        # Check termination conditions
        terminated, truncated, info = self._check_termination(collision)
        
        # Add reward info to info dict
        info.update(reward_info)
        
        # 添加分解的奖励组件到info中，更方便日志记录
        info['reward'] = reward
        info['reward_components'] = reward_info
        
        return observation, reward, terminated, truncated, info
    
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Scale normalized actions to actual values.
        
        Args:
            action: Normalized action array
            
        Returns:
            Scaled action array (as standard Python float types)
        """
        # Clip action to ensure it's within bounds
        action = np.clip(action, -1.0, 1.0)
        
        # Scale to actual values
        scaled_action = np.zeros_like(action)
        scaled_action[:3] = action[:3] * self.max_velocity
        scaled_action[3] = action[3] * self.max_yaw_rate
        
        # 重要：返回标准Python浮点类型数组，避免NumPy类型与AirSim序列化不兼容
        return np.array([float(x) for x in scaled_action], dtype=np.float32)
    
    def _execute_action(self, action: np.ndarray):
        """
        Execute action in AirSim.
        
        Args:
            action: Scaled action array [vx, vy, vz, yaw_rate]
        """
        vx, vy, vz, yaw_rate = action
        
        # Convert to AirSim's coordinate system if necessary
        # Note: AirSim uses NED (North-East-Down) coordinates
        # 重要：转换为标准Python类型，避免NumPy类型导致的序列化问题
        vx_ned = float(vx)
        vy_ned = float(vy)
        vz_ned = float(vz)
        yaw_rate_float = float(yaw_rate)
        
        # Send velocity command to AirSim
        try:
            self.client.moveByVelocityAsync(
                vx_ned, vy_ned, vz_ned, 
                duration=0.5,  # Control for 0.5 second (adjust as needed)
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_float)
            )
        except Exception as e:
            print(f"AirSim控制命令错误: {str(e)}")
            print(f"vx={vx_ned}({type(vx_ned)}), vy={vy_ned}({type(vy_ned)}), vz={vz_ned}({type(vz_ned)}), yaw_rate={yaw_rate_float}({type(yaw_rate_float)})")
        
        # Wait a small amount of time to simulate control frequency
        time.sleep(0.05)  # 20Hz control rate
    
    def _get_observation(self) -> Dict:
        """
        Get current observation (images, state, target).
        
        Returns:
            Dict containing observation components
        """
        observation = {}
        
        # Get images
        if self.image_requests:
            observation['image'] = self._get_image_observation()
        
        # Get state
        observation['state'] = self._get_state_observation()
        
        # Get target
        observation['target'] = self._get_target_observation()
        
        return observation
    
    def _get_image_observation(self) -> np.ndarray:
        """
        Get image observation and handle frame stacking.
        
        Returns:
            Stacked image observation
        """
        # Get images from AirSim
        responses = self.client.simGetImages(self.image_requests)
        
        # Process images
        current_frame = None
        
        for i, response in enumerate(responses):
            if response.pixels_as_float:
                # Convert depth image
                img1d = np.array(response.image_data_float, dtype=np.float32)
                depth = img1d.reshape(response.height, response.width)
                depth = np.array(depth * 255, dtype=np.uint8)  # Normalize to 0-255
                if current_frame is None:
                    current_frame = np.zeros((response.height, response.width, 3), dtype=np.uint8)
                # Use depth for all channels
                current_frame[:, :, 0] = depth
                current_frame[:, :, 1] = depth
                current_frame[:, :, 2] = depth
            else:
                # Convert RGB image
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                current_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        # Handle frame stacking
        if self.frame_buffer is None:
            # Initialize frame buffer with copies of the first frame
            self.frame_buffer = np.zeros(
                (current_frame.shape[0], current_frame.shape[1], current_frame.shape[2] * self.frame_stack),
                dtype=np.uint8
            )
            for i in range(self.frame_stack):
                start_idx = i * current_frame.shape[2]
                end_idx = (i + 1) * current_frame.shape[2]
                self.frame_buffer[:, :, start_idx:end_idx] = current_frame
        else:
            # Shift frames (remove oldest, add newest)
            for i in range(self.frame_stack - 1):
                start_dest = i * current_frame.shape[2]
                end_dest = (i + 1) * current_frame.shape[2]
                start_src = (i + 1) * current_frame.shape[2]
                end_src = (i + 2) * current_frame.shape[2]
                
                # Handle last frame case
                if end_src <= self.frame_buffer.shape[2]:
                    self.frame_buffer[:, :, start_dest:end_dest] = self.frame_buffer[:, :, start_src:end_src]
            
            # Add new frame
            start_idx = (self.frame_stack - 1) * current_frame.shape[2]
            end_idx = self.frame_stack * current_frame.shape[2]
            self.frame_buffer[:, :, start_idx:end_idx] = current_frame
        
        return self.frame_buffer
    
    def _get_state_observation(self) -> np.ndarray:
        """
        Get state observation (position, velocity, acceleration, orientation).
        
        Returns:
            State observation array
        """
        # Get drone state
        drone_state = self.client.getMultirotorState()
        
        # Initialize state vector
        state = []
        
        # 安全获取状态空间配置
        state_space_config = {}
        
        # 首先尝试不同的配置路径
        if 'environment' in self.config and 'state_space' in self.config['environment']:
            # 原始的配置路径
            state_space_config = self.config['environment']['state_space']
        elif 'state_space' in self.config:
            # 直接在配置根目录下
            state_space_config = self.config['state_space']
        else:
            # 没有找到配置，使用默认值
            print("警告: 未找到状态空间配置，使用默认值")
            state_space_config = {
                'use_position': True,
                'use_velocity': True,
                'use_acceleration': False,
                'use_orientation': True,
                'use_angular_velocity': False
            }
        
        # 添加位置（如果配置）
        if state_space_config.get('use_position', True):
            pos = drone_state.kinematics_estimated.position
            state.extend([pos.x_val, pos.y_val, pos.z_val])
        
        # 添加速度（如果配置）
        if state_space_config.get('use_velocity', True):
            vel = drone_state.kinematics_estimated.linear_velocity
            state.extend([vel.x_val, vel.y_val, vel.z_val])
        
        # 添加加速度（如果配置）
        if state_space_config.get('use_acceleration', False):
            acc = drone_state.kinematics_estimated.linear_acceleration
            state.extend([acc.x_val, acc.y_val, acc.z_val])
        
        # 添加方向（如果配置）
        if state_space_config.get('use_orientation', True):
            ori = drone_state.kinematics_estimated.orientation
            # Convert quaternion to euler angles (roll, pitch, yaw)
            roll, pitch, yaw = airsim.to_eularian_angles(ori)
            state.extend([roll, pitch, yaw])
        
        # 添加角速度（如果配置）
        if state_space_config.get('use_angular_velocity', False):
            ang_vel = drone_state.kinematics_estimated.angular_velocity
            state.extend([ang_vel.x_val, ang_vel.y_val, ang_vel.z_val])
        
        return np.array(state, dtype=np.float32)
    
    def _get_target_observation(self) -> np.ndarray:
        """
        Get target observation (distance and direction to target).
        
        Returns:
            Target observation array
        """
        # Get current position
        drone_state = self.client.getMultirotorState()
        drone_position = drone_state.kinematics_estimated.position
        current_position = np.array([drone_position.x_val, drone_position.y_val, drone_position.z_val])
        
        # 转换为标准Python类型，避免序列化问题
        current_position_list = [float(x) for x in current_position]
        target_position_list = self.target_position
        if isinstance(self.target_position, np.ndarray):
            target_position_list = [float(x) for x in self.target_position]
        
        # Calculate vector to target
        vector_to_target = np.array([
            target_position_list[0] - current_position_list[0],
            target_position_list[1] - current_position_list[1],
            target_position_list[2] - current_position_list[2]
        ])
        
        # Calculate distance to target
        distance_to_target = float(np.linalg.norm(vector_to_target))
        
        # Get drone orientation
        drone_orientation = drone_state.kinematics_estimated.orientation
        _, _, yaw = airsim.to_eularian_angles(drone_orientation)
        
        # Calculate relative direction to target
        # We need to transform the vector to drone's local coordinate frame
        # First calculate the heading vector of the drone (in world frame)
        heading_x = math.cos(yaw)
        heading_y = math.sin(yaw)
        
        # Calculate angle between drone's heading and target vector (in x-y plane)
        target_angle_xy = math.atan2(vector_to_target[1], vector_to_target[0]) - yaw
        # Normalize to [-pi, pi]
        target_angle_xy = (target_angle_xy + math.pi) % (2 * math.pi) - math.pi
        
        # Calculate vertical angle to target
        horizontal_distance = math.sqrt(vector_to_target[0]**2 + vector_to_target[1]**2)
        target_angle_z = math.atan2(vector_to_target[2], horizontal_distance)
        
        # 确保返回标准Python浮点类型的数组，但同时也保持NumPy类型供内部使用
        return_array = np.array([float(distance_to_target), float(target_angle_xy), float(target_angle_z)], dtype=np.float32)
        return return_array
    
    def _compute_reward(self, action: np.ndarray, collision: bool) -> Tuple[float, Dict]:
        """
        Compute reward based on current state, action, and target.
        
        Args:
            action: Scaled action that was executed
            collision: Whether a collision occurred
            
        Returns:
            Total reward and dictionary of reward components
        """
        # Get current state
        drone_state = self.client.getMultirotorState()
        position = drone_state.kinematics_estimated.position
        velocity = drone_state.kinematics_estimated.linear_velocity
        orientation = drone_state.kinematics_estimated.orientation
        
        current_position = np.array([position.x_val, position.y_val, position.z_val])
        
        # Calculate distance to target
        vector_to_target = self.target_position - current_position
        distance_to_target = np.linalg.norm(vector_to_target)
        
        # Get previous distance if available
        previous_distance = getattr(self, '_previous_distance', None)
        if previous_distance is None:
            previous_distance = distance_to_target
        
        # Calculate progress towards target
        distance_improvement = previous_distance - distance_to_target
        # Store current distance for next step
        self._previous_distance = distance_to_target
        
        # Check if reached target
        reached_target = distance_to_target < self.success_distance
        
        # 安全获取奖励配置
        reward_config = {}
        
        # 尝试不同的配置路径
        if 'environment' in self.config and 'reward' in self.config['environment']:
            # 原始的配置路径
            reward_config = self.config['environment']['reward']
        elif 'reward' in self.config:
            # 直接在配置根目录下
            reward_config = self.config['reward']
        else:
            # 没有找到配置，使用默认值
            print("警告: 未找到奖励配置，使用默认值")
        
        # 安全地获取各奖励权重，提供默认值
        distance_weight = reward_config.get('distance_weight', 1.0)
        velocity_weight = reward_config.get('velocity_weight', 0.5)
        orientation_weight = reward_config.get('orientation_weight', 0.3)
        energy_weight = reward_config.get('energy_weight', 0.2)
        collision_penalty = reward_config.get('collision_penalty', 10.0)
        success_reward = reward_config.get('success_reward', 20.0)
        step_penalty = reward_config.get('step_penalty', 0.01)
        height_penalty = reward_config.get('height_penalty', 5.0)  # 高度惩罚权重
        
        # Calculate reward components
        reward_components = {}
        
        # Distance reward
        reward_components['distance'] = distance_improvement * distance_weight
        
        # Velocity reward - reward for moving toward the target
        # Project velocity onto vector to target for directional component
        velocity_vector = np.array([velocity.x_val, velocity.y_val, velocity.z_val])
        target_direction = vector_to_target / (distance_to_target + 1e-8)  # Normalized vector to target
        velocity_projection = np.dot(velocity_vector, target_direction)
        reward_components['velocity'] = velocity_projection * velocity_weight
        
        # 检查当前高度是否超过初始高度太多
        # 获取初始高度，如果在重置时没有存储，则存储当前高度
        if not hasattr(self, '_initial_height'):
            self._initial_height = current_position[2]  # Z坐标为高度
            print(f"\n初始高度设置为: {self._initial_height:.2f}")
            
        # 计算高度差值 - 在AirSim中，负值表示向上，所以我们需要考虑符号
        # 由于高度增加会让z值变得更负，所以使用初始高度-当前高度
        # 这样，当飞行器上升时，高度差为正号，下降时为负号
        height_diff = self._initial_height - current_position[2]  # 正值表示当前高度高于初始高度
        
        # 计算与目标高度的差值
        target_height_diff = 0.0
        if isinstance(self.target_position, np.ndarray):
            target_height_diff = self.target_position[2] - current_position[2]  # 正值表示远离目标
        else:
            target_height_diff = self.target_position[2] - current_position[2]  # 正值表示远离目标
        
        # 打印当前高度信息（每100步打印一次）
        if self.step_count % 100 == 0:
            print(f"\n当前高度: {current_position[2]:.2f}, 初始高度: {self._initial_height:.2f}, "
                  f"高度差值: {height_diff:.2f}, 目标高度差值: {target_height_diff:.2f}")
        
        # 高度惩罚逻辑：
        # 1. 如果无人机与初始高度差大于50m，给予惩罚
        # 2. 如果无人机与初始高度差小于-20m，给予惩罚
        # 3. 如果无人机与目标高度差超过30m，给予額外惩罚
        
        height_penalty_value = 0.0
        
        # 高空惩罚 - 高于初始位置50m
        if height_diff > 50.0:  
            excess_height = height_diff - 50.0
            high_altitude_penalty = excess_height * height_penalty / 10.0
            height_penalty_value -= high_altitude_penalty
            
            # 打印高空惩罚详情
            if high_altitude_penalty > 0.01:
                print(f"  高空惩罚: {high_altitude_penalty:.4f}, 超高: {excess_height:.2f}m")
            
        # 低空惩罚 - 低于初始位置20m
        if height_diff < -20.0:  
            deficit_height = abs(height_diff) - 20.0
            low_altitude_penalty = deficit_height * height_penalty / 5.0
            height_penalty_value -= low_altitude_penalty
            
            # 打印低空惩罚详情
            if low_altitude_penalty > 0.01:
                print(f"  低空惩罚: {low_altitude_penalty:.4f}, 下降过度: {deficit_height:.2f}m")
            
        # 与目标高度相差过大的惩罚
        if abs(target_height_diff) > 30.0:  
            target_height_excess = abs(target_height_diff) - 30.0
            target_height_penalty = target_height_excess * height_penalty / 20.0
            height_penalty_value -= target_height_penalty
            
            # 打印目标高度差异惩罚详情
            if target_height_penalty > 0.01:
                print(f"  目标高度差异惩罚: {target_height_penalty:.4f}, 高度差: {target_height_excess:.2f}m")
        
        reward_components['height_penalty'] = height_penalty_value
        
        # Orientation reward - reward for facing the target
        _, _, yaw = airsim.to_eularian_angles(orientation)
        heading_vector = np.array([math.cos(yaw), math.sin(yaw), 0])
        target_direction_xy = np.array([target_direction[0], target_direction[1], 0])
        target_direction_xy /= np.linalg.norm(target_direction_xy) + 1e-8
        orientation_alignment = np.dot(heading_vector, target_direction_xy)
        reward_components['orientation'] = orientation_alignment * orientation_weight
        
        # Energy penalty - penalize high energy actions
        energy_usage = np.sum(np.square(action))
        reward_components['energy'] = -energy_usage * energy_weight
        
        # 改进碰撞检测和惩罚
        # 更精确地检测碰撞，包括检查碰撞对象和碰撞位置
        collision_info_str = ''
        if collision and hasattr(self.collision_info, 'object_name') and self.collision_info.object_name:
            collision_info_str = f"与{self.collision_info.object_name}碰撞"
        
        # 增加碰撞位置打印，打印到控制台
        if collision and not hasattr(self, '_collision_reported'):
            collision_pos = None
            if hasattr(self.collision_info, 'impact_point'):
                collision_pos = self.collision_info.impact_point
            
            print(f"\n检测到碰撞! {collision_info_str}")
            if collision_pos:
                print(f"碰撞位置: {collision_pos.x_val:.2f}, {collision_pos.y_val:.2f}, {collision_pos.z_val:.2f}")
                
            # 设置标志避免重复报告同一次碰撞
            self._collision_reported = True
        
        # 如果无碎计数器为0，则重置碰撞报告标志
        if not collision and hasattr(self, '_collision_reported'):
            delattr(self, '_collision_reported')
            
        # 计算碰撞惩罚
        reward_components['collision'] = -collision_penalty if collision else 0.0
        
        # Success reward
        reward_components['success'] = success_reward if reached_target else 0.0
        
        # Step penalty
        reward_components['step'] = -step_penalty
        
        # Calculate total reward
        total_reward = sum(reward_components.values())
        
        return total_reward, reward_components
    
    def _check_termination(self, collision: bool) -> Tuple[bool, bool, Dict]:
        """
        Check termination conditions.
        
        Args:
            collision: Whether a collision occurred
            
        Returns:
            Tuple of (terminated, truncated, info)
        """
        terminated = False
        truncated = False
        info = {}
        
        # Get current position
        drone_state = self.client.getMultirotorState()
        position = drone_state.kinematics_estimated.position
        current_position = np.array([position.x_val, position.y_val, position.z_val])
        
        # 安全地计算到目标的距离，处理类型兼容问题
        # 确保在处理target_position时方式兼容标准Python类型和NumPy数组
        if isinstance(self.target_position, np.ndarray):
            vector_to_target = self.target_position - current_position
            distance_to_target = float(np.linalg.norm(vector_to_target))
        else:
            # 如果是标准Python列表，手动计算
            vector_to_target = np.array([
                self.target_position[0] - current_position[0],
                self.target_position[1] - current_position[1],
                self.target_position[2] - current_position[2]
            ])
            distance_to_target = float(np.linalg.norm(vector_to_target))
        
        # Check termination conditions
        if collision:
            terminated = True
            info['termination_reason'] = 'collision'
        elif distance_to_target < self.success_distance:
            terminated = True
            info['termination_reason'] = 'success'
        elif self.step_count >= self.max_steps:
            truncated = True
            info['termination_reason'] = 'max_steps'
        
        # Additional info
        info['collision'] = collision
        info['distance_to_target'] = distance_to_target
        
        # 安全处理目标位置，处理NumPy数组和标准Python列表的兼容性
        if isinstance(self.target_position, np.ndarray):
            info['target_position'] = self.target_position.tolist()
        else:
            # 如果已经是标准Python列表，直接使用
            info['target_position'] = self.target_position
        
        return terminated, truncated, info
    
    def close(self):
        """
        Close environment and release resources.
        """
        if self.client is not None:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            # Additional cleanup if needed
        return super().close()