import numpy as np
import random
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import json

class DomainRandomizer:
    """
    Domain randomization for reinforcement learning environments.
    Implements the framework's "域随机化：光照、风速、场景纹理" feature.
    
    Provides various randomization techniques to enhance model generalization:
    - Image randomization (brightness, contrast, noise, etc.)
    - Physics randomization (wind, gravity, friction, etc.)
    - Environment randomization (obstacles, textures, etc.)
    """
    def __init__(self, config: Dict = None):
        """
        Initialize domain randomizer with configuration.
        
        Args:
            config: Configuration dictionary with randomization parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.defaults = {
            # Image randomization
            'image': {
                'enabled': True,
                'brightness_range': [0.8, 1.2],  # Brightness adjustment factor
                'contrast_range': [0.8, 1.2],    # Contrast adjustment factor
                'saturation_range': [0.8, 1.2],  # Saturation adjustment factor
                'hue_range': [-0.1, 0.1],        # Hue adjustment in fraction of circle
                'noise_range': [0.0, 0.05],      # Gaussian noise standard deviation
                'blur_range': [0.0, 1.0],        # Blur kernel size factor
                'dropout_range': [0.0, 0.05],    # Fraction of pixels to drop
                'cutout_prob': 0.2,              # Probability of applying cutout
                'cutout_size_range': [0.1, 0.2], # Size of cutout as fraction of image size
            },
            
            # Physics randomization
            'physics': {
                'enabled': True,
                'wind_enabled': True,
                'wind_speed_range': [0.0, 5.0],   # Wind speed in m/s
                'wind_direction_range': [-180, 180], # Wind direction in degrees
                'wind_gust_range': [0.0, 2.0],   # Wind gust strength
                'gravity_range': [9.7, 9.9],     # Gravity in m/s^2
                'density_range': [1.1, 1.3],      # Air density in kg/m^3
                'friction_range': [0.9, 1.1],     # Friction coefficient multiplier
            },
            
            # Environment randomization
            'environment': {
                'enabled': True,
                'time_of_day_range': [0, 24],     # Time of day in hours
                'cloud_cover_range': [0.0, 0.8],  # Cloud cover fraction
                'fog_range': [0.0, 0.3],          # Fog density
                'rain_range': [0.0, 0.5],         # Rain intensity
                'texture_variation_prob': 0.3,    # Probability of texture variation
                'obstacle_variation_prob': 0.2,   # Probability of obstacle variation
                'terrain_variation_prob': 0.2,    # Probability of terrain variation
            },
            
            # Sensor randomization
            'sensor': {
                'enabled': True,
                'camera_noise_range': [0.0, 0.02],  # Camera sensor noise
                'camera_dropout_prob': 0.05,        # Camera pixel dropout probability
                'sensor_bias_range': [-0.05, 0.05], # Sensor bias as fraction of range
                'sensor_drift_range': [0.0, 0.02],  # Sensor drift per step
                'sensor_delay_range': [0, 2],       # Sensor delay in timesteps
            },
            
            # Curriculum difficulty (0.0 = easy, 1.0 = hard)
            'curriculum': {
                'enabled': True,
                'difficulty': 0.0,                # Initial difficulty level
                'adaptation_rate': 0.001,         # Rate of difficulty increase
                'max_difficulty': 1.0,            # Maximum difficulty level
                'success_threshold': 0.7,         # Success rate threshold for increasing difficulty
            }
        }
        
        # Combine default and user configuration
        self.init_from_config()
        
        # Initialize curriculum
        self.current_difficulty = self.curriculum_config.get('difficulty', 0.0)
        self.success_history = []
        self.success_window = 100  # Number of episodes to consider for success rate
        
        # Cache for current randomization parameters
        self.current_params = {}
        self.reset_randomization()
    
    def init_from_config(self):
        """
        Initialize randomizer from configuration.
        """
        # Get configuration sections with fallbacks to defaults
        self.image_config = self.config.get('image', self.defaults['image'])
        self.physics_config = self.config.get('physics', self.defaults['physics'])
        self.environment_config = self.config.get('environment', self.defaults['environment'])
        self.sensor_config = self.config.get('sensor', self.defaults['sensor'])
        self.curriculum_config = self.config.get('curriculum', self.defaults['curriculum'])
        
        # Check if randomization is enabled
        self.image_enabled = self.image_config.get('enabled', True)
        self.physics_enabled = self.physics_config.get('enabled', True)
        self.environment_enabled = self.environment_config.get('enabled', True)
        self.sensor_enabled = self.sensor_config.get('enabled', True)
        self.curriculum_enabled = self.curriculum_config.get('enabled', True)
    
    def reset_randomization(self):
        """
        Reset randomization parameters for a new episode.
        """
        # Generate new randomization parameters for this episode
        self.current_params = {
            'image': self._generate_image_params() if self.image_enabled else {},
            'physics': self._generate_physics_params() if self.physics_enabled else {},
            'environment': self._generate_environment_params() if self.environment_enabled else {},
            'sensor': self._generate_sensor_params() if self.sensor_enabled else {},
        }
    
    def _lerp(self, a: float, b: float, t: float) -> float:
        """
        Linear interpolation between a and b by factor t.
        
        Args:
            a: Start value
            b: End value
            t: Interpolation factor (0-1)
            
        Returns:
            Interpolated value
        """
        return a + (b - a) * t
    
    def _random_range(self, range_min: float, range_max: float, difficulty: float = None) -> float:
        """
        Generate random value within range, adjusted by difficulty.
        
        Args:
            range_min: Minimum value
            range_max: Maximum value
            difficulty: Optional difficulty factor (0-1)
            
        Returns:
            Random value within range
        """
        if difficulty is None:
            difficulty = self.current_difficulty if self.curriculum_enabled else 1.0
            
        # At difficulty 0, range is narrow around mean
        # At difficulty 1, full range is used
        mean = (range_min + range_max) / 2
        half_range = (range_max - range_min) / 2
        
        # Adjust range based on difficulty
        effective_half_range = half_range * (0.1 + 0.9 * difficulty)
        
        # Generate random value within effective range
        return random.uniform(mean - effective_half_range, mean + effective_half_range)
    
    def _generate_image_params(self) -> Dict:
        """
        Generate randomization parameters for image processing.
        
        Returns:
            Dictionary of image randomization parameters
        """
        # Get configuration with fallbacks
        brightness_range = self.image_config.get('brightness_range', self.defaults['image']['brightness_range'])
        contrast_range = self.image_config.get('contrast_range', self.defaults['image']['contrast_range'])
        saturation_range = self.image_config.get('saturation_range', self.defaults['image']['saturation_range'])
        hue_range = self.image_config.get('hue_range', self.defaults['image']['hue_range'])
        noise_range = self.image_config.get('noise_range', self.defaults['image']['noise_range'])
        blur_range = self.image_config.get('blur_range', self.defaults['image']['blur_range'])
        dropout_range = self.image_config.get('dropout_range', self.defaults['image']['dropout_range'])
        cutout_prob = self.image_config.get('cutout_prob', self.defaults['image']['cutout_prob'])
        cutout_size_range = self.image_config.get('cutout_size_range', self.defaults['image']['cutout_size_range'])
        
        # Generate random parameters
        params = {
            'brightness': self._random_range(brightness_range[0], brightness_range[1]),
            'contrast': self._random_range(contrast_range[0], contrast_range[1]),
            'saturation': self._random_range(saturation_range[0], saturation_range[1]),
            'hue': self._random_range(hue_range[0], hue_range[1]),
            'noise': self._random_range(noise_range[0], noise_range[1]),
            'blur': self._random_range(blur_range[0], blur_range[1]),
            'dropout': self._random_range(dropout_range[0], dropout_range[1]),
            'use_cutout': random.random() < cutout_prob,
            'cutout_size': self._random_range(cutout_size_range[0], cutout_size_range[1]),
            'cutout_x': random.random(),  # Random x position as fraction of width
            'cutout_y': random.random(),  # Random y position as fraction of height
        }
        
        return params
    
    def _generate_physics_params(self) -> Dict:
        """
        Generate randomization parameters for physics simulation.
        
        Returns:
            Dictionary of physics randomization parameters
        """
        # Get configuration with fallbacks
        wind_enabled = self.physics_config.get('wind_enabled', self.defaults['physics']['wind_enabled'])
        wind_speed_range = self.physics_config.get('wind_speed_range', self.defaults['physics']['wind_speed_range'])
        wind_direction_range = self.physics_config.get('wind_direction_range', self.defaults['physics']['wind_direction_range'])
        wind_gust_range = self.physics_config.get('wind_gust_range', self.defaults['physics']['wind_gust_range'])
        gravity_range = self.physics_config.get('gravity_range', self.defaults['physics']['gravity_range'])
        density_range = self.physics_config.get('density_range', self.defaults['physics']['density_range'])
        friction_range = self.physics_config.get('friction_range', self.defaults['physics']['friction_range'])
        
        # Generate random parameters
        params = {
            'wind_enabled': wind_enabled,
            'wind_speed': self._random_range(wind_speed_range[0], wind_speed_range[1]),
            'wind_direction': self._random_range(wind_direction_range[0], wind_direction_range[1]),
            'wind_gust': self._random_range(wind_gust_range[0], wind_gust_range[1]),
            'gravity': self._random_range(gravity_range[0], gravity_range[1]),
            'density': self._random_range(density_range[0], density_range[1]),
            'friction': self._random_range(friction_range[0], friction_range[1]),
        }
        
        return params
    
    def _generate_environment_params(self) -> Dict:
        """
        Generate randomization parameters for environment.
        
        Returns:
            Dictionary of environment randomization parameters
        """
        # Get configuration with fallbacks
        time_of_day_range = self.environment_config.get('time_of_day_range', self.defaults['environment']['time_of_day_range'])
        cloud_cover_range = self.environment_config.get('cloud_cover_range', self.defaults['environment']['cloud_cover_range'])
        fog_range = self.environment_config.get('fog_range', self.defaults['environment']['fog_range'])
        rain_range = self.environment_config.get('rain_range', self.defaults['environment']['rain_range'])
        texture_variation_prob = self.environment_config.get('texture_variation_prob', self.defaults['environment']['texture_variation_prob'])
        obstacle_variation_prob = self.environment_config.get('obstacle_variation_prob', self.defaults['environment']['obstacle_variation_prob'])
        terrain_variation_prob = self.environment_config.get('terrain_variation_prob', self.defaults['environment']['terrain_variation_prob'])
        
        # Generate random parameters
        params = {
            'time_of_day': self._random_range(time_of_day_range[0], time_of_day_range[1]),
            'cloud_cover': self._random_range(cloud_cover_range[0], cloud_cover_range[1]),
            'fog': self._random_range(fog_range[0], fog_range[1]),
            'rain': self._random_range(rain_range[0], rain_range[1]),
            'texture_variation': random.random() < texture_variation_prob,
            'obstacle_variation': random.random() < obstacle_variation_prob,
            'terrain_variation': random.random() < terrain_variation_prob,
            'texture_id': random.randint(0, 9),  # Random texture ID (0-9)
            'obstacle_id': random.randint(0, 5),  # Random obstacle set ID (0-5)
            'terrain_id': random.randint(0, 3),   # Random terrain ID (0-3)
        }
        
        return params
    
    def _generate_sensor_params(self) -> Dict:
        """
        Generate randomization parameters for sensors.
        
        Returns:
            Dictionary of sensor randomization parameters
        """
        # Get configuration with fallbacks
        camera_noise_range = self.sensor_config.get('camera_noise_range', self.defaults['sensor']['camera_noise_range'])
        camera_dropout_prob = self.sensor_config.get('camera_dropout_prob', self.defaults['sensor']['camera_dropout_prob'])
        sensor_bias_range = self.sensor_config.get('sensor_bias_range', self.defaults['sensor']['sensor_bias_range'])
        sensor_drift_range = self.sensor_config.get('sensor_drift_range', self.defaults['sensor']['sensor_drift_range'])
        sensor_delay_range = self.sensor_config.get('sensor_delay_range', self.defaults['sensor']['sensor_delay_range'])
        
        # Generate random parameters
        params = {
            'camera_noise': self._random_range(camera_noise_range[0], camera_noise_range[1]),
            'camera_dropout': camera_dropout_prob,
            'position_bias': [self._random_range(sensor_bias_range[0], sensor_bias_range[1]) for _ in range(3)],
            'velocity_bias': [self._random_range(sensor_bias_range[0], sensor_bias_range[1]) for _ in range(3)],
            'orientation_bias': [self._random_range(sensor_bias_range[0], sensor_bias_range[1]) for _ in range(3)],
            'position_drift': [self._random_range(sensor_drift_range[0], sensor_drift_range[1]) for _ in range(3)],
            'velocity_drift': [self._random_range(sensor_drift_range[0], sensor_drift_range[1]) for _ in range(3)],
            'orientation_drift': [self._random_range(sensor_drift_range[0], sensor_drift_range[1]) for _ in range(3)],
            'sensor_delay': int(self._random_range(sensor_delay_range[0], sensor_delay_range[1])),
        }
        
        return params
    
    def randomize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply randomization to image.
        
        Args:
            image: Input image as numpy array (H, W, C), RGB format
            
        Returns:
            Randomized image
        """
        if not self.image_enabled or image is None:
            return image
        
        # Get randomization parameters
        params = self.current_params.get('image', {})
        if not params:
            return image
        
        # Make a copy of the image
        img = image.copy()
        
        # Convert to float32 for processing
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        
        # Apply brightness adjustment
        if 'brightness' in params:
            img = img * params['brightness']
        
        # Apply contrast adjustment
        if 'contrast' in params:
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            img = (img - mean) * params['contrast'] + mean
        
        # Apply HSV adjustments
        if any(k in params for k in ['hue', 'saturation']):
            # Convert to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            # Apply saturation adjustment
            if 'saturation' in params:
                img_hsv[:, :, 1] = img_hsv[:, :, 1] * params['saturation']
            
            # Apply hue adjustment
            if 'hue' in params:
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + params['hue'] * 180) % 180
            
            # Convert back to RGB
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        # Apply Gaussian noise
        if 'noise' in params and params['noise'] > 0:
            noise = np.random.normal(0, params['noise'], img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)
        
        # Apply blur
        if 'blur' in params and params['blur'] > 0:
            kernel_size = int(params['blur'] * 5) * 2 + 1  # Odd kernel size
            if kernel_size > 1:
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Apply dropout (randomly set pixels to black)
        if 'dropout' in params and params['dropout'] > 0:
            mask = np.random.random(img.shape[:2]) < params['dropout']
            img[mask] = 0
        
        # Apply cutout (set a rectangular region to black)
        if params.get('use_cutout', False):
            h, w = img.shape[:2]
            size_h = int(h * params['cutout_size'])
            size_w = int(w * params['cutout_size'])
            x = int(params['cutout_x'] * (w - size_w))
            y = int(params['cutout_y'] * (h - size_h))
            
            img[y:y+size_h, x:x+size_w] = 0
        
        # Clip values to valid range
        img = np.clip(img, 0, 1)
        
        # Convert back to original data type
        if image.dtype != np.float32:
            img = (img * 255).astype(image.dtype)
        
        return img
    
    def get_physics_params(self) -> Dict:
        """
        Get current physics randomization parameters.
        
        Returns:
            Dictionary of physics parameters
        """
        if not self.physics_enabled:
            return {}
        
        return self.current_params.get('physics', {})
    
    def get_environment_params(self) -> Dict:
        """
        Get current environment randomization parameters.
        
        Returns:
            Dictionary of environment parameters
        """
        if not self.environment_enabled:
            return {}
        
        return self.current_params.get('environment', {})
    
    def get_sensor_params(self) -> Dict:
        """
        Get current sensor randomization parameters.
        
        Returns:
            Dictionary of sensor parameters
        """
        if not self.sensor_enabled:
            return {}
        
        return self.current_params.get('sensor', {})
    
    def apply_sensor_noise(self, sensor_data: Dict) -> Dict:
        """
        Apply sensor noise and biases to sensor data.
        
        Args:
            sensor_data: Dictionary of sensor readings
            
        Returns:
            Noisy sensor data
        """
        if not self.sensor_enabled:
            return sensor_data
        
        params = self.current_params.get('sensor', {})
        if not params:
            return sensor_data
        
        result = sensor_data.copy()
        
        # Apply position bias and noise
        if 'position' in result and params.get('position_bias'):
            bias = np.array(params['position_bias'])
            drift = np.array(params.get('position_drift', [0, 0, 0]))
            position = np.array(result['position'])
            
            # Apply bias and drift
            max_position = np.max(np.abs(position)) + 1e-6
            position += bias * max_position + drift * max_position * np.random.randn(3)
            
            result['position'] = position.tolist()
        
        # Apply velocity bias and noise
        if 'velocity' in result and params.get('velocity_bias'):
            bias = np.array(params['velocity_bias'])
            drift = np.array(params.get('velocity_drift', [0, 0, 0]))
            velocity = np.array(result['velocity'])
            
            # Apply bias and drift
            max_velocity = np.max(np.abs(velocity)) + 1e-6
            velocity += bias * max_velocity + drift * max_velocity * np.random.randn(3)
            
            result['velocity'] = velocity.tolist()
        
        # Apply orientation bias and noise
        if 'orientation' in result and params.get('orientation_bias'):
            bias = np.array(params['orientation_bias'])
            drift = np.array(params.get('orientation_drift', [0, 0, 0]))
            orientation = np.array(result['orientation'])
            
            # Apply bias and drift (in radians)
            orientation += bias * 0.1 + drift * 0.1 * np.random.randn(3)
            
            result['orientation'] = orientation.tolist()
        
        return result
    
    def update_curriculum(self, success: bool):
        """
        Update curriculum difficulty based on agent's success.
        
        Args:
            success: Whether the episode was successful
        """
        if not self.curriculum_enabled:
            return
        
        # Add success to history
        self.success_history.append(float(success))
        
        # Keep history within window
        if len(self.success_history) > self.success_window:
            self.success_history.pop(0)
        
        # Calculate success rate
        success_rate = sum(self.success_history) / len(self.success_history) if self.success_history else 0
        
        # Get curriculum parameters
        adaptation_rate = self.curriculum_config.get('adaptation_rate', self.defaults['curriculum']['adaptation_rate'])
        success_threshold = self.curriculum_config.get('success_threshold', self.defaults['curriculum']['success_threshold'])
        max_difficulty = self.curriculum_config.get('max_difficulty', self.defaults['curriculum']['max_difficulty'])
        
        # Adjust difficulty based on success rate
        if success_rate > success_threshold:
            # Increase difficulty if success rate is high
            self.current_difficulty = min(self.current_difficulty + adaptation_rate, max_difficulty)
        elif len(self.success_history) >= self.success_window:
            # Decrease difficulty if success rate is low and we have enough history
            self.current_difficulty = max(self.current_difficulty - adaptation_rate, 0.0)
    
    def save_config(self, filepath: str):
        """
        Save randomization configuration to file.
        
        Args:
            filepath: Path to save configuration file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Combine configuration with current difficulty
        config_to_save = self.config.copy()
        if 'curriculum' in config_to_save:
            config_to_save['curriculum']['difficulty'] = self.current_difficulty
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(config_to_save, f, indent=4)
    
    def load_config(self, filepath: str):
        """
        Load randomization configuration from file.
        
        Args:
            filepath: Path to configuration file
        """
        if not os.path.exists(filepath):
            print(f"Config file {filepath} not found, using defaults")
            return
        
        try:
            with open(filepath, 'r') as f:
                self.config = json.load(f)
            
            # Initialize from loaded config
            self.init_from_config()
            
            # Set current difficulty from config
            if 'curriculum' in self.config and 'difficulty' in self.config['curriculum']:
                self.current_difficulty = self.config['curriculum']['difficulty']
                
            print(f"Loaded randomization config from {filepath}")
        except Exception as e:
            print(f"Error loading config from {filepath}: {e}")
    
    def get_current_difficulty(self) -> float:
        """
        Get current curriculum difficulty level.
        
        Returns:
            Current difficulty (0-1)
        """
        return self.current_difficulty