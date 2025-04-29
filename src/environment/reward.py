import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Any

class RewardFunction:
    """
    Reward function for UAV reinforcement learning.
    Provides modular reward components that can be combined and weighted.
    """
    def __init__(self, config: Dict):
        """
        Initialize reward function with configuration.
        
        Args:
            config: Dictionary with reward weights and parameters
        """
        # Load reward weights from config
        self.weights = {
            'distance': config.get('weight_distance', 1.0),
            'heading': config.get('weight_heading', 0.5),
            'velocity': config.get('weight_velocity', 0.3),
            'altitude': config.get('weight_altitude', 0.5),
            'stability': config.get('weight_stability', 0.4),
            'collision': config.get('weight_collision', 10.0),
            'goal_reached': config.get('weight_goal_reached', 10.0),
            'energy': config.get('weight_energy', 0.2),
            'progress': config.get('weight_progress', 0.8)
        }
        
        # Reward function parameters
        self.distance_threshold = config.get('distance_threshold', 3.0)  # Distance considered "close" to goal
        self.goal_threshold = config.get('goal_threshold', 1.0)  # Distance to consider goal reached
        self.collision_threshold = config.get('collision_threshold', 0.5)  # Distance to consider collision
        self.min_altitude = config.get('min_altitude', 1.0)  # Minimum safe altitude
        self.max_altitude = config.get('max_altitude', 100.0)  # Maximum desired altitude
        self.optimal_velocity = config.get('optimal_velocity', 5.0)  # Optimal velocity in m/s
        self.max_velocity_penalty = config.get('max_velocity_penalty', 20.0)  # Velocity threshold for max penalty
        self.stability_window = config.get('stability_window', 5)  # Number of steps to consider for stability
        
        # Reward shaping parameters
        self.use_potential_based = config.get('use_potential_based', True)  # Use potential-based reward shaping
        self.gamma = config.get('gamma', 0.99)  # Discount factor for potential-based shaping
        
        # State history for stability and progress calculation
        self.position_history = []
        self.velocity_history = []
        self.rotation_history = []
        self.last_distance_to_goal = None
        
        # Flag for first update
        self.first_update = True
    
    def reset(self):
        """
        Reset reward function state between episodes.
        """
        self.position_history = []
        self.velocity_history = []
        self.rotation_history = []
        self.last_distance_to_goal = None
        self.first_update = True
    
    def distance_reward(self, current_position: np.ndarray, goal_position: np.ndarray) -> float:
        """
        Calculate reward component based on distance to goal.
        Reward increases as UAV gets closer to goal.
        
        Args:
            current_position: Current UAV position (x, y, z)
            goal_position: Goal position (x, y, z)
            
        Returns:
            Distance reward component
        """
        # Calculate Euclidean distance to goal
        distance = np.linalg.norm(np.array(current_position) - np.array(goal_position))
        
        # Store for potential-based shaping
        prev_distance = self.last_distance_to_goal
        self.last_distance_to_goal = distance
        
        # Exponential decay reward based on distance
        base_reward = np.exp(-distance / self.distance_threshold)
        
        # Apply potential-based shaping if enabled and not first update
        if self.use_potential_based and not self.first_update and prev_distance is not None:
            potential_current = -distance  # Negative distance as potential
            potential_prev = -prev_distance
            shaped_reward = self.gamma * potential_current - potential_prev
            return shaped_reward * self.weights['distance']
        
        return base_reward * self.weights['distance']
    
    def heading_reward(self, current_position: np.ndarray, current_velocity: np.ndarray, goal_position: np.ndarray) -> float:
        """
        Calculate reward component based on UAV heading relative to goal.
        Reward is higher when UAV is moving toward the goal.
        
        Args:
            current_position: Current UAV position (x, y, z)
            current_velocity: Current UAV velocity (vx, vy, vz)
            goal_position: Goal position (x, y, z)
            
        Returns:
            Heading reward component
        """
        # Calculate direction vector to goal
        direction_to_goal = np.array(goal_position) - np.array(current_position)
        direction_to_goal_norm = np.linalg.norm(direction_to_goal)
        
        # If already at goal or zero velocity, no heading reward
        velocity_norm = np.linalg.norm(current_velocity)
        if direction_to_goal_norm < 1e-6 or velocity_norm < 1e-6:
            return 0.0
        
        # Normalize vectors
        direction_to_goal = direction_to_goal / direction_to_goal_norm
        current_velocity_norm = np.array(current_velocity) / velocity_norm
        
        # Calculate cosine similarity between velocity and direction to goal
        cosine_similarity = np.dot(direction_to_goal, current_velocity_norm)
        
        # Map from [-1, 1] to [0, 1]
        heading_reward = (cosine_similarity + 1) / 2
        
        return heading_reward * self.weights['heading']
    
    def velocity_reward(self, current_velocity: np.ndarray) -> float:
        """
        Calculate reward component based on velocity.
        Reward is highest at optimal velocity and decreases as it deviates.
        
        Args:
            current_velocity: Current UAV velocity (vx, vy, vz)
            
        Returns:
            Velocity reward component
        """
        # Calculate velocity magnitude
        velocity_norm = np.linalg.norm(current_velocity)
        
        # Store in history
        self.velocity_history.append(velocity_norm)
        if len(self.velocity_history) > self.stability_window:
            self.velocity_history.pop(0)
        
        # Calculate reward based on deviation from optimal velocity
        velocity_deviation = abs(velocity_norm - self.optimal_velocity)
        
        # Gaussian-like reward centered at optimal velocity
        reward = np.exp(-(velocity_deviation ** 2) / (2 * (self.optimal_velocity / 2) ** 2))
        
        # Add penalty for very high velocities (could be dangerous)
        if velocity_norm > self.max_velocity_penalty:
            penalty = (velocity_norm - self.max_velocity_penalty) / self.max_velocity_penalty
            reward -= penalty
        
        return max(0, reward) * self.weights['velocity']
    
    def altitude_reward(self, current_position: np.ndarray) -> float:
        """
        Calculate reward component based on altitude.
        Penalizes flying too low or too high.
        
        Args:
            current_position: Current UAV position (x, y, z)
            
        Returns:
            Altitude reward component
        """
        altitude = current_position[2]  # z-coordinate is altitude
        
        # Penalty for flying too low
        if altitude < self.min_altitude:
            return -((self.min_altitude - altitude) / self.min_altitude) * self.weights['altitude']
        
        # Penalty for flying too high
        if altitude > self.max_altitude:
            return -((altitude - self.max_altitude) / self.max_altitude) * self.weights['altitude']
        
        # Reward for good altitude range
        normalized_altitude = (altitude - self.min_altitude) / (self.max_altitude - self.min_altitude)
        altitude_reward = 1.0 - 2.0 * abs(normalized_altitude - 0.5)  # Optimal at middle of range
        
        return altitude_reward * self.weights['altitude']
    
    def stability_reward(self, current_velocity: np.ndarray, current_rotation: np.ndarray) -> float:
        """
        Calculate reward component based on flight stability.
        Rewards smooth changes in velocity and rotation.
        
        Args:
            current_velocity: Current UAV velocity (vx, vy, vz)
            current_rotation: Current UAV rotation (roll, pitch, yaw)
            
        Returns:
            Stability reward component
        """
        # Add current rotation to history
        self.rotation_history.append(current_rotation)
        if len(self.rotation_history) > self.stability_window:
            self.rotation_history.pop(0)
        
        # If not enough history, return neutral reward
        if len(self.rotation_history) < 2 or len(self.velocity_history) < 2:
            return 0.0
        
        # Calculate velocity stability - penalize rapid changes
        velocity_changes = np.array([abs(self.velocity_history[i] - self.velocity_history[i-1]) 
                                     for i in range(1, len(self.velocity_history))])
        mean_velocity_change = np.mean(velocity_changes)
        velocity_stability = np.exp(-mean_velocity_change)  # Exponential decay for instability
        
        # Calculate rotation stability - penalize rapid changes in attitude
        rotation_changes = np.array([np.linalg.norm(np.array(self.rotation_history[i]) - np.array(self.rotation_history[i-1]))
                                     for i in range(1, len(self.rotation_history))])
        mean_rotation_change = np.mean(rotation_changes)
        rotation_stability = np.exp(-mean_rotation_change / (2 * np.pi))  # Normalized by full rotation
        
        # Combine stability metrics
        stability = (velocity_stability + rotation_stability) / 2
        
        return stability * self.weights['stability']
    
    def collision_penalty(self, collision_info: Dict) -> float:
        """
        Calculate penalty for collisions.
        
        Args:
            collision_info: Dictionary with collision information
            
        Returns:
            Collision penalty (negative reward)
        """
        if collision_info.get('has_collided', False):
            return -self.weights['collision']
        return 0.0
    
    def goal_reward(self, current_position: np.ndarray, goal_position: np.ndarray) -> float:
        """
        Calculate reward for reaching the goal.
        
        Args:
            current_position: Current UAV position (x, y, z)
            goal_position: Goal position (x, y, z)
            
        Returns:
            Goal reward component
        """
        distance = np.linalg.norm(np.array(current_position) - np.array(goal_position))
        if distance <= self.goal_threshold:
            return self.weights['goal_reached']
        return 0.0
    
    def energy_efficiency_reward(self, current_velocity: np.ndarray, current_rotation: np.ndarray) -> float:
        """
        Calculate reward component based on energy efficiency.
        Penalizes high energy maneuvers.
        
        Args:
            current_velocity: Current UAV velocity (vx, vy, vz)
            current_rotation: Current UAV rotation (roll, pitch, yaw)
            
        Returns:
            Energy efficiency reward component
        """
        # Approximate energy use based on velocity and rotation changes
        velocity_norm = np.linalg.norm(current_velocity)
        
        # Energy increases quadratically with velocity (air resistance)
        velocity_energy = (velocity_norm / self.optimal_velocity) ** 2
        
        # Energy also increases with rotation magnitude (attitude adjustments)
        rotation_norm = np.linalg.norm(current_rotation)
        rotation_energy = rotation_norm / (2 * np.pi)  # Normalize by full rotation
        
        # Combined energy metric (0 = best, higher = worse)
        total_energy = (velocity_energy + rotation_energy) / 2
        
        # Convert to reward (highest at low energy)
        energy_reward = np.exp(-total_energy)  # Exponential decay for high energy use
        
        return energy_reward * self.weights['energy']
    
    def progress_reward(self, current_position: np.ndarray, goal_position: np.ndarray) -> float:
        """
        Calculate reward component based on progress toward goal.
        Rewards making progress toward the goal over time.
        
        Args:
            current_position: Current UAV position (x, y, z)
            goal_position: Goal position (x, y, z)
            
        Returns:
            Progress reward component
        """
        # Store current position in history
        self.position_history.append(current_position)
        if len(self.position_history) > self.stability_window:
            self.position_history.pop(0)
        
        # If not enough history or first update, return neutral reward
        if len(self.position_history) < 2 or self.first_update:
            self.first_update = False
            return 0.0
        
        # Calculate previous and current distances to goal
        prev_position = self.position_history[-2]
        prev_distance = np.linalg.norm(np.array(prev_position) - np.array(goal_position))
        current_distance = np.linalg.norm(np.array(current_position) - np.array(goal_position))
        
        # Calculate progress (positive if closer, negative if farther)
        progress = prev_distance - current_distance
        
        # Scale progress reward
        if progress > 0:
            # Positive progress scaled by distance to goal
            progress_reward = progress / (prev_distance + 1e-6)  # Normalize by previous distance
        else:
            # Negative progress (penalize moving away from goal)
            progress_reward = progress / (self.distance_threshold / 2)  # Scale penalty to be less harsh
        
        return progress_reward * self.weights['progress']
    
    def compute_reward(
        self, 
        current_position: np.ndarray, 
        goal_position: np.ndarray, 
        current_velocity: np.ndarray, 
        current_rotation: np.ndarray, 
        collision_info: Dict,
        info: Dict = None
    ) -> Tuple[float, Dict]:
        """
        Compute total reward combining all components.
        
        Args:
            current_position: Current UAV position (x, y, z)
            goal_position: Goal position (x, y, z)
            current_velocity: Current UAV velocity (vx, vy, vz)
            current_rotation: Current UAV rotation (roll, pitch, yaw)
            collision_info: Dictionary with collision information
            info: Optional dictionary to store reward components
            
        Returns:
            Tuple of (total_reward, reward_info_dict)
        """
        # Calculate individual reward components
        reward_distance = self.distance_reward(current_position, goal_position)
        reward_heading = self.heading_reward(current_position, current_velocity, goal_position)
        reward_velocity = self.velocity_reward(current_velocity)
        reward_altitude = self.altitude_reward(current_position)
        reward_stability = self.stability_reward(current_velocity, current_rotation)
        reward_collision = self.collision_penalty(collision_info)
        reward_goal = self.goal_reward(current_position, goal_position)
        reward_energy = self.energy_efficiency_reward(current_velocity, current_rotation)
        reward_progress = self.progress_reward(current_position, goal_position)
        
        # Sum all reward components
        total_reward = (
            reward_distance + 
            reward_heading + 
            reward_velocity + 
            reward_altitude + 
            reward_stability + 
            reward_collision + 
            reward_goal + 
            reward_energy + 
            reward_progress
        )
        
        # Store reward components in info dictionary
        reward_info = {
            'reward_distance': reward_distance,
            'reward_heading': reward_heading,
            'reward_velocity': reward_velocity,
            'reward_altitude': reward_altitude,
            'reward_stability': reward_stability,
            'reward_collision': reward_collision,
            'reward_goal': reward_goal,
            'reward_energy': reward_energy,
            'reward_progress': reward_progress,
            'total_reward': total_reward
        }
        
        # Update info dictionary if provided
        if info is not None:
            info.update(reward_info)
        
        return total_reward, reward_info