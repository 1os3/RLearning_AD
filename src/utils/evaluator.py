import os
import time
import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque

class Evaluator:
    """
    Evaluator class for assessing agent performance in the environment.
    Includes utilities for recording videos and computing statistics.
    """
    def __init__(
        self, 
        env, 
        num_episodes: int = 5, 
        render: bool = False,
        record: bool = False,
        record_dir: Optional[str] = None,
        fps: int = 30,
        max_steps_per_episode: Optional[int] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            env: Environment to evaluate agent in
            num_episodes: Number of episodes to run for evaluation
            render: Whether to render episodes during evaluation
            record: Whether to record videos of evaluation episodes
            record_dir: Directory to save recorded videos
            fps: Frames per second for recorded videos
            max_steps_per_episode: Maximum steps per episode (None for unlimited)
        """
        self.env = env
        self.num_episodes = num_episodes
        self.render = render
        self.record = record
        self.record_dir = record_dir
        self.fps = fps
        self.max_steps_per_episode = max_steps_per_episode
        
        # Create recording directory if needed
        if self.record and self.record_dir is not None:
            os.makedirs(self.record_dir, exist_ok=True)
    
    def evaluate(self, agent, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate agent performance over multiple episodes.
        
        Args:
            agent: Agent to evaluate
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary of evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_collisions = []
        episode_timeouts = []
        
        if self.record and not hasattr(self.env, 'render'):
            print("Warning: Environment does not support rendering, recording will be disabled")
            self.record = False
        
        for i in range(self.num_episodes):
            # Initialize video writer if recording
            video_writer = None
            frames = []
            
            if self.record:
                # Create video writer based on first frame
                test_frame = self.env.render(mode='rgb_array')
                if test_frame is not None:
                    if self.record_dir is not None:
                        # Get frame size
                        frame_height, frame_width = test_frame.shape[:2]
                        video_path = os.path.join(self.record_dir, f"eval_episode_{i}.mp4")
                        video_writer = cv2.VideoWriter(
                            video_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            self.fps,
                            (frame_width, frame_height)
                        )
                    else:
                        # If no record_dir, store frames in memory
                        frames = []
            
            # Reset environment and agent
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            success = False
            collision = False
            timeout = False
            
            # Episode loop
            while not done:
                # Select action
                action = agent.select_action(state, deterministic=deterministic)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Update episode stats
                episode_reward += reward
                episode_length += 1
                
                # Check for success, collision, or timeout in info
                if info.get('success', False):
                    success = True
                if info.get('collision', False):
                    collision = True
                if info.get('timeout', False):
                    timeout = True
                
                # Render if requested
                if self.render:
                    self.env.render()
                    time.sleep(0.01)  # Small delay for visualization
                
                # Record frame if requested
                if self.record:
                    frame = self.env.render(mode='rgb_array')
                    if frame is not None:
                        if video_writer is not None:
                            # Convert RGB to BGR for OpenCV
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            video_writer.write(frame_bgr)
                        else:
                            frames.append(frame)
                
                # Check for episode termination
                if self.max_steps_per_episode is not None and episode_length >= self.max_steps_per_episode:
                    done = True
                    timeout = True
                
                # Update state
                state = next_state
            
            # Close video writer if created
            if video_writer is not None:
                video_writer.release()
            
            # Record episode stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(float(success))
            episode_collisions.append(float(collision))
            episode_timeouts.append(float(timeout))
            
            print(f"Evaluation Episode {i+1}/{self.num_episodes}: "
                  f"Reward = {episode_reward:.2f}, Length = {episode_length}, "
                  f"Success = {success}, Collision = {collision}, Timeout = {timeout}")
        
        # Compute aggregate metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'median_length': np.median(episode_lengths),
            'success_rate': np.mean(episode_successes),
            'collision_rate': np.mean(episode_collisions),
            'timeout_rate': np.mean(episode_timeouts),
        }
        
        return metrics


class EpisodeBuffer:
    """
    Buffer for storing episode data for analysis and visualization.
    Useful for post-hoc analysis of agent behavior.
    """
    def __init__(self, capacity: int = 10000):
        """
        Initialize episode buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.infos = deque(maxlen=capacity)
        
        # Episode tracking
        self.episode_starts = [0]  # Indices where episodes start
        self.episode_rewards = []  # Total reward for each episode
        self.episode_lengths = []  # Length of each episode
        
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def __len__(self) -> int:
        """
        Return number of transitions in buffer.
        """
        return len(self.states)
    
    def add(self, state, action, reward, next_state, done, info=None):
        """
        Add transition to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            info: Additional information
        """
        # Add transition
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.infos.append(info if info is not None else {})
        
        # Update episode stats
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # If episode is done, record stats
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_starts.append(len(self.states))
            
            # Reset current episode stats
            self.current_episode_reward = 0
            self.current_episode_length = 0
    
    def get_episode(self, episode_idx: int) -> Dict[str, List]:
        """
        Get transitions for a specific episode.
        
        Args:
            episode_idx: Index of episode to retrieve
            
        Returns:
            Dictionary with episode data
        """
        if episode_idx >= len(self.episode_lengths):
            raise ValueError(f"Episode index {episode_idx} out of range (max {len(self.episode_lengths)-1})")
        
        start_idx = self.episode_starts[episode_idx]
        end_idx = self.episode_starts[episode_idx + 1]
        
        return {
            'states': list(self.states)[start_idx:end_idx],
            'actions': list(self.actions)[start_idx:end_idx],
            'rewards': list(self.rewards)[start_idx:end_idx],
            'next_states': list(self.next_states)[start_idx:end_idx],
            'dones': list(self.dones)[start_idx:end_idx],
            'infos': list(self.infos)[start_idx:end_idx],
            'total_reward': self.episode_rewards[episode_idx],
            'length': self.episode_lengths[episode_idx]
        }
    
    def get_all_episodes(self) -> List[Dict[str, List]]:
        """
        Get all episodes in the buffer.
        
        Returns:
            List of dictionaries with episode data
        """
        return [self.get_episode(i) for i in range(len(self.episode_lengths))]
    
    def get_episode_stats(self) -> Dict[str, List]:
        """
        Get statistics for all episodes.
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'num_episodes': len(self.episode_rewards)
        }
    
    def clear(self):
        """
        Clear buffer.
        """
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.infos.clear()
        
        self.episode_starts = [0]
        self.episode_rewards = []
        self.episode_lengths = []
        
        self.current_episode_reward = 0
        self.current_episode_length = 0
