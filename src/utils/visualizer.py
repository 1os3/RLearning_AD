import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image

class Visualizer:
    """
    Visualizer for reinforcement learning agent.
    Provides utilities for visualizing states, actions, attention maps, etc.
    """
    def __init__(self, output_dir: str, max_frames: int = 1000):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            max_frames: Maximum number of frames to store
        """
        self.output_dir = output_dir
        self.max_frames = max_frames
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'attention'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'activations'), exist_ok=True)
        
        # Store for frames
        self.frames = []
        self.attention_maps = []
        self.frame_count = 0
    
    def plot_attention_map(self, attention_weights: torch.Tensor, title: str = "Attention Map") -> np.ndarray:
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weight tensor
            title: Plot title
            
        Returns:
            Plot as numpy array
        """
        # Convert to numpy array if tensor
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attention_weights, cmap='viridis')
        plt.colorbar(im)
        
        # Add labels
        ax.set_title(title)
        ax.set_xlabel("Key/Value")
        ax.set_ylabel("Query")
        
        # Save figure to buffer
        fig.tight_layout()
        fig.canvas.draw()
        
        # Convert to numpy array
        plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return plot
    
    def visualize_image_with_attention(self, image: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
        """
        Overlay attention map on image.
        
        Args:
            image: RGB image (H, W, 3)
            attention_map: Attention map (H, W)
            
        Returns:
            Image with attention overlay
        """
        # Ensure attention map is proper shape
        if attention_map.shape != image.shape[:2]:
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        
        # Normalize attention map to [0, 1]
        attention_map = attention_map - attention_map.min()
        attention_map = attention_map / (attention_map.max() + 1e-8)
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Convert image to BGR if it's RGB
        if image.shape[2] == 3:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = image
        
        # Overlay
        overlay = cv2.addWeighted(bgr_image, 0.7, heatmap, 0.3, 0)
        
        # Convert back to RGB
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        return overlay_rgb
    
    def visualize_action(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Visualize action on state.
        
        Args:
            action: Action vector
            state: State/observation
            
        Returns:
            Visualization as numpy array
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot state
        if len(state.shape) == 3 and state.shape[2] == 3:  # RGB image
            ax1.imshow(state)
            ax1.set_title("Current State (Image)")
            ax1.axis('off')
        else:
            # Plot as matrix or feature visualization
            ax1.imshow(np.atleast_2d(state), cmap='viridis')
            ax1.set_title("Current State (Feature)")
            fig.colorbar(ax1.imshow(np.atleast_2d(state), cmap='viridis'), ax=ax1)
        
        # Plot action as bar chart
        action_dim = len(action)
        action_labels = [f"A{i}" for i in range(action_dim)]
        ax2.bar(action_labels, action)
        ax2.set_title("Action Values")
        ax2.set_ylim([-1.1, 1.1])  # Assuming actions are in [-1, 1]
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_ylabel("Action Value")
        
        # Add annotations with values
        for i, v in enumerate(action):
            ax2.text(i, v + 0.05 if v >= 0 else v - 0.1, f"{v:.2f}", 
                    ha='center', va='bottom' if v >= 0 else 'top',
                    fontweight='bold')
        
        # Save figure to buffer
        fig.tight_layout()
        fig.canvas.draw()
        
        # Convert to numpy array
        plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return plot
    
    def visualize_q_values(self, q_values: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Visualize Q-values and taken action.
        
        Args:
            q_values: Q-values for different actions
            action: Action taken
            
        Returns:
            Visualization as numpy array
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Q-values
        x = np.arange(len(q_values))
        ax.bar(x, q_values, label='Q-values')
        
        # Highlight chosen action
        action_idx = np.argmax(np.abs(action - q_values))
        ax.bar(action_idx, q_values[action_idx], color='red', label='Chosen Action')
        
        # Add labels
        ax.set_title("Q-values and Chosen Action")
        ax.set_xlabel("Action Index")
        ax.set_ylabel("Q-value")
        ax.legend()
        
        # Save figure to buffer
        fig.tight_layout()
        fig.canvas.draw()
        
        # Convert to numpy array
        plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return plot
    
    def extract_attention_weights(self, agent, state: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract attention weights from agent's forward pass.
        This is a placeholder that needs to be customized for your specific agent implementation.
        
        Args:
            agent: RL agent
            state: Current state
            
        Returns:
            Attention weights or None if not available
        """
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if hasattr(agent, 'device'):
                state_tensor = state_tensor.to(agent.device)
        else:
            state_tensor = state.unsqueeze(0) if state.dim() == 3 else state
        
        # This is just a placeholder - you need to modify this to extract
        # attention weights from your specific agent/model architecture
        try:
            # Try to access attention weights from model
            # This assumes the agent has a method to extract attention weights
            with torch.no_grad():
                # This is just a placeholder example
                if hasattr(agent, 'ac_network') and hasattr(agent.ac_network, 'get_attention_weights'):
                    attention_weights = agent.ac_network.get_attention_weights(state_tensor)
                    if attention_weights is not None:
                        return attention_weights.squeeze(0).cpu().numpy()
            
            return None
        except:
            return None
    
    def update(self, state: np.ndarray, action: np.ndarray, agent: Any, step: int, episode: int, show_attention: bool = False):
        """
        Update visualizer with current state and action.
        
        Args:
            state: Current state/observation
            action: Action taken
            agent: RL agent (for extracting internal representations)
            step: Current step in episode
            episode: Current episode number
            show_attention: Whether to extract and visualize attention maps
        """
        # Skip if we've reached max frames
        if len(self.frames) >= self.max_frames:
            return
        
        # Visualize action
        action_viz = self.visualize_action(action, state)
        self.frames.append(action_viz)
        
        # Save frame
        frame_path = os.path.join(self.output_dir, 'frames', f'frame_{episode}_{step}.png')
        Image.fromarray(action_viz).save(frame_path)
        
        # Visualize attention if requested
        if show_attention:
            attention_weights = self.extract_attention_weights(agent, state)
            if attention_weights is not None:
                # Plot attention map
                attention_viz = self.plot_attention_map(attention_weights, f"Attention Map (Episode {episode}, Step {step})")
                self.attention_maps.append(attention_viz)
                
                # Save attention map
                attention_path = os.path.join(self.output_dir, 'attention', f'attention_{episode}_{step}.png')
                Image.fromarray(attention_viz).save(attention_path)
                
                # If attention is spatial, try overlaying on image
                if len(attention_weights.shape) == 2 and attention_weights.shape[0] == attention_weights.shape[1]:
                    # Check if state is an image
                    if len(state.shape) == 3 and state.shape[2] == 3:
                        overlay = self.visualize_image_with_attention(state, attention_weights)
                        overlay_path = os.path.join(self.output_dir, 'attention', f'overlay_{episode}_{step}.png')
                        Image.fromarray(overlay).save(overlay_path)
        
        self.frame_count += 1
    
    def create_video(self, output_path: str, fps: int = 30) -> bool:
        """
        Create video from stored frames.
        
        Args:
            output_path: Path to save video
            fps: Frames per second
            
        Returns:
            True if video was created successfully
        """
        if not self.frames:
            print("No frames to create video.")
            return False
        
        try:
            # Get dimensions from first frame
            height, width = self.frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Add frames to video
            for frame in self.frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
            # Close video writer
            video_writer.release()
            
            print(f"Video created at {output_path}")
            return True
        except Exception as e:
            print(f"Error creating video: {e}")
            return False
    
    def save_summary(self, metrics: Optional[Dict] = None):
        """
        Save visualization summary and metrics.
        
        Args:
            metrics: Dictionary of metrics to include in summary
        """
        # Create summary figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot metrics if provided
        if metrics is not None and isinstance(metrics, dict):
            # Plot episode rewards if available
            if 'episode_rewards' in metrics:
                rewards = metrics['episode_rewards']
                ax.plot(rewards, label='Episode Rewards')
                ax.set_title("Training Progress")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                ax.legend()
            
            # Add text box with summary statistics
            stat_text = "\n".join([f"{k}: {v}" for k, v in metrics.items() 
                                  if not isinstance(v, (list, np.ndarray))])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        else:
            ax.set_title("Visualization Summary")
            ax.text(0.5, 0.5, f"Total Frames: {self.frame_count}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
        
        # Save summary figure
        summary_path = os.path.join(self.output_dir, 'summary.png')
        fig.savefig(summary_path)
        plt.close(fig)
        
        # Create video from frames
        video_path = os.path.join(self.output_dir, 'visualization.mp4')
        self.create_video(video_path)
        
        # Also create video from attention maps if available
        if self.attention_maps:
            attention_video_path = os.path.join(self.output_dir, 'attention.mp4')
            
            # Get dimensions from first attention map
            height, width = self.attention_maps[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(attention_video_path, fourcc, 10, (width, height))
            
            # Add frames to video
            for frame in self.attention_maps:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
            # Close video writer
            video_writer.release()
            
        print(f"Visualization summary saved to {self.output_dir}")
    
    def close(self, metrics: Optional[Dict] = None):
        """
        Close visualizer and save summary.
        
        Args:
            metrics: Dictionary of metrics to include in summary
        """
        self.save_summary(metrics)
        
        # Clear frames to free memory
        self.frames.clear()
        self.attention_maps.clear()
