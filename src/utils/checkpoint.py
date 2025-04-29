import os
import time
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import glob
import shutil
import yaml
from datetime import datetime
import logging

class CheckpointManager:
    """
    Comprehensive checkpoint manager for saving and loading model states.
    Supports multiple checkpoints, automatic versioning, and metadata tracking.
    Implements the framework's "断点续训" feature.
    """
    def __init__(
        self, 
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_freq: int = 10000,
        save_best: bool = True,
        verbose: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_freq: Frequency of checkpoints (in steps)
            save_best: Whether to save best model based on eval metrics
            verbose: Whether to print checkpoint info
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_freq = save_freq
        self.save_best = save_best
        self.verbose = verbose
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Checkpoint metadata
        self.checkpoints = []
        self.best_reward = -float('inf')
        self.best_checkpoint = None
        
        # Load checkpoint metadata if it exists
        self._load_metadata()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)
    
    def _load_metadata(self):
        """
        Load checkpoint metadata from file.
        """
        metadata_path = os.path.join(self.checkpoint_dir, 'checkpoint_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.checkpoints = metadata.get('checkpoints', [])
                self.best_reward = metadata.get('best_reward', -float('inf'))
                self.best_checkpoint = metadata.get('best_checkpoint', None)
                
                if self.verbose:
                    self.logger.info(f"Loaded checkpoint metadata: {len(self.checkpoints)} checkpoints found")
                    if self.best_checkpoint:
                        self.logger.info(f"Best checkpoint: {self.best_checkpoint}, reward: {self.best_reward}")
            except Exception as e:
                self.logger.error(f"Error loading checkpoint metadata: {e}")
    
    def _save_metadata(self):
        """
        Save checkpoint metadata to file.
        """
        metadata_path = os.path.join(self.checkpoint_dir, 'checkpoint_metadata.json')
        metadata = {
            'checkpoints': self.checkpoints,
            'best_reward': self.best_reward,
            'best_checkpoint': self.best_checkpoint,
            'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving checkpoint metadata: {e}")
    
    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints exceeding max_checkpoints.
        """
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort checkpoints by step
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.get('step', 0))
        
        # Keep newest checkpoints and best checkpoint
        checkpoints_to_remove = sorted_checkpoints[:-(self.max_checkpoints)]
        
        # Don't remove best checkpoint
        if self.best_checkpoint:
            checkpoints_to_remove = [c for c in checkpoints_to_remove 
                                   if c.get('version') != self.best_checkpoint]
        
        # Remove old checkpoints
        for checkpoint in checkpoints_to_remove:
            version = checkpoint.get('version')
            if version:
                checkpoint_path = os.path.join(self.checkpoint_dir, version)
                try:
                    if os.path.isdir(checkpoint_path):
                        shutil.rmtree(checkpoint_path)
                    self.checkpoints.remove(checkpoint)
                    if self.verbose:
                        self.logger.info(f"Removed old checkpoint: {version}")
                except Exception as e:
                    self.logger.error(f"Error removing checkpoint {version}: {e}")
        
        # Update metadata
        self._save_metadata()
    
    def should_save(self, step: int) -> bool:
        """
        Check if a checkpoint should be saved at current step.
        
        Args:
            step: Current training step
            
        Returns:
            Whether to save checkpoint
        """
        return step % self.save_freq == 0
    
    def save(self, 
             model: Any, 
             optimizer: Optional[Any] = None,
             scheduler: Optional[Any] = None,
             replay_buffer: Optional[Any] = None,
             env_state: Optional[Dict] = None,
             step: int = 0,
             episode: int = 0,
             metrics: Optional[Dict] = None,
             config: Optional[Dict] = None,
             extra_data: Optional[Dict] = None
            ) -> str:
        """
        Save checkpoint including model, optimizer, scheduler, config, and metrics.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Learning rate scheduler to save
            replay_buffer: Replay buffer to save (optional)
            env_state: Environment state (optional)
            step: Current training step
            episode: Current episode number
            metrics: Training metrics (optional)
            config: Training configuration (optional)
            extra_data: Any additional data to save
            
        Returns:
            Checkpoint version string
        """
        # Create timestamp and version
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        version = f"checkpoint_{timestamp}"
        
        # Create checkpoint directory
        checkpoint_path = os.path.join(self.checkpoint_dir, version)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model state
        try:
            if hasattr(model, 'module'):  # Handle distributed models
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
                
            torch.save(model_state, os.path.join(checkpoint_path, 'model.pt'))
            
            # Save optimizer state
            if optimizer is not None:
                if isinstance(optimizer, (list, tuple)):
                    # Handle multiple optimizers
                    optim_states = [opt.state_dict() for opt in optimizer]
                    torch.save(optim_states, os.path.join(checkpoint_path, 'optimizer.pt'))
                else:
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer.pt'))
            
            # Save scheduler state
            if scheduler is not None:
                if isinstance(scheduler, (list, tuple)):
                    # Handle multiple schedulers
                    scheduler_states = [sched.state_dict() for sched in scheduler]
                    torch.save(scheduler_states, os.path.join(checkpoint_path, 'scheduler.pt'))
                else:
                    torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, 'scheduler.pt'))
            
            # Save replay buffer state if provided
            if replay_buffer is not None:
                if hasattr(replay_buffer, 'save_to_file'):
                    # If replay buffer has built-in save method
                    replay_buffer.save_to_file(os.path.join(checkpoint_path, 'replay_buffer.pt'))
                else:
                    # Try to pickle the replay buffer
                    try:
                        torch.save(replay_buffer, os.path.join(checkpoint_path, 'replay_buffer.pt'))
                    except Exception as e:
                        self.logger.warning(f"Could not save replay buffer: {e}")
            
            # Save environment state if provided
            if env_state is not None:
                torch.save(env_state, os.path.join(checkpoint_path, 'env_state.pt'))
            
            # Save training state
            training_state = {
                'step': step,
                'episode': episode,
                'timestamp': timestamp,
                'metrics': metrics if metrics is not None else {}
            }
            
            if extra_data is not None:
                training_state.update(extra_data)
                
            torch.save(training_state, os.path.join(checkpoint_path, 'training_state.pt'))
            
            # Save configuration
            if config is not None:
                with open(os.path.join(checkpoint_path, 'config.yaml'), 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            
            # Update checkpoint list
            checkpoint_info = {
                'version': version,
                'step': step,
                'episode': episode,
                'timestamp': timestamp,
                'metrics': metrics if metrics is not None else {}
            }
            self.checkpoints.append(checkpoint_info)
            
            # Check if this is the best checkpoint
            is_best = False
            if self.save_best and metrics is not None and 'mean_reward' in metrics:
                mean_reward = metrics['mean_reward']
                if mean_reward > self.best_reward:
                    self.best_reward = mean_reward
                    self.best_checkpoint = version
                    is_best = True
                    
                    # Create a symbolic link or copy for best checkpoint
                    best_path = os.path.join(self.checkpoint_dir, 'best')
                    if os.path.exists(best_path):
                        if os.path.islink(best_path):
                            os.unlink(best_path)
                        else:
                            shutil.rmtree(best_path)
                    
                    # Create link or copy files
                    try:
                        os.symlink(checkpoint_path, best_path, target_is_directory=True)
                    except:
                        # If symlink fails, copy files
                        os.makedirs(best_path, exist_ok=True)
                        for file in os.listdir(checkpoint_path):
                            src = os.path.join(checkpoint_path, file)
                            dst = os.path.join(best_path, file)
                            shutil.copy2(src, dst)
            
            # Save metadata
            self._save_metadata()
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            # Create latest link
            latest_path = os.path.join(self.checkpoint_dir, 'latest')
            if os.path.exists(latest_path):
                if os.path.islink(latest_path):
                    os.unlink(latest_path)
                else:
                    shutil.rmtree(latest_path)
            
            # Create link or copy files
            try:
                os.symlink(checkpoint_path, latest_path, target_is_directory=True)
            except:
                # If symlink fails, copy files
                os.makedirs(latest_path, exist_ok=True)
                for file in os.listdir(checkpoint_path):
                    src = os.path.join(checkpoint_path, file)
                    dst = os.path.join(latest_path, file)
                    shutil.copy2(src, dst)
            
            if self.verbose:
                self.logger.info(f"Saved checkpoint {version}, step {step}, episode {episode}")
                if is_best:
                    self.logger.info(f"New best checkpoint with mean reward {self.best_reward}")
            
            return version
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def load(self, 
             model: Any, 
             optimizer: Optional[Any] = None,
             scheduler: Optional[Any] = None,
             replay_buffer: Optional[Any] = None,
             version: str = 'latest',
             load_replay_buffer: bool = True,
             map_location: Optional[Any] = None
            ) -> Dict:
        """
        Load checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Learning rate scheduler to load state into
            replay_buffer: Replay buffer to load state into
            version: Checkpoint version to load (latest, best, or specific version)
            load_replay_buffer: Whether to load replay buffer
            map_location: Device to map model tensors to
            
        Returns:
            Training state dictionary
        """
        if version == 'latest':
            checkpoint_path = os.path.join(self.checkpoint_dir, 'latest')
            if not os.path.exists(checkpoint_path):
                # If latest link doesn't exist, find most recent checkpoint
                if self.checkpoints:
                    sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.get('step', 0), reverse=True)
                    version = sorted_checkpoints[0]['version']
                    checkpoint_path = os.path.join(self.checkpoint_dir, version)
                else:
                    raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
        elif version == 'best':
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best')
            if not os.path.exists(checkpoint_path):
                if self.best_checkpoint:
                    checkpoint_path = os.path.join(self.checkpoint_dir, self.best_checkpoint)
                else:
                    raise FileNotFoundError(f"No best checkpoint found in {self.checkpoint_dir}")
        else:
            # Specific version
            checkpoint_path = os.path.join(self.checkpoint_dir, version)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint {version} not found in {self.checkpoint_dir}")
        
        try:
            # Load model state
            model_path = os.path.join(checkpoint_path, 'model.pt')
            if os.path.exists(model_path):
                model_state = torch.load(model_path, map_location=map_location)
                
                if hasattr(model, 'module'):  # Handle distributed models
                    model.module.load_state_dict(model_state)
                else:
                    model.load_state_dict(model_state)
                    
                if self.verbose:
                    self.logger.info(f"Loaded model state from {model_path}")
            else:
                self.logger.warning(f"Model state not found at {model_path}")
            
            # Load optimizer state
            if optimizer is not None:
                optim_path = os.path.join(checkpoint_path, 'optimizer.pt')
                if os.path.exists(optim_path):
                    optim_state = torch.load(optim_path, map_location=map_location)
                    
                    if isinstance(optimizer, (list, tuple)):
                        # Handle multiple optimizers
                        if isinstance(optim_state, list) and len(optim_state) == len(optimizer):
                            for opt, state in zip(optimizer, optim_state):
                                opt.load_state_dict(state)
                        else:
                            self.logger.warning(f"Optimizer state format mismatch, skipping")
                    else:
                        optimizer.load_state_dict(optim_state)
                        
                    if self.verbose:
                        self.logger.info(f"Loaded optimizer state from {optim_path}")
                else:
                    self.logger.warning(f"Optimizer state not found at {optim_path}")
            
            # Load scheduler state
            if scheduler is not None:
                scheduler_path = os.path.join(checkpoint_path, 'scheduler.pt')
                if os.path.exists(scheduler_path):
                    scheduler_state = torch.load(scheduler_path, map_location=map_location)
                    
                    if isinstance(scheduler, (list, tuple)):
                        # Handle multiple schedulers
                        if isinstance(scheduler_state, list) and len(scheduler_state) == len(scheduler):
                            for sched, state in zip(scheduler, scheduler_state):
                                sched.load_state_dict(state)
                        else:
                            self.logger.warning(f"Scheduler state format mismatch, skipping")
                    else:
                        scheduler.load_state_dict(scheduler_state)
                        
                    if self.verbose:
                        self.logger.info(f"Loaded scheduler state from {scheduler_path}")
                else:
                    self.logger.warning(f"Scheduler state not found at {scheduler_path}")
            
            # Load replay buffer state
            if load_replay_buffer and replay_buffer is not None:
                replay_buffer_path = os.path.join(checkpoint_path, 'replay_buffer.pt')
                if os.path.exists(replay_buffer_path):
                    try:
                        if hasattr(replay_buffer, 'load_from_file'):
                            # If replay buffer has built-in load method
                            replay_buffer.load_from_file(replay_buffer_path)
                        else:
                            # Try to load directly
                            loaded_buffer = torch.load(replay_buffer_path, map_location=map_location)
                            replay_buffer = loaded_buffer
                            
                        if self.verbose:
                            self.logger.info(f"Loaded replay buffer from {replay_buffer_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not load replay buffer: {e}")
                else:
                    self.logger.warning(f"Replay buffer not found at {replay_buffer_path}")
            
            # Load training state
            training_state_path = os.path.join(checkpoint_path, 'training_state.pt')
            training_state = {}
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path, map_location=map_location)
                if self.verbose:
                    step = training_state.get('step', 0)
                    episode = training_state.get('episode', 0)
                    self.logger.info(f"Loaded training state: step={step}, episode={episode}")
            else:
                self.logger.warning(f"Training state not found at {training_state_path}")
            
            # Load environment state
            env_state_path = os.path.join(checkpoint_path, 'env_state.pt')
            if os.path.exists(env_state_path):
                env_state = torch.load(env_state_path, map_location=map_location)
                training_state['env_state'] = env_state
                if self.verbose:
                    self.logger.info(f"Loaded environment state from {env_state_path}")
            
            return training_state
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the latest checkpoint version.
        
        Returns:
            Latest checkpoint version or None if no checkpoints available
        """
        if not self.checkpoints:
            return None
            
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.get('step', 0), reverse=True)
        return sorted_checkpoints[0]['version']
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get the best checkpoint version.
        
        Returns:
            Best checkpoint version or None if no best checkpoint available
        """
        return self.best_checkpoint
    
    def list_checkpoints(self) -> List[Dict]:
        """
        List all available checkpoints with metadata.
        
        Returns:
            List of checkpoint dictionaries
        """
        return self.checkpoints
    
    def delete_checkpoint(self, version: str) -> bool:
        """
        Delete a specific checkpoint.
        
        Args:
            version: Checkpoint version to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, version)
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint {version} not found for deletion")
            return False
        
        try:
            # Remove directory
            shutil.rmtree(checkpoint_path)
            
            # Update checkpoint list
            self.checkpoints = [c for c in self.checkpoints if c.get('version') != version]
            
            # Update best checkpoint if needed
            if self.best_checkpoint == version:
                # Find new best checkpoint
                best_reward = -float('inf')
                self.best_checkpoint = None
                
                for checkpoint in self.checkpoints:
                    metrics = checkpoint.get('metrics', {})
                    if 'mean_reward' in metrics and metrics['mean_reward'] > best_reward:
                        best_reward = metrics['mean_reward']
                        self.best_checkpoint = checkpoint.get('version')
                self.best_reward = best_reward
            
            # Save metadata
            self._save_metadata()
            
            if self.verbose:
                self.logger.info(f"Deleted checkpoint {version}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error deleting checkpoint {version}: {e}")
            return False