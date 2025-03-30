"""
VizDoom Environment Wrapper

Provides an interface for interacting with VizDoom environments,
including observation processing and action handling.
"""

import os
import numpy as np
import vizdoom as vzd
from skimage.transform import resize
import matplotlib.pyplot as plt

class DoomEnvironment:
    """
    Environment wrapper for VizDoom
    
    Provides a standardized interface for the agent to interact with
    VizDoom environments, including observation processing, action handling,
    and scenario configuration.
    """
    
    # Available actions
    ACTIONS = [
        [1, 0, 0, 0],  # MOVE_FORWARD
        [0, 1, 0, 0],  # MOVE_RIGHT
        [0, 0, 1, 0],  # MOVE_LEFT
        [0, 0, 0, 1],  # ATTACK
        [0, 0, 0, 0],  # NO_ACTION
    ]
    
    ACTION_NAMES = ["FORWARD", "RIGHT", "LEFT", "ATTACK", "NONE"]
    
    def __init__(self, scenario="basic", frame_skip=4, visible=False):
        """
        Initialize VizDoom environment
        
        Args:
            scenario: Name of scenario to load ('basic', 'deadly_corridor', etc.)
            frame_skip: Number of frames to skip between observations
            visible: Whether to render the game window
        """
        self.scenario = scenario
        self.frame_skip = frame_skip
        self.visible = visible
        
        # Path to VizDoom scenarios
        self.scenarios_path = os.path.join(os.path.dirname(vzd.__file__), "scenarios")
        
        # Create the VizDoom environment
        self.game = vzd.DoomGame()
        
        # Set window visibility
        self.game.set_window_visible(visible)
        
        # Initialize the game
        self._setup_game()
        
        # Store the screen size
        self.screen_height = self.game.get_screen_height()
        self.screen_width = self.game.get_screen_width()
        
        # Pre-process observation dimensions
        self.observation_shape = (120, 160, 3)  # default resized observation
        
        # Last frames for frame stacking / motion detection
        self.last_frames = []
        
    def _setup_game(self):
        """Setup the VizDoom game with the specified scenario"""
        # Set common settings
        self.game.set_doom_scenario_path(os.path.join(self.scenarios_path, f"{self.scenario}.wad"))
        self.game.set_doom_map("map01")
        
        # Configure game settings
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        
        # Set available buttons
        self.game.set_available_buttons([
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.MOVE_LEFT,
            vzd.Button.ATTACK
        ])
        
        # Set additional game variables
        self.game.set_available_game_variables([
            vzd.GameVariable.AMMO2,
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.KILLCOUNT
        ])
        
        # Max episode length
        self.game.set_episode_timeout(2000)
        
        # Disable auto-shooting
        self.game.set_episode_start_time(10)
        self.game.set_sound_enabled(False)
        
        # Initialize the game
        self.game.init()
        
    def reset(self):
        """
        Reset the environment
        
        Returns:
            Initial observation
        """
        self.game.new_episode()
        self.last_frames = []
        
        # Get initial observation
        raw_obs = self.game.get_state().screen_buffer
        processed_obs = self._preprocess_frame(raw_obs)
        
        # Initialize frame history with copied first frame
        for _ in range(4):  # Stack 4 initial frames
            self.last_frames.append(processed_obs.copy())
            
        return self._get_observation()
        
    def step(self, action_idx):
        """
        Take a step in the environment
        
        Args:
            action_idx: Index of the action to take
            
        Returns:
            (observation, reward, done, info) tuple
        """
        # Convert action index to VizDoom action
        action = self.ACTIONS[action_idx]
        
        # Take action in environment with frame skipping
        reward = self.game.make_action(action, self.frame_skip)
        
        # Check if episode is done
        done = self.game.is_episode_finished()
        
        # Get new observation if not done
        if not done:
            raw_obs = self.game.get_state().screen_buffer
            processed_obs = self._preprocess_frame(raw_obs)
            self.last_frames.append(processed_obs)
            
            # Keep only last 4 frames
            if len(self.last_frames) > 4:
                self.last_frames.pop(0)
                
            # Get game variables
            game_vars = {
                "health": self.game.get_game_variable(vzd.GameVariable.HEALTH),
                "ammo": self.game.get_game_variable(vzd.GameVariable.AMMO2),
                "kills": self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            }
        else:
            # If episode is done, use the last observation
            game_vars = {}
            
        # Return SARS tuple
        return self._get_observation(), reward, done, game_vars
        
    def _preprocess_frame(self, frame):
        """Preprocess raw frame from the game"""
        # Resize frame
        resized = resize(frame, self.observation_shape, anti_aliasing=True)
        
        # Convert to range [0, 255] and uint8 type
        return (resized * 255).astype(np.uint8)
        
    def _get_observation(self):
        """
        Create observation from frame stack
        
        Returns:
            Processed observation with shape (120, 160, 3)
        """
        if not self.last_frames:
            # Return zeros if no frames yet
            return np.zeros(self.observation_shape, dtype=np.uint8)
            
        # Return the latest frame
        return self.last_frames[-1]
        
    def get_frame_stack(self):
        """
        Get stack of the last 4 frames
        
        Returns:
            Stacked frames with shape (120, 160, 12)
        """
        if len(self.last_frames) < 4:
            # Pad with copies of first frame if needed
            frames = [self.last_frames[0]] * (4 - len(self.last_frames)) + self.last_frames
        else:
            frames = self.last_frames
            
        # Stack along channel dimension
        return np.concatenate(frames, axis=2)
        
    def get_motion_frames(self):
        """
        Calculate motion between last frames
        
        Returns:
            Motion frame representing differences
        """
        if len(self.last_frames) < 2:
            return np.zeros(self.observation_shape, dtype=np.uint8)
            
        # Calculate absolute difference between latest frames
        frame1 = self.last_frames[-2].astype(np.int16)
        frame2 = self.last_frames[-1].astype(np.int16)
        
        # Compute absolute difference and scale
        diff = np.abs(frame2 - frame1)
        
        return diff.astype(np.uint8)
        
    def close(self):
        """Close the VizDoom environment"""
        self.game.close()
        
    def render(self, mode='human'):
        """
        Render the current state
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            Rendered frame if mode is 'rgb_array', else None
        """
        if not self.visible:
            self.game.set_window_visible(True)
            self.visible = True
            
        if mode == 'rgb_array' and not self.game.is_episode_finished():
            frame = self.game.get_state().screen_buffer
            return frame
        return None
        
    def visualize_observation(self, observation=None):
        """
        Visualize the current observation
        
        Args:
            observation: Optional observation to visualize
                         If None, uses the latest observation
        """
        if observation is None:
            observation = self._get_observation()
            
        plt.figure(figsize=(10, 8))
        plt.imshow(observation)
        plt.title(f'Doom Observation - {self.scenario}')
        plt.axis('off')
        plt.show()
        
    def get_action_names(self):
        """Get list of action names"""
        return self.ACTION_NAMES
        
    def get_action_space_size(self):
        """Get size of action space"""
        return len(self.ACTIONS) 