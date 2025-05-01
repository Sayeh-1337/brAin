"""
Hyperdimensional Computing and Spiking Neural Network Agent

Core implementation of the brain-inspired agent that combines:
- Hyperdimensional computing for perception encoding
- Spiking neural networks for temporal processing and decision making
- Cellular automata for emergent pattern formation
- Episodic and semantic memory for learning and recall

Enhanced with biologically-inspired brain systems:
- Thalamic gating for sensory filtering
- Basal ganglia loop for action selection
- Cerebellar correction for motor error prediction
- Autonomic regulation for homeostatic control
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import time
import os
import pickle
import torch
import cv2

from brain.encoders.hdc_encoder import HDCEncoder
from brain.networks.snn import SpikingNeuralNetwork
from brain.networks.cellular_automata import CellularAutomata
from brain.memory.episodic import EpisodicMemory
from brain.memory.semantic import SemanticMemory
from brain.systems.thalamic_gating import ThalamicGating
from brain.systems.basal_ganglia import BasalGangliaLoop
from brain.systems.cerebellum import CerebellarCorrection
from brain.systems.autonomic import AutonomicSystem

class HDCSNNAgent:
    """
    Brain-inspired agent that integrates:
    - Visual cortex analog (HDC encoding for perception)
    - Basal ganglia / motor cortex analog (SNN for action selection)
    - Cortical sheet analog (Cellular automata for pattern processing)
    - Hippocampus analog (Episodic memory for experience storage)
    - Neocortex analog (Semantic memory for knowledge consolidation)
    - Visual Object Recognition analog (YOLO detector for object recognition)
    
    Enhanced with new components:
    - Thalamus analog (Thalamic gating for sensory filtering)
    - Basal ganglia loop (Direct/Indirect pathways for action selection)
    - Cerebellum analog (Motor error correction) 
    - Autonomic system (Homeostatic regulation)
    """
    
    def __init__(self, 
                 input_shape=(120, 160, 3),
                 hd_dim=1000,
                 snn_neurons=500,
                 num_actions=5,
                 ca_width=30,
                 ca_height=20,
                 memory_capacity=10000,
                 learning_rate=0.01,
                 use_yolo=False):
        """
        Initialize the agent components
        
        Args:
            input_shape: Shape of input observations
            hd_dim: Dimensionality of hyperdimensional vectors
            snn_neurons: Number of neurons in the SNN
            num_actions: Number of possible actions
            ca_width: Width of cellular automata grid
            ca_height: Height of cellular automata grid
            memory_capacity: Capacity of episodic memory
            learning_rate: Learning rate for the SNN
            use_yolo: Whether to use YOLO object detection
        """
        self.input_shape = input_shape
        self.hd_dim = hd_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.use_yolo = use_yolo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.hdc_encoder = HDCEncoder(D=hd_dim, use_yolo=use_yolo)
        self.snn = SpikingNeuralNetwork(
            input_size=hd_dim,
            num_neurons=snn_neurons,
            num_actions=num_actions
        )
        self.ca = CellularAutomata(
            width=ca_width,
            height=ca_height,
            state_levels=5
        )
        self.episodic_memory = EpisodicMemory(capacity=memory_capacity)
        self.semantic_memory = SemanticMemory(vector_dim=hd_dim)
        
        # Initialize new brain systems
        ca_size = ca_width * ca_height
        sensory_dim = max(hd_dim, ca_size)
        
        self.thalamic_gate = ThalamicGating(
            input_dim=sensory_dim,
            output_dim=sensory_dim,
            n_channels=4
        ).to(self.device)
        
        self.basal_ganglia = BasalGangliaLoop(
            input_size=hd_dim,
            action_size=num_actions,
            hidden_size=snn_neurons // 2
        ).to(self.device)
        
        self.cerebellum = CerebellarCorrection(
            input_size=hd_dim,
            output_size=num_actions
        ).to(self.device)
        
        self.autonomic = AutonomicSystem(n_drives=5).to(self.device)
        
        # Performance tracking
        self.episode_rewards = []
        self.recent_rewards = deque(maxlen=100)
        self.step_count = 0
        self.episode_count = 0
        
        # Visualization settings
        self.visualize_internals = False
        
        # YOLO visualizations
        self.show_yolo_detections = False
        self.last_frame = None
        self.last_detections = None
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        
        # Neuromodulator levels
        self.neuromodulators = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'norepinephrine': 0.5,
            'acetylcholine': 0.5
        }
        
    def act(self, observation, motion=None, deterministic=False):
        """
        Select an action based on the current observation
        
        Args:
            observation: Current environment observation
            motion: Optional motion information (frame difference)
            deterministic: Whether to act deterministically or explore
            
        Returns:
            Selected action index
        """
        # Preprocess and encode observation
        processed_obs = self._preprocess(observation)
        
        # Save current frame for visualization if needed
        if self.visualize_internals and self.show_yolo_detections:
            self.last_frame = processed_obs
        
        # Encode observation with HDC (and potentially YOLO)
        hd_vector = self.hdc_encoder.encode_observation(processed_obs, motion)
        
        if hd_vector is None:
            return random.randint(0, self.num_actions - 1)
        
        # Convert to torch tensor
        hd_tensor = torch.tensor(hd_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Update cellular automata with observation
        ca_input = self._observation_to_ca_input(processed_obs)
        self.ca.update(ca_input)
        
        # Combine CA features with HD vector
        ca_features = self.ca.extract_features()
        ca_features_tensor = torch.tensor(ca_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Apply thalamic gating for sensory filtering
        filtered_hd, salience = self.thalamic_gate(hd_tensor)
        
        # Update autonomic system and get neuromodulator levels
        _, urgency, neuromod_levels = self.autonomic(None)
        
        # Update agent's neuromodulator levels
        for key, value in neuromod_levels.items():
            self.neuromodulators[key] = value
        
        # Process through basal ganglia for action selection
        bg_output, bg_value = self.basal_ganglia(filtered_hd)
        
        # Also get SNN output for comparison
        snn_probs, _ = self.snn.simulate(hd_vector)
        snn_tensor = torch.tensor(snn_probs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Apply cerebellar correction
        _, predicted_error, corrected_command = self.cerebellum(filtered_hd, snn_tensor)
        
        # Determine final action
        if corrected_command is not None:
            action_probs = corrected_command.cpu().detach().numpy().flatten()
        else:
            # Use basal ganglia output
            action_probs = bg_output.cpu().detach().numpy().flatten()
        
        # Select action
        if deterministic:
            action = np.argmax(action_probs)
        else:
            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                action = random.randint(0, self.num_actions - 1)
            else:
                # Check semantic memory first
                semantic_action = self.semantic_memory.get_best_action(hd_vector, threshold=0.7)
                if semantic_action is not None:
                    action = semantic_action
                else:
                    # Otherwise use action probabilities
                    action = np.argmax(action_probs)
                
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        self.step_count += 1
        return action
        
    def learn(self, state, action, reward, next_state, done):
        """
        Update agent's knowledge based on experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Skip if state/next_state is None
        if state is None or next_state is None:
            return
            
        # Preprocess and encode states
        processed_state = self._preprocess(state)
        processed_next_state = self._preprocess(next_state)
        
        hd_state = self.hdc_encoder.encode_observation(processed_state)
        hd_next_state = self.hdc_encoder.encode_observation(processed_next_state)
        
        if hd_state is None or hd_next_state is None:
            return
            
        # Store in episodic memory
        self.episodic_memory.store(hd_state, action, reward, hd_next_state, done)
        
        # Store in semantic memory
        self.semantic_memory.store_experience(hd_state, action, reward, hd_next_state)
        
        # Create tensors for torch components
        hd_state_tensor = torch.tensor(hd_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Create target for SNN training
        target = np.zeros(self.num_actions)
        if reward > 0:
            target[action] = 0.9
        elif reward < 0:
            target[action] = 0.1
        else:
            target[action] = 0.5
        
        # Convert to tensor for cerebellum
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Train the SNN normally
        self.snn.train(hd_state, target, learning_rate=self.learning_rate)
        
        # Update basal ganglia with reward
        self.basal_ganglia.update_dopamine(reward)
        
        # Update cerebellum with error (difference between target and actual output)
        action_tensor = torch.zeros(1, self.num_actions, device=self.device)
        action_tensor[0, action] = 1.0
        
        # Calculate error for cerebellar learning
        # Higher error for unexpected outcomes
        if reward > 0 and target[action] < 0.8:  # Unexpected positive reward
            error = torch.ones(1, self.num_actions, device=self.device) * 0.1
            error[0, action] = -0.8  # Negative error (action was better than expected)
        elif reward < 0 and target[action] > 0.2:  # Unexpected negative reward
            error = torch.zeros(1, self.num_actions, device=self.device)
            error[0, action] = 0.8  # Positive error (action was worse than expected)
        else:
            # Expected outcome, small error
            error = torch.zeros(1, self.num_actions, device=self.device)
            error[0, action] = 0.1 * -np.sign(reward)
        
        # Update cerebellar model
        self.cerebellum.update_error(
            context=hd_state_tensor,
            command=action_tensor,
            observed_error=error
        )
        
        # Update thalamic attention based on reward
        novelty = 0.3  # TODO: Calculate actual novelty from episodic memory
        self.thalamic_gate.update_attention(reward=reward, novelty=novelty)
        
        # Update autonomic system
        self.autonomic.update_drives(
            rewards=reward,
            actions=np.eye(self.num_actions)[action]  # One-hot encoding of action
        )
            
    def replay_experience(self, batch_size=32):
        """
        Replay past experiences for learning
        
        Args:
            batch_size: Number of experiences to replay
        """
        # Skip if not enough memories
        if len(self.episodic_memory) < batch_size:
            return
            
        # Sample experiences
        experiences = self.episodic_memory.sample(batch_size)
        
        for state, action, reward, next_state, done in experiences:
            # Create target based on reward
            target = np.zeros(self.num_actions)
            if reward > 0:
                target[action] = 0.9
            elif reward < 0:
                target[action] = 0.1
            else:
                target[action] = 0.5
                
            # Train SNN on this experience
            self.snn.train(state, target, learning_rate=self.learning_rate * 0.5)
            
            # Also update new components with a small learning rate
            # Convert to tensors
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Update basal ganglia with smaller learning rate
            self.basal_ganglia.update_dopamine(reward * 0.3)
            
            # Simplified cerebellar update
            if reward != 0:  # Only learn from rewarded experiences
                action_tensor = torch.zeros(1, self.num_actions, device=self.device)
                action_tensor[0, action] = 1.0
                
                error = torch.zeros(1, self.num_actions, device=self.device)
                error[0, action] = -0.1 * np.sign(reward)  # Small error signal
                
                self.cerebellum.update_error(
                    context=state_tensor,
                    command=action_tensor,
                    observed_error=error,
                    learning_rate=0.005  # Very small learning rate for replays
                )
            
    def _preprocess(self, observation):
        """
        Preprocess observation for encoding
        """
        # Simple normalization and type conversion
        return observation.astype(np.float32) / 255.0
        
    def _observation_to_ca_input(self, obs):
        """
        Convert observation to input for cellular automata
        
        Args:
            obs: Normalized observation
            
        Returns:
            numpy.ndarray: Resized and normalized grid for CA
        """
        # Get dimensions
        grid_height = self.ca.height
        grid_width = self.ca.width
        
        # Simple downsampling (average pooling)
        ca_input = np.zeros((grid_height, grid_width))
        
        obs_height, obs_width = obs.shape[:2]
        
        # Compute scaling factors
        h_scale = obs_height / grid_height
        w_scale = obs_width / grid_width
        
        # Average pooling
        for i in range(grid_height):
            for j in range(grid_width):
                # Calculate the corresponding region in the observation
                start_h = int(i * h_scale)
                end_h = int((i + 1) * h_scale)
                start_w = int(j * w_scale)
                end_w = int((j + 1) * w_scale)
                
                # Average the region
                region = obs[start_h:end_h, start_w:end_w]
                ca_input[i, j] = np.mean(region)
                
        return ca_input

    def save(self, filename):
        """
        Save the agent's state
        
        Args:
            filename: Path to save the agent state
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create state dict for torch components
        torch_state = {
            'thalamic_gate': self.thalamic_gate.state_dict(),
            'basal_ganglia': self.basal_ganglia.state_dict(),
            'cerebellum': self.cerebellum.state_dict(),
            'autonomic': self.autonomic.state_dict()
        }
        
        # Save torch components separately
        torch.save(torch_state, filename + '.torch')
        
        # Components that use pickle
        pickle_state = {
            'hdc_encoder': self.hdc_encoder,
            'snn': self.snn,
            'ca': self.ca,
            'episodic_memory': self.episodic_memory,
            'semantic_memory': self.semantic_memory,
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'episode_rewards': self.episode_rewards,
            'recent_rewards': list(self.recent_rewards),
            'neuromodulators': self.neuromodulators
        }
        
        # Save pickle components
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(pickle_state, f)
            
        print(f"Agent saved to {filename}.torch and {filename}.pkl")
        
    def load(self, filename):
        """
        Load the agent's state
        
        Args:
            filename: Path to load the agent state from
        """
        # Load torch components
        if os.path.exists(filename + '.torch'):
            torch_state = torch.load(filename + '.torch', map_location=self.device)
            self.thalamic_gate.load_state_dict(torch_state['thalamic_gate'])
            self.basal_ganglia.load_state_dict(torch_state['basal_ganglia'])
            self.cerebellum.load_state_dict(torch_state['cerebellum'])
            self.autonomic.load_state_dict(torch_state['autonomic'])
        else:
            print(f"Warning: {filename}.torch not found, torch components not loaded")
        
        # Load pickle components
        if os.path.exists(filename + '.pkl'):
            with open(filename + '.pkl', 'rb') as f:
                pickle_state = pickle.load(f)
                
            self.hdc_encoder = pickle_state['hdc_encoder']
            self.snn = pickle_state['snn']
            self.ca = pickle_state['ca']
            self.episodic_memory = pickle_state['episodic_memory']
            self.semantic_memory = pickle_state['semantic_memory']
            self.epsilon = pickle_state['epsilon']
            self.step_count = pickle_state['step_count']
            self.episode_count = pickle_state['episode_count']
            self.episode_rewards = pickle_state['episode_rewards']
            self.recent_rewards = deque(pickle_state['recent_rewards'], maxlen=100)
            
            if 'neuromodulators' in pickle_state:
                self.neuromodulators = pickle_state['neuromodulators']
                
            print(f"Agent loaded from {filename}.pkl")
        else:
            print(f"Warning: {filename}.pkl not found, pickle components not loaded")
        
    def visualize(self, observation=None):
        """
        Visualize internal representations
        
        Args:
            observation: Optional current observation to visualize
        """
        if not self.visualize_internals:
            return
            
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        # Flatten axes for easier indexing
        axs = axs.flatten()
        
        # 1. Show current observation if provided
        if observation is not None:
            processed_obs = self._preprocess(observation)
            axs[0].imshow(processed_obs)
            axs[0].set_title("Current Observation")
            axs[0].axis('off')
            
            # Save for YOLO visualization
            self.last_frame = processed_obs
            
        elif self.last_frame is not None:
            axs[0].imshow(self.last_frame)
            axs[0].set_title("Last Observation")
            axs[0].axis('off')
        else:
            axs[0].set_title("No Observation Available")
            axs[0].axis('off')
            
        # 2. Visualize YOLO detections if enabled
        if self.show_yolo_detections and self.use_yolo and hasattr(self.hdc_encoder, 'last_detections'):
            if self.last_frame is not None and self.hdc_encoder.last_detections is not None:
                # Draw detection boxes
                detection_img = self.last_frame.copy()
                for det in self.hdc_encoder.last_detections:
                    x1, y1, x2, y2 = [int(val) for val in det[:4]]
                    confidence = det[4]
                    class_id = int(det[5])
                    
                    # Draw rectangle
                    color = (0, 1, 0)  # Green
                    cv2_rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                          linewidth=2, edgecolor=color, 
                                          facecolor='none')
                    axs[1].add_patch(cv2_rect)
                    
                    # Add label
                    label = f"Class {class_id}: {confidence:.2f}"
                    axs[1].text(x1, y1-5, label, color=color)
                    
                axs[1].imshow(detection_img)
                axs[1].set_title("YOLO Detections")
                axs[1].axis('off')
            else:
                axs[1].set_title("No YOLO Detections")
                axs[1].axis('off')
        else:
            # 2. Alternative: Show cellular automata state
            ca_state = self.ca.get_grid()
            axs[1].imshow(ca_state, cmap='viridis')
            axs[1].set_title("Cellular Automata State")
            axs[1].axis('off')
            
        # 3. Visualize SNN activity
        if hasattr(self.snn, 'neuron_activity'):
            activity = np.array(self.snn.neuron_activity)
            if activity.size > 0:
                # Reshape to 2D grid if possible
                size = activity.shape[0]
                grid_size = int(np.sqrt(size))
                if grid_size**2 == size:
                    activity = activity.reshape(grid_size, grid_size)
                    axs[2].imshow(activity, cmap='hot')
                else:
                    # Plot as 1D heatmap
                    axs[2].imshow(activity.reshape(1, -1), cmap='hot', aspect='auto')
                axs[2].set_title("SNN Neuron Activity")
                axs[2].axis('off')
        else:
            axs[2].set_title("No SNN Activity Data")
            axs[2].axis('off')
            
        # 4. Visualize action probabilities from both sources
        if hasattr(self.snn, 'last_output'):
            # SNN action probabilities
            snn_probs = self.snn.last_output
            if snn_probs is not None:
                axs[3].bar(range(len(snn_probs)), snn_probs, alpha=0.7, label='SNN')
                
                # Also show basal ganglia output if available
                if hasattr(self.basal_ganglia, 'd1_output'):
                    bg_probs = self.basal_ganglia.d1_output.cpu().detach().numpy().flatten()
                    axs[3].bar(range(len(bg_probs)), bg_probs, alpha=0.5, label='BG')
                
                axs[3].set_title("Action Probabilities")
                axs[3].set_xlabel("Action Index")
                axs[3].set_ylabel("Probability")
                axs[3].legend()
        else:
            axs[3].set_title("No Action Probability Data")
            
        # 5. Visualize episodic memory statistics
        if hasattr(self.episodic_memory, 'size'):
            # Memory usage
            memory_usage = len(self.episodic_memory) / self.episodic_memory.capacity
            axs[4].bar(['Memory Usage'], [memory_usage])
            axs[4].set_ylim(0, 1)
            axs[4].set_title(f"Episodic Memory: {len(self.episodic_memory)} / {self.episodic_memory.capacity}")
            
        else:
            axs[4].set_title("No Memory Statistics")
            
        # 6. Visualize neuromodulator levels
        if hasattr(self, 'neuromodulators'):
            labels = list(self.neuromodulators.keys())
            values = list(self.neuromodulators.values())
            axs[5].bar(labels, values)
            axs[5].set_ylim(0, 1)
            axs[5].set_title("Neuromodulator Levels")
            
            # Add autonomic drive values
            if hasattr(self.autonomic, 'drive_values'):
                drive_values = self.autonomic.drive_values.cpu().detach().numpy()
                drive_names = self.autonomic.drive_names
                
                # Create a twin axis
                ax2 = axs[5].twinx()
                ax2.bar(drive_names, drive_values, color='lightgreen', alpha=0.5)
                ax2.set_ylim(0, 1)
                ax2.set_ylabel('Drive Levels', color='green')
        else:
            axs[5].set_title("No Neuromodulator Data")
            
        # Show the figure
        plt.tight_layout()
        plt.show()
        
    def reset(self):
        """Reset agent state between episodes"""
        # Reset neural components
        self.snn.reset()
        self.ca.reset()
        
        # Reset new components
        self.basal_ganglia.dopamine_factor.data = torch.tensor(1.0)
        
        self.episode_count += 1
        
    def get_metrics(self):
        """Get current agent metrics"""
        metrics = {
            'epsilon': self.epsilon,
            'memory_usage': len(self.episodic_memory) / self.episodic_memory.capacity 
                if hasattr(self.episodic_memory, 'capacity') else 0,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'avg_reward': sum(self.recent_rewards) / max(1, len(self.recent_rewards)),
        }
        
        # Add neuromodulator levels
        for key, value in self.neuromodulators.items():
            metrics[f'neuromod_{key}'] = value
            
        # Add autonomic drive levels if available
        if hasattr(self.autonomic, 'drive_values'):
            drive_values = self.autonomic.drive_values.cpu().detach().numpy()
            drive_names = self.autonomic.drive_names
            for i, name in enumerate(drive_names):
                metrics[f'drive_{name}'] = drive_values[i]
                
        return metrics 