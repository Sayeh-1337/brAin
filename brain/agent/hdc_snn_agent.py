"""
Hyperdimensional Computing and Spiking Neural Network Agent

Core implementation of the brain-inspired agent that combines:
- Hyperdimensional computing for perception encoding
- Spiking neural networks for temporal processing and decision making
- Cellular automata for emergent pattern formation
- Episodic and semantic memory for learning and recall
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import time
import os
import pickle

from brain.encoders.hdc_encoder import HDCEncoder
from brain.networks.snn import SpikingNeuralNetwork
from brain.networks.cellular_automata import CellularAutomata
from brain.memory.episodic import EpisodicMemory
from brain.memory.semantic import SemanticMemory

class HDCSNNAgent:
    """
    Brain-inspired agent that integrates:
    - Visual cortex analog (HDC encoding for perception)
    - Basal ganglia / motor cortex analog (SNN for action selection)
    - Cortical sheet analog (Cellular automata for pattern processing)
    - Hippocampus analog (Episodic memory for experience storage)
    - Neocortex analog (Semantic memory for knowledge consolidation)
    - Visual Object Recognition analog (YOLO detector for object recognition)
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
        
        # Update cellular automata with observation
        ca_input = self._observation_to_ca_input(processed_obs)
        self.ca.update(ca_input)
        
        # Combine CA features with HD vector
        ca_features = self.ca.extract_features()
        ca_features_scaled = 2 * ca_features - 1  # Convert [0,1] to [-1,1]
        
        # Create combined representation
        combined_vector = hd_vector
        
        # Query semantic memory for similar situations
        best_action = self.semantic_memory.get_best_action(combined_vector, threshold=0.7)
        
        # Run SNN to get action probabilities
        action_probs, _ = self.snn.simulate(combined_vector)
        
        # Select action
        if deterministic:
            action = np.argmax(action_probs)
        else:
            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                action = random.randint(0, self.num_actions - 1)
            else:
                # Use semantic memory if available, otherwise use SNN
                action = best_action if best_action is not None else np.argmax(action_probs)
                
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
        
        # Create target for SNN training
        target = np.zeros(self.num_actions)
        
        # Higher values for actions with positive rewards
        if reward > 0:
            target[action] = 0.9
        elif reward < 0:
            target[action] = 0.1
        else:
            target[action] = 0.5
            
        # Update SNN weights
        self.snn.update_weights(hd_state, target, learning_rate=self.learning_rate)
        
        # Performance tracking
        if done:
            episode_reward = sum(self.recent_rewards)
            self.episode_rewards.append(episode_reward)
            self.recent_rewards.clear()
            self.episode_count += 1
            
            # Reset SNN state at episode end
            self.snn.reset_state()
            self.ca.reset()
        else:
            self.recent_rewards.append(reward)
            
    def _preprocess(self, observation):
        """Preprocess observation before encoding"""
        # Ensure observation is in proper format
        if observation is None:
            return None
            
        # Return the observation directly if already in correct format
        return observation
        
    def _observation_to_ca_input(self, obs):
        """Convert observation to cellular automata input format"""
        # Simple conversion: grayscale and downscale to CA dimensions
        if obs is None:
            return None
            
        # Convert to grayscale
        gray = np.mean(obs, axis=2)
        
        # Resize to CA dimensions
        h_scale = self.ca.height / gray.shape[0]
        w_scale = self.ca.width / gray.shape[1]
        
        ca_input = np.zeros((self.ca.height, self.ca.width))
        
        for y in range(self.ca.height):
            for x in range(self.ca.width):
                orig_y = min(int(y / h_scale), gray.shape[0] - 1)
                orig_x = min(int(x / w_scale), gray.shape[1] - 1)
                ca_input[y, x] = int(gray[orig_y, orig_x] / 51)  # Scale to 0-5 range
                
        return ca_input
        
    def replay_experience(self, batch_size=32):
        """
        Learn from past experiences using replay
        
        Args:
            batch_size: Number of experiences to replay
        """
        # Skip if not enough experiences
        if len(self.episodic_memory) < batch_size:
            return
            
        # Get experiences with priority on high rewards
        experiences = self.episodic_memory.prioritized_sample(batch_size)
        
        for state, action, reward, next_state, done in experiences:
            # Create target for SNN training
            target = np.zeros(self.num_actions)
            
            # Higher values for actions with positive rewards
            if reward > 0:
                target[action] = 0.9
            elif reward < 0:
                target[action] = 0.1
            else:
                target[action] = 0.5
                
            # Update SNN weights with lower learning rate for replay
            self.snn.update_weights(state, target, learning_rate=self.learning_rate * 0.5)
            
    def save(self, filename):
        """
        Save agent state to a file
        
        Args:
            filename: Base filename to save agent components
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save agent state
        state = {
            'input_shape': self.input_shape,
            'hd_dim': self.hd_dim,
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'use_yolo': self.use_yolo,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'episode_rewards': self.episode_rewards,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }
        
        # Save components
        components = {
            'hdc_encoder': self.hdc_encoder,
            'snn': self.snn,
            'ca': self.ca,
            'episodic_memory': self.episodic_memory,
            'semantic_memory': self.semantic_memory
        }
        
        # Save state and components
        with open(f"{filename}_state.pkl", 'wb') as f:
            pickle.dump(state, f)
            
        with open(f"{filename}_components.pkl", 'wb') as f:
            pickle.dump(components, f)
            
        print(f"Saved agent state to {filename}_state.pkl")
        print(f"Saved agent components to {filename}_components.pkl")
        
    def load(self, filename):
        """
        Load agent state from a file
        
        Args:
            filename: Base filename to load agent components
        """
        # Load state
        with open(f"{filename}_state.pkl", 'rb') as f:
            state = pickle.load(f)
            
        # Load components
        with open(f"{filename}_components.pkl", 'rb') as f:
            components = pickle.load(f)
            
        # Restore state
        self.input_shape = state['input_shape']
        self.hd_dim = state['hd_dim']
        self.num_actions = state['num_actions']
        self.learning_rate = state['learning_rate']
        self.use_yolo = state['use_yolo']
        self.epsilon = state['epsilon']
        self.epsilon_min = state['epsilon_min']
        self.epsilon_decay = state['epsilon_decay']
        self.episode_rewards = state['episode_rewards']
        self.step_count = state['step_count']
        self.episode_count = state['episode_count']
        
        # Restore components
        self.hdc_encoder = components['hdc_encoder']
        self.snn = components['snn']
        self.ca = components['ca']
        self.episodic_memory = components['episodic_memory']
        self.semantic_memory = components['semantic_memory']
        
        print(f"Loaded agent state from {filename}_state.pkl")
        print(f"Loaded agent components from {filename}_components.pkl")
        
    def visualize(self, observation=None):
        """
        Visualize agent's internal state
        
        Args:
            observation: Current observation
        """
        if not self.visualize_internals:
            return
            
        # Create figure for visualization
        if self.show_yolo_detections and self.use_yolo and hasattr(self.hdc_encoder, 'yolo_detector'):
            # 3x2 layout with YOLO detection visualization
            fig = plt.figure(figsize=(18, 12))
            
            # Plot current observation
            if observation is not None:
                ax1 = fig.add_subplot(2, 3, 1)
                ax1.imshow(observation)
                ax1.set_title("Current Observation")
                ax1.axis('off')
                
            # Plot YOLO detections if available
            ax2 = fig.add_subplot(2, 3, 2)
            if self.last_frame is not None and hasattr(self.hdc_encoder.yolo_detector, 'visualize_detections'):
                # Get detections
                try:
                    detections = self.hdc_encoder.yolo_detector.detect(self.last_frame)
                    # Create visualization
                    vis_frame = self.hdc_encoder.yolo_detector.visualize_detections(self.last_frame, detections)
                    ax2.imshow(vis_frame)
                    ax2.set_title(f"YOLO Detections: {len(detections)} objects")
                except Exception as e:
                    ax2.text(0.5, 0.5, f"YOLO error: {str(e)}", ha='center', va='center')
            else:
                ax2.text(0.5, 0.5, "YOLO not available", ha='center', va='center')
            ax2.axis('off')
            
            # Plot cellular automata state
            ax3 = fig.add_subplot(2, 3, 3)
            self.ca.visualize(ax=ax3)
            
            # Plot attention map if available
            ax4 = fig.add_subplot(2, 3, 4)
            if self.last_frame is not None and hasattr(self.hdc_encoder.yolo_detector, 'create_attention_map'):
                try:
                    detections = self.hdc_encoder.yolo_detector.detect(self.last_frame)
                    attention = self.hdc_encoder.yolo_detector.create_attention_map(self.last_frame, detections)
                    ax4.imshow(attention, cmap='hot')
                    ax4.set_title("Attention Map")
                except Exception as e:
                    ax4.text(0.5, 0.5, f"Attention error: {str(e)}", ha='center', va='center')
            else:
                ax4.text(0.5, 0.5, "Attention map not available", ha='center', va='center')
            ax4.axis('off')
            
            # Plot SNN activations
            ax5 = fig.add_subplot(2, 3, 5)
            ax5.bar(range(len(self.snn.last_activations)), self.snn.last_activations)
            ax5.set_title("SNN Neuron Activations")
            ax5.set_xlabel("Neuron")
            ax5.set_ylabel("Activation")
            
            # Plot action probabilities
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.bar(range(len(self.snn.last_output)), self.snn.last_output)
            ax6.set_title("Action Probabilities")
            ax6.set_xlabel("Action")
            ax6.set_ylabel("Probability")
            ax6.set_xticks(range(self.num_actions))
            ax6.set_xticklabels(["FWD", "RIGHT", "LEFT", "ATTACK", "NONE"])
        else:
            # Standard 2x2 layout
            fig = plt.figure(figsize=(15, 10))
            
            # Plot current observation
            if observation is not None:
                ax1 = fig.add_subplot(2, 2, 1)
                ax1.imshow(observation)
                ax1.set_title("Current Observation")
                ax1.axis('off')
                
            # Plot cellular automata state
            ax2 = fig.add_subplot(2, 2, 2)
            self.ca.visualize(ax=ax2)
            
            # Plot SNN activations
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.bar(range(len(self.snn.last_activations)), self.snn.last_activations)
            ax3.set_title("SNN Neuron Activations")
            ax3.set_xlabel("Neuron")
            ax3.set_ylabel("Activation")
            
            # Plot action probabilities
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.bar(range(len(self.snn.last_output)), self.snn.last_output)
            ax4.set_title("Action Probabilities")
            ax4.set_xlabel("Action")
            ax4.set_ylabel("Probability")
            ax4.set_xticks(range(self.num_actions))
            ax4.set_xticklabels(["FWD", "RIGHT", "LEFT", "ATTACK", "NONE"])
        
        plt.tight_layout()
        plt.pause(0.01)
        plt.close()
        
    def reset(self):
        """Reset agent state between episodes"""
        self.snn.reset_state()
        self.ca.reset()
        self.recent_rewards.clear()
        
    def get_metrics(self):
        """
        Get agent performance metrics
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "memory_size": len(self.episodic_memory),
            "epsilon": self.epsilon
        }
        
        # Add episode rewards if available
        if self.episode_rewards:
            metrics["last_episode_reward"] = self.episode_rewards[-1]
            metrics["average_reward"] = np.mean(self.episode_rewards[-100:])
            
        return metrics 