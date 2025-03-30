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
    """
    
    def __init__(self, 
                 input_shape=(120, 160, 3),
                 hd_dim=1000,
                 snn_neurons=500,
                 num_actions=5,
                 ca_width=30,
                 ca_height=20,
                 memory_capacity=10000,
                 learning_rate=0.01):
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
        """
        self.input_shape = input_shape
        self.hd_dim = hd_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        
        # Initialize components
        self.hdc_encoder = HDCEncoder(D=hd_dim)
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
        Save agent state
        
        Args:
            filename: Base filename to save agent components
        """
        # TODO: Implement save functionality
        pass
        
    def load(self, filename):
        """
        Load agent state
        
        Args:
            filename: Base filename to load agent components
        """
        # TODO: Implement load functionality
        pass
        
    def visualize(self, observation=None):
        """
        Visualize agent's internal state
        
        Args:
            observation: Current observation
        """
        if not self.visualize_internals:
            return
            
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