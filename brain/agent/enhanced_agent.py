"""
Enhanced Agent that combines Hyperdimensional Computing with BrainCog components
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from collections import deque
import random
import time

# Add BrainCog to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Brain-Cog'))

# Import from BrainCog
from braincog.base.node import LIFNode
from braincog.base.connection import CustomLinear

# Import our own components
from brain.networks.snn_braincog import BrainCogSNN
from brain.systems.basal_ganglia_braincog import BasalGangliaBrainCog
from brain.utils.config import config

class EnhancedAgent:
    """
    Enhanced Agent combining HDC with BrainCog components
    
    Features:
    - Biologically plausible SNN based on BrainCog's LIF neurons
    - Basal ganglia model with direct/indirect pathways
    - Hyperdimensional computing for robust representation
    - Episodic memory for experience storage
    - Multiple learning mechanisms (STDP, RL, HDC binding)
    """
    
    def __init__(self,
                 input_shape=(120, 160, 3),
                 hd_dim=1000,
                 snn_hidden=500,
                 num_actions=5,
                 memory_capacity=10000,
                 learning_rate=0.01,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=0.995,
                 use_yolo=False,
                 device=None):
        """
        Initialize the enhanced agent
        
        Args:
            input_shape: Shape of input observations (height, width, channels)
            hd_dim: Dimension of hypervectors
            snn_hidden: Number of hidden neurons in SNN
            num_actions: Number of possible actions
            memory_capacity: Capacity of episodic memory
            learning_rate: Learning rate
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of epsilon decay
            use_yolo: Whether to use YOLO for enhanced perception
            device: Computing device (cpu/cuda)
        """
        # Set device
        self.device = device if device is not None else config.device
        
        # Store parameters
        self.input_shape = input_shape
        self.hd_dim = hd_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_yolo = use_yolo
        
        # Flattened input size
        self.input_size = input_shape[0] * input_shape[1] * input_shape[2]
        self.compressed_input_size = 256  # Compressed input representation size
        
        # Initialize episodic memory
        self.memory = deque(maxlen=memory_capacity)
        self.memory_capacity = memory_capacity
        
        # Initialize step counter
        self.steps = 0
        
        # Initialize networks
        self._init_networks(snn_hidden)
        
        # Initialize HD vectors
        self._init_hd_vectors()
        
        # Initialize YOLO model if needed
        if self.use_yolo:
            self._init_yolo()
            
        # Visualization flag
        self.visualize_internals = False
        self.show_yolo_detections = False
        
        # For timing visualization
        self.forward_times = []
        self.learning_times = []
        
        # Initialize neuromodulators
        self.neuromodulators = {
            'dopamine': 0.5,  # Reward prediction
            'serotonin': 0.5,  # Mood, behavioral inhibition
            'noradrenaline': 0.5,  # Arousal, attention
            'acetylcholine': 0.5  # Learning, memory
        }
    
    def _init_networks(self, snn_hidden):
        """Initialize neural networks"""
        # Input compression network
        self.input_compression = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.compressed_input_size),
            nn.ReLU()
        ).to(self.device)
        
        # Enhanced SNN using BrainCog
        self.snn = BrainCogSNN(
            input_size=self.compressed_input_size,
            hidden_size=snn_hidden,
            output_size=self.compressed_input_size,
            learning_rate=self.learning_rate,
            device=self.device
        )
        
        # Basal ganglia for action selection
        self.basal_ganglia = BasalGangliaBrainCog(
            input_size=self.compressed_input_size,
            num_actions=self.num_actions,
            d1_learning_rate=self.learning_rate,
            d2_learning_rate=self.learning_rate * 0.5,
            device=self.device
        )
        
        # Q-network for value estimation
        self.q_network = nn.Sequential(
            nn.Linear(self.compressed_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        ).to(self.device)
        
        # Target Q-network
        self.target_q_network = nn.Sequential(
            nn.Linear(self.compressed_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        ).to(self.device)
        
        # Copy weights from Q-network to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Q-network optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
    
    def _init_hd_vectors(self):
        """Initialize hyperdimensional vectors"""
        # Create random item memory (dictionary of random hypervectors)
        self.item_memory = torch.randn(self.compressed_input_size, self.hd_dim, device=self.device)
        self.item_memory = self.item_memory / torch.norm(self.item_memory, dim=1, keepdim=True)
        
        # Create action hypervectors
        self.action_vectors = torch.randn(self.num_actions, self.hd_dim, device=self.device)
        self.action_vectors = self.action_vectors / torch.norm(self.action_vectors, dim=1, keepdim=True)
        
        # Create value hypervectors (for different reward levels)
        num_value_levels = 21  # -1.0 to 1.0 in steps of 0.1
        self.value_vectors = torch.randn(num_value_levels, self.hd_dim, device=self.device)
        self.value_vectors = self.value_vectors / torch.norm(self.value_vectors, dim=1, keepdim=True)
        
        # Create memory hypervectors for episodic memory
        self.memory_vectors = []
    
    def _init_yolo(self):
        """Initialize YOLO model for object detection"""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
        except ImportError:
            print("Warning: ultralytics package not found. YOLO detection disabled.")
            self.use_yolo = False
    
    def reset(self):
        """Reset agent state for a new episode"""
        # Reset SNN state
        self.snn.reset_state()
        
        # Reset basal ganglia
        self.basal_ganglia.reset()
        
        # Reset neuromodulators to baseline
        self.neuromodulators = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'noradrenaline': 0.5,
            'acetylcholine': 0.5
        }
    
    def preprocess(self, observation):
        """
        Preprocess the observation
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            Preprocessed observation tensor
        """
        # Normalize pixel values to 0-1
        obs = observation.astype(np.float32) / 255.0
        
        # Process with YOLO if enabled
        if self.use_yolo:
            yolo_features = self._process_yolo(observation)
            
            # Visualize detections if requested
            if self.show_yolo_detections and hasattr(self, 'yolo_model'):
                # Show detection visualization
                for result in self.yolo_results:
                    result.show()
        
        # Flatten and convert to tensor
        obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32, device=self.device)
        
        # Compress input
        with torch.no_grad():
            compressed = self.input_compression(obs_tensor)
        
        return compressed
    
    def _process_yolo(self, observation):
        """Process observation with YOLO for object detection"""
        if not hasattr(self, 'yolo_model'):
            return None
            
        # Run inference
        self.yolo_results = self.yolo_model(observation)
        
        # Extract features from detections
        yolo_features = []
        
        # Get detections
        for result in self.yolo_results:
            # Extract boxes, confidence, and class information
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    # Add normalized box coordinates, confidence, and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = box.cls[0].cpu().numpy()
                    
                    # Normalize coordinates
                    h, w = observation.shape[:2]
                    x1, x2 = x1/w, x2/w
                    y1, y2 = y1/h, y2/h
                    
                    yolo_features.extend([x1, y1, x2, y2, confidence, class_id])
        
        # Pad or truncate features to fixed length
        max_features = 30  # Support up to 5 objects (6 values each)
        if len(yolo_features) > max_features:
            yolo_features = yolo_features[:max_features]
        else:
            yolo_features.extend([0] * (max_features - len(yolo_features)))
            
        return torch.tensor(yolo_features, dtype=torch.float32, device=self.device)
    
    def encode_state(self, state_features):
        """
        Encode state features into a hypervector using HDC principles
        
        Args:
            state_features: Compressed state features
            
        Returns:
            state_hypervector: HDC encoding of the state
        """
        # Quantize features to create index pattern
        quantized = (state_features * 10).long().clamp(0, 9)
        
        # Build state hypervector through bundling
        state_hypervector = torch.zeros(self.hd_dim, device=self.device)
        
        for i, q in enumerate(quantized):
            # Add contribution from each feature (with item memory)
            # We use permutation (cyclic shift) for position encoding
            item_vec = self.item_memory[i]
            # Apply cyclic shift proportional to feature value
            shifted = torch.roll(item_vec, shifts=int(q.item()) + 1)
            state_hypervector += shifted
            
        # Normalize the resulting vector
        state_hypervector = state_hypervector / torch.norm(state_hypervector)
        
        return state_hypervector
    
    def act(self, observation, motion=None, deterministic=False):
        """
        Select an action based on the current observation
        
        Args:
            observation: Environment observation
            motion: Motion information (ignored in this agent)
            deterministic: Whether to act deterministically (no exploration)
            
        Returns:
            Selected action
        """
        # Start timing
        start_time = time.time()
        
        # Preprocess observation
        state_features = self.preprocess(observation)
        
        # Encode state with spiking neural network
        # First encode input as spike trains
        spike_trains = self.snn.encode_input(state_features.unsqueeze(0), time_steps=10)
        
        # Forward through SNN
        with torch.no_grad():
            snn_output_spikes = self.snn(spike_trains, training=False)
            snn_output = self.snn.get_firing_rates(snn_output_spikes).squeeze()
        
        # Encode state with hyperdimensional computing
        state_hypervector = self.encode_state(state_features)
        
        # Compute Q values from state features
        with torch.no_grad():
            q_values = self.q_network(state_features).cpu().numpy()
        
        # Get action probabilities from basal ganglia
        with torch.no_grad():
            action_probs = self.basal_ganglia.forward(state_features.unsqueeze(0)).squeeze().cpu().numpy()
        
        # Combine Q-values and basal ganglia output
        # Normalize Q-values to probabilities using softmax
        q_probs = self._softmax(q_values)
        
        # Weight between Q-values and basal ganglia proportionally to dopamine (more dopamine = more Q-values)
        dopamine = float(self.neuromodulators['dopamine'])
        combined_probs = dopamine * q_probs + (1 - dopamine) * action_probs
        
        # Select action (epsilon-greedy)
        # Make sure epsilon is a scalar
        try:
            epsilon_value = float(self.epsilon)
        except:
            epsilon_value = 0.1  # Default if conversion fails
            
        # Exploration vs exploitation
        if not deterministic:
            random_choice = random.random() < epsilon_value
            if random_choice:
                action = random.randint(0, self.num_actions - 1)
            else:
                action = np.argmax(combined_probs)
        else:
            action = np.argmax(combined_probs)
        
        # End timing
        self.forward_times.append(time.time() - start_time)
        if len(self.forward_times) > 100:
            self.forward_times.pop(0)
        
        return action
    
    def _softmax(self, x):
        """Compute softmax values for array x"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def store(self, state, action, reward, next_state, done):
        """
        Store experience in episodic memory
        
        Args:
            state: Current state
            action: Selected action
            reward: Received reward
            next_state: Next state
            done: Whether episode is done
        """
        # Preprocess states
        state_features = self.preprocess(state)
        next_state_features = self.preprocess(next_state)
        
        # Store in replay memory
        self.memory.append((
            state_features.cpu().numpy(),
            action,
            reward,
            next_state_features.cpu().numpy(),
            done
        ))
        
        # Encode experience as hypervector and store
        state_hv = self.encode_state(state_features)
        action_hv = self.action_vectors[action]
        
        # Quantize reward to index in value_vectors (-1.0 to 1.0 in steps of 0.1)
        reward_idx = int((reward + 1.0) * 10)
        reward_idx = max(0, min(20, reward_idx))  # Clamp to valid range
        reward_hv = self.value_vectors[reward_idx]
        
        # Bind state, action, and reward hypervectors
        # We use element-wise multiplication for binding
        memory_hv = state_hv * action_hv * reward_hv
        
        # Store in memory vectors (up to a capacity)
        if len(self.memory_vectors) >= self.memory_capacity:
            self.memory_vectors.pop(0)
        self.memory_vectors.append(memory_hv.cpu().numpy())
    
    def learn(self, state=None, action=None, reward=None, next_state=None, done=None):
        """
        Update the agent's knowledge from experiences
        
        If parameters are provided, store the experience first.
        Then learn from replay buffer.
        
        Args:
            state: Current state (optional)
            action: Action taken (optional)
            reward: Reward received (optional)
            next_state: Next state (optional)
            done: Whether episode is done (optional)
        """
        # Store experience if provided
        if state is not None and action is not None and reward is not None and next_state is not None and done is not None:
            self.store(state, action, reward, next_state, done)
            
        # Start timing
        start_time = time.time()
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Need enough samples in replay memory
        if len(self.memory) < 32:
            return
        
        # Sample random minibatch from replay memory
        minibatch = random.sample(self.memory, 32)
        
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool, device=self.device)
        
        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            max_next_q = self.target_q_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update basal ganglia with a random experience
        idx = random.randint(0, len(minibatch) - 1)
        state = states[idx]
        action = actions[idx].item()
        reward = rewards[idx].item()
        
        # Update basal ganglia weights
        self.basal_ganglia.learn(state, action, reward)
        
        # Update neuromodulators based on rewards
        self._update_neuromodulators(rewards.mean().item())
        
        # Periodically update target network
        if self.steps % 10 == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            
        # Update SNN with random experiences (STDP learning)
        # Encode states as spike trains
        spike_trains = self.snn.encode_input(states[:8], time_steps=10)
        
        # Forward through SNN with training
        self.snn(spike_trains, training=True, reward_signal=rewards[:8].mean().item())
        
        # Increment step counter
        self.steps += 1
        
        # End timing
        self.learning_times.append(time.time() - start_time)
        if len(self.learning_times) > 100:
            self.learning_times.pop(0)
    
    def _update_neuromodulators(self, reward):
        """Update neuromodulator levels based on reward"""
        # Update dopamine based on reward (rises with positive rewards, falls with negative)
        self.neuromodulators['dopamine'] = min(1.0, max(0.1, 
            self.neuromodulators['dopamine'] + 0.1 * reward))
        
        # Update serotonin - slowly rises with positive rewards, falls with negative
        self.neuromodulators['serotonin'] = min(1.0, max(0.1,
            self.neuromodulators['serotonin'] + 0.05 * reward))
        
        # Update noradrenaline - rises with reward magnitude (positive or negative)
        reward_magnitude = abs(reward)
        self.neuromodulators['noradrenaline'] = min(1.0, max(0.1,
            0.8 * self.neuromodulators['noradrenaline'] + 0.2 * reward_magnitude))
        
        # Update acetylcholine - rises with novelty/learning
        self.neuromodulators['acetylcholine'] = min(1.0, max(0.1,
            0.9 * self.neuromodulators['acetylcholine'] + 0.1 * reward_magnitude))
    
    def visualize(self, observation):
        """
        Generate visualization data for the agent's internal representations
        
        Args:
            observation: Current observation
            
        Returns:
            vis_data: Dictionary of visualization data
        """
        if not self.visualize_internals:
            return {}
            
        # Create visualization data
        vis_data = {}
        
        # Process observation
        state_features = self.preprocess(observation)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_features).cpu().numpy()
            vis_data['q_values'] = q_values
        
        # Get basal ganglia activity
        bg_data = self.basal_ganglia.visualize_activity(state_features)
        vis_data.update(bg_data)
        
        # Add neuromodulator data
        vis_data['neuromodulators'] = self.neuromodulators
        
        # Add memory statistics
        vis_data['episodic_memory'] = {
            'size': len(self.memory),
            'capacity': self.memory_capacity
        }
        
        # Add exploration rate
        vis_data['epsilon'] = self.epsilon
        
        # Add timing information if available
        if self.forward_times:
            vis_data['avg_forward_time'] = sum(self.forward_times) / len(self.forward_times)
        if self.learning_times:
            vis_data['avg_learning_time'] = sum(self.learning_times) / len(self.learning_times)
        
        return vis_data
    
    def save(self, path):
        """
        Save agent state to disk
        
        Args:
            path: Path to save the model
        """
        # Create save dictionary
        save_dict = {
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'input_compression': self.input_compression.state_dict(),
            'item_memory': self.item_memory,
            'action_vectors': self.action_vectors,
            'value_vectors': self.value_vectors,
            'epsilon': self.epsilon,
            'steps': self.steps,
            'neuromodulators': self.neuromodulators
        }
        
        # Save to disk
        torch.save(save_dict, path)
        
    def load(self, path):
        """
        Load agent state from disk
        
        Args:
            path: Path to load the model from
        """
        # Load from disk
        save_dict = torch.load(path, map_location=self.device)
        
        # Load network weights
        self.q_network.load_state_dict(save_dict['q_network'])
        self.target_q_network.load_state_dict(save_dict['target_q_network'])
        self.input_compression.load_state_dict(save_dict['input_compression'])
        
        # Load HDC vectors
        self.item_memory = save_dict['item_memory']
        self.action_vectors = save_dict['action_vectors']
        self.value_vectors = save_dict['value_vectors']
        
        # Load agent state
        self.epsilon = save_dict['epsilon']
        self.steps = save_dict['steps']
        self.neuromodulators = save_dict['neuromodulators']
    
    def replay_experience(self, batch_size=32):
        """
        Replay experiences from memory buffer
        
        This is an alias for the learn method without parameters
        to maintain compatibility with the trainer interface.
        
        Args:
            batch_size: Size of the batch to sample (ignored, using internal batch size)
        """
        self.learn() 