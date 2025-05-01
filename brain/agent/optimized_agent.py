"""
Optimized Brain-Inspired AI Agent

Core implementation of the optimized brain-inspired agent that combines:
- GPU-accelerated hyperdimensional computing
- Optimized spiking neural networks
- Efficient cellular automata processing
- Fast memory systems with batch processing
- JIT compilation for critical operations
"""

import torch
import numpy as np
import time
import os
import pickle
import cv2
import random
from typing import Dict, List, Tuple, Any, Optional, Union

from brain.encoders.optimized_hdc_encoder import OptimizedHDCEncoder
from brain.networks.optimized_snn import OptimizedSpikingNeuralNetwork
from brain.networks.optimized_cellular_automata import OptimizedCellularAutomata
from brain.memory.optimized_episodic_memory import OptimizedEpisodicMemory
from brain.memory.semantic import SemanticMemory  # Reuse existing semantic memory
from brain.systems.thalamic_gating import ThalamicGating
from brain.systems.basal_ganglia import BasalGangliaLoop
from brain.systems.cerebellum import CerebellarCorrection
from brain.systems.autonomic import AutonomicSystem
from brain.utils.config import config

class OptimizedAgent:
    """
    Optimized brain-inspired agent that integrates:
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
    
    Features:
    - GPU acceleration with PyTorch
    - JIT compilation for critical operations
    - Efficient batch processing
    - Optimized memory operations
    """
    
    def __init__(self, 
                 input_shape=(120, 160, 3),
                 hd_dim=10000,
                 snn_neurons=500,
                 num_actions=5,
                 ca_width=30,
                 ca_height=20,
                 memory_capacity=10000,
                 learning_rate=0.01,
                 use_yolo=False,
                 device=None):
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
            device: Computation device ('cpu', 'cuda')
        """
        # Update config if needed
        if hd_dim != config.hd_dimension:
            config.hd_dimension = hd_dim
            
        self.input_shape = input_shape
        self.hd_dim = hd_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.use_yolo = use_yolo
        self.device = device if device is not None else config.device
        
        # Initialize components with optimized implementations
        print(f"Initializing optimized agent on device: {self.device}")
        
        # Initialize encoders and networks
        self.hdc_encoder = OptimizedHDCEncoder(
            dimension=hd_dim, 
            binary=config.use_binary_hvs,
            use_yolo=use_yolo,
            device=self.device
        )
        
        # Use multi-layer SNN with optimized implementation
        self.snn = OptimizedSpikingNeuralNetwork(
            input_size=hd_dim,
            hidden_sizes=[snn_neurons],
            output_size=num_actions,
            device=self.device
        )
        
        # Optimized cellular automata
        self.ca = OptimizedCellularAutomata(
            grid_size=max(ca_width, ca_height),
            num_states=5,
            rule="brain_wave"
        )
        
        # Memory systems
        self.episodic_memory = OptimizedEpisodicMemory(
            capacity=memory_capacity,
            device=self.device
        )
        
        self.semantic_memory = SemanticMemory(vector_dim=hd_dim)
        
        # Initialize brain systems (reuse existing implementations)
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
        self.recent_rewards = []
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
        
        # Performance metrics
        self.forward_times = []
        self.learning_times = []
        
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
        start_time = time.time()
        
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
        if isinstance(hd_vector, np.ndarray):
            hd_tensor = torch.from_numpy(hd_vector).float().to(self.device)
        else:
            hd_tensor = hd_vector.to(self.device)
            
        if hd_tensor.dim() == 1:
            hd_tensor = hd_tensor.unsqueeze(0)
        
        # Update cellular automata with observation
        ca_input = self._observation_to_ca_input(processed_obs)
        ca_features = self.ca(ca_input)
        
        # Convert CA features to tensor
        ca_features_tensor = ca_features.unsqueeze(0)
        
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
        snn_probs, _ = self.snn(hd_tensor)
        
        # Apply cerebellar correction
        _, predicted_error, corrected_command = self.cerebellum(filtered_hd, snn_probs)
        
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
        
        # Track performance
        self.forward_times.append(time.time() - start_time)
        
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
        start_time = time.time()
        
        # Skip if state/next_state is None
        if state is None or next_state is None:
            return
            
        # Preprocess and encode states
        if not isinstance(state, torch.Tensor):
            state_hd = self.hdc_encoder.encode_observation(self._preprocess(state))
        else:
            state_hd = state
            
        if not isinstance(next_state, torch.Tensor):
            next_state_hd = self.hdc_encoder.encode_observation(self._preprocess(next_state))
        else:
            next_state_hd = next_state
            
        # Create target for SNN learning
        target = np.zeros(self.num_actions)
        target[action] = 1.0
        
        # Update SNN weights
        loss, _ = self.snn.learn(
            state_hd, 
            target, 
            learning_rate=self.learning_rate,
            dopamine=self.neuromodulators['dopamine'] * 2.0  # Scale dopamine effect
        )
        
        # Update basal ganglia
        # Compute reward prediction error
        prediction_error = self.basal_ganglia.update_dopamine(reward)
        
        # Update cerebellar correction
        action_tensor = torch.zeros(self.num_actions, device=torch.device(self.device))
        action_tensor[action] = 1.0
        self.cerebellum.update(state_hd, action_tensor, reward)
        
        # Update autonomic system
        self.autonomic.process_reward(reward)
        
        # Store in episodic memory
        self.episodic_memory.store(
            state_hd, 
            action, 
            reward, 
            next_state_hd, 
            done, 
            priority=abs(prediction_error * reward)
        )
        
        # Update semantic memory if significant reward
        if abs(reward) > 0.5:
            # Encode action as vector
            action_vector = self.hdc_encoder.encode_action(action)
            # Store in semantic memory with reward-based confidence
            confidence = min(1.0, 0.5 + 0.5 * abs(reward))
            self.semantic_memory.add_association(state_hd, action_vector, confidence)
        
        # Track learning time
        self.learning_times.append(time.time() - start_time)
        
    def replay_experience(self, batch_size=32):
        """
        Replay experiences from memory for additional learning
        
        Args:
            batch_size: Number of experiences to replay
        """
        # Sample experiences with prioritization
        experiences = self.episodic_memory.prioritized_sample(batch_size)
        
        if not experiences:
            return
        
        # Learn from each experience
        for state, action, reward, next_state, done in experiences:
            if state is not None and next_state is not None:
                # Create target
                target = np.zeros(self.num_actions)
                target[action] = 1.0
                
                # Update SNN (with lower learning rate for replay)
                self.snn.learn(
                    state, 
                    target, 
                    learning_rate=self.learning_rate * 0.5
                )
                
        # Occasionally consolidate memories
        if random.random() < 0.1:  # 10% chance
            self.episodic_memory.consolidate_memories()
        
    def _preprocess(self, observation):
        """Preprocess observation for encoding"""
        # Simple preprocessing - just ensure correct shape and type
        if observation is not None and not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        return observation
        
    def _observation_to_ca_input(self, obs):
        """Convert observation to CA input format"""
        if obs is None:
            return None
            
        # Convert to grayscale if needed
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
            
        # Resize to match CA dimensions
        if gray.shape != (self.ca.grid_size, self.ca.grid_size):
            resized = cv2.resize(gray, (self.ca.grid_size, self.ca.grid_size))
        else:
            resized = gray
            
        # Normalize to 0-1 range
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        input_tensor = torch.from_numpy(normalized).float().to(self.device)
        
        return input_tensor
        
    def save(self, filename):
        """
        Save agent to file
        
        Args:
            filename: Path to save the agent
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Prepare serializable state dict
        state_dict = {
            "hd_dim": self.hd_dim,
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "use_yolo": self.use_yolo,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "neuromodulators": self.neuromodulators,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "episode_rewards": self.episode_rewards
        }
        
        # Save PyTorch modules separately
        torch.save({
            "snn_state": self.snn.state_dict(),
            "thalamic_gate_state": self.thalamic_gate.state_dict(),
            "basal_ganglia_state": self.basal_ganglia.state_dict(),
            "cerebellum_state": self.cerebellum.state_dict(),
            "autonomic_state": self.autonomic.state_dict(),
        }, filename + ".pt")
        
        # Save semantic memory
        with open(filename + ".sem", "wb") as f:
            pickle.dump(self.semantic_memory, f)
            
        # Save remaining state
        with open(filename + ".st", "wb") as f:
            pickle.dump(state_dict, f)
            
        print(f"Agent saved to {filename}")
        
    def load(self, filename):
        """
        Load agent from file
        
        Args:
            filename: Path to load the agent from
        """
        # Load PyTorch modules
        checkpoint = torch.load(filename + ".pt", map_location=torch.device(self.device))
        
        # Load SNN
        self.snn.load_state_dict(checkpoint["snn_state"])
        
        # Load other PyTorch modules
        self.thalamic_gate.load_state_dict(checkpoint["thalamic_gate_state"])
        self.basal_ganglia.load_state_dict(checkpoint["basal_ganglia_state"])
        self.cerebellum.load_state_dict(checkpoint["cerebellum_state"])
        self.autonomic.load_state_dict(checkpoint["autonomic_state"])
        
        # Load semantic memory
        with open(filename + ".sem", "rb") as f:
            self.semantic_memory = pickle.load(f)
            
        # Load remaining state
        with open(filename + ".st", "rb") as f:
            state_dict = pickle.load(f)
            
        # Update agent state
        self.hd_dim = state_dict["hd_dim"]
        self.num_actions = state_dict["num_actions"]
        self.learning_rate = state_dict["learning_rate"]
        self.use_yolo = state_dict["use_yolo"]
        self.epsilon = state_dict["epsilon"]
        self.epsilon_min = state_dict["epsilon_min"]
        self.epsilon_decay = state_dict["epsilon_decay"]
        self.neuromodulators = state_dict["neuromodulators"]
        self.step_count = state_dict["step_count"]
        self.episode_count = state_dict["episode_count"]
        self.episode_rewards = state_dict["episode_rewards"]
        
        print(f"Agent loaded from {filename}")
        
    def visualize(self, observation=None):
        """
        Visualize internal states of the agent
        
        Args:
            observation: Optional observation to process
            
        Returns:
            Dict of visualization data
        """
        if observation is not None:
            # Process observation to update internal states
            self.act(observation, deterministic=True)
            
        vis_data = {
            "neuromodulators": self.neuromodulators,
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "episode_count": self.episode_count
        }
        
        # Add memory statistics
        vis_data["episodic_memory"] = self.episodic_memory.get_stats()
        
        # Add timing statistics
        if self.forward_times:
            vis_data["avg_forward_time"] = sum(self.forward_times[-100:]) / len(self.forward_times[-100:])
        if self.learning_times:
            vis_data["avg_learning_time"] = sum(self.learning_times[-100:]) / len(self.learning_times[-100:])
            
        # Add CA visualization if requested
        if config.enable_visualization:
            vis_data["ca_grid"] = self.ca.grid.cpu().numpy()
            
        return vis_data
        
    def reset(self):
        """Reset agent states"""
        self.snn.reset_state()
        self.ca.reset()
        
        # Reset brain systems
        if hasattr(self.thalamic_gate, 'reset'):
            self.thalamic_gate.reset()
        if hasattr(self.basal_ganglia, 'reset'):
            self.basal_ganglia.reset()
        if hasattr(self.cerebellum, 'reset'):
            self.cerebellum.reset()
        if hasattr(self.autonomic, 'reset'):
            self.autonomic.reset()
            
    def get_metrics(self):
        """Get performance metrics"""
        metrics = {
            "avg_reward": sum(self.episode_rewards[-100:]) / max(1, len(self.episode_rewards[-100:])),
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "memory_size": len(self.episodic_memory),
            "epsilon": self.epsilon
        }
        
        # Add timing metrics
        if self.forward_times:
            metrics["forward_time"] = sum(self.forward_times[-100:]) / len(self.forward_times[-100:])
        if self.learning_times:
            metrics["learning_time"] = sum(self.learning_times[-100:]) / len(self.learning_times[-100:])
            
        return metrics 