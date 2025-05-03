"""
Self-awareness system based on BrainCog's mirror test implementation
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from collections import deque

# Import BrainCog components
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../Brain-Cog'))
from braincog.base.node import LIFNode
from braincog.base.connection import LinearConnection
from braincog.model_zoo.tom.mirror_test import MirrorTest as BrainCogMirrorTest

class MirrorTestSelfAwareness(nn.Module):
    """
    Self-awareness module based on BrainCog's mirror test implementation
    
    The mirror test is a classic measure of self-awareness in animals:
    - An animal is marked with a dye in a location only visible through a mirror
    - If the animal recognizes itself in the mirror and touches the mark, it demonstrates self-awareness
    
    This model simulates this cognitive capability through:
    1. Self-model: Internal representation of the agent's own appearance
    2. Visual processor: Processes visual input (from environment or mirror)
    3. Comparator: Detects discrepancies between self-model and observed reflection
    4. Action predictor: Predicts how actions will affect appearance in mirror
    """
    
    def __init__(self, 
                 input_size=256, 
                 hidden_size=128, 
                 self_model_size=64, 
                 action_size=5,
                 learning_rate=0.01,
                 device='cpu'):
        """
        Initialize the mirror test self-awareness module
        
        Args:
            input_size: Size of visual input
            hidden_size: Size of hidden representations
            self_model_size: Size of self-model representation
            action_size: Number of possible actions
            learning_rate: Learning rate
            device: Computing device (cpu/cuda)
        """
        super(MirrorTestSelfAwareness, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.self_model_size = self_model_size
        self.action_size = action_size
        self.device = device
        
        # Initialize the BrainCog mirror test model
        self.mirror_test = BrainCogMirrorTest(
            input_dim=input_size,
            hidden_dim=hidden_size,
            output_dim=self_model_size
        )
        
        # Add additional components specific to our implementation
        
        # Visual processor (processes input to detect self and others)
        self.visual_processor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(device)
        
        # Self-model (representation of agent's own appearance)
        self.self_model = torch.zeros(self_model_size, device=device)
        self.self_model_confidence = 0.1  # Initial confidence in self-model
        
        # Action predictor (predicts how actions affect appearance)
        self.action_predictor = nn.Sequential(
            nn.Linear(self_model_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self_model_size)
        ).to(device)
        
        # Attention controller (modulates attention to self vs. environment)
        self.attention_controller = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Memory of past observations for consistency tracking
        self.observation_memory = deque(maxlen=100)
        
        # Self-recognition score and history
        self.self_recognition_score = 0.0
        self.recognition_history = []
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.visual_processor.parameters()) +
            list(self.action_predictor.parameters()) +
            list(self.attention_controller.parameters()) +
            list(self.mirror_test.parameters()),
            lr=learning_rate
        )
    
    def reset(self):
        """Reset stateful components"""
        self.observation_memory.clear()
        self.recognition_history.clear()
        self.self_recognition_score = 0.0
        
        # Reset the BrainCog mirror test model if it has a reset method
        if hasattr(self.mirror_test, 'reset'):
            self.mirror_test.reset()
    
    def update_self_model(self, observation, is_mirror=False, action_taken=None):
        """
        Update internal self-model based on observation
        
        Args:
            observation: Visual observation
            is_mirror: Whether the observation is from a mirror
            action_taken: Action taken before this observation (for action-effect learning)
            
        Returns:
            recognition_score: Self-recognition confidence score (0-1)
        """
        # Convert observation to tensor if needed
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
            
        # Process observation with visual processor
        visual_features = self.visual_processor(observation)
        
        # Process with BrainCog mirror test model
        with torch.no_grad():
            braincog_output = self.mirror_test(observation.unsqueeze(0)).squeeze(0)
        
        # Calculate self-recognition using BrainCog model output
        braincog_recognition = torch.sigmoid(braincog_output.mean()).item()
        
        # If this is a mirror observation, update self-model
        if is_mirror:
            # Compute attention weight (how much to focus on self vs environment)
            attention_input = torch.cat([
                visual_features, 
                torch.tensor(self.self_model, device=self.device).unsqueeze(0).expand(visual_features.shape)
            ], dim=1)
            
            attention = self.attention_controller(attention_input).item()
            
            # Update self-model as weighted average
            self.self_model = (1 - attention) * self.self_model + attention * braincog_output.detach()
            
            # Increase confidence in self-model
            self.self_model_confidence = min(1.0, self.self_model_confidence + 0.01)
            
            # If action was taken, train action predictor
            if action_taken is not None:
                # Create one-hot action encoding
                action_one_hot = torch.zeros(self.action_size, device=self.device)
                action_one_hot[action_taken] = 1.0
                
                # Get previous observation from memory
                if len(self.observation_memory) > 0:
                    prev_obs = self.observation_memory[-1]
                    
                    # Process previous observation
                    prev_visual = self.visual_processor(prev_obs)
                    prev_braincog = self.mirror_test(prev_obs.unsqueeze(0)).squeeze(0)
                    
                    # Train action predictor to predict how actions affect appearance
                    action_input = torch.cat([prev_braincog, action_one_hot])
                    predicted_new_appearance = self.action_predictor(action_input.unsqueeze(0))
                    
                    # Loss is the difference between predicted and actual new appearance
                    action_loss = nn.MSELoss()(predicted_new_appearance.squeeze(0), braincog_output)
                    
                    # Update models
                    self.optimizer.zero_grad()
                    action_loss.backward()
                    self.optimizer.step()
        
        # Store observation in memory
        self.observation_memory.append(observation.detach().clone())
        
        # Calculate combined self-recognition score
        feature_similarity = torch.cosine_similarity(
            braincog_output.unsqueeze(0), 
            torch.tensor(self.self_model, device=self.device).unsqueeze(0)
        ).item()
        
        # Weighted combination of braincog model and similarity to self-model
        self.self_recognition_score = (
            0.5 * braincog_recognition + 
            0.5 * feature_similarity * self.self_model_confidence
        )
        
        # Add to history
        self.recognition_history.append(self.self_recognition_score)
        
        return self.self_recognition_score
    
    def detect_discrepancy(self, observation):
        """
        Detect discrepancies between self-model and observation
        (e.g., detecting a mark on yourself in a mirror)
        
        Args:
            observation: Visual observation
            
        Returns:
            discrepancy: Detected discrepancy regions
            discrepancy_score: Overall discrepancy magnitude
        """
        # Convert observation to tensor if needed
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
            
        # Process observation
        with torch.no_grad():
            processed_obs = self.mirror_test(observation.unsqueeze(0)).squeeze(0)
            
        # Compare to self-model
        diff = processed_obs - torch.tensor(self.self_model, device=self.device)
        
        # Overall discrepancy magnitude
        discrepancy_score = torch.norm(diff).item()
        
        # Return difference map and score
        return diff.detach().cpu().numpy(), discrepancy_score
    
    def predict_action_effect(self, action):
        """
        Predict how taking an action will affect appearance in mirror
        
        Args:
            action: Action to take
            
        Returns:
            predicted_appearance: Predicted appearance after action
        """
        # Convert action to one-hot
        action_one_hot = torch.zeros(self.action_size, device=self.device)
        action_one_hot[action] = 1.0
        
        # Combine with current self-model
        action_input = torch.cat([
            torch.tensor(self.self_model, device=self.device),
            action_one_hot
        ])
        
        # Predict new appearance
        with torch.no_grad():
            predicted_appearance = self.action_predictor(action_input.unsqueeze(0)).squeeze(0)
            
        return predicted_appearance.cpu().numpy()
    
    def evaluate_self_awareness(self):
        """
        Evaluate the current level of self-awareness
        
        Returns:
            awareness_metrics: Dictionary of self-awareness metrics
        """
        # Calculate metrics based on recognition history and model state
        awareness_metrics = {
            'recognition_score': self.self_recognition_score,
            'self_model_confidence': self.self_model_confidence,
            'recognition_stability': np.std(self.recognition_history[-20:]) if len(self.recognition_history) >= 20 else 1.0,
            'visual_adaptation': 1.0 - np.exp(-len(self.observation_memory) / 20)  # Approaches 1 with more observations
        }
        
        # Calculate overall self-awareness score (weighted combination)
        awareness_metrics['overall_awareness'] = (
            0.4 * awareness_metrics['recognition_score'] +
            0.3 * awareness_metrics['self_model_confidence'] +
            0.2 * (1.0 - awareness_metrics['recognition_stability']) +  # Lower variation is better
            0.1 * awareness_metrics['visual_adaptation']
        )
        
        return awareness_metrics
    
    def visualize(self):
        """
        Generate visualization data for the self-awareness system
        
        Returns:
            vis_data: Dictionary containing visualization data
        """
        vis_data = {
            'self_model': self.self_model.cpu().numpy(),
            'recognition_score': self.self_recognition_score,
            'recognition_history': self.recognition_history[-50:] if len(self.recognition_history) > 0 else [0],
            'self_model_confidence': self.self_model_confidence,
            'awareness_metrics': self.evaluate_self_awareness()
        }
        
        # If we have observations, add the latest processed observation
        if len(self.observation_memory) > 0:
            latest_obs = self.observation_memory[-1]
            with torch.no_grad():
                processed_obs = self.mirror_test(latest_obs.unsqueeze(0)).squeeze(0)
            vis_data['latest_observation'] = processed_obs.cpu().numpy()
            
            # Also compute discrepancy
            diff, score = self.detect_discrepancy(latest_obs)
            vis_data['discrepancy'] = diff
            vis_data['discrepancy_score'] = score
            
        return vis_data