"""
Enhanced Basal Ganglia model using BrainCog components
Implements more biologically accurate direct and indirect pathways
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Import BrainCog components
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Brain-Cog'))
from braincog.base.node import LIFNode
from braincog.base.connection import LinearConnection
from braincog.model_zoo.brainarea.bg import BasalGanglia, BG_PathWay

class BasalGangliaBrainCog(nn.Module):
    """
    Enhanced Basal Ganglia model using BrainCog components
    
    Implements the direct and indirect pathways of the basal ganglia circuit:
    - Direct pathway (D1): Striatum (D1) → GPi/SNr → Thalamus → Cortex (facilitates action)
    - Indirect pathway (D2): Striatum (D2) → GPe → STN → GPi/SNr → Thalamus → Cortex (inhibits action)
    
    This more biologically accurate model can exhibit realistic properties like:
    - Action selection
    - Reinforcement learning through dopamine modulation
    - Inhibition of competing actions
    - Pathological behavior patterns (e.g., Parkinson's, Huntington's)
    """
    
    def __init__(self, 
                 input_size, 
                 num_actions,
                 d1_learning_rate=0.01,
                 d2_learning_rate=0.005,
                 dopamine_baseline=0.5,
                 tau=2.0,
                 threshold=1.0,
                 device='cpu'):
        """
        Initialize the enhanced basal ganglia model
        
        Args:
            input_size: Size of the input from cortex
            num_actions: Number of possible actions
            d1_learning_rate: Learning rate for D1 pathway
            d2_learning_rate: Learning rate for D2 pathway
            dopamine_baseline: Baseline dopamine level (0-1)
            tau: Membrane time constant
            threshold: Firing threshold
            device: Computing device (cpu/cuda)
        """
        super(BasalGangliaBrainCog, self).__init__()
        
        self.input_size = input_size
        self.num_actions = num_actions
        self.device = device
        self.dopamine_baseline = dopamine_baseline
        self.dopamine_level = dopamine_baseline
        
        # Learning rates for different pathways
        self.d1_learning_rate = d1_learning_rate
        self.d2_learning_rate = d2_learning_rate
        
        # Neuron configuration
        self.tau = tau
        self.threshold = threshold
        
        # Create the full basal ganglia model using BrainCog's implementation
        self.bg_model = BasalGanglia(
            input_size=input_size,
            hidden_size=num_actions * 2,  # Larger internal representation
            output_size=num_actions,
            pathway=BG_PathWay.Both,  # Use both direct (D1) and indirect (D2) pathways
            tau=tau,
            threshold=threshold,
            decay=0.2,
            requires_grad=True
        )
        
        # Additional neuromodulator tracking
        self.reward_history = []
        self.dopamine_history = []
        
        # Weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize connection weights with appropriate patterns"""
        # Initialize cortex → striatum (D1) weights
        if hasattr(self.bg_model, 'ctx_str_d1'):
            nn.init.normal_(self.bg_model.ctx_str_d1.weight, mean=0.5, std=0.1)
            
        # Initialize cortex → striatum (D2) weights
        if hasattr(self.bg_model, 'ctx_str_d2'):
            nn.init.normal_(self.bg_model.ctx_str_d2.weight, mean=0.5, std=0.1)
            
        # Initialize striatum → GPi weights (direct pathway)
        if hasattr(self.bg_model, 'str_gpi'):
            # Inhibitory connection - negative weights
            nn.init.normal_(self.bg_model.str_gpi.weight, mean=-0.5, std=0.1)
            
        # Initialize striatum → GPe weights (indirect pathway)
        if hasattr(self.bg_model, 'str_gpe'):
            # Inhibitory connection - negative weights
            nn.init.normal_(self.bg_model.str_gpe.weight, mean=-0.5, std=0.1)
            
        # Initialize GPe → STN weights (indirect pathway)
        if hasattr(self.bg_model, 'gpe_stn'):
            # Inhibitory connection - negative weights
            nn.init.normal_(self.bg_model.gpe_stn.weight, mean=-0.5, std=0.1)
            
        # Initialize STN → GPi weights (indirect pathway)
        if hasattr(self.bg_model, 'stn_gpi'):
            # Excitatory connection - positive weights
            nn.init.normal_(self.bg_model.stn_gpi.weight, mean=0.5, std=0.1)
            
        # Initialize GPi → Thalamus weights
        if hasattr(self.bg_model, 'gpi_th'):
            # Inhibitory connection - negative weights
            nn.init.normal_(self.bg_model.gpi_th.weight, mean=-0.5, std=0.1)
        
    def reset(self):
        """Reset the network state"""
        if hasattr(self.bg_model, 'reset'):
            self.bg_model.reset()
        
        # Reset dopamine to baseline
        self.dopamine_level = self.dopamine_baseline
        
    def forward(self, x, time_steps=10):
        """
        Forward pass through the basal ganglia circuit
        
        Args:
            x: Input from cortex (batch_size, input_size)
            time_steps: Number of simulation time steps
            
        Returns:
            action_probs: Action probabilities after basal ganglia processing
        """
        batch_size = x.shape[0]
        
        # Expand input for time steps
        x_expanded = x.unsqueeze(1).repeat(1, time_steps, 1)
        
        # Forward through BrainCog BasalGanglia model
        output_spikes = self.bg_model(x_expanded)
        
        # Compute firing rates (summing across time steps and normalizing)
        firing_rates = output_spikes.sum(dim=1) / time_steps
        
        # Transform firing rates to action probabilities
        action_probs = self._firing_rates_to_probabilities(firing_rates)
        
        return action_probs
    
    def _firing_rates_to_probabilities(self, firing_rates):
        """Convert firing rates to action probabilities"""
        # Apply softmax transformation
        scaled_rates = firing_rates * 5.0  # Scale for sharper distribution
        return torch.softmax(scaled_rates, dim=-1)
    
    def update_dopamine(self, reward):
        """
        Update dopamine level based on reward prediction error
        
        Args:
            reward: Reward value (-1 to 1)
        """
        # Simple reward prediction error model
        reward_prediction_error = reward - self.dopamine_baseline
        
        # Update dopamine level (limited to 0-1 range)
        self.dopamine_level = torch.clamp(
            self.dopamine_baseline + 0.5 * reward_prediction_error, 
            min=0.0, 
            max=1.0
        )
        
        # Record history
        self.reward_history.append(reward.item() if isinstance(reward, torch.Tensor) else reward)
        self.dopamine_history.append(self.dopamine_level.item() if isinstance(self.dopamine_level, torch.Tensor) else self.dopamine_level)
    
    def learn(self, state, action, reward):
        """
        Apply dopamine-modulated learning to basal ganglia connections
        
        Args:
            state: Input state
            action: Selected action
            reward: Reward value
        """
        # Update dopamine level
        self.update_dopamine(reward)
        
        # Compute dopamine modulation factors
        d1_modulation = self.dopamine_level - self.dopamine_baseline  # Enhanced by dopamine
        d2_modulation = self.dopamine_baseline - self.dopamine_level  # Reduced by dopamine
        
        # Skip learning if no BrainCog interface present
        if not hasattr(self.bg_model, 'ctx_str_d1') or not hasattr(self.bg_model, 'ctx_str_d2'):
            return
        
        # Get current weights
        d1_weights = self.bg_model.ctx_str_d1.weight.data
        d2_weights = self.bg_model.ctx_str_d2.weight.data
        
        # Create action mask (one-hot encoding of the selected action)
        action_mask = torch.zeros(self.num_actions, device=self.device)
        action_mask[action] = 1.0
        
        # Create state tensor
        state_tensor = state.to(self.device) if isinstance(state, torch.Tensor) else torch.tensor(state, device=self.device)
        
        # Reshape if needed
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        # Compute weight updates for D1 pathway (direct pathway) - reinforced by dopamine
        if d1_modulation > 0:
            for a in range(self.num_actions):
                # Strengthen connections for the chosen action, weakened for others
                action_factor = 1.0 if a == action else -0.2
                d1_weights[a] += self.d1_learning_rate * d1_modulation * action_factor * state_tensor
        
        # Compute weight updates for D2 pathway (indirect pathway) - inhibited by dopamine
        if d2_modulation > 0:
            for a in range(self.num_actions):
                # Strengthen connections for non-chosen actions, weakened for chosen
                action_factor = -0.2 if a == action else 0.5
                d2_weights[a] += self.d2_learning_rate * d2_modulation * action_factor * state_tensor
                
        # Apply updated weights
        self.bg_model.ctx_str_d1.weight.data = d1_weights
        self.bg_model.ctx_str_d2.weight.data = d2_weights
    
    def visualize_activity(self, input_state=None):
        """
        Visualize basal ganglia activity
        
        Args:
            input_state: Optional input state to process
            
        Returns:
            activity_data: Dictionary containing visualization data
        """
        # Create visualization data dictionary
        vis_data = {
            'dopamine_level': self.dopamine_level,
            'dopamine_history': self.dopamine_history[-50:] if len(self.dopamine_history) > 0 else [self.dopamine_baseline],
            'reward_history': self.reward_history[-50:] if len(self.reward_history) > 0 else [0],
        }
        
        # If input provided, compute and add neuronal activity
        if input_state is not None:
            # Ensure tensor format
            if not isinstance(input_state, torch.Tensor):
                input_state = torch.tensor(input_state, device=self.device).float()
            
            # Reshape if needed
            if len(input_state.shape) == 1:
                input_state = input_state.unsqueeze(0)
                
            # Forward pass with multiple steps to get activity data
            with torch.no_grad():
                x_expanded = input_state.unsqueeze(1).repeat(1, 10, 1)
                output = self.bg_model(x_expanded)
                
                # Extract firing rates for different nuclei if accessible
                vis_data['action_probs'] = self._firing_rates_to_probabilities(output.sum(dim=1) / 10).squeeze().cpu().numpy()
                
                # Add weights visualization if available
                if hasattr(self.bg_model, 'ctx_str_d1') and hasattr(self.bg_model, 'ctx_str_d2'):
                    vis_data['d1_weights'] = self.bg_model.ctx_str_d1.weight.data.cpu().numpy()
                    vis_data['d2_weights'] = self.bg_model.ctx_str_d2.weight.data.cpu().numpy()
        
        return vis_data
    
    def simulate_conditions(self, condition, severity=0.5):
        """
        Simulate pathological conditions of basal ganglia
        
        Args:
            condition: 'normal', 'parkinsons', 'huntingtons', 'dyskinesia'
            severity: Severity of the condition (0-1)
            
        Returns:
            Modified basal ganglia model
        """
        # Reset to normal condition first
        self.reset()
        
        if condition == 'normal':
            # Normal condition - do nothing
            return
            
        elif condition == 'parkinsons':
            # Parkinson's disease - reduced dopamine, stronger indirect pathway
            self.dopamine_baseline = max(0.1, self.dopamine_baseline * (1 - severity))
            self.dopamine_level = self.dopamine_baseline
            
            # Strengthen STN-GPi connection (hyperactive indirect pathway)
            if hasattr(self.bg_model, 'stn_gpi'):
                self.bg_model.stn_gpi.weight.data *= (1 + severity)
                
            # Weaken direct pathway
            if hasattr(self.bg_model, 'str_gpi'):
                self.bg_model.str_gpi.weight.data *= (1 - 0.5 * severity)
                
        elif condition == 'huntingtons':
            # Huntington's disease - degeneration of striatal neurons (especially D2)
            
            # Weaken D2 pathway connections
            if hasattr(self.bg_model, 'ctx_str_d2'):
                self.bg_model.ctx_str_d2.weight.data *= (1 - severity)
                
            if hasattr(self.bg_model, 'str_gpe'):
                self.bg_model.str_gpe.weight.data *= (1 - severity)
                
        elif condition == 'dyskinesia':
            # Dyskinesia - overactive direct pathway, excessive dopamine
            self.dopamine_baseline = min(1.0, self.dopamine_baseline * (1 + severity))
            self.dopamine_level = self.dopamine_baseline
            
            # Strengthen direct pathway
            if hasattr(self.bg_model, 'str_gpi'):
                self.bg_model.str_gpi.weight.data *= (1 + severity)
                
            # Weaken indirect pathway
            if hasattr(self.bg_model, 'ctx_str_d2'):
                self.bg_model.ctx_str_d2.weight.data *= (1 - 0.3 * severity)
                
        # Update dopamine level
        self.dopamine_level = self.dopamine_baseline 