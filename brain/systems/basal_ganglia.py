"""
Basal Ganglia Action Selection Loop

Implementation of a basal ganglia circuit for action selection with direct (Go)
and indirect (NoGo) pathways, dopamine modulation, and reinforcement learning.
"""

import torch
import torch.nn as nn
import numpy as np

class BasalGangliaLoop(nn.Module):
    """Action selection through direct (Go) and indirect (NoGo) pathways"""
    
    def __init__(self, input_size, action_size, hidden_size=128):
        super().__init__()
        self.input_size = input_size
        self.action_size = action_size
        
        # Striatum: Direct (D1/Go) and Indirect (D2/NoGo) pathways
        self.d1_pathway = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Sigmoid()
        )
        
        self.d2_pathway = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Sigmoid()
        )
        
        # GPi/SNr (output nuclei)
        self.gpi_snr = nn.Sequential(
            nn.Linear(action_size*2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
            nn.Sigmoid()
        )
        
        # Striatal Critic for value estimation
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Tonic activity (baseline inhibition)
        self.tonic_activity = nn.Parameter(torch.ones(action_size) * 0.8)
        
        # Neuromodulatory influence
        self.dopamine_factor = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('d1_trace', torch.zeros(action_size))
        self.register_buffer('d2_trace', torch.zeros(action_size))
        
        # Initialize weights (critical for BG)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize with appropriate striatal weights"""
        # Striatum D1 starts somewhat inhibited
        for layer in self.d1_pathway:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.8)
                nn.init.constant_(layer.bias, -0.1)
        
        # Striatum D2 starts somewhat active
        for layer in self.d2_pathway:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.2)
                nn.init.constant_(layer.bias, 0.1)
        
        # GPi/SNr defaults to inhibiting thalamus
        for layer in self.gpi_snr:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.5)
                
    def update_dopamine(self, reward):
        """
        Update dopamine factor based on reward prediction error
        
        Args:
            reward: Scalar reward value
            
        Returns:
            float: Prediction error
        """
        predicted_value = self.critic(self.last_state).item() if hasattr(self, 'last_state') else 0
        prediction_error = reward - predicted_value
        
        # Scale effect by prediction error 
        dopamine_change = torch.clamp(torch.tensor(prediction_error), -0.5, 0.5)
        self.dopamine_factor.data += 0.1 * dopamine_change
        self.dopamine_factor.data = torch.clamp(self.dopamine_factor, 0.5, 2.0)
        
        # Update action traces based on dopamine
        if hasattr(self, 'd1_output') and hasattr(self, 'd2_output'):
            # Hebbian update for D1 (strengthen on positive PE, weaken on negative)
            d1_update = 0.1 * prediction_error * self.d1_output
            self.d1_trace = 0.9 * self.d1_trace + d1_update
            
            # Anti-Hebbian update for D2 (strengthen on negative PE)
            d2_update = -0.1 * prediction_error * self.d2_output
            self.d2_trace = 0.9 * self.d2_trace + d2_update
            
        return prediction_error
        
    def forward(self, x):
        """
        Process input through the basal ganglia circuit
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            
        Returns:
            tuple: (action_output, value_estimate)
        """
        # Save state for learning
        self.last_state = x.detach()
        
        # Compute activity in both pathways
        d1_activity = self.d1_pathway(x) * self.dopamine_factor  # Enhanced by dopamine
        d2_activity = self.d2_pathway(x) * (2.0 - self.dopamine_factor)  # Suppressed by dopamine
        
        # Save outputs for learning
        self.d1_output = d1_activity.detach()
        self.d2_output = d2_activity.detach()
        
        # Combine pathways (D1 excites thalamus by inhibiting GPi, D2 inhibits thalamus)
        striatum_combined = torch.cat([d1_activity, d2_activity], dim=1)
        
        # GPi/SNr processing (output nuclei) - inverted for thalamic disinhibition
        gpi_output = self.gpi_snr(striatum_combined)
        
        # Thalamic disinhibition (GPi tonically inhibits thalamus, so less GPi = more thalamus)
        # We invert to get activation (subtracted from tonic inhibition)
        thalamic_output = torch.ones_like(gpi_output) * self.tonic_activity - gpi_output
        
        # Ensure positive values
        thalamic_output = torch.relu(thalamic_output)
        
        # Value estimation for learning
        value = self.critic(x)
        
        return thalamic_output, value 