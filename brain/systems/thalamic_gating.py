"""
Thalamic Gating System for Sensory Filtering

Implementation of a thalamic gating system that filters sensory inputs based
on attention and relevance, simulating the thalamus role in sensory processing.
"""

import torch
import torch.nn as nn
import numpy as np

class ThalamicGating(nn.Module):
    """Filters sensory inputs using attention-based relevance mechanisms"""
    
    def __init__(self, input_dim, output_dim, n_channels=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_channels = n_channels
        
        # Attention mechanism
        self.attention = nn.Parameter(torch.ones(input_dim) / input_dim)
        
        # Channel-specific filters (different sensory modalities)
        self.channel_weights = nn.Parameter(torch.randn(n_channels, input_dim) * 0.1)
        
        # Salience detection network
        self.salience_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Gating mechanism
        self.gate_threshold = nn.Parameter(torch.tensor(0.3))
        
        # Working memory trace (short-term signal buffer)
        self.register_buffer('signal_trace', torch.zeros(input_dim))
        self.trace_decay = 0.9
        
    def update_attention(self, reward=0.0, novelty=0.0):
        """Update attention based on reward and novelty signals"""
        # Increase attention for rewarding or novel inputs
        attention_mod = torch.sigmoid(torch.tensor(reward + 0.5*novelty))
        self.attention.data = (1-attention_mod) * self.attention + attention_mod * self.signal_trace
        self.attention.data = self.attention / (self.attention.sum() + 1e-6)  # Normalize
        
    def forward(self, x):
        """
        Process input through thalamic gating
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            tuple: (gated_output, salience)
        """
        batch_size = x.size(0)
        
        # Update signal trace with decay
        self.signal_trace = self.trace_decay * self.signal_trace + (1-self.trace_decay) * x.mean(0)
        
        # Compute salience
        salience = self.salience_network(x).view(batch_size, 1)
        
        # Apply attentional modulation
        attended = x * self.attention
        
        # Process through different sensory channels
        channel_outputs = []
        for i in range(self.n_channels):
            channel_out = attended * torch.sigmoid(self.channel_weights[i])
            channel_outputs.append(channel_out)
        
        # Combine channel outputs
        combined = torch.stack(channel_outputs).sum(0)
        
        # Apply salience-based gating
        gated_output = torch.where(
            salience > self.gate_threshold,
            combined,  # Pass through if salient
            0.2 * combined  # Attenuate if not salient
        )
        
        # Ensure output dimensionality matches expected output
        if self.output_dim != self.input_dim:
            # Use adaptive pooling to adjust dimension
            reshaped = gated_output.view(batch_size, 1, -1)
            output = torch.nn.functional.adaptive_avg_pool1d(reshaped, self.output_dim).squeeze(1)
        else:
            output = gated_output
            
        return output, salience 