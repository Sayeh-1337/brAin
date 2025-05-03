"""
Enhanced Spiking Neural Network implementation using BrainCog components
Leverages BrainCog's biologically plausible neuron models and learning rules
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Import BrainCog components
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Brain-Cog'))
from braincog.base.node import LIFNode
from braincog.base.connection import LinearConnection
from braincog.base.learningrule import STDP, ReinforcementLearning

class BrainCogSNN(nn.Module):
    """
    Enhanced Spiking Neural Network using BrainCog components
    
    Features:
    - More biologically plausible LIF neurons
    - STDP learning rule for unsupervised learning
    - Reinforcement learning capability
    - Multiple encoding options
    """
    
    def __init__(self, input_size, hidden_size, output_size, 
                 learning_rate=0.01, 
                 tau_m=2.0,
                 tau_s=1.0,
                 threshold=1.0,
                 device='cpu'):
        """
        Initialize the BrainCog enhanced SNN
        
        Args:
            input_size: Number of input neurons
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons
            learning_rate: Learning rate for weight updates
            tau_m: Membrane time constant
            tau_s: Synaptic time constant
            threshold: Firing threshold
            device: Computing device (cpu/cuda)
        """
        super(BrainCogSNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Create BrainCog LIF neuron layers
        self.hidden_neurons = LIFNode(
            tau_m=tau_m,
            tau_s=tau_s,
            threshold=threshold,
            v_reset=0.0,
            step_mode='m'
        )
        
        self.output_neurons = LIFNode(
            tau_m=tau_m,
            tau_s=tau_s,
            threshold=threshold,
            v_reset=0.0,
            step_mode='m'
        )
        
        # Create connections between layers using BrainCog's connections
        self.input_hidden = LinearConnection(input_size, hidden_size)
        self.hidden_output = LinearConnection(hidden_size, output_size)
        
        # Initialize STDP learning rule for hidden layer
        self.stdp = STDP(
            connection=self.input_hidden,
            learning_rate=learning_rate,
            w_max=1.0,
            w_min=0.0
        )
        
        # Initialize reinforcement learning rule for output layer
        self.rl = ReinforcementLearning(
            connection=self.hidden_output,
            learning_rate=learning_rate
        )
        
        # Initialize membrane potentials and spike history
        self.reset_state()
        
    def reset_state(self):
        """Reset the network state (membrane potentials and spike history)"""
        self.hidden_neurons.reset()
        self.output_neurons.reset()
        
        # Initialize spike history for STDP
        self.input_spikes_history = torch.zeros(1, self.input_size, device=self.device)
        self.hidden_spikes_history = torch.zeros(1, self.hidden_size, device=self.device)
        self.output_spikes_history = torch.zeros(1, self.output_size, device=self.device)
    
    def forward(self, x, training=True, reward_signal=None):
        """
        Forward pass through the network
        
        Args:
            x: Input spike train (batch_size, time_steps, input_size)
            training: Whether to update weights using learning rules
            reward_signal: Reward signal for reinforcement learning
            
        Returns:
            output_spikes: Output spike train (batch_size, time_steps, output_size)
        """
        batch_size, time_steps, _ = x.shape
        
        # Initialize outputs
        hidden_spikes = torch.zeros(batch_size, time_steps, self.hidden_size, device=self.device)
        output_spikes = torch.zeros(batch_size, time_steps, self.output_size, device=self.device)
        
        # Process each time step
        for t in range(time_steps):
            # Get current input spikes
            input_spikes = x[:, t, :].float()
            
            # Forward through hidden layer
            hidden_input = self.input_hidden(input_spikes)
            hidden_spikes_t = self.hidden_neurons(hidden_input)
            hidden_spikes[:, t, :] = hidden_spikes_t
            
            # Forward through output layer
            output_input = self.hidden_output(hidden_spikes_t)
            output_spikes_t = self.output_neurons(output_input)
            output_spikes[:, t, :] = output_spikes_t
            
            # Apply learning rules if in training mode
            if training:
                # Update weights using STDP
                self.stdp.update(input_spikes, hidden_spikes_t)
                
                # Update weights using reinforcement learning if reward is provided
                if reward_signal is not None:
                    self.rl.update(hidden_spikes_t, output_spikes_t, reward_signal)
                
                # Store spike history for learning rules
                self.input_spikes_history = input_spikes.detach()
                self.hidden_spikes_history = hidden_spikes_t.detach()
                self.output_spikes_history = output_spikes_t.detach()
        
        return output_spikes
    
    def get_firing_rates(self, output_spikes):
        """
        Calculate firing rates from spike trains
        
        Args:
            output_spikes: Output spike train (batch_size, time_steps, output_size)
            
        Returns:
            firing_rates: Firing rates for each output neuron (batch_size, output_size)
        """
        # Sum spikes across time dimension and normalize
        time_steps = output_spikes.shape[1]
        firing_rates = output_spikes.sum(dim=1) / time_steps
        return firing_rates
    
    def encode_input(self, x, encoding_method='rate', time_steps=20):
        """
        Encode continuous input as spike trains
        
        Args:
            x: Input data (batch_size, input_size)
            encoding_method: 'rate', 'temporal', or 'phase'
            time_steps: Number of time steps
            
        Returns:
            spike_trains: Encoded spike trains (batch_size, time_steps, input_size)
        """
        batch_size = x.shape[0]
        spike_trains = torch.zeros(batch_size, time_steps, self.input_size, device=self.device)
        
        if encoding_method == 'rate':
            # Rate coding - higher values produce more spikes
            for t in range(time_steps):
                spike_trains[:, t, :] = torch.bernoulli(x.clamp(0, 1))
                
        elif encoding_method == 'temporal':
            # Temporal coding - higher values spike earlier
            spike_time = ((1.0 - x.clamp(0, 1)) * time_steps).long()
            for b in range(batch_size):
                for i in range(self.input_size):
                    if spike_time[b, i] < time_steps:
                        spike_trains[b, spike_time[b, i], i] = 1.0
                        
        elif encoding_method == 'phase':
            # Phase coding - encode values as spike phases
            for t in range(time_steps):
                phase = (2 * np.pi * t / time_steps)
                intensity = 0.5 * (1 + torch.sin(phase + np.pi * x.clamp(0, 1)))
                spike_trains[:, t, :] = torch.bernoulli(intensity)
        
        return spike_trains
    
    def decode_output(self, output_spikes, decoding_method='rate'):
        """
        Decode spike trains to continuous values
        
        Args:
            output_spikes: Output spike train (batch_size, time_steps, output_size)
            decoding_method: 'rate', 'first_spike', or 'population'
            
        Returns:
            decoded_values: Decoded continuous values (batch_size, output_size)
        """
        if decoding_method == 'rate':
            # Rate decoding - spike frequency
            return self.get_firing_rates(output_spikes)
            
        elif decoding_method == 'first_spike':
            # First spike time decoding
            batch_size, time_steps, output_size = output_spikes.shape
            first_spike = torch.zeros(batch_size, output_size, device=self.device)
            
            # Find first spike for each output neuron
            for b in range(batch_size):
                for i in range(output_size):
                    spike_times = torch.nonzero(output_spikes[b, :, i])
                    if len(spike_times) > 0:
                        first_spike[b, i] = 1.0 - (spike_times[0] / time_steps)
                    else:
                        first_spike[b, i] = 0.0
            
            return first_spike
            
        elif decoding_method == 'population':
            # Population decoding - weighted average
            firing_rates = self.get_firing_rates(output_spikes)
            population_vector = torch.arange(self.output_size, device=self.device)
            decoded = torch.sum(firing_rates * population_vector, dim=1) / (torch.sum(firing_rates, dim=1) + 1e-10)
            return decoded.unsqueeze(1)  # Add dimension for consistency 