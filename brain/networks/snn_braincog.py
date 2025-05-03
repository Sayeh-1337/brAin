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
from braincog.base.connection import CustomLinear
from braincog.base.learningrule import STDP

# Let's check if ReinforcementLearning exists in learningrule
try:
    from braincog.base.learningrule import ReinforcementLearning
except ImportError:
    # Custom ReinforcementLearning implementation as fallback
    class ReinforcementLearning:
        """
        Custom Reinforcement Learning rule for spiking neural networks
        
        Modulates weights based on reward signal and pre/post-synaptic activity
        """
        
        def __init__(self, connection, learning_rate=0.01, w_min=0.0, w_max=1.0):
            """
            Initialize the RL learning rule
            
            Args:
                connection: The synaptic connection to modify
                learning_rate: Rate of weight updates
                w_min: Minimum weight value
                w_max: Maximum weight value
            """
            self.connection = connection
            self.learning_rate = learning_rate
            self.w_min = w_min
            self.w_max = w_max
        
        def update(self, pre_spikes, post_spikes, reward):
            """
            Update weights based on pre-post activity and reward
            
            Args:
                pre_spikes: Pre-synaptic activity
                post_spikes: Post-synaptic activity
                reward: Reward signal (scalar or tensor)
            """
            # Safely handle dimension mismatch
            try:
                # Convert reward to tensor if needed
                if not isinstance(reward, torch.Tensor):
                    reward = torch.tensor(reward, device=pre_spikes.device)
                
                # Ensure reward has proper shape
                if reward.dim() == 0:
                    reward = reward.expand_as(post_spikes.sum(dim=0))
                
                # Compute eligibility trace (outer product of pre and post activity)
                pre_acts = pre_spikes.detach()
                post_acts = post_spikes.detach()
                
                # Ensure dimensions match
                if pre_acts.dim() == 1 and post_acts.dim() == 1:
                    # Reshape for outer product
                    pre_acts = pre_acts.unsqueeze(0)
                    post_acts = post_acts.unsqueeze(0)
                
                # Compute weight update based on Hebbian-like rule modulated by reward
                try:
                    dw = torch.mm(post_acts.t(), pre_acts) * self.learning_rate * reward.view(-1, 1)
                    
                    # Apply update
                    with torch.no_grad():
                        self.connection.weight += dw
                        
                        # Clip weights to specified range
                        self.connection.weight.data = torch.clamp(
                            self.connection.weight.data,
                            min=self.w_min,
                            max=self.w_max
                        )
                except RuntimeError as e:
                    # If dimensions don't match, use a simpler update rule
                    with torch.no_grad():
                        # Get activity levels
                        pre_activity = pre_acts.mean().item()
                        post_activity = post_acts.mean().item()
                        reward_val = float(reward.mean().item())
                        
                        # Simple Hebbian-like update
                        update_factor = self.learning_rate * reward_val * pre_activity * post_activity
                        
                        # Scale weights uniformly
                        if reward_val > 0:
                            # Strengthen weights for positive reward
                            self.connection.weight.data *= (1.0 + 0.01 * update_factor)
                        else:
                            # Weaken weights for negative reward
                            self.connection.weight.data *= (1.0 - 0.01 * abs(update_factor))
                        
                        # Clip weights
                        self.connection.weight.data = torch.clamp(
                            self.connection.weight.data,
                            min=self.w_min,
                            max=self.w_max
                        )
            except Exception as e:
                # If anything goes wrong, just skip the update
                print(f"Warning: Skipping RL update due to: {e}")

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
        
        # Create weight tensors for connections
        input_hidden_weights = torch.randn(input_size, hidden_size, device=device) * 0.1
        hidden_output_weights = torch.randn(hidden_size, output_size, device=device) * 0.1
        
        # Create connections between layers using BrainCog's connections
        self.input_hidden = CustomLinear(input_hidden_weights)
        self.hidden_output = CustomLinear(hidden_output_weights)
        
        # Initialize STDP learning rule for hidden layer
        self.stdp = STDP(
            node=self.hidden_neurons,
            connection=self.input_hidden,
            decay=0.99
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
        if hasattr(self.hidden_neurons, 'reset'):
            self.hidden_neurons.reset()
        elif hasattr(self.hidden_neurons, 'n_reset'):
            self.hidden_neurons.n_reset()
            
        if hasattr(self.output_neurons, 'reset'):
            self.output_neurons.reset()
        elif hasattr(self.output_neurons, 'n_reset'):
            self.output_neurons.n_reset()
        
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
            # Get current input spikes for this time step (for all batch samples)
            # Shape should be [batch_size, input_size]
            input_spikes = x[:, t, :].float()
            
            # Process each sample in the batch separately
            # to avoid dimension mismatches
            for b in range(batch_size):
                # Get input for this batch sample
                # Shape should be [input_size]
                sample_input = input_spikes[b]
                
                # Forward through hidden layer
                # Use unsqueeze to add batch dimension for network layers
                hidden_input = self.input_hidden(sample_input.unsqueeze(0))
                hidden_output = self.hidden_neurons(hidden_input)
                
                # Store hidden layer output
                hidden_spikes[b, t, :] = hidden_output.squeeze(0)
                
                # Forward through output layer
                output_input = self.hidden_output(hidden_output)
                output_output = self.output_neurons(output_input)
                
                # Store output layer output
                output_spikes[b, t, :] = output_output.squeeze(0)
                
                # Apply learning rules if in training mode
                if training:
                    try:
                        # Process through STDP rule for this sample
                        # This will internally update weights
                        stdp_output, dw = self.stdp(sample_input.unsqueeze(0))
                    except Exception as e:
                        # If STDP fails, print warning
                        print(f"Warning: STDP learning failed: {e}")
                    
                    # Update weights using reinforcement learning if reward is provided
                    if reward_signal is not None and hasattr(self.rl, 'update'):
                        # Convert scalar reward to tensor if needed
                        sample_reward = reward_signal
                        if isinstance(reward_signal, torch.Tensor) and reward_signal.dim() > 0:
                            sample_reward = reward_signal[b] if b < len(reward_signal) else reward_signal[0]
                        
                        try:
                            self.rl.update(hidden_output, output_output, sample_reward)
                        except Exception as e:
                            print(f"Warning: RL update failed: {e}")
        
        # Store spike history for visualization
        if batch_size > 0:
            self.input_spikes_history = x[0, -1, :].detach()
            self.hidden_spikes_history = hidden_spikes[0, -1, :].detach()
            self.output_spikes_history = output_spikes[0, -1, :].detach()
        
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