"""
Optimized Spiking Neural Network (SNN)

Implements optimized spiking neural networks for temporal processing using:
- PyTorch for GPU acceleration
- JIT compilation for performance
- Batch processing for efficiency
- STDP learning with dopamine modulation
"""

import torch
import torch.nn as nn
import torch.jit as jit
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from brain.utils.config import config

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron model with optimized batch processing
    
    Features:
    - Batched processing for efficiency
    - Configurable parameters for biological plausibility
    - JIT compilation support
    """
    
    def __init__(self, threshold=1.0, leak_factor=0.9, refractory_period=5):
        """
        Initialize LIF neuron
        
        Args:
            threshold: Firing threshold
            leak_factor: Membrane potential leak factor
            refractory_period: Refractory period duration
        """
        super().__init__()
        self.threshold = threshold
        self.leak_factor = leak_factor
        self.refractory_period = refractory_period
        self.register_buffer('membrane_potential', torch.tensor(0.0))
        self.register_buffer('refractory_count', torch.tensor(0))
        
    def reset_state(self):
        """Reset neuron state"""
        self.membrane_potential.fill_(0.0)
        self.refractory_count.fill_(0)
        
    def forward(self, x):
        """
        Process input through the neuron
        
        Args:
            x: Input current tensor [batch_size, ...]
            
        Returns:
            Spike tensor of same shape as input
        """
        # Process entire batch at once for efficiency
        batch_size = x.size(0)
        
        # Initialize output spikes tensor
        spikes = torch.zeros_like(x)
        
        # Expand membrane potential and refractory count to match batch size
        if self.membrane_potential.dim() == 0:
            self.membrane_potential = self.membrane_potential.expand(batch_size)
        if self.refractory_count.dim() == 0:
            self.refractory_count = self.refractory_count.expand(batch_size)
            
        # Update neurons not in refractory period
        active_mask = (self.refractory_count <= 0)
        
        # Only update active neurons
        if active_mask.any():
            # Update membrane potential (apply leak and add input)
            self.membrane_potential[active_mask] = (
                self.leak_factor * self.membrane_potential[active_mask] + x[active_mask]
            )
            
            # Check for spikes
            spike_mask = (self.membrane_potential >= self.threshold) & active_mask
            
            # Generate output spikes
            spikes[spike_mask] = 1.0
            
            # Reset membrane potential after spike
            self.membrane_potential[spike_mask] = 0.0
            
            # Set refractory period
            self.refractory_count[spike_mask] = self.refractory_period
            
        # Decrement refractory counters
        self.refractory_count = torch.max(
            self.refractory_count - 1, 
            torch.tensor(0, device=self.refractory_count.device)
        )
            
        return spikes

class SpikingLayer(nn.Module):
    """
    Layer of spiking neurons with batch processing and STDP learning
    
    Features:
    - Efficient batch processing
    - STDP learning with trace-based updates
    - Support for dopamine modulation
    - Configurable parameters
    """
    
    def __init__(self, input_size, output_size, threshold=1.0, leak_factor=0.9):
        """
        Initialize spiking layer
        
        Args:
            input_size: Size of input
            output_size: Number of neurons
            threshold: Firing threshold
            leak_factor: Membrane potential leak factor
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = nn.Parameter(torch.randn(input_size, output_size) * 0.1)
        
        # Create neurons with shared parameters for efficiency
        self.threshold = threshold
        self.leak_factor = leak_factor
        self.register_buffer('membrane_potentials', torch.zeros(output_size))
        self.register_buffer('refractory_counts', torch.zeros(output_size))
        
        # STDP learning traces
        self.register_buffer('pre_trace', torch.zeros(input_size))
        self.register_buffer('post_trace', torch.zeros(output_size))
        
    def reset_state(self):
        """Reset layer state"""
        self.membrane_potentials.fill_(0.0)
        self.refractory_counts.fill_(0)
        self.pre_trace.fill_(0.0)
        self.post_trace.fill_(0.0)
        
    @torch.jit.export
    def forward_step(self, x):
        """
        Single step of spiking neural processing
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Spike tensor [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # Calculate input current to neurons
        inputs = torch.mm(x, self.weights)
        
        # Initialize output spikes
        spikes = torch.zeros(batch_size, self.output_size, device=x.device)
        
        # Process each sample in batch
        for b in range(batch_size):
            # Update neurons not in refractory period
            active_mask = (self.refractory_counts <= 0)
            
            # Update membrane potentials
            self.membrane_potentials[active_mask] = (
                self.leak_factor * self.membrane_potentials[active_mask] + 
                inputs[b, active_mask]
            )
            
            # Check for spikes
            spike_mask = (self.membrane_potentials >= self.threshold) & active_mask
            
            # Generate output spikes
            spikes[b, spike_mask] = 1.0
            
            # Reset membrane potential after spike
            self.membrane_potentials[spike_mask] = 0.0
            
            # Set refractory period
            self.refractory_counts[spike_mask] = 5  # Refractory period length
            
            # Update traces for STDP (pre-synaptic)
            self.pre_trace = 0.95 * self.pre_trace
            self.pre_trace += x[b]
            
            # Update traces for STDP (post-synaptic)
            self.post_trace = 0.95 * self.post_trace
            self.post_trace += spikes[b]
            
            # Decrement refractory counters
            self.refractory_counts = torch.max(
                self.refractory_counts - 1,
                torch.tensor(0, device=self.refractory_counts.device)
            )
            
        return spikes
        
    def forward(self, x):
        """
        Full forward pass
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Spike tensor [batch_size, output_size]
        """
        if config.use_jit_compile:
            return self.forward_step(x)
        else:
            batch_size = x.size(0)
            inputs = torch.mm(x, self.weights)
            spikes = torch.zeros(batch_size, self.output_size, device=x.device)
            
            for b in range(batch_size):
                # Similar implementation without JIT compilation
                active_mask = (self.refractory_counts <= 0)
                self.membrane_potentials[active_mask] = (
                    self.leak_factor * self.membrane_potentials[active_mask] + 
                    inputs[b, active_mask]
                )
                
                spike_mask = (self.membrane_potentials >= self.threshold) & active_mask
                spikes[b, spike_mask] = 1.0
                self.membrane_potentials[spike_mask] = 0.0
                self.refractory_counts[spike_mask] = 5
                
                # Traces & refractory updates
                self.pre_trace = 0.95 * self.pre_trace + x[b]
                self.post_trace = 0.95 * self.post_trace + spikes[b]
                self.refractory_counts = torch.max(self.refractory_counts - 1, torch.tensor(0))
                
            return spikes
    
    def stdp_update(self, learning_rate=0.01, dopamine_factor=1.0):
        """
        Spike-Timing-Dependent Plasticity weight update
        
        Args:
            learning_rate: Base learning rate
            dopamine_factor: Neuromodulation factor (reward scaling)
            
        Returns:
            Average weight change magnitude
        """
        # Scale learning rate by dopamine (reward modulation)
        effective_lr = learning_rate * dopamine_factor
        
        # Compute weight updates based on traces
        # Pre before post (causal) -> strengthen connection
        # Post before pre (acausal) -> weaken connection
        dw = effective_lr * (
            torch.outer(self.pre_trace, self.post_trace) - 
            torch.outer(self.post_trace, self.pre_trace).t()
        )
        
        # Apply weight changes with constraints
        self.weights.data += dw
        
        # Ensure weights stay in reasonable range
        self.weights.data.clamp_(-1.0, 1.0)
        
        return dw.abs().mean().item()  # Return average weight change magnitude

class OptimizedSpikingNeuralNetwork(nn.Module):
    """
    Optimized implementation of a spiking neural network
    
    Features:
    - PyTorch-based for GPU acceleration
    - JIT compilation for performance
    - Efficient batch processing
    - STDP learning with dopamine modulation
    - Multi-layer architecture
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, device=None):
        """
        Initialize SNN with multiple layers
        
        Args:
            input_size: Size of input
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output
            device: Computation device ('cpu', 'cuda')
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.device = device if device is not None else config.device
        
        # Build network layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(SpikingLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i+1]
            ))
            
        # Create softmax output layer for action probabilities
        self.output_softmax = nn.Softmax(dim=1)
        
        # Move to device
        self.to(torch.device(self.device))
        
    def reset_state(self):
        """Reset network state"""
        for layer in self.layers:
            layer.reset_state()
            
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Tuple of (output probabilities, all spikes)
        """
        # Ensure input is tensor with batch dimension
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
            
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Convert to proper device
        x = x.to(self.device)
        
        # Process through layers
        spikes = []
        current_input = x
        
        for layer in self.layers:
            current_spikes = layer(current_input)
            spikes.append(current_spikes)
            current_input = current_spikes
            
        # Apply softmax to get output probabilities
        output_probs = self.output_softmax(spikes[-1])
        
        return output_probs, spikes
    
    def learn(self, input_data, target, learning_rate=0.01, dopamine=1.0):
        """
        Update network weights using learning
        
        Args:
            input_data: Input data tensor
            target: Target output tensor
            learning_rate: Base learning rate
            dopamine: Reward modulation factor
            
        Returns:
            Loss and average weight change magnitude
        """
        # Forward pass
        output, _ = self.forward(input_data)
        
        # Compute error
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float().to(self.device)
            
        if target.dim() == 1:
            target = target.unsqueeze(0)
            
        target = target.to(self.device)
        error = target - output
        loss = torch.mean(error ** 2)
        
        # Update each layer with dopamine modulation
        weight_changes = []
        for layer in self.layers:
            weight_change = layer.stdp_update(learning_rate, dopamine)
            weight_changes.append(weight_change)
            
        return loss.item(), np.mean(weight_changes)
        
    def save(self, path):
        """Save network to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'architecture': {
                'input_size': self.input_size,
                'hidden_sizes': self.hidden_sizes,
                'output_size': self.output_size
            }
        }, path)
        
    @classmethod
    def load(cls, path, device=None):
        """Load network from file"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        arch = checkpoint['architecture']
        
        # Create model
        model = cls(
            input_size=arch['input_size'],
            hidden_sizes=arch['hidden_sizes'],
            output_size=arch['output_size'],
            device=device
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model 