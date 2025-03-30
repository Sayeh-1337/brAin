"""
Spiking Neural Network (SNN) implementation

Implements a brain-inspired spiking neural network for temporal processing
and action selection, using biologically plausible dynamics like:
- Leaky integrate-and-fire neurons
- Refractory periods
- Spike-timing-dependent weight updates
"""

import numpy as np
import random

class SpikingNeuralNetwork:
    """
    BASAL GANGLIA / MOTOR CORTEX ANALOG

    Brain-inspired neural network with spiking dynamics.
    Implements temporal integration and action selection:
    - Leaky integrate-and-fire dynamics similar to biological neurons
    - Refractory periods and spike timing integration
    - Activity-dependent weight updates (Hebbian-like learning)
    - Competition dynamics similar to lateral inhibition

    This network provides the action selection mechanism similar to
    how the basal ganglia and motor cortex coordinate decision making
    and action execution.
    """

    def __init__(self, input_size, num_neurons, num_actions):
        # Network architecture
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.num_actions = num_actions
        
        # Initialize weights with normal distribution
        self.w_input = np.random.normal(0, 0.1, (num_neurons, input_size))
        self.w_output = np.random.normal(0, 0.1, (num_actions, num_neurons))
        
        # Neuron state variables
        self.membrane_potential = np.zeros(num_neurons)
        self.last_spike_time = np.zeros(num_neurons)
        self.refractory_period = np.random.uniform(2, 5, num_neurons)  # Different refractory periods
        
        # Neuron parameters
        self.threshold = 1.0
        self.leak = 0.9
        
        # Network state for visualization
        self.last_activations = np.zeros(num_neurons)
        self.last_output = np.zeros(num_actions)
        
        # Timestep counter
        self.time = 0

    def reset_state(self):
        """Reset the state of all neurons"""
        self.membrane_potential = np.zeros(self.num_neurons)
        self.last_spike_time = np.zeros(self.num_neurons)
        self.time = 0
        self.last_activations = np.zeros(self.num_neurons)
        self.last_output = np.zeros(self.num_actions)

    def simulate(self, input_signal, dt=1.0):
        """
        Run one step of network simulation

        Args:
            input_signal: Input HD vector
            dt: Time step size
            
        Returns:
            (output, spikes): Output activation vector and neuron spike vector
        """
        self.time += dt
        
        # Calculate input current for each neuron
        input_current = np.dot(self.w_input, input_signal)
        
        # Update membrane potential with leak
        self.membrane_potential = self.membrane_potential * self.leak + input_current
        
        # Determine which neurons spike
        spikes = np.zeros(self.num_neurons)
        
        for i in range(self.num_neurons):
            # Check if neuron is in refractory period
            if (self.time - self.last_spike_time[i]) > self.refractory_period[i]:
                # Check if potential exceeds threshold
                if self.membrane_potential[i] > self.threshold:
                    spikes[i] = 1
                    self.last_spike_time[i] = self.time
                    self.membrane_potential[i] = 0  # Reset potential after spike
        
        # Calculate output based on spikes
        output = np.dot(self.w_output, spikes)
        
        # Apply softmax to output
        exp_output = np.exp(output - np.max(output))  # Subtract max for numerical stability
        output = exp_output / np.sum(exp_output)
        
        # Store for visualization
        self.last_activations = self.membrane_potential.copy()
        self.last_output = output.copy()
        
        return output, spikes

    def update_weights(self, input_signal, target, learning_rate=0.01):
        """
        Update network weights using supervised learning
        
        Args:
            input_signal: Input HD vector
            target: Target output vector
            learning_rate: Learning rate for weight updates
        """
        # Run the network
        output, spikes = self.simulate(input_signal)
        
        # Calculate output error
        error = target - output
        
        # Update output weights using error
        delta_w_output = learning_rate * np.outer(error, spikes)
        self.w_output += delta_w_output
        
        # Calculate hidden layer error (simplified backprop)
        hidden_error = np.dot(error, self.w_output)
        
        # Update input weights
        delta_w_input = learning_rate * np.outer(hidden_error * (self.membrane_potential > 0.5), input_signal)
        self.w_input += delta_w_input 