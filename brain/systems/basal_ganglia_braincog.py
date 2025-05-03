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
from braincog.base.connection import CustomLinear

# Define Basal Ganglia pathway types
class BG_PathWay:
    Direct = 0
    Indirect = 1
    Both = 2

# Custom BasalGanglia implementation 
class BasalGanglia(nn.Module):
    """
    Custom implementation of BasalGanglia
    
    Implements a biologically inspired basal ganglia circuit with:
    - Direct pathway: Cortex → Striatum (D1) → GPi → Thalamus
    - Indirect pathway: Cortex → Striatum (D2) → GPe → STN → GPi → Thalamus
    
    This enables both 'Go' (direct) and 'NoGo' (indirect) learning for optimal action selection
    """
    
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size,
                 pathway=BG_PathWay.Both,
                 tau=2.0,
                 threshold=1.0,
                 decay=0.2,
                 requires_grad=True):
        """
        Initialize BasalGanglia model
        
        Args:
            input_size: Size of input from cortex
            hidden_size: Size of intermediate layers
            output_size: Size of output (number of actions)
            pathway: Which pathway(s) to use (Direct, Indirect, or Both)
            tau: Membrane time constant
            threshold: Firing threshold
            decay: Potential decay rate
            requires_grad: Whether weights require gradients
        """
        super(BasalGanglia, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pathway = pathway
        
        # Create neuron types with appropriate parameters
        self.lif_params = {
            'tau_m': tau,
            'threshold': threshold,
            'v_reset': 0.0,
            'detach_reset': True,
            'step_mode': 'm'
        }
        
        # Create connections and neurons for the direct pathway (D1, Go)
        if pathway in [BG_PathWay.Direct, BG_PathWay.Both]:
            # Cortex → Striatum (D1)
            d1_weights = torch.randn(input_size, output_size, requires_grad=requires_grad) * 0.1
            self.ctx_str_d1 = CustomLinear(d1_weights)
            
            # Striatum (D1) neurons
            self.str_d1_neurons = LIFNode(**self.lif_params)
            
            # Striatum → GPi (inhibitory)
            gpi_weights = torch.randn(output_size, output_size, requires_grad=requires_grad) * -0.1  # Negative for inhibition
            self.str_gpi = CustomLinear(gpi_weights)
            
            # GPi neurons
            self.gpi_neurons = LIFNode(**self.lif_params)
            
        # Create connections and neurons for the indirect pathway (D2, NoGo)
        if pathway in [BG_PathWay.Indirect, BG_PathWay.Both]:
            # Cortex → Striatum (D2)
            d2_weights = torch.randn(input_size, output_size, requires_grad=requires_grad) * 0.1
            self.ctx_str_d2 = CustomLinear(d2_weights)
            
            # Striatum (D2) neurons
            self.str_d2_neurons = LIFNode(**self.lif_params)
            
            # Striatum → GPe (inhibitory)
            gpe_weights = torch.randn(output_size, hidden_size, requires_grad=requires_grad) * -0.1  # Negative for inhibition
            self.str_gpe = CustomLinear(gpe_weights)
            
            # GPe neurons
            self.gpe_neurons = LIFNode(**self.lif_params)
            
            # GPe → STN (inhibitory)
            stn_weights = torch.randn(hidden_size, hidden_size, requires_grad=requires_grad) * -0.1  # Negative for inhibition
            self.gpe_stn = CustomLinear(stn_weights)
            
            # STN neurons
            self.stn_neurons = LIFNode(**self.lif_params)
            
            # STN → GPi (excitatory)
            stn_gpi_weights = torch.randn(hidden_size, output_size, requires_grad=requires_grad) * 0.1  # Positive for excitation
            self.stn_gpi = CustomLinear(stn_gpi_weights)
        
        # GPi → Thalamus (inhibitory)
        th_weights = torch.randn(output_size, output_size, requires_grad=requires_grad) * -0.1  # Negative for inhibition
        self.gpi_th = CustomLinear(th_weights)
        
        # Thalamus neurons
        self.th_neurons = LIFNode(**self.lif_params)
    
    def reset(self):
        """Reset all neuron states"""
        for module in self.modules():
            if isinstance(module, LIFNode):
                if hasattr(module, 'reset'):
                    module.reset()
                elif hasattr(module, 'n_reset'):
                    module.n_reset()
    
    def forward(self, x):
        """
        Forward pass through the basal ganglia circuit
        
        Args:
            x: Input spike train (batch_size, time_steps, input_size)
            
        Returns:
            output_spikes: Output spike train (batch_size, time_steps, output_size)
        """
        batch_size, time_steps, _ = x.shape
        device = x.device
        
        # Initialize output spikes
        output_spikes = torch.zeros(batch_size, time_steps, self.output_size, device=device)
        
        # Process each time step
        for t in range(time_steps):
            # Get current input
            current_input = x[:, t, :]
            
            # Initialize GPi activity
            gpi_input = torch.zeros(batch_size, self.output_size, device=device)
            
            # Process direct pathway (D1)
            if self.pathway in [BG_PathWay.Direct, BG_PathWay.Both]:
                # Cortex → Striatum (D1)
                str_d1_input = self.ctx_str_d1(current_input)
                str_d1_spikes = self.str_d1_neurons(str_d1_input)
                
                # Striatum → GPi (inhibitory, negative weights already applied)
                gpi_input_d1 = self.str_gpi(str_d1_spikes)
                
                # Add to GPi input
                gpi_input = gpi_input + gpi_input_d1
            
            # Process indirect pathway (D2)
            if self.pathway in [BG_PathWay.Indirect, BG_PathWay.Both]:
                # Cortex → Striatum (D2)
                str_d2_input = self.ctx_str_d2(current_input)
                str_d2_spikes = self.str_d2_neurons(str_d2_input)
                
                # Striatum → GPe (inhibitory)
                gpe_input = self.str_gpe(str_d2_spikes)
                gpe_spikes = self.gpe_neurons(gpe_input)
                
                # GPe → STN (inhibitory)
                stn_input = self.gpe_stn(gpe_spikes)
                stn_spikes = self.stn_neurons(stn_input)
                
                # STN → GPi (excitatory)
                gpi_input_indirect = self.stn_gpi(stn_spikes)
                
                # Add to GPi input
                gpi_input = gpi_input + gpi_input_indirect
            
            # Process GPi
            gpi_spikes = self.gpi_neurons(gpi_input)
            
            # GPi → Thalamus (inhibitory)
            th_input = self.gpi_th(gpi_spikes)
            th_spikes = self.th_neurons(th_input)
            
            # Store output spikes
            output_spikes[:, t, :] = th_spikes
        
        return output_spikes

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
        
        # Create the full basal ganglia model using our implementation
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
        
        # Forward through our BasalGanglia model
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
        # Convert values to tensor if they are scalars
        if isinstance(reward_prediction_error, (int, float)):
            reward_prediction_error = torch.tensor(reward_prediction_error, device=self.device)
            
        if isinstance(self.dopamine_baseline, (int, float)):
            self.dopamine_baseline = torch.tensor(self.dopamine_baseline, device=self.device)
            
        # Calculate new dopamine level
        new_dopamine = self.dopamine_baseline + 0.5 * reward_prediction_error
        
        # Clamp values using min and max function for compatibility
        self.dopamine_level = min(1.0, max(0.0, new_dopamine.item() if hasattr(new_dopamine, 'item') else new_dopamine))
        
        # Record history
        self.reward_history.append(reward if isinstance(reward, (int, float)) else reward.item())
        self.dopamine_history.append(self.dopamine_level if isinstance(self.dopamine_level, (int, float)) else self.dopamine_level.item())
    
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
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
        
        # Deal with dimension mismatch - create weight update matrices with proper dimensions
        if d1_modulation > 0:
            for a in range(self.num_actions):
                # Strengthen connections for the chosen action, weakened for others
                action_factor = 1.0 if a == action else -0.2
                
                # Update each weight row separately to handle dimension issues
                # For each output, we update based on the entire input
                update_scale = self.d1_learning_rate * float(d1_modulation) * action_factor
                # Each weight connects an input element to an output element
                # Shape might be (output_dim, input_dim)
                
                # Create an expanded update matrix for each weight
                # Try first just updating the weights
                try:
                    d1_weights[:, a] += update_scale * state_tensor.squeeze()
                except RuntimeError:
                    # If the dimensions don't match, use a more basic update rule
                    # Just scale the weights by a small factor proportional to reward
                    if action == a:
                        # Strengthen weights for the chosen action
                        d1_weights[:, a] *= (1.0 + update_scale * 0.01)
                    else:
                        # Weaken weights for other actions
                        d1_weights[:, a] *= (1.0 - abs(update_scale) * 0.005)
        
        # Compute weight updates for D2 pathway (indirect pathway) - inhibited by dopamine
        if d2_modulation > 0:
            for a in range(self.num_actions):
                # Strengthen connections for non-chosen actions, weakened for chosen
                action_factor = -0.2 if a == action else 0.5
                
                # Similar approach as above
                update_scale = self.d2_learning_rate * float(d2_modulation) * action_factor
                
                try:
                    d2_weights[:, a] += update_scale * state_tensor.squeeze()
                except RuntimeError:
                    # Basic scaling if dimensions don't match
                    if action == a:
                        # Weaken weights for the chosen action (less inhibition)
                        d2_weights[:, a] *= (1.0 - update_scale * 0.01)
                    else:
                        # Strengthen weights for other actions (more inhibition)
                        d2_weights[:, a] *= (1.0 + abs(update_scale) * 0.005)
        
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