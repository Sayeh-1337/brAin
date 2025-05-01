"""
Cerebellar Error Correction in Motor Outputs

Implementation of a cerebellar circuit for predictive error correction in
motor commands, based on neurobiological principles of cerebellar learning.
"""

import torch
import torch.nn as nn
import numpy as np

class CerebellarCorrection(nn.Module):
    """Predictive error correction for motor outputs"""
    
    def __init__(self, input_size, output_size, memory_size=100):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Cerebellar cortex models
        # Granule cells (massive expansion for pattern separation)
        self.granule_cells = nn.Sequential(
            nn.Linear(input_size, input_size*10),
            nn.ReLU(),
            nn.Dropout(0.3)  # Sparse coding
        )
        
        # Purkinje cells (main output cells, one per output dimension)
        self.purkinje_cells = nn.Linear(input_size*10, output_size)
        
        # Parallel fiber activity trace (for learning)
        self.register_buffer('pf_trace', torch.zeros(input_size*10))
        self.pf_trace_decay = 0.8
        
        # Error prediction model
        self.error_predictor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, output_size)
        )
        
        # Motor correction memory (stores recent errors)
        self.memory_size = memory_size
        self.register_buffer('error_memory', torch.zeros(memory_size, output_size))
        self.register_buffer('context_memory', torch.zeros(memory_size, input_size))
        self.register_buffer('memory_strength', torch.zeros(memory_size))
        self.memory_index = 0
        
    def forward(self, x, command=None):
        """
        Process input through cerebellar circuitry
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            command: Optional motor command to correct
            
        Returns:
            tuple: (purkinje_output, predicted_error, corrected_command)
        """
        # Process through granule cell layer (expansion)
        granule_out = self.granule_cells(x)
        
        # Update parallel fiber activity trace
        self.pf_trace = self.pf_trace_decay * self.pf_trace + (1-self.pf_trace_decay) * granule_out.detach().mean(0)
        
        # Generate Purkinje cell output (inhibitory)
        purkinje_out = self.purkinje_cells(granule_out)
        
        # Predict potential error
        predicted_error = self.error_predictor(x)
        
        # If command provided, generate corrected command
        corrected = None
        if command is not None:
            # Look up similar contexts in memory
            similarities = torch.nn.functional.cosine_similarity(
                x.view(1, -1), 
                self.context_memory.view(self.memory_size, -1),
                dim=1
            )
            
            # Weight by memory strength
            weighted_similarities = similarities * self.memory_strength
            
            # Find most similar memory above threshold
            best_match = torch.argmax(weighted_similarities)
            
            if weighted_similarities[best_match] > 0.7:
                # Apply correction from memory
                remembered_error = self.error_memory[best_match]
                correction_magnitude = weighted_similarities[best_match]
                
                # Combine with current prediction
                correction = correction_magnitude * remembered_error + (1-correction_magnitude) * predicted_error
            else:
                # Use only predicted correction
                correction = predicted_error
            
            # Apply correction
            corrected = command - correction
            
        return purkinje_out, predicted_error, corrected
    
    def update(self, context, command, reward, learning_rate=0.01):
        """
        Process reward into error signal and update cerebellar model
        
        Args:
            context: Input context tensor
            command: Action tensor/command that was executed
            reward: Scalar reward received
            learning_rate: Learning rate for updates
            
        Returns:
            float: Error loss
        """
        # Convert reward to error signal
        # For positive reward, reduce error for chosen action
        # For negative reward, increase error for chosen action
        error = torch.zeros_like(command)
        
        # Handle reward as float, tensor, or batch
        if isinstance(reward, (int, float)):
            reward_value = reward
        elif isinstance(reward, torch.Tensor):
            if reward.dim() == 0:  # Scalar tensor
                reward_value = reward.item()
            else:
                # Use mean for batched rewards
                reward_value = reward.mean().item()
        else:
            # Handle numpy arrays or other types
            reward_value = float(reward)
            
        # Create error signal based on reward
        if reward_value > 0:
            # Positive reward - signal was better than expected
            error = -0.2 * torch.ones_like(command)
            # Emphasize the specific action taken
            _, max_idx = torch.max(command, dim=-1)
            if error.dim() > 1:
                for i in range(error.size(0)):
                    error[i, max_idx[i]] = -0.8  # Stronger negative error (better than expected)
            else:
                error[max_idx] = -0.8
        elif reward_value < 0:
            # Negative reward - signal was worse than expected
            error = 0.2 * torch.ones_like(command)
            # Emphasize the specific action taken
            _, max_idx = torch.max(command, dim=-1)
            if error.dim() > 1:
                for i in range(error.size(0)):
                    error[i, max_idx[i]] = 0.8  # Stronger positive error (worse than expected)
            else:
                error[max_idx] = 0.8
        else:
            # No reward - small corrective signal based on action
            error = 0.1 * torch.ones_like(command)
        
        # Ensure context is properly dimensioned for the update_error call
        if context.dim() == 1 and command.dim() > 1:
            # If context is 1D but command is batched, expand context
            context = context.unsqueeze(0)
        elif context.dim() > 1 and command.dim() == 1:
            # If context is batched but command is 1D, expand command
            command = command.unsqueeze(0)
        
        # Update model with computed error
        return self.update_error(context, command, error, learning_rate)
    
    def update_error(self, context, command, observed_error, learning_rate=0.01):
        """
        Update cerebellar model with observed error
        
        Args:
            context: Input context tensor
            command: Motor command that was executed
            observed_error: Observed error after execution
            learning_rate: Learning rate for updates
            
        Returns:
            float: Error loss
        """
        # Ensure all inputs have proper dimensions
        if context.dim() == 1:
            context = context.unsqueeze(0)  # Add batch dimension
        
        if observed_error.dim() == 1:
            observed_error = observed_error.unsqueeze(0)  # Add batch dimension
            
        if command.dim() == 1:
            command = command.unsqueeze(0)  # Add batch dimension
            
        # Get batch size
        batch_size = context.size(0)
            
        # Store error in memory (average across batch if needed)
        self.context_memory[self.memory_index] = context.detach().mean(0)
        self.error_memory[self.memory_index] = observed_error.detach().mean(0)
        self.memory_strength[self.memory_index] = 1.0  # New memory is strong
        self.memory_index = (self.memory_index + 1) % self.memory_size
        
        # Decay other memory strengths slightly
        self.memory_strength = 0.99 * self.memory_strength
        
        # Process through granule cell layer (expansion)
        granule_out = self.granule_cells(context)  # [batch_size, input_size*10]
        
        # Update parallel fiber activity trace
        self.pf_trace = self.pf_trace_decay * self.pf_trace + (1-self.pf_trace_decay) * granule_out.detach().mean(0)
        
        # Update Purkinje cells based on error
        with torch.no_grad():
            # Reshape for multiplication 
            # observed_error: [batch_size, output_size]
            # granule_out: [batch_size, input_size*10]
            
            # We need to compute weight updates for each output dimension based on error correlation
            weight_updates = torch.zeros((self.output_size, granule_out.size(1)), 
                                       device=granule_out.device)
            
            # For each output dimension
            for i in range(self.output_size):
                # Extract errors for this output dimension across batch
                errors_i = observed_error[:, i].view(batch_size, 1)  # [batch_size, 1]
                
                # Weight update is proportional to error and granule activity
                # High error + high activity = decrease weight (LTD)
                update_i = -learning_rate * (errors_i * granule_out)  # [batch_size, input_size*10]
                
                # Average across batch
                weight_updates[i] = update_i.mean(0)  # [input_size*10]
                
            # Apply updates to Purkinje cells
            self.purkinje_cells.weight.data += weight_updates
            
        # Also update error predictor through backprop
        predicted = self.error_predictor(context)  # [batch_size, output_size]
        error_loss = torch.nn.functional.mse_loss(predicted, observed_error)
        error_loss.backward()
        
        return error_loss.item() 