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
        # Store error in memory
        self.context_memory[self.memory_index] = context.detach().mean(0)
        self.error_memory[self.memory_index] = observed_error.detach().mean(0)
        self.memory_strength[self.memory_index] = 1.0  # New memory is strong
        self.memory_index = (self.memory_index + 1) % self.memory_size
        
        # Decay other memory strengths slightly
        self.memory_strength = 0.99 * self.memory_strength
        
        # Update Purkinje cells based on error
        # This is simplified LTD/LTP at parallel fiber-Purkinje cell synapses
        with torch.no_grad():
            # Get granule cell output
            granule_out = self.granule_cells(context)
            
            # Update Purkinje weights based on error correlation with granule activity
            error_expanded = observed_error.unsqueeze(1).expand(-1, granule_out.size(1))
            
            # Weight update proportional to error and granule activity (LTD)
            # Multiply activity by error to get direction (high error + high activity = decrease weight)
            weight_updates = -learning_rate * torch.mean(error_expanded * granule_out, dim=0)
            
            # Apply updates to Purkinje cells
            self.purkinje_cells.weight.data += weight_updates.unsqueeze(0).expand(self.output_size, -1)
            
        # Also update error predictor through backprop
        predicted = self.error_predictor(context)
        error_loss = torch.nn.functional.mse_loss(predicted, observed_error)
        error_loss.backward()
        
        return error_loss.item() 