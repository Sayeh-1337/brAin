"""
Autonomic System for Homeostatic Regulation

Implementation of an autonomic regulatory system that manages homeostatic
drives and influences neuromodulatory systems, similar to the role of
hypothalamus and brainstem in biological systems.
"""

import torch
import torch.nn as nn
import numpy as np

class AutonomicSystem(nn.Module):
    """Homeostatic regulation through autonomic control"""
    
    def __init__(self, n_drives=5):
        super().__init__()
        self.n_drives = n_drives
        
        # Homeostatic drives (energy, safety, curiosity, social, etc.)
        self.register_buffer('drive_values', torch.ones(n_drives) * 0.5)  # Start at middle
        self.register_buffer('drive_targets', torch.ones(n_drives) * 0.7)  # Optimal levels
        self.register_buffer('drive_decay_rates', torch.ones(n_drives) * 0.99)  # Natural decay
        
        # Drive names for monitoring
        self.drive_names = ['energy', 'safety', 'curiosity', 'social', 'growth'][:n_drives]
        
        # Urgency detection network
        self.urgency_detector = nn.Sequential(
            nn.Linear(n_drives, 32),
            nn.ReLU(),
            nn.Linear(32, n_drives),
            nn.Sigmoid()
        )
        
        # Drive-action mapping (what actions satisfy which drives)
        self.drive_action_map = nn.Parameter(torch.randn(n_drives, 10) * 0.1)
        
        # Drive influence on neuromodulators
        self.neuromodulator_map = nn.Parameter(torch.tensor([
            [0.7, -0.2, 0.3, 0.1],  # energy → [DA, 5HT, NE, ACh]
            [-0.5, 0.8, 0.2, -0.1],  # safety → [DA, 5HT, NE, ACh]
            [0.4, -0.1, 0.6, 0.5],  # curiosity → [DA, 5HT, NE, ACh]
            [0.2, 0.6, 0.1, 0.3],  # social → [DA, 5HT, NE, ACh]
            [0.3, 0.2, 0.3, 0.6],  # growth → [DA, 5HT, NE, ACh]
        ][:n_drives]))
    
    def update_drives(self, rewards=None, actions=None, contexts=None):
        """
        Update drive values based on rewards, actions, and contexts
        
        Args:
            rewards: Reward values (scalar or array)
            actions: Action values taken
            contexts: Environmental context
            
        Returns:
            torch.Tensor: Drive error values
        """
        # Natural decay toward lower values
        self.drive_values = self.drive_values * self.drive_decay_rates
        
        # Update based on rewards if provided
        if rewards is not None:
            reward_tensor = torch.tensor(rewards, device=self.drive_values.device)
            # Different rewards affect different drives
            if reward_tensor.dim() == 0:  # Single scalar reward
                # Default influence: reward increases energy, safety, growth
                self.drive_values[0] += 0.1 * reward_tensor  # Energy
                self.drive_values[1] += 0.08 * reward_tensor  # Safety
                self.drive_values[min(4, self.n_drives-1)] += 0.05 * reward_tensor  # Growth
            else:
                # Multiple dimension rewards map to different drives
                for i in range(min(len(reward_tensor), self.n_drives)):
                    self.drive_values[i] += 0.1 * reward_tensor[i]
        
        # Update based on actions if provided
        if actions is not None:
            action_tensor = torch.tensor(actions, device=self.drive_values.device)
            # Map actions to drive satisfaction
            drive_satisfaction = torch.matmul(self.drive_action_map, action_tensor.unsqueeze(-1)).squeeze(-1)
            self.drive_values += 0.05 * drive_satisfaction
        
        # Update based on contexts if provided (e.g., environmental threats decrease safety)
        if contexts is not None:
            context_tensor = torch.tensor(contexts, device=self.drive_values.device)
            # For simplicity, just use first n_drives elements of context
            context_effect = 0.03 * context_tensor[:self.n_drives]
            self.drive_values += context_effect
        
        # Ensure values stay in valid range
        self.drive_values = torch.clamp(self.drive_values, 0.0, 1.0)
        
        # Compute homeostatic errors (deviation from targets)
        drive_errors = torch.abs(self.drive_targets - self.drive_values)
        
        return drive_errors
    
    def compute_drive_urgency(self):
        """
        Compute which drives need most urgent attention
        
        Returns:
            torch.Tensor: Urgency values for each drive
        """
        # Compare current values to targets
        drive_errors = torch.abs(self.drive_targets - self.drive_values)
        
        # Get urgency for each drive
        urgency = self.urgency_detector(drive_errors.unsqueeze(0)).squeeze(0)
        
        return urgency
    
    def get_neuromodulator_levels(self):
        """
        Compute neuromodulator levels based on drive states
        
        Returns:
            dict: Dictionary of neuromodulator levels
        """
        # Compute drive errors (deviation from targets)
        drive_errors = torch.abs(self.drive_targets - self.drive_values)
        
        # Map drive errors to neuromodulator changes
        neuromod_influence = torch.matmul(drive_errors, self.neuromodulator_map)
        
        # Scale to valid range
        neuromod_levels = torch.sigmoid(neuromod_influence)
        
        # Return as dictionary
        return {
            'dopamine': neuromod_levels[0].item(),
            'serotonin': neuromod_levels[1].item(),
            'norepinephrine': neuromod_levels[2].item(),
            'acetylcholine': neuromod_levels[3].item()
        }
    
    def forward(self, x=None):
        """
        Process input through autonomic system
        
        Args:
            x: Optional input tensor to modulate
            
        Returns:
            tuple: (modulated_input, urgency, neuromodulator_levels)
        """
        # Compute drive urgency
        urgency = self.compute_drive_urgency()
        
        # Get neuromodulator levels
        neuromod_levels = self.get_neuromodulator_levels()
        
        # If input provided, modulate it based on autonomic state
        if x is not None:
            # Modulate based on arousal (NE)
            arousal = neuromod_levels['norepinephrine']
            
            # Amplify input based on arousal
            modulated_input = x * (0.5 + arousal)
            
            return modulated_input, urgency, neuromod_levels
        
        return None, urgency, neuromod_levels 