"""
Optimized Cellular Automata Neural Processing

Implements optimized cellular automata for emergent pattern processing using:
- PyTorch for GPU acceleration
- Convolution operations for efficient neighborhood computation
- Multiple rule types for different pattern dynamics
- Efficient feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
from brain.utils.config import config

class OptimizedCellularAutomata(nn.Module):
    """
    Efficient implementation of 2D cellular automata with custom rules
    
    Features:
    - Convolutional operations for efficient neighborhood computation
    - Multiple rule types for different pattern dynamics
    - PyTorch-based for GPU acceleration
    - Visualization capabilities
    """
    
    def __init__(self, grid_size: int, num_states: int = 2, rule: Optional[str] = None):
        """
        Initialize the CA module
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            num_states: Number of possible cell states
            rule: Rule type ("conway", "nerve_growth", "brain_wave")
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_states = num_states
        
        # Initialize grid state
        self.register_buffer('grid', torch.zeros(grid_size, grid_size))
        
        # Tracking for visualization
        self.register_buffer('activity_history', torch.zeros(100, grid_size, grid_size))
        self.history_index = 0
        
        # Default to Conway's Game of Life rules if none provided
        self.rule_name = "conway" if rule is None else rule
        
        # Precomputed neighbor kernels for efficiency
        self._init_kernels()
        
        # Move to configured device
        self.to(torch.device(config.device))
        
    def _init_kernels(self):
        """Initialize convolution kernels for different CA rules"""
        device = torch.device(config.device)
        
        # Conway's Game of Life kernel
        kernel_conway = torch.ones(3, 3, device=device)
        kernel_conway[1, 1] = 0  # Center cell isn't its own neighbor
        self.register_buffer('kernel_conway', kernel_conway.view(1, 1, 3, 3))
        
        # Von Neumann neighborhood (4-connected)
        kernel_neumann = torch.zeros(3, 3, device=device)
        kernel_neumann[0, 1] = kernel_neumann[1, 0] = 1
        kernel_neumann[1, 2] = kernel_neumann[2, 1] = 1
        self.register_buffer('kernel_neumann', kernel_neumann.view(1, 1, 3, 3))
        
        # Moore neighborhood (8-connected) with distance weighting
        kernel_moore = torch.ones(3, 3, device=device)
        kernel_moore[1, 1] = 0  # Center cell isn't its own neighbor
        # Diagonal cells have less influence (sqrt(2) distance)
        kernel_moore[0, 0] = kernel_moore[0, 2] = kernel_moore[2, 0] = kernel_moore[2, 2] = 0.7
        self.register_buffer('kernel_moore', kernel_moore.view(1, 1, 3, 3))
        
    def initialize_random(self, density=0.3):
        """
        Initialize grid with random active cells
        
        Args:
            density: Probability of a cell being active
            
        Returns:
            Initialized grid
        """
        device = self.grid.device
        self.grid = (torch.rand(self.grid_size, self.grid_size, device=device) < density).float()
        return self.grid
        
    @torch.jit.export
    def conway_rule(self, grid, neighbors):
        """
        Conway's Game of Life update rule
        
        Args:
            grid: Current grid state
            neighbors: Neighbor count tensor
            
        Returns:
            Updated grid
        """
        # Any live cell with 2 or 3 live neighbors survives
        survive = grid & ((neighbors == 2) | (neighbors == 3))
        # Any dead cell with exactly 3 live neighbors becomes alive
        birth = (~grid.bool()) & (neighbors == 3)
        
        return (survive | birth).float()
        
    @torch.jit.export
    def nerve_growth_rule(self, grid, neighbors):
        """
        Neural-inspired cellular automaton rule
        Simulates growth and pruning dynamics of neural networks
        
        Args:
            grid: Current grid state
            neighbors: Neighbor count tensor
            
        Returns:
            Updated grid
        """
        # Active cells with 1-3 neighbors stay active (survival)
        survive = grid & ((neighbors >= 1) & (neighbors <= 3))
        
        # Inactive cells activate with probability based on neighbor count
        activation_prob = neighbors / 8.0  # Normalized by max possible neighbors
        random_mask = torch.rand_like(grid) < activation_prob
        birth = (~grid.bool()) & random_mask & (neighbors >= 2)
        
        # Activity wave propagation
        wave = (~grid.bool()) & (neighbors == 4)
        
        return (survive | birth | wave).float()
        
    @torch.jit.export
    def brain_wave_rule(self, grid, neighbors):
        """
        Rule simulating wave-like activation patterns similar to brain waves
        
        Args:
            grid: Current grid state
            neighbors: Neighbor count tensor
            
        Returns:
            Updated grid
        """
        # Waves need specific neighbor counts to propagate
        wave_prop = (neighbors >= 2) & (neighbors <= 4)
        
        # Activity decays if too many or too few neighbors
        decay = (neighbors < 2) | (neighbors > 4)
        
        # Current state affects transition probability
        state_factor = 0.8 if grid.mean() > 0.5 else 0.3
        
        # Apply probabilistic update
        random_mask = torch.rand_like(grid) < state_factor
        
        new_state = grid.clone()
        new_state[wave_prop & random_mask] = 1.0
        new_state[decay] = 0.0
        
        return new_state
    
    @torch.jit.export
    def update_grid(self):
        """
        Update the CA grid based on the selected rule
        
        Returns:
            Updated grid
        """
        # Reshape grid for convolution
        grid_expanded = self.grid.unsqueeze(0).unsqueeze(0)
        
        # Use convolution to count neighbors efficiently (with circular padding)
        padded = F.pad(grid_expanded, (1, 1, 1, 1), mode='circular')
        
        # Use the appropriate kernel
        if self.rule_name == "conway":
            kernel = self.kernel_conway
        elif self.rule_name == "neumann":
            kernel = self.kernel_neumann
        elif self.rule_name == "moore":
            kernel = self.kernel_moore
        else:
            kernel = self.kernel_conway  # Default
        
        # Count neighbors using convolution
        neighbors = F.conv2d(padded, kernel)
        neighbors = neighbors.squeeze(0).squeeze(0)
        
        # Apply the selected rule
        if self.rule_name == "conway":
            self.grid = self.conway_rule(self.grid, neighbors)
        elif self.rule_name == "nerve_growth":
            self.grid = self.nerve_growth_rule(self.grid, neighbors)
        elif self.rule_name == "brain_wave":
            self.grid = self.brain_wave_rule(self.grid, neighbors)
        else:
            # Default to Conway
            self.grid = self.conway_rule(self.grid, neighbors)
            
        # Update history for visualization (if needed)
        if config.enable_visualization:
            self.activity_history[self.history_index % 100] = self.grid.clone()
            self.history_index += 1
            
        return self.grid
        
    def forward(self, x=None):
        """
        Process input through cellular automaton
        
        Args:
            x: Optional input tensor to incorporate into the grid
            
        Returns:
            Flattened grid for downstream processing
        """
        # If input provided, incorporate it into the grid
        if x is not None:
            # Convert numpy input to tensor if needed
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float().to(self.grid.device)
                
            # Ensure input has appropriate dimensions
            if x.dim() == 1:
                # Reshape vector to grid 
                grid_cells = self.grid_size * self.grid_size
                if x.size(0) >= grid_cells:
                    # Use first grid_cells elements
                    reshaped_x = x[:grid_cells].view(self.grid_size, self.grid_size)
                else:
                    # Pad with zeros
                    padded = torch.zeros(grid_cells, device=x.device)
                    padded[:x.size(0)] = x
                    reshaped_x = padded.view(self.grid_size, self.grid_size)
            elif x.dim() == 2:
                # Already 2D, reshape if needed
                reshaped_x = F.interpolate(
                    x.unsqueeze(0).unsqueeze(0), 
                    size=(self.grid_size, self.grid_size),
                    mode='bilinear'
                ).squeeze(0).squeeze(0)
            else:
                # Higher dimensional input - extract 2D slice if possible
                reshaped_x = x.view(-1)[:self.grid_size * self.grid_size].view(self.grid_size, self.grid_size)
            
            # Apply input stimulus (threshold to determine active cells)
            input_mask = reshaped_x > 0.1
            self.grid[input_mask] = 1.0
            
        # Run multiple update steps for richer dynamics
        for _ in range(config.ca_update_steps):
            self.update_grid()
            
        # Return flattened grid for downstream processing
        return self.grid.view(-1)
    
    def extract_features(self, num_features=10):
        """
        Extract features from the current CA state
        
        Args:
            num_features: Number of features to extract
            
        Returns:
            Feature vector representing the CA state
        """
        features = []
        
        # Feature 1: Regional activity (divide grid into regions)
        num_regions = min(4, num_features // 2)
        region_size = self.grid_size // num_regions
        
        for i in range(num_regions):
            for j in range(num_regions):
                # Extract region
                region = self.grid[
                    i*region_size:(i+1)*region_size, 
                    j*region_size:(j+1)*region_size
                ]
                # Compute average activation
                features.append(torch.mean(region).item())
                
        # Feature 2: Activity statistics
        features.append(torch.mean(self.grid).item())  # Average activity
        features.append(torch.std(self.grid).item())   # Activity variability
        
        # Feature 3: Pattern complexity (gradient-based)
        gx = self.grid[:, 1:] - self.grid[:, :-1]  # Horizontal gradient
        gy = self.grid[1:, :] - self.grid[:-1, :]  # Vertical gradient
        features.append(torch.mean(torch.abs(gx)).item())  # Horizontal complexity
        features.append(torch.mean(torch.abs(gy)).item())  # Vertical complexity
        
        # Feature 4: Activity trends (if history available)
        if self.history_index >= 3:
            idx = self.history_index % 100
            prev_idx = (idx - 1) % 100
            prev2_idx = (idx - 2) % 100
            
            # Activity change rate
            current_activity = torch.mean(self.grid)
            prev_activity = torch.mean(self.activity_history[prev_idx])
            prev2_activity = torch.mean(self.activity_history[prev2_idx])
            
            # Activity direction and acceleration
            features.append((current_activity - prev_activity).item())  # Direction
            features.append((current_activity - 2*prev_activity + prev2_activity).item())  # Acceleration
            
        # Add extra features if needed
        while len(features) < num_features:
            features.append(0.0)
            
        # Limit to requested number
        features = features[:num_features]
            
        return torch.tensor(features, device=self.grid.device)
    
    def reset(self):
        """Reset CA to initial state"""
        self.grid.fill_(0)
        self.activity_history.fill_(0)
        self.history_index = 0
        
    def visualize(self):
        """
        Create visualization of the current CA state
        
        Returns:
            Figure with grid visualization
        """
        grid_np = self.grid.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(grid_np, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Cell State')
        ax.set_title(f'Cellular Automata ({self.rule_name} rule)')
        plt.tight_layout()
        
        return fig
        
    def create_animation_data(self, steps=100):
        """
        Run CA for multiple steps and collect animation data
        
        Args:
            steps: Number of steps to run
            
        Returns:
            Tensor of grid states for animation [steps, grid_size, grid_size]
        """
        frames = torch.zeros(steps, self.grid_size, self.grid_size)
        
        # Store initial state
        frames[0] = self.grid.clone()
        
        # Run simulation and collect frames
        for i in range(1, steps):
            self.update_grid()
            frames[i] = self.grid.clone()
            
        return frames 