"""
Cellular Automata Neural Network

Implements a brain-inspired processing system using:
- Grid of cells with state transitions
- Local neighborhood interactions
- Emergent pattern formation
- State-based information processing
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CellularAutomata:
    """
    CORTICAL SHEET ANALOG

    Brain-inspired cellular automata system that mimics cortical sheet dynamics.
    Features:
    - Local inhibitory and excitatory interactions
    - Spatial pattern formation
    - Self-organizing dynamics
    
    This model simulates how information might spread and process 
    across a cortical sheet, with local interactions governing global behavior.
    """

    def __init__(self, width=30, height=20, state_levels=5):
        """
        Initialize the cellular automata grid
        
        Args:
            width: Width of the CA grid
            height: Height of the CA grid
            state_levels: Number of possible state levels for each cell
        """
        self.width = width
        self.height = height
        self.state_levels = state_levels
        
        # Initialize grid with zeros
        self.grid = np.zeros((height, width), dtype=int)
        
        # Keep track of age of each cell state
        self.age = np.zeros((height, width), dtype=int)
        
        # History for visualization
        self.history = []
        
    def reset(self):
        """Reset the cellular automata grid"""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.age = np.zeros((self.height, self.width), dtype=int)
        self.history = []
        
    def _get_neighborhood(self, x, y, radius=1):
        """Get the values of cells in the neighborhood"""
        neighborhood = []
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Skip the center cell
                if dx == 0 and dy == 0:
                    continue
                    
                # Calculate neighbor coordinates with wrapping
                nx = (x + dx) % self.width
                ny = (y + dy) % self.height
                
                neighborhood.append(self.grid[ny, nx])
                
        return neighborhood
        
    def update(self, input_data=None):
        """
        Update the cellular automata grid for one step
        
        Args:
            input_data: Optional external input to influence the grid
        """
        new_grid = np.copy(self.grid)
        
        # Apply input data if provided
        if input_data is not None:
            # Reshape input data to grid dimensions if needed
            if input_data.shape != self.grid.shape:
                # Take a subset or expand as needed
                h = min(input_data.shape[0], self.height)
                w = min(input_data.shape[1], self.width)
                new_grid[:h, :w] = np.maximum(new_grid[:h, :w], input_data[:h, :w])
            else:
                new_grid = np.maximum(new_grid, input_data)
                
        # Update each cell based on its neighborhood
        for y in range(self.height):
            for x in range(self.width):
                neighborhood = self._get_neighborhood(x, y)
                
                # Calculate the average activation of neighbors
                avg_neighbors = np.mean(neighborhood)
                
                # Current cell state
                current = self.grid[y, x]
                
                # Update cell age
                if current > 0:
                    self.age[y, x] += 1
                else:
                    self.age[y, x] = 0
                
                # Rules for state transition:
                
                # Rule 1: If neighbors are very active, cell becomes more active
                if avg_neighbors > 3 and current < self.state_levels - 1:
                    new_grid[y, x] = current + 1
                    
                # Rule 2: If cell has been in high state too long, decrease
                elif current > 0 and self.age[y, x] > 3:
                    new_grid[y, x] = current - 1
                    
                # Rule 3: If neighbors are inactive, cell gradually decreases
                elif avg_neighbors < 1 and current > 0:
                    new_grid[y, x] = current - 1
                    
                # Rule 4: Random spontaneous activity
                elif np.random.random() < 0.01:
                    new_grid[y, x] = np.random.randint(1, self.state_levels)
                    
        # Update the grid
        self.grid = new_grid
                
        # Store history for visualization
        self.history.append(np.copy(self.grid))
        
        # Keep history bounded
        if len(self.history) > 100:
            self.history.pop(0)
            
        return self.grid
        
    def extract_features(self):
        """
        Extract features from the current CA state
        
        Returns:
            Feature vector representing the CA state
        """
        features = []
        
        # Feature 1: Average activation in different regions
        regions = [
            (0, 0, self.width//2, self.height//2),  # top-left
            (self.width//2, 0, self.width, self.height//2),  # top-right
            (0, self.height//2, self.width//2, self.height),  # bottom-left
            (self.width//2, self.height//2, self.width, self.height)  # bottom-right
        ]
        
        for x1, y1, x2, y2 in regions:
            region_avg = np.mean(self.grid[y1:y2, x1:x2])
            features.append(region_avg)
            
        # Feature 2: Overall activation
        features.append(np.mean(self.grid))
        
        # Feature 3: Pattern complexity (approximate entropy)
        transitions = 0
        for y in range(self.height):
            for x in range(1, self.width):
                if self.grid[y, x] != self.grid[y, x-1]:
                    transitions += 1
        features.append(transitions / (self.width * self.height))
        
        # Feature 4: Active cell ratio
        active_cells = np.sum(self.grid > 0)
        features.append(active_cells / (self.width * self.height))
        
        # Feature 5: Maximum activation cluster size
        # (simplified - just using the maximum value)
        features.append(np.max(self.grid) / self.state_levels)
        
        return np.array(features)
        
    def get_state_representation(self):
        """
        Get a normalized representation of the current state
        
        Returns:
            Flattened and normalized array of the grid
        """
        # Normalize grid values to [0, 1]
        normalized = self.grid.astype(float) / self.state_levels
        
        # Flatten to 1D array
        return normalized.flatten()
        
    def visualize(self, ax=None):
        """
        Visualize the current state of the CA
        
        Args:
            ax: Optional matplotlib axis to plot on
        """
        if ax is None:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()
            
        ax.clear()
        ax.imshow(self.grid, cmap='viridis', interpolation='nearest')
        ax.set_title('Cellular Automata State')
        ax.set_xticks([])
        ax.set_yticks([])
        
        return ax
        
    def create_animation(self, interval=200):
        """
        Create an animation of the CA history
        
        Args:
            interval: Time between frames in milliseconds
            
        Returns:
            Matplotlib animation object
        """
        if not self.history:
            raise ValueError("No history available. Run update() multiple times first.")
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        def animate(i):
            ax.clear()
            ax.imshow(self.history[i], cmap='viridis', interpolation='nearest')
            ax.set_title(f'Step {i}')
            ax.set_xticks([])
            ax.set_yticks([])
            
        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.history), 
            interval=interval, blit=False
        )
        
        return anim 