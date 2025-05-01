"""
Global configuration settings for brAin system.

This module provides centralized configuration for optimization settings
and other system-wide parameters.
"""

import torch
from dataclasses import dataclass, field, asdict

@dataclass
class SimulationConfig:
    """Configuration settings for the brAin system"""
    
    # Hyperdimensional computing settings
    hd_dimension: int = 10000      # Hypervector dimensionality
    use_binary_hvs: bool = True    # Use binary (-1/1) or continuous hypervectors
    
    # Cellular automata settings
    ca_update_steps: int = 3       # Number of CA update steps per forward pass
    
    # Optimization settings
    use_jit_compile: bool = True   # Use PyTorch JIT compilation
    batch_process: bool = True     # Use batch processing for vector operations
    enable_visualization: bool = False  # Enable visualizations for debugging
    
    # Compute device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory settings
    memory_consolidation: bool = True  # Enable memory consolidation
    
    # Logging settings
    verbose: bool = False   # Enable verbose logging

# Create global config instance
config = SimulationConfig()

def set_config(**kwargs):
    """
    Update configuration settings
    
    Args:
        **kwargs: Configuration parameters to update
    """
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
            
def get_config():
    """
    Get a dictionary of all configuration settings
    
    Returns:
        Dictionary of configuration settings
    """
    return asdict(config)
    
def reset_config():
    """Reset configuration to default values"""
    global config
    config = SimulationConfig() 