"""
Brain module - Core brain-inspired cognitive architecture

This module contains the brain-inspired components of the agent architecture,
including perception, memory, neural networks, and agent implementations.
It is organized to mimic functional regions of the brain, with each
submodule representing a different brain region or function.
"""

# Version info
__version__ = '0.1.0'

# Import core components for easy access
from brain.agent.hdc_snn_agent import HDCSNNAgent
from brain.encoders.hdc_encoder import HDCEncoder
from brain.networks.snn import SpikingNeuralNetwork
from brain.networks.cellular_automata import CellularAutomata
from brain.memory.episodic import EpisodicMemory
from brain.memory.semantic import SemanticMemory 