"""
Brain-inspired AI Architecture

This module implements a brain-inspired AI architecture combining:
- Hyperdimensional computing (HDC) for efficient encoding
- Spiking neural networks (SNN) for biologically plausible learning
- Cellular automata (CA) for emergent pattern formation
- Episodic and semantic memory systems
- Brain systems for biologically plausible information processing
"""

from brain import agent
from brain import encoders
from brain import memory
from brain import networks
from brain import utils
from brain import systems

__all__ = [
    'agent',
    'encoders',
    'memory',
    'networks',
    'utils',
    'systems',
]

# Version info
__version__ = '0.1.0'

# Import core components for easy access
from brain.agent.hdc_snn_agent import HDCSNNAgent
from brain.encoders.hdc_encoder import HDCEncoder
from brain.networks.snn import SpikingNeuralNetwork
from brain.networks.cellular_automata import CellularAutomata
from brain.memory.episodic import EpisodicMemory
from brain.memory.semantic import SemanticMemory 