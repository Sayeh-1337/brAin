# brAin: Brain-Inspired Doom Agent with HDC-SNN Architecture

A biologically-inspired artificial intelligence agent that combines **Hyperdimensional Computing** and **Spiking Neural Networks** for playing Doom scenarios. This project implements a brain-inspired cognitive architecture with modules analogous to different brain regions.

## Architecture Overview

The agent architecture is designed to mimic the functional organization of the brain:

- **Visual Cortex Analog** (HDCEncoder): Processes visual input using hyperdimensional computing, inspired by the visual processing stream from V1-V5.
- **Basal Ganglia / Motor Cortex Analog** (SpikingNeuralNetwork): Handles action selection and temporal processing using spiking neural dynamics.
- **Cortical Sheet Analog** (CellularAutomata): Implements pattern formation and emergent dynamics through cellular automata.
- **Hippocampus Analog** (EpisodicMemory): Stores experiences and enables episodic learning.
- **Neocortex Analog** (SemanticMemory): Stores knowledge and associations for semantic understanding.
- **Visual Object Recognition Analog** (YOLODetector): Implements object detection using YOLO, mimicking the ventral stream's object recognition capabilities.

## Project Structure

```
brAin/
├── brain/                   # Core brain-inspired modules
│   ├── agent/               # Agent implementations
│   │   ├── hdc_snn_agent.py # Main agent class combining all components
│   │   └── trainer.py       # Training utilities for agents
│   ├── encoders/            # Perception modules
│   │   └── hdc_encoder.py   # Hyperdimensional Computing encoder
│   ├── memory/              # Memory systems
│   │   ├── episodic.py      # Episodic memory (hippocampus-inspired)
│   │   └── semantic.py      # Semantic memory (neocortex-inspired)
│   ├── networks/            # Neural processing
│   │   ├── cellular_automata.py  # Cellular automata for pattern processing
│   │   └── snn.py           # Spiking Neural Network implementation
│   ├── perception/          # Advanced perception components
│   │   └── yolo_detector.py # YOLO-based object detection
│   └── utils/               # Utility functions
├── config/                  # Configuration files
│   └── scenarios.py         # Scenario definitions for VizDoom
├── environment/             # Environment wrappers
│   └── doom_environment.py  # VizDoom environment interface
├── evaluation/              # Evaluation tools
│   └── metrics.py           # Performance metrics and visualization
├── results/                 # Training/testing results (generated)
├── main.py                  # Main script for training and testing
└── README.md                # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brAin.git
cd brAin
```

2. Install dependencies:
```bash
pip install numpy matplotlib vizdoom scikit-image scipy pandas seaborn tqdm
```

3. For YOLO object detection support:
```bash
pip install torch torchvision ultralytics
```

## Usage

### Training an Agent

To train an agent on the basic scenario:

```bash
python main.py train --scenario basic --episodes 1000 --output-dir results/basic
```

With YOLO object detection:
```bash
python main.py train --scenario basic --episodes 1000 --use-yolo --output-dir results/basic_yolo
```

Optional parameters:
- `--render`: Show the game window during training
- `--hd-dim 800`: Set the dimensionality of hyperdimensional vectors
- `--snn-neurons 500`: Set the number of neurons in the SNN
- `--learning-rate 0.01`: Set the learning rate
- `--model results/my_model`: Path to save the trained model
- `--use-yolo`: Enable YOLO object detection for enhanced perception
- `--show-yolo-detections`: Visualize YOLO detections during training/testing

### Testing an Agent

To test a trained agent:

```bash
python main.py test --scenario basic --model results/basic/agent_final --render
```

With YOLO visualization:
```bash
python main.py test --scenario basic --model results/basic/agent_final --render --use-yolo --show-yolo-detections
```

### Evaluating Generalization

To evaluate an agent's performance across multiple scenarios:

```bash
python main.py eval --model results/my_model --output-dir results/evaluation
```

### Visualizing Agent's Internals

To visualize the agent's internal representations:

```bash
python main.py visualize --scenario basic --model results/my_model --render --visualize-internals
```

With YOLO detection visualization:
```bash
python main.py visualize --scenario basic --model results/my_model --render --visualize-internals --use-yolo --show-yolo-detections
```

## Scenarios

The agent can be trained and tested on various VizDoom scenarios, each requiring different skills:

- **basic**: Simple navigation and shooting
- **defend_center**: Defend against enemies coming from all sides
- **deadly_corridor**: Navigate corridor while eliminating enemies
- **health_gathering**: Find and collect health packs to survive
- **defend_line**: Defend a line against approaching enemies

## Components

### HDCEncoder

Implements a brain-inspired encoding mechanism for visual and state information, similar to the visual processing stream from V1-V5 in the brain.

### SpikingNeuralNetwork

A biologically plausible neural network with leaky integrate-and-fire dynamics, refractory periods, and spike-timing-dependent weight updates.

### CellularAutomata

Simulates cortical sheet dynamics with local inhibitory and excitatory interactions, spatial pattern formation and self-organizing dynamics.

### EpisodicMemory

Stores experiences in a way analogous to hippocampal function, with pattern completion and temporal sequence learning capabilities.

### SemanticMemory

Stores semantic knowledge and associations, similar to how the neocortex consolidates information from episodic memory.

### YOLODetector

YOLO-based object detector that mimics the ventral stream's object recognition capabilities in the visual cortex. Enhances perception by detecting objects like enemies, health packs, weapons, and more.

## YOLO Object Detection

The agent integrates YOLO (You Only Look Once) object detection to enhance its perception capabilities:

- Detects objects in the game environment (enemies, health packs, weapons, etc.)
- Creates attention maps focused on important objects
- Generates object-centric representations for improved decision making
- Visualizes detections during training/testing

### Fine-tuning YOLO for Doom

For optimal performance, you can fine-tune YOLO with Doom-specific data:

1. **Collect Images**: Capture frames from different scenarios
2. **Label Data**: Use tools like CVAT or LabelImg to label objects
3. **Organize Data** in YOLO format according to ultralytics requirements
4. **Fine-tune** using the provided `fine_tune` method in YOLODetector

## Evaluation Metrics

The agent's performance is evaluated using various metrics:

- **Reward Distribution**: Statistical analysis of rewards across episodes
- **Action Distribution**: Analysis of action selection patterns
- **Learning Curve**: Progression of learning over time
- **Generalization**: How well the agent transfers to different scenarios

## Future Work

- Implement save/load functionality for agent models
- Add more sophisticated memory consolidation mechanisms
- Improve the integration between episodic and semantic memory
- Implement attention mechanisms for more effective visual processing
- Support for additional environments beyond VizDoom
- Create a Doom-specific dataset for YOLO fine-tuning

## License

[MIT License](LICENSE)

## Acknowledgments

This project takes inspiration from various fields including:
- Hyperdimensional Computing
- Spiking Neural Networks
- Neuroscience and cognitive architecture
- Cellular Automata
- The VizDoom environment
- YOLO object detection 