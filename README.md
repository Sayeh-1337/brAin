# brAin - Brain-Inspired Artificial Intelligence

A brain-inspired AI architecture that integrates multiple cognitive mechanisms to create a more human-like learning agent.

## Architecture

The architecture combines:

1. **Hyperdimensional Computing (HDC)** - For symbolic and distributed representation of sensory inputs
2. **Spiking Neural Networks (SNN)** - For temporal processing and learning
3. **Cellular Automata (CA)** - For emergent pattern formation and processing
4. **Memory Systems** - For episodic and semantic memory
5. **Brain Systems** - For biologically-inspired information processing

### Core Brain Systems

The system incorporates these key biological components:

- **Thalamic Gating** - Filters sensory inputs based on attention and relevance
- **Basal Ganglia Loop** - Action selection through direct (Go) and indirect (NoGo) pathways
- **Cerebellar Error Correction** - Motor command refinement and error prediction
- **Autonomic System** - Homeostatic regulation of drives and needs

## Key Features

- Combines multiple brain-inspired approaches to create a more cognitive architecture
- Learns from few examples using HDC semantic binding and SNN temporal dynamics
- Forms emergent patterns through cellular automata processing
- Stores and recalls episodic memories for experience replay
- Builds semantic knowledge through memory consolidation
- Implements biologically-plausible action selection and sensory filtering
- Corrects motor errors through cerebellar predictive processing
- Maintains homeostatic balance through autonomic regulation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/brAin.git
cd brAin

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from brain.agent.hdc_snn_agent import HDCSNNAgent

# Create agent
agent = HDCSNNAgent(
    input_shape=(120, 160, 3),
    hd_dim=1000,
    snn_neurons=500,
    num_actions=5
)

# Train agent
for episode in range(1000):
    observation = environment.reset()
    done = False
    
    while not done:
        # Select action
        action = agent.act(observation)
        
        # Execute action in environment
        next_observation, reward, done, info = environment.step(action)
        
        # Learn from experience
        agent.learn(observation, action, reward, next_observation, done)
        
        observation = next_observation
        
    # Process episodic memory
    agent.replay_experience(batch_size=32)
```

## Brain Systems Details

### Thalamic Gating System

The thalamic gating system implements:
- Attention-based filtering of sensory inputs
- Multi-channel sensory processing
- Salience detection for information flow control
- Dynamic working memory trace

### Basal Ganglia Action Selection

The basal ganglia loop implements:
- Direct pathway (D1/Go) for action facilitation
- Indirect pathway (D2/NoGo) for action inhibition
- Dopamine-modulated learning through prediction errors
- Action selection through disinhibition mechanism

### Cerebellar Error Correction

The cerebellar correction system implements:
- Predictive error correction for motor commands
- Massive granule cell expansion for pattern separation
- Purkinje cell-based error learning
- Context-based error memory system

### Autonomic Regulation

The autonomic regulation system implements:
- Homeostatic drive regulation (energy, safety, curiosity, etc.)
- Neuromodulator level coordination (dopamine, serotonin, etc.)
- Drive-action mapping for intrinsic motivation
- Adaptive urgency detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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

## Acknowledgments

This project takes inspiration from various fields including:
- Hyperdimensional Computing
- Spiking Neural Networks
- Neuroscience and cognitive architecture
- Cellular Automata
- The VizDoom environment
- YOLO object detection 