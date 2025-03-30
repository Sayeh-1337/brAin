import numpy as np
import cv2
from collections import deque, defaultdict
import random
import math
import vizdoom as vzd
from IPython.display import clear_output, HTML, Image, display
import time
import os
import urllib.request
from datetime import datetime
import base64
import io
import matplotlib.pyplot as plt
import zipfile
import sys
import pickle

# Helper function to download and locate WAD file
def get_wad_file(wad_name="defend_the_center.wad"):
    """
    Helper function to locate or download a WAD file.
    Looks in common ViZDoom directories and downloads if necessary.
    """
    # Check common paths
    common_paths = [
        ".",  # Current directory
        "./scenarios",  # Local scenarios directory
        os.path.expanduser("~/.vizdoom/"),  # User's ViZDoom directory
    ]

    # Check if we're in Google Colab
    is_colab = 'google.colab' in sys.modules
    if is_colab:
        # Add Colab-specific paths
        common_paths.append("/content/scenarios")
        os.makedirs("/content/scenarios", exist_ok=True)
    else:
        # Only try to use __file__ in non-Colab environments
        try:
            script_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")
            common_paths.append(script_dir_path)
        except NameError:
            # __file__ is not defined (likely running in interactive environment)
            print("Running in interactive environment, skipping __file__ based path")

    # Try to find the WAD file in common paths
    for path in common_paths:
        wad_path = os.path.join(path, wad_name)
        if os.path.exists(wad_path):
            print(f"Found WAD file at: {wad_path}")
            return wad_path

    # If not found, download it
    print(f"WAD file {wad_name} not found. Downloading...")

    # Create scenarios directory if it doesn't exist
    save_dir = "./scenarios"
    if is_colab:
        save_dir = "/content/scenarios"

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, wad_name)

    # URL for the WAD file
    base_url = "https://github.com/mwydmuch/ViZDoom/raw/master/scenarios/"
    wad_url = f"{base_url}{wad_name}"

    try:
        # Download the file
        print(f"Downloading from {wad_url}...")
        urllib.request.urlretrieve(wad_url, save_path)
        print(f"Downloaded WAD file to {save_path}")
        return save_path
    except Exception as e:
        print(f"Error downloading WAD file: {e}")
        raise FileNotFoundError(f"Could not find or download {wad_name}")

# Try to import PyTorch for YOLO
try:
    import torch
    TORCH_AVAILABLE = True
    print("PyTorch is available, will use YOLO for enhanced monster detection")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Will use traditional detection methods only.")

# Check for Colab environment early
IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab environment")

    # Colab sometimes has compatibility issues with YOLO due to CUDA/cuDNN conflicts
    # If we're in Colab, we'll be extra cautious with YOLO initialization
    try:
        # Test if YOLO and TensorFlow can coexist
        if TORCH_AVAILABLE:
            import tensorflow as tf
            # If both are imported successfully, still mark as potentially unstable
            print("Both PyTorch and TensorFlow loaded - potential conflicts may occur")
            # We'll still allow YOLO but with additional safeguards
    except Exception as e:
        print(f"Detected potential compatibility issue in Colab: {e}")
        print("Disabling YOLO to prevent crashes")
        TORCH_AVAILABLE = False  # Disable even if PyTorch is available
except:
    IN_COLAB = False
    print("Not running in Google Colab environment")

"""
NEUROSCIENCE-INSPIRED AGENT ARCHITECTURE

This agent implementation maps to biological brain regions and functions:

1. VISUAL CORTEX - HDCEncoder
   - Primary Visual Cortex (V1): Edge detection and basic visual feature extraction
   - Visual Association Areas (V2-V5): Feature binding and motion processing
   - Visual-Spatial Integration: Position encoding and spatial awareness

2. LIMBIC SYSTEM - Memory Systems
   - Hippocampus: Episodic memory for experiences and spatial navigation
   - Amygdala: Threat detection and emotional responses through CA pattern recognition
   - Emotion Processing: Reward system and feedback integration

3. BASAL GANGLIA - Action Selection
   - Striatum: Action selection and decision making in SNN
   - Substantia Nigra: Dopaminergic modulation (reward signaling)
   - Motor Learning: Weight updates in the SNN (like STDP)

4. CEREBELLUM - Movement Coordination
   - Motor Timing: Refining actions and predicting outcomes
   - Error Correction: Adjusting movement based on feedback

5. PREFRONTAL CORTEX - Executive Control
   - Working Memory: Current state representation and planning
   - Decision Making: Integrating perception, memory, and action selection
   - Inhibitory Control: Exploration vs. exploitation balance (epsilon parameter)

6. THALAMUS - Sensory Integration
   - Sensory Processing: Integration of different input modalities
   - Attention Mechanisms: Focus on relevant threats and targets
   - Relay Functions: Information flow between perception and action

7. BRAINSTEM & MIDBRAIN - Primitive Responses
   - Superior Colliculus: Visual target detection and orientation
   - Periaqueductal Gray: Defensive behaviors when threatened
   - Reticular Formation: Arousal and alertness (exploration patterns)

8. DEEP VISUAL CORTEX - YOLO-based Object Detection
   - Ventral Visual Stream: Deep hierarchical object recognition
   - Inferotemporal Cortex: High-level object categorization
   - Feedback Connections: Top-down modulation of visual processing
   - Attention Networks: Focused processing of relevant visual features

The agent's architecture implements these brain-inspired mechanisms through:
- Distributed representation (HD vectors analogous to neural population coding)
- Temporal dynamics (spiking neurons with refractory periods)
- Associative learning (STDP-like weight updates)
- Spatiotemporal pattern recognition (cellular automata for situation awareness)
- Hierarchical processing from perception to action
- Deep convolutional networks for object detection (modeling hierarchical visual cortex)
"""

def setup_virtual_display(visible=0, size=(1400, 900)):
    """Setup a virtual display for headless environments like Google Colab"""
    if IN_COLAB:
        try:
            from pyvirtualdisplay import Display
            display = Display(visible=visible, size=size)
            display.start()
            print(f"Virtual display started with resolution {size[0]}x{size[1]}")
            return display
        except Exception as e:
            print(f"Error setting up virtual display: {e}")
            return None
    else:
        print("Virtual display not needed outside of Colab")
        return None

class HDCEncoder:
    """
    VISUAL CORTEX ANALOG

    Hyperdimensional Computing encoder for visual and state information.
    Functions similar to the visual processing stream from V1-V5:
    - Edge detection (analogous to V1 simple cells)
    - Position encoding (analogous to spatial maps in parietal cortex)
    - Motion processing (analogous to MT/V5 motion selective neurons)
    - Feature binding (analogous to integration in higher visual areas)

    The distributed representation in high-dimensional space mimics
    population coding in the brain, where information is encoded across
    large groups of neurons rather than individual cells.
    """

    def __init__(self, D=1000):  # D is the dimensionality of HD vectors
        self.D = D
        self.item_memory = {}
        self.base_vectors = {}
        self._initialize_base_vectors()

    def _initialize_base_vectors(self):
        """Initialize base vectors for different features"""
        # Create random bipolar vectors for basic features
        features = ['edge', 'motion', 'shape', 'position', 'action']
        for feature in features:
            self.base_vectors[feature] = np.random.choice([-1, 1], size=self.D)

    def create_position_vector(self, x, y, resolution=(120, 160)):
        """Create HD vector for position information"""
        h, w = resolution
        # Normalize coordinates
        x_norm = x / w
        y_norm = y / h

        # Create position vector through binding of x and y components
        x_vec = self._continuous_to_hd(x_norm)
        y_vec = self._continuous_to_hd(y_norm)
        return self._bind(x_vec, y_vec)

    def _continuous_to_hd(self, value):
        """Convert continuous value to HD vector"""
        # Create a weighted sum of base vectors
        phases = np.linspace(0, 2*np.pi, self.D)
        return np.sign(np.cos(2*np.pi*value + phases))

    def _bind(self, vec1, vec2):
        """Bind two HD vectors using element-wise multiplication"""
        return vec1 * vec2

    def _bundle(self, vectors):
        """Bundle multiple HD vectors using majority sum"""
        if not vectors:
            return np.zeros(self.D)
        summed = np.sum(vectors, axis=0)
        return np.sign(summed)

    def encode_observation(self, frame, motion=None):
        """Encode visual observation into HD vector"""
        if frame is None:
            return np.zeros(self.D)

        # Extract visual features - use lower thresholds for efficiency
        edges = cv2.Canny(frame, 50, 100)

        # Create HD vectors for each feature with sampling to reduce computation
        edge_vectors = []
        motion_vectors = []

        # Process edges with sampling (process only a subset of points)
        edge_points = np.where(edges > 0)
        # Sample only up to 100 edge points to reduce computation
        num_edge_points = min(len(edge_points[0]), 100)
        if num_edge_points > 0:
            indices = np.random.choice(len(edge_points[0]), num_edge_points, replace=False)
            for i in indices:
                y, x = edge_points[0][i], edge_points[1][i]
                pos_vec = self.create_position_vector(x, y)
                edge_vectors.append(self._bind(self.base_vectors['edge'], pos_vec))

        # Process motion if available (with more aggressive sampling)
        if motion is not None:
            # Use larger step size (16 instead of 8) to process fewer blocks
            for y in range(0, motion.shape[0], 16):
                for x in range(0, motion.shape[1], 16):
                    if np.any(motion[y:y+16, x:x+16] > 40):  # Higher threshold
                        pos_vec = self.create_position_vector(x, y)
                        motion_vectors.append(self._bind(self.base_vectors['motion'], pos_vec))

        # Bundle all features
        edge_hd = self._bundle(edge_vectors)
        motion_hd = self._bundle(motion_vectors)

        # Combine all features
        return self._bundle([edge_hd, motion_hd])

    def encode_action(self, action):
        """Encode action into HD vector"""
        if action not in self.item_memory:
            self.item_memory[action] = self._bind(
                self.base_vectors['action'],
                np.random.choice([-1, 1], size=self.D)
            )
        return self.item_memory[action]

    def similarity(self, vec1, vec2):
        """Compute similarity between two HD vectors"""
        return np.sum(vec1 * vec2) / self.D

class SpikingNeuralNetwork:
    """
    BASAL GANGLIA & MOTOR CORTEX ANALOG

    Simplified Spiking Neural Network for action selection.
    Implements functions similar to the basal ganglia and motor circuits:
    - Action selection (similar to striatum direct/indirect pathways)
    - STDP-like plasticity (similar to corticostriatal plasticity)
    - Thresholded activation (similar to tonic inhibition in basal ganglia)
    - Membrane dynamics (similar to neuronal integration in cortical neurons)

    The sparse connectivity pattern mimics the specialized circuitry of
    the basal ganglia, while the learning mechanism approximates dopamine-
    modulated plasticity in corticostriatal synapses.
    """

    def __init__(self, input_size, num_neurons, num_actions):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.num_actions = num_actions

        # Initialize weights with sparse connections (only 50% of connections are active)
        self.weights_input = np.random.randn(input_size, num_neurons) * 0.1 * (np.random.random((input_size, num_neurons)) > 0.5)
        self.weights_output = np.random.randn(num_neurons, num_actions) * 0.1

        # Neuron states
        self.membrane_potential = np.zeros(num_neurons)
        self.spike_history = deque(maxlen=50)  # Reduced from 100

        # Parameters
        self.threshold = 1.0
        self.tau = 0.1  # Time constant
        self.refractory_period = 3  # Reduced from 5
        self.last_spike = np.zeros(num_neurons)

        # Initialize with some spikes for better initial exploration
        self.membrane_potential = np.random.random(num_neurons) * 0.5

    def reset_state(self):
        """Reset neuron states"""
        self.membrane_potential = np.zeros(self.num_neurons)
        self.last_spike = np.zeros(self.num_neurons)

    def simulate(self, input_signal, dt=1.0):
        """Simulate network for one timestep"""
        # Input integration
        input_current = np.dot(input_signal, self.weights_input)

        # Update membrane potentials
        self.membrane_potential += dt * (
            -self.membrane_potential/self.tau + input_current
        )

        # Generate spikes
        spikes = self.membrane_potential >= self.threshold

        # Reset membrane potential for spiking neurons
        self.membrane_potential[spikes] = 0

        # Update spike history
        self.last_spike[spikes] = 0
        self.last_spike[~spikes] += 1

        # Calculate output
        output_current = np.dot(spikes.astype(float), self.weights_output)

        return output_current, spikes

    def update_weights(self, input_signal, target, learning_rate=0.01):
        """Update weights using simplified STDP"""
        output, spikes = self.simulate(input_signal)

        # Compute errors
        output_error = target - output
        hidden_error = np.dot(output_error, self.weights_output.T)

        # Update weights
        self.weights_output += learning_rate * np.outer(spikes.astype(float), output_error)
        self.weights_input += learning_rate * np.outer(input_signal, hidden_error)

class CellularAutomata:
    """
    LIMBIC SYSTEM ANALOG - SPATIAL MEMORY & PATTERN RECOGNITION

    Cellular automata for pattern detection and spatial memory.

    Functions similar to:
    - Hippocampus: Map-like representation of environment
    - Place cells and grid cells: Spatial encoding
    - Pattern completion: Recognition of similar situations
    - Memory trace formation: Activation persists after stimuli

    The emergent pattern formation in CA mirrors how the hippocampus
    and associated limbic structures form cognitive maps of environments
    and recognize patterns across experiences.
    """

    def __init__(self, width=20, height=15):  # Reduced dimensions for efficiency
        self.width = width
        self.height = height
        self.state = np.zeros((height, width))

        # Define different patterns to recognize (like hippocampal place fields)
        self.patterns = {
            'threat': np.random.randint(0, 2, (height//3, width//3)),
            'safe': np.random.randint(0, 2, (height//3, width//3)),
            'explore': np.random.randint(0, 2, (height//3, width//3)),
            'action': np.random.randint(0, 2, (height//3, width//3))
        }

        # Create pattern activation history (like memory traces)
        self.pattern_activations = {k: 0.0 for k in self.patterns.keys()}

    def update(self, observation, reward):
        """
        Update cellular automata state based on observation and reward.
        This simulates how the hippocampus forms and updates cognitive maps.

        observation: The current visual input
        reward: The reward signal (influences pattern formation)
        """
        # Convert observation to binary format suitable for CA
        # First ensure observation is in the right format (2D binary)
        if observation is None:
            return

        # Handle different observation formats
        if len(observation.shape) == 3:  # Color image (H,W,C)
            # Convert to grayscale first
            if observation.shape[2] == 3:  # RGB/BGR
                gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            else:
                gray = observation[:,:,0]  # Just take first channel
        elif len(observation.shape) == 2:  # Already grayscale
            gray = observation
        else:
            # Unexpected format, return without updating
            print(f"Warning: Unexpected observation shape: {observation.shape}")
            return

        # Resize to match CA dimensions
        small_obs = cv2.resize(gray, (self.width, self.height))

        # Threshold to binary (0 or 1)
        _, binary = cv2.threshold(small_obs, 127, 1, cv2.THRESH_BINARY)

        # Now update state with proper binary observation
        self.state = np.logical_or(self.state, binary).astype(float)

        # Apply Game of Life rules (simulates neural network lateral interactions)
        new_state = self.state.copy()
        for y in range(self.height):
            for x in range(self.width):
                neighbors = self._count_neighbors(x, y)

                # Survival rules (requires 2-3 neighbors)
                if self.state[y, x] > 0:
                    if neighbors < 2 or neighbors > 3:
                        new_state[y, x] = 0
                # Birth rule (exactly 3 neighbors)
                else:
                    if neighbors == 3:
                        new_state[y, x] = 1

        self.state = new_state

        # Update pattern activations (simulates memory trace formation)
        decay_rate = 0.1
        for pattern_name, pattern in self.patterns.items():
            # Compute pattern activation (correlation with current state)
            activation = self._pattern_match(pattern)

            # Apply reward modulation (emotional valence affects memory)
            if pattern_name == 'threat' and reward < 0:
                activation *= 1.5  # Negative reward strengthens threat pattern
            elif pattern_name == 'safe' and reward > 0:
                activation *= 1.5  # Positive reward strengthens safe pattern

            # Update activation with decay (memory trace dynamics)
            self.pattern_activations[pattern_name] = (
                (1 - decay_rate) * self.pattern_activations[pattern_name] +
                decay_rate * activation
            )

    def _count_neighbors(self, x, y):
        """Count live neighbors for a cell - kept for compatibility but no longer used"""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                count += self.state[ny, nx]
        return count

    def _pattern_match(self, pattern):
        """Compute the correlation between the pattern and the current state"""
        # Ensure pattern can be compared with state
        ph, pw = pattern.shape

        # If pattern is smaller than state, we need to find the best matching region
        best_match = 0

        # Sample some positions to check (for efficiency)
        num_samples = 10
        for _ in range(num_samples):
            # Random position to check
            y = random.randint(0, max(0, self.height - ph))
            x = random.randint(0, max(0, self.width - pw))

            # Extract region and compute similarity
            region = self.state[y:min(y+ph, self.height), x:min(x+pw, self.width)]
            if region.shape == pattern.shape:
                similarity = np.sum(region * pattern) / (ph * pw)
                best_match = max(best_match, similarity)

        return best_match

    def get_pattern_activations(self):
        """
        Get activation levels for known patterns.
        This simulates how the hippocampus and amygdala recognize
        specific environmental patterns that signal safety, threat, etc.

        Returns:
            Dictionary mapping pattern names to activation levels (0-1)
        """
        # Simply return the current activation levels
        return self.pattern_activations.copy()

class YOLODetector:
    """
    VISUAL CORTEX DEEP NETWORK ANALOG

    YOLO-based object detector for monster recognition.
    Functions similar to the ventral visual stream and inferotemporal cortex:
    - Hierarchical visual processing (analogous to V1-V4-IT pathway)
    - Object category recognition (similar to inferotemporal neurons)
    - Parallel feature extraction (like retinotopic maps in visual cortex)
    - Invariant object recognition (similar to IT representation)

    The deep convolutional architecture mimics the hierarchical organization
    of the primate visual system, with early layers detecting basic features
    and deeper layers representing complex objects and categories.
    """

    def __init__(self, confidence_threshold=0.25):
        """
        Initialize the YOLO detector.

        Args:
            confidence_threshold: Minimum confidence level for detections
        """
        self.confidence_threshold = confidence_threshold
        self.is_initialized = False
        self.model = None
        self.class_names = None
        self.target_classes = ['person', 'monster']  # Classes we consider threats
        self.monster_template = None
        self.version = "yolov11"  # Using the latest version
        self.model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "yolo_models")

        # Check if we're in Colab
        self.is_colab = 'google.colab' in sys.modules

        # Create cache directory if it doesn't exist
        os.makedirs(self.model_cache_dir, exist_ok=True)

        # Check if PyTorch is available
        if not TORCH_AVAILABLE:
            print("PyTorch not available. YOLO detector will be disabled.")
            self.is_initialized = False
            return

        # Try to initialize the model
        try:
            self.init_model()
        except Exception as e:
            print(f"Failed to initialize YOLO model: {e}")
            # Create monster template for fallback detection
            self.monster_template = self._create_monster_template()

    def init_model(self):
        """
        Initialize the YOLO model.
        Analogous to the development of specialized visual neurons through
        experience and learning.
        """
        if self.is_initialized:
            return

        if not TORCH_AVAILABLE:
            return

        try:
            model_path = os.path.join(self.model_cache_dir, f"{self.version}.pt")

            # Check if we need to download the model
            if not os.path.exists(model_path):
                print(f"Downloading {self.version} model...")

                # Use different approaches based on environment
                if self.is_colab:
                    # In Colab, use torch hub to download
                    self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
                    # Save model for future use
                    torch.save(self.model.state_dict(), model_path)
                else:
                    # For local environment, use direct download
                    import urllib.request
                    model_url = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt"
                    urllib.request.urlretrieve(model_url, model_path)
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            else:
                # Load existing model
                print(f"Loading existing {self.version} model from {model_path}")
                try:
                    # First try loading directly from path
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
                except Exception as e:
                    print(f"Error loading from path: {e}, trying from hub...")
                    # Fallback to hub
                    self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

            # Move model to GPU if available
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

            # Set inference parameters
            self.model.conf = self.confidence_threshold  # Confidence threshold
            self.model.iou = 0.45  # NMS IoU threshold
            self.model.agnostic = False  # NMS class-agnostic
            self.model.multi_label = False  # NMS multiple labels per box
            self.model.max_det = 10  # Maximum detections per image

            self.is_initialized = True
            print("YOLO model successfully initialized")

            # Try a test inference to ensure everything works
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            self.model(test_image)

        except Exception as e:
            print(f"Failed to initialize YOLO model: {e}")
            traceback.print_exc()
            self.is_initialized = False

            # Create a template for traditional detection as fallback
            self.monster_template = self._create_monster_template()

    def _create_monster_template(self):
        """
        Create a template for traditional matching when YOLO is unavailable.
        This is analogous to innate visual recognition patterns in the brain.
        """
        # Create a simple template of a monster-like shape
        template = np.zeros((30, 20, 3), dtype=np.uint8)
        # Add monster body (reddish)
        cv2.rectangle(template, (5, 5), (15, 25), (0, 0, 180), -1)
        # Add monster eyes (greenish)
        cv2.circle(template, (8, 12), 2, (0, 180, 0), -1)
        cv2.circle(template, (12, 12), 2, (0, 180, 0), -1)
        return template

    def detect(self, image):
        """
        Detect monsters in an image using YOLO or fallback to traditional methods.
        Mimics the ventral stream's object recognition process.

        Args:
            image: RGB image to detect monsters in

        Returns:
            List of dictionaries with detection info: {'bbox': (x, y, w, h), 'confidence': conf, 'class_name': class_name}
        """
        # If YOLO is not initialized, use traditional detection
        if not self.is_initialized or self.model is None:
            return self._detect_traditional(image)

        try:
            # Ensure image is in RGB format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]  # Drop alpha channel

            # Run inference
            results = self.model(image)

            # Process detections
            detections = []

            # Extract predictions from model output
            if hasattr(results, 'xyxy'):  # YOLOv5 format
                boxes = results.xyxy[0].cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2, conf, class_id = box
                    class_name = results.names[int(class_id)]

                    # Only keep detections of target classes with high confidence
                    if class_name in self.target_classes and conf >= self.confidence_threshold:
                        # Convert to center format with normalized coordinates
                        h, w = image.shape[:2]
                        center_x = ((x1 + x2) / 2) / w
                        center_y = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h

                        # Create detection dictionary
                        detections.append({
                            'bbox': (center_x, center_y, width, height),
                            'confidence': float(conf),
                            'class_name': class_name
                        })
            else:  # Generic processing for other YOLO versions
                pred = results.pred[0].cpu().numpy() if hasattr(results, 'pred') else results.xyxy[0].cpu().numpy()

                for *xyxy, conf, cls in pred:
                    if conf >= self.confidence_threshold:
                        class_name = results.names[int(cls)]
                        if class_name in self.target_classes:
                            x1, y1, x2, y2 = xyxy
                            h, w = image.shape[:2]
                            center_x = ((x1 + x2) / 2) / w
                            center_y = ((y1 + y2) / 2) / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h

                            # Create detection dictionary
                            detections.append({
                                'bbox': (center_x, center_y, width, height),
                                'confidence': float(conf),
                                'class_name': class_name
                            })

            # If YOLO found nothing, fall back to traditional detection
            if len(detections) == 0:
                traditional_detections = self._detect_traditional(image)
                detections = traditional_detections

            return detections

        except Exception as e:
            print(f"YOLO detection error: {e}")
            return self._detect_traditional(image)

    def _detect_traditional(self, image):
        """
        Use traditional computer vision to detect monsters when YOLO fails.
        This is analogous to lower-level visual processing when higher-level
        recognition fails.
        """
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Copy image for processing
        img = image.copy()
        h, w = img.shape[:2]

        # Create monster color mask (reddish colors for DOOM monsters)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Define color ranges for monsters in DOOM (reddish and brown tones)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks and combine
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        monster_mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        monster_mask = cv2.morphologyEx(monster_mask, cv2.MORPH_OPEN, kernel)
        monster_mask = cv2.morphologyEx(monster_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(monster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size and shape
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Skip very small contours
            if area < 50:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Skip if too wide or too tall (likely a wall)
            aspect_ratio = float(w) / h
            if aspect_ratio > 3 or aspect_ratio < 0.3:
                continue

            # Normalize coordinates
            center_x = (x + w/2) / float(image.shape[1])
            center_y = (y + h/2) / float(image.shape[0])
            width = w / float(image.shape[1])
            height = h / float(image.shape[0])

            # Calculate a confidence score based on area and color match
            roi = monster_mask[y:y+h, x:x+w]
            fill_ratio = np.sum(roi) / (255.0 * w * h)
            confidence = min(0.9, fill_ratio * area / 5000)  # Cap at 0.9

            # Create detection dictionary
            detections.append({
                'bbox': (center_x, center_y, width, height),
                'confidence': confidence,
                'class_name': 'monster',
                'contour': contour  # Save contour for visualization
            })

        return detections

    def detect_with_fallback(self, image, fallback_detections):
        """
        Combines YOLO detections with fallback detection methods.
        This demonstrates how the brain integrates multiple visual pathways
        for robust perception.

        Args:
            image: RGB image to detect monsters in
            fallback_detections: Detections from another method to use if YOLO fails

        Returns:
            Combined list of detections
        """
        # First try YOLO detection
        yolo_detections = self.detect(image)

        # If YOLO found something with high confidence, trust it
        high_conf_detections = [d for d in yolo_detections if d['confidence'] > 0.6]
        if high_conf_detections:
            return high_conf_detections

        # If YOLO found something but with lower confidence, combine with fallbacks
        if yolo_detections:
            # For overlapping detections, keep the one with higher confidence
            combined_detections = yolo_detections.copy()

            for fb_det in fallback_detections:
                # Extract bbox data from dictionary
                fb_x, fb_y, fb_w, fb_h = fb_det['bbox']
                fb_conf = fb_det['confidence']

                # Check if this fallback detection overlaps with any YOLO detection
                overlaps = False
                for y_det in yolo_detections:
                    # Extract YOLO detection data
                    y_x, y_y, y_w, y_h = y_det['bbox']
                    y_conf = y_det['confidence']

                    # Calculate IoU (Intersection over Union)
                    x_left = max(fb_x - fb_w/2, y_x - y_w/2)
                    y_top = max(fb_y - fb_h/2, y_y - y_h/2)
                    x_right = min(fb_x + fb_w/2, y_x + y_w/2)
                    y_bottom = min(fb_y + fb_h/2, y_y + y_h/2)

                    if x_right < x_left or y_bottom < y_top:
                        intersection = 0
                    else:
                        intersection = (x_right - x_left) * (y_bottom - y_top)

                    fb_area = fb_w * fb_h
                    y_area = y_w * y_h
                    union = fb_area + y_area - intersection

                    iou = intersection / union if union > 0 else 0

                    if iou > 0.3:  # Significant overlap
                        overlaps = True
                        break

                # If no overlap, add the fallback detection
                if not overlaps:
                    # Give fallback detections slightly lower confidence
                    adjusted_conf = fb_conf * 0.8  # Reduce confidence by 20%

                    # Create a copy with adjusted confidence
                    modified_det = fb_det.copy()
                    modified_det['confidence'] = adjusted_conf
                    combined_detections.append(modified_det)

            return combined_detections

        # If YOLO found nothing, use fallback detections
        return fallback_detections

class DoomHDCSNNAgent:
    """
    INTEGRATED BRAIN SYSTEM WITH PREFRONTAL EXECUTIVE FUNCTION

    DOOM agent using HDC, SNN, and CA for decision making.
    Integrates different brain-inspired subsystems:
    - Prefrontal Cortex: Decision making with exploration/exploitation balance
    - Working Memory: Short-term storage in action and reward history
    - Episodic Memory: Experience storage and replay (hippocampal system)
    - Semantic Memory: Knowledge of actions and their meanings (cortical)
    - Executive Control: Coordination of perception, memory and action systems
    - Deep Visual Cortex: YOLO-based object detection for precise targeting

    The agent architecture mirrors the brain's hierarchical organization,
    with perception (HDC) feeding into memory systems (CA) and action selection
    (SNN), coordinated by higher-level executive functions (prefrontal analog).
    """

    def __init__(self, n_actions=8, hd_dim=500):  # Reduced from 1000 to 500
        # Initialize components
        self.hdc = HDCEncoder(D=hd_dim)
        self.snn = SpikingNeuralNetwork(input_size=hd_dim, num_neurons=50, num_actions=n_actions)  # Reduced from 100 to 50
        self.ca = CellularAutomata()

        # Initialize YOLO if available
        self.yolo = YOLODetector(confidence_threshold=0.3) if TORCH_AVAILABLE else None
        self.using_yolo = self.yolo is not None and self.yolo.is_initialized
        if self.using_yolo:
            print("Agent will use YOLO for enhanced monster detection")
        else:
            print("Agent will use traditional detection methods only")

        self.n_actions = n_actions
        self.previous_frame = None
        self.action_history = deque(maxlen=10)
        self.reward_history = deque(maxlen=100)

        # Memory systems
        self.episodic_memory = deque(maxlen=1000)
        self.semantic_memory = defaultdict(lambda: np.zeros(hd_dim))

        # Learning parameters
        self.learning_rate = 0.01
        self.gamma = 0.95
        self.epsilon = 0.1

    def process_observation(self, observation):
        """
        THALAMIC SENSORY INTEGRATION

        Process new observation through all systems.
        Similar to thalamic functions:
        - Sensory preprocessing and filtering
        - Integration of multiple input modalities
        - Relay of information to higher centers

        This process mimics the way the thalamus serves as a hub
        for sensory information before it reaches cortical areas.
        """
        if observation is None:
            return None

        # Convert to grayscale if RGB
        if len(observation.shape) == 3 and observation.shape[2] == 3:
            gray_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        else:
            gray_observation = observation

        # Downsample observation to reduce computation if needed
        if gray_observation.shape[0] > 60:  # If height > 60, downsample
            gray_observation = cv2.resize(gray_observation, (80, 60))

        # Calculate motion if possible
        motion = None
        if self.previous_frame is not None:
            # Ensure both frames have the same shape and type
            if self.previous_frame.shape != gray_observation.shape:
                self.previous_frame = cv2.resize(self.previous_frame, (gray_observation.shape[1], gray_observation.shape[0]))

            # Calculate motion on downsampled grayscale images
            motion = cv2.absdiff(gray_observation, self.previous_frame)

        # Create HD vector for observation
        obs_vector = self.hdc.encode_observation(gray_observation, motion)

        # Update CA with observation - but not on every step to save computation
        if random.random() < 0.5:  # Only update CA 50% of the time
            self.ca.update(gray_observation, 0)  # Initial reward of 0

        # Store frame for next iteration
        self.previous_frame = gray_observation.copy()

        return obs_vector

    def select_action(self, observation):
        """
        MULTI-REGION DECISION CIRCUIT

        Select action based on current state with improved shooting precision.
        Involves interaction of multiple brain-like systems:
        - Superior Colliculus: Visual target detection and orientation
        - Basal Ganglia: Action selection and prioritization
        - Prefrontal Cortex: Strategic decision making
        - Amygdala: Threat assessment and response
        - Motor Cortex: Execution of selected actions
        - Deep Visual Cortex: YOLO-based object detection for precise targeting

        The complex decision process mimics the interactions between
        cortical, limbic, and midbrain regions for goal-directed behavior.
        """
        # Process observation
        obs_vector = self.process_observation(observation)
        if obs_vector is None:
            return random.randint(0, self.n_actions - 1)

        # Get CA pattern activations
        ca_patterns = self.ca.get_pattern_activations()

        # Run observation through SNN
        output, _ = self.snn.simulate(obs_vector)

        # Enhanced monster detection for better shooting precision
        # Initialize monster detection info for visualization
        monster_info = {
            'detected': False,
            'rect': (0, 0, 0, 0),
            'area': 0,
            'offset': 0
        }

        """
        DORSAL/VENTRAL VISUAL STREAM PROCESSING

        The following section implements visual processing analogous to:
        - Ventral stream ("what" pathway): Object recognition and classification
        - Dorsal stream ("where/how" pathway): Spatial localization and motion
        - V4-like processing: Shape and color analysis
        - MT/MST-like processing: Motion detection and analysis
        - Inferior temporal cortex: Complex object recognition (monsters)

        The integration of shape, color, motion, and position information
        mirrors the hierarchical processing in primate visual cortex.
        """

        if len(observation.shape) == 3 and observation.shape[2] == 3:
            gray_obs = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            # Also get color version for color-based detection
            color_obs = observation.copy()
        else:
            gray_obs = observation.copy()
            color_obs = cv2.cvtColor(gray_obs, cv2.COLOR_GRAY2BGR)

        # Detect potential monster regions using traditional image processing methods first
        # This serves as both a fallback and a complementary approach to YOLO

        # 1. Resize for faster processing
        small_obs = cv2.resize(gray_obs, (160, 120))

        # 2. Apply preprocessing to enhance monster visibility
        # Use a combination of contrast enhancement and edge detection
        enhanced = cv2.equalizeHist(small_obs)  # Enhance contrast

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Create edge map - helps differentiate monsters from flat walls
        edges = cv2.Canny(blurred, 50, 150)

        # 3. Apply advanced thresholding with better parameters
        # Use local adaptive thresholding for better handling of different lighting
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Inverse to highlight monsters (usually darker)
            11,  # Block size
            2    # Constant subtracted from mean
        )

        # 4. Clean up binary image to remove noise
        # Apply morphological operations to clean up the image
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill small holes

        # 5. Extract color features if available (for RGB observations)
        color_mask = None
        motion_weight_map = None

        if color_obs is not None:
            # Resize color observation
            small_color = cv2.resize(color_obs, (160, 120))

            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(small_color, cv2.COLOR_RGB2HSV)

            # Define color ranges for typical DOOM monsters
            # Cacodemon - red/blue monster (more specific ranges)
            lower_caco1 = np.array([0, 70, 50])    # Red component
            upper_caco1 = np.array([10, 255, 255])

            lower_caco2 = np.array([160, 70, 50])  # Pink/purple component
            upper_caco2 = np.array([180, 255, 255])

            # Create masks for monster colors
            mask_caco1 = cv2.inRange(hsv, lower_caco1, upper_caco1)
            mask_caco2 = cv2.inRange(hsv, lower_caco2, upper_caco2)

            # Combine specific monster masks
            color_mask = cv2.bitwise_or(mask_caco1, mask_caco2)

            # Clean up color mask more aggressively
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

            # Create a more focused detection binary using both color and structure
            # Only allow detection of objects with both good structure and color match
            if color_mask is not None:
                # Weight the binary image to prefer regions that have monster colors
                binary_enhanced = cv2.bitwise_and(binary, color_mask)
                # But still keep some of the original binary to avoid missing monsters
                # with slight color variations
                binary = cv2.addWeighted(binary, 0.6, binary_enhanced, 0.4, 0)

            # Motion detection for moving monsters (if previous frame exists)
            if self.previous_frame is not None:
                # Ensure proper resize for previous frame
                prev_small = cv2.resize(self.previous_frame, (160, 120))
                # Calculate motion
                motion = cv2.absdiff(small_obs, prev_small)
                # Threshold motion
                _, motion_thresh = cv2.threshold(motion, 15, 255, cv2.THRESH_BINARY)
                # Clean motion map
                motion_thresh = cv2.morphologyEx(motion_thresh, cv2.MORPH_OPEN, kernel)
                # Use motion as a weighting factor - moving objects are more likely monsters
                motion_weight_map = motion_thresh

        # 6. Edge density filtering - monsters have more edges than flat walls
        edge_density_map = np.zeros_like(binary)
        if edges is not None:
            # Dilate edges to connect nearby edges
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            # Calculate edge density using a sliding window
            for y in range(0, edges.shape[0]-10, 5):
                for x in range(0, edges.shape[1]-10, 5):
                    roi = dilated_edges[y:y+10, x:x+10]
                    density = np.sum(roi) / 255.0 / 100.0  # Normalize by window size
                    # Only mark regions with significant edge density
                    if density > 0.15:  # More than 15% edges in window
                        edge_density_map[y:y+10, x:x+10] = 255

            # Add edge density information to binary detection map
            binary = cv2.bitwise_and(binary, binary, mask=edge_density_map)

        # 7. Find contours from the improved binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 8. Analyze contours with improved filtering
        monster_detected = False
        monster_center_x = 0
        best_monster_score = 0
        best_monster_rect = None
        best_monster_area = 0
        best_contour = None

        # Store preprocessed images for debug visualization
        self.debug_images = {
            'binary': binary,
            'color_mask': color_mask,
            'edges': edges,
            'edge_density': edge_density_map,
            'motion': motion_weight_map
        }

        # List to store traditional detections for YOLO integration
        traditional_detections = []

        if contours:
            for contour in contours:
                # Filter by size (monsters are typically medium-sized in frame)
                area = cv2.contourArea(contour)
                if 200 < area < 5000:  # Adjusted thresholds based on DOOM monsters
                    # Get bounding rect and check aspect ratio
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h

                    # Skip areas at the very bottom (likely the weapon)
                    if y > 90:  # Bottom 25% of screen
                        continue

                    # Skip areas at the very top (likely ceiling)
                    if y < 10:
                        continue

                    # Skip very wide or very tall shapes (likely walls/textures)
                    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                        continue

                    # Check for horizontal or vertical alignment - walls are typically perfectly aligned
                    rect = cv2.minAreaRect(contour)
                    angle = abs(rect[2] % 90)  # Normalize angle to 0-90 degrees
                    # If angle is very close to 0 or 90 degrees, it's likely a wall
                    if angle < 3 or angle > 87:
                        # Skip perfectly aligned rectangles
                        rect_area = w * h
                        # But only if they're also large and rectangular (wall-like)
                        if rect_area > 1000 and min(w, h) > 20:
                            continue

                    # Skip very rectangular shapes (likely wall edges)
                    rect_area = w * h
                    extent = float(area) / rect_area  # Area ratio
                    if extent < 0.5:  # Very sparse/hollow shapes are likely not monsters
                        continue

                    # Calculate convexity - monsters are typically more convex than wall textures
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0

                    # Skip shapes that are not solid enough (likely wall textures)
                    if solidity < 0.75:
                        continue

                    # Calculate perimeter-to-area ratio - monsters have more complex boundaries
                    perimeter = cv2.arcLength(contour, True)
                    compactness = perimeter**2 / (4 * np.pi * area) if area > 0 else 0
                    # Skip shapes that are too simple (likely walls) or too complex (noise)
                    if compactness < 1.2 or compactness > 5.0:
                        continue

                    # Calculate a comprehensive score based on multiple factors
                    center_x = x + w//2
                    center_y = y + h//2

                    # Distance from center of screen (prioritize centered targets)
                    center_offset_x = abs(center_x - small_obs.shape[1]//2)
                    center_offset_y = abs(center_y - small_obs.shape[0]//2)
                    center_dist = np.sqrt(center_offset_x**2 + center_offset_y**2)
                    center_score = 1.0 - (center_dist / (np.sqrt(small_obs.shape[1]**2 + small_obs.shape[0]**2) / 2))

                    # Prefer larger objects (closer monsters)
                    size_score = min(1.0, area / 1500)

                    # Prefer more monster-like shapes (complex but solid)
                    shape_score = solidity * min(1.0, compactness / 2.0)

                    # Check color feature for color observations
                    color_score = 0
                    if color_mask is not None:
                        # Calculate overlap between contour and color mask
                        mask = np.zeros_like(binary)
                        cv2.drawContours(mask, [contour], 0, 255, -1)
                        color_overlap = cv2.bitwise_and(color_mask, mask)
                        overlap_ratio = np.sum(color_overlap) / (np.sum(mask) + 1e-6)
                        color_score = overlap_ratio

                    # Motion score - prefer moving objects
                    motion_score = 0
                    if motion_weight_map is not None:
                        # Ensure mask is defined before using it
                        if not 'mask' in locals() or mask is None:
                            mask = np.zeros_like(binary)
                            cv2.drawContours(mask, [contour], 0, 255, -1)
                        motion_overlap = cv2.bitwise_and(motion_weight_map, mask)
                        motion_ratio = np.sum(motion_overlap) / (np.sum(mask) + 1e-6)
                        motion_score = motion_ratio

                    # Edge density score - prefer objects with good edge structure
                    edge_score = 0
                    if edge_density_map is not None:
                        # Ensure mask is defined before using it
                        if not 'mask' in locals() or mask is None:
                            mask = np.zeros_like(binary)
                            cv2.drawContours(mask, [contour], 0, 255, -1)
                        edge_overlap = cv2.bitwise_and(edge_density_map, mask)
                        edge_ratio = np.sum(edge_overlap) / (np.sum(mask) + 1e-6)
                        edge_score = edge_ratio

                    # Combine scores with appropriate weights
                    # Prioritize center position, size, and color match
                    monster_score = (
                        0.30 * center_score +
                        0.20 * size_score +
                        0.15 * shape_score +
                        0.15 * color_score +
                        0.10 * motion_score +
                        0.10 * edge_score
                    ) * (area / 1000.0)  # Scale by area to prefer larger objects

                    # Add to traditional detections list
                    traditional_detections.append({
                        'bbox': (x, y, w, h),  # Store as bbox for compatibility with detect_with_fallback
                        'confidence': monster_score,
                        'class_name': 'monster',
                        'contour': contour
                    })

                    if monster_score > best_monster_score:
                        monster_detected = True
                        best_monster_score = monster_score
                        monster_center_x = center_x
                        best_monster_rect = (x, y, w, h)
                        best_monster_area = area
                        best_contour = contour

        # Store monster detection for visualization (traditional method)
        if monster_detected:
            monster_info['detected'] = True
            monster_info['bbox'] = best_monster_rect
            monster_info['area'] = best_monster_area
            center_x = small_obs.shape[1] // 2
            monster_info['offset'] = monster_center_x - center_x
            monster_info['confidence'] = best_monster_score
            monster_info['class_name'] = 'monster'
            monster_info['contour'] = best_contour

            # Store in agent state for visualization
            self.detected_monster = monster_info
        else:
            self.detected_monster = {'detected': False}

        # Now use YOLO to detect monsters more accurately (if available)
        if self.using_yolo:
            yolo_detections = self.yolo.detect_with_fallback(color_obs, traditional_detections)
            self.yolo_detections = yolo_detections  # Store for visualization

            # Update monster info with best YOLO detection if available
            if yolo_detections:
                # Sort by confidence
                sorted_detections = sorted(yolo_detections, key=lambda x: x['confidence'], reverse=True)
                best_det = sorted_detections[0]

                # Extract data from detection dictionary
                x, y, w, h = best_det['bbox']
                confidence = best_det['confidence']
                class_name = best_det.get('class_name', 'monster')

                # Scale coordinates back to small_obs size for consistency with traditional method
                scale_x = 160.0 / color_obs.shape[1]
                scale_y = 120.0 / color_obs.shape[0]

                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                scaled_w = int(w * scale_x)
                scaled_h = int(h * scale_y)

                # Update monster position info
                center_x = small_obs.shape[1] // 2
                monster_center_x = scaled_x + scaled_w // 2
                monster_offset = monster_center_x - center_x

                # Store improved detection
                improved_info = {
                    'detected': True,
                    'bbox': (scaled_x, scaled_y, scaled_w, scaled_h),
                    'area': scaled_w * scaled_h,
                    'offset': monster_offset,
                    'confidence': confidence,
                    'class_name': class_name,
                    'contour': best_det.get('contour', None),
                    'yolo': True  # Flag that this was detected by YOLO
                }

                # Store in agent state for visualization
                self.detected_monster = improved_info

                # Set targeting variables for action selection
                monster_detected = True
                monster_center_x = monster_center_x
                best_monster_area = scaled_w * scaled_h
        else:
            # Not using YOLO, set empty list for visualization
            self.yolo_detections = []

        """
        BASAL GANGLIA & BRAINSTEM MOTOR CONTROL

        This section implements action selection analogous to:
        - Basal Ganglia: Action selection and inhibition of competing actions
        - Superior Colliculus: Orientation toward targets (turning)
        - Motor Cortex: Action execution and planning
        - Cerebellum: Fine motor coordination and timing

        The balance between offensive and exploratory behaviors mimics
        the interaction between cortical planning and subcortical reflexive circuits.
        """

        # Adjust output based on detected threats and spatial information
        if ca_patterns['threat'] > 0.6 or monster_detected:
            # Increase probability of shooting
            # Check if SHOOT action exists (usually at index 2)
            if 2 < len(output):
                output[2] += 1.5  # Boost SHOOT action (index 2)

            # If monster is detected with position info, adjust turning to aim
            if monster_detected:
                # Determine if monster is left or right of center
                center_x = small_obs.shape[1] // 2  # Center of screen
                offset = monster_center_x - center_x

                # Adjust turning amount based on offset magnitude
                # The further off-center, the stronger the turn
                turn_strength = min(2.0, abs(offset) / 10.0)

                # With YOLO, we can be more precise in our aiming
                aim_threshold = 6  # Reduced threshold for more precise aiming (was 8)

                # Check if TURN_LEFT (4) and TURN_RIGHT (5) actions exist in this scenario
                # In defend_center, they are typically at indices 0 and 1
                turn_left_idx = 0  # Default for defend_center
                turn_right_idx = 1  # Default for defend_center

                # For scenarios with more actions, they might be at different indices
                if len(output) > 4:
                    turn_left_idx = 4
                if len(output) > 5:
                    turn_right_idx = 5

                if offset < -aim_threshold:  # Monster is to the left
                    output[turn_left_idx] += turn_strength  # TURN_LEFT
                    # Reduce shooting probability when far off-center
                    if 2 < len(output):
                        output[2] *= max(0.5, 1.0 - abs(offset)/(center_x))
                elif offset > aim_threshold:  # Monster is to the right
                    output[turn_right_idx] += turn_strength  # TURN_RIGHT
                    # Reduce shooting probability when far off-center
                    if 2 < len(output):
                        output[2] *= max(0.5, 1.0 - abs(offset)/(center_x))
                else:
                    # Monster is centered - shoot!
                    if 2 < len(output):
                        output[2] += 3.0  # Extra boost for SHOOT (increased from 2.5)

                # Add proper aiming logic - adjust aim based on distance to target
                if best_monster_area < 300:  # Far away - lead the target
                    if self.previous_frame is not None:
                        # Try to predict movement direction
                        if offset < 0:  # Moving left
                            output[turn_left_idx] += 0.5  # Lead target by turning left
                        elif offset > 0:  # Moving right
                            output[turn_right_idx] += 0.5  # Lead target by turning right
        else:
            # No threat detected, prioritize movement and exploration
            # Check if FORWARD action exists (usually at index 3)
            if 3 < len(output):
                output[3] += 0.5  # FORWARD

            # Random turning to explore
            if random.random() < 0.3:
                # Determine indices for turn actions based on scenario
                turn_left_idx = 0  # Default for defend_center
                turn_right_idx = 1  # Default for defend_center

                # For scenarios with more actions, they might be at different indices
                if len(output) > 4:
                    turn_left_idx = 4
                if len(output) > 5:
                    turn_right_idx = 5

                if random.random() < 0.5:
                    output[turn_left_idx] += 0.3  # TURN_LEFT
                else:
                    output[turn_right_idx] += 0.3  # TURN_RIGHT

        # Epsilon-greedy action selection with reduced randomness for better precision
        if random.random() < self.epsilon * 0.7:  # Further reduced epsilon for more exploitation
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            action_idx = np.argmax(output)

        # Store action in history
        self.action_history.append(action_idx)

        # Convert action index to one-hot encoded list for ViZDoom
        # ViZDoom expects a list/array of 0s and 1s for each possible action
        action_list = [0] * self.n_actions
        action_list[action_idx] = 1

        return action_list

    def update(self, observation, action, reward, next_observation=None):
        """
        MEMORY CONSOLIDATION AND LEARNING SYSTEM

        Update agent's internal state and learning systems, analogous to:
        - Hippocampal memory consolidation
        - Basal ganglia reward learning
        - Cerebellum error-based learning
        - Cortical semantic memory formation

        Represents how the brain integrates new experiences with reward signals
        to update memory systems and refine future behavior.

        Parameters:
            observation: Current observation/state
            action: Action taken in current state (one-hot encoded list)
            reward: Reward received for action
            next_observation: Next observation/state (optional)
        """
        # Store reward in history
        self.reward_history.append(reward)

        # Update Cellular Automata with current observation and reward
        self.ca.update(observation, reward)

        # Convert observation to HD vector
        obs_vector = self.process_observation(observation)

        # Skip learning if observation cannot be processed
        if obs_vector is None:
            return

        # Process next observation if provided
        next_obs_vector = None
        if next_observation is not None:
            next_obs_vector = self.process_observation(next_observation)

        # If action is a one-hot encoded list, convert to index
        if isinstance(action, (list, np.ndarray)) and len(action) > 1:
            action_idx = np.argmax(action)
        else:
            action_idx = action

        # Create a one-hot target vector for the chosen action
        target = np.zeros(self.n_actions)
        target[action_idx] = 1.0

        # If reward is positive, boost the target for that action
        if reward > 0:
            target[action_idx] = 1.5  # Boost for positive reward
        elif reward < 0:
            target[action_idx] = 0.5  # Reduce for negative reward

        # Update SNN weights based on current observation and action taken
        self.snn.update_weights(obs_vector, target, self.learning_rate)

        # Store experience in episodic memory
        # Only store if we have both current and next observation
        if next_obs_vector is not None:
            experience = (obs_vector, action_idx, reward, next_obs_vector)
            self.episodic_memory.append(experience)
        else:
            # If no next_observation, store with None as next_obs_vector
            experience = (obs_vector, action_idx, reward, None)
            self.episodic_memory.append(experience)

        # Update semantic memory (action representations)
        action_vector = self.hdc.encode_action(action_idx)
        # Associate this action with its observation context
        self.semantic_memory[action_idx] = self.semantic_memory[action_idx] * 0.95 + obs_vector * 0.05

    def get_state(self):
        """Get current state of the agent"""
        state = {
            'reward_history': list(self.reward_history),
            'action_history': list(self.action_history),
            'ca_patterns': self.ca.get_pattern_activations(),
            'agent': self  # Include reference to agent for debug images
        }

        # Add monster detection information if available
        if hasattr(self, 'detected_monster'):
            state['detected_monster'] = self.detected_monster

        return state

class Visualizer:
    """
    CONSCIOUSNESS & INTROSPECTION ANALOG

    Visualization helper for the DOOM agent.

    While not directly implementing neural processes, this class represents
    functions analogous to:
    - Consciousness: Making internal representations available for observation
    - Self-monitoring: Tracking of internal states and decisions
    - Introspection: Ability to examine and visualize thought processes
    - Executive awareness: Monitoring of decision factors and outcomes

    The visualization acts as a "global workspace" in which multiple
    brain-inspired processes become integrated and available for observation,
    similar to theories of how consciousness emerges from neural activity.
    """

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.colors = {
            'threat': (0, 0, 255),    # Blue in BGR (appears as red)
            'safe': (0, 255, 0),      # Green in BGR
            'explore': (255, 255, 255), # White in BGR
            'action': (0, 255, 255),   # Yellow in BGR
            'person': (255, 0, 0),     # Blue in BGR
            'monster': (0, 0, 255),    # Red in BGR
            'yolo': (255, 0, 255)      # Magenta in BGR for YOLO detections
        }

    def draw_agent_state(self, frame, agent_state, observation):
        """Draw agent's internal state on the frame"""
        # Create a copy to avoid modifying the original
        display_frame = frame.copy()

        # Always ensure we have a BGR frame for drawing
        if len(display_frame.shape) == 2:
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
        elif display_frame.shape[2] == 3:
            # Make sure we're in BGR for OpenCV drawing operations
            # No-op if already BGR
            pass

        # Force resize to ensure consistent overlay positioning
        display_frame = cv2.resize(display_frame, (640, 480))

        # Define scale factors early to avoid undefined variable errors
        x_scale = display_frame.shape[1] / 160  # Assuming small_obs width is 160px
        y_scale = display_frame.shape[0] / 120  # Assuming small_obs height is 120px

        # Draw detection debug images if available
        if hasattr(agent_state.get('agent', {}), 'debug_images'):
            debug_images = agent_state['agent'].debug_images

            # Show binary detection image
            if 'binary' in debug_images and debug_images['binary'] is not None:
                # Create small overlay for debug binary image
                binary = debug_images['binary']
                binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                binary_resized = cv2.resize(binary_bgr, (120, 90))

                # Place in top-right corner
                display_frame[10:100, 510:630] = binary_resized
                cv2.rectangle(display_frame, (509, 9), (631, 101), (0, 255, 255), 1)
                cv2.putText(display_frame, "Detection", (518, 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Show edge detection overlay
            if 'edges' in debug_images and debug_images['edges'] is not None:
                # Create small overlay for debug edges
                edges = debug_images['edges']
                edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                edges_resized = cv2.resize(edges_bgr, (120, 90))

                # Place below binary image
                display_frame[110:200, 510:630] = edges_resized
                cv2.rectangle(display_frame, (509, 109), (631, 201), (0, 255, 0), 1)
                cv2.putText(display_frame, "Edges", (518, 122),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw YOLO detections if available
        if hasattr(agent_state.get('agent', {}), 'yolo_detections'):
            yolo_detections = agent_state['agent'].yolo_detections

            # Only proceed if we have detections
            if yolo_detections:
                # Scale from original frame size to display frame size
                frame_x_scale = display_frame.shape[1] / observation.shape[1]
                frame_y_scale = display_frame.shape[0] / observation.shape[0]

                for det in yolo_detections:
                    # Handle both tuple format (x, y, w, h, conf) and dict format
                    if isinstance(det, tuple):
                        x, y, w, h = det[0], det[1], det[2], det[3]
                        confidence = det[4] if len(det) > 4 else 0.0
                        class_name = 'monster'  # Default class name for tuple format
                    elif isinstance(det, dict):
                        if 'bbox' in det:
                            x, y, w, h = det['bbox']
                        else:
                            # Try to extract from rect if available
                            if 'rect' in det:
                                x, y, w, h = det['rect']
                            else:
                                continue  # Skip this detection if no bbox info
                        confidence = det.get('confidence', det.get('score', 0.0))
                        class_name = det.get('class_name', 'unknown')
                    else:
                        continue  # Skip unknown detection format

                    # Scale to display frame size
                    scaled_x = int(x * frame_x_scale)
                    scaled_y = int(y * frame_y_scale)
                    scaled_w = int(w * frame_x_scale)
                    scaled_h = int(h * frame_y_scale)

                    # Select color based on class (default to magenta for YOLO detections)
                    color = self.colors.get(class_name, self.colors['yolo'])

                    # Draw rectangle with slightly different style for YOLO detections
                    cv2.rectangle(display_frame,
                                 (scaled_x, scaled_y),
                                 (scaled_x + scaled_w, scaled_y + scaled_h),
                                 color, 2)

                    # Add class name and confidence as label
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(display_frame, label,
                               (scaled_x, scaled_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Draw center point
                    center_x = scaled_x + scaled_w // 2
                    center_y = scaled_y + scaled_h // 2
                    cv2.circle(display_frame, (center_x, center_y), 4, color, -1)

                    # Draw line from screen center to detection center for aiming
                    screen_center_x = display_frame.shape[1] // 2
                    screen_center_y = display_frame.shape[0] // 2
                    cv2.line(display_frame,
                            (screen_center_x, screen_center_y),
                            (center_x, center_y),
                            (0, 255, 255), 1, cv2.LINE_AA)

        # Draw monster detection overlay if available in agent_state
        if 'detected_monster' in agent_state:
            monster_info = agent_state['detected_monster']
            if monster_info.get('detected', False):
                # Draw rectangle around detected monster
                bbox_available = False
                if 'bbox' in monster_info:
                    x, y, w, h = monster_info['bbox']
                    bbox_available = True
                elif 'rect' in monster_info:  # For backward compatibility
                    x, y, w, h = monster_info['rect']
                    bbox_available = True

                if bbox_available:
                    # Scale coordinates to match display frame size
                    scaled_x = int(x * x_scale)
                    scaled_y = int(y * y_scale)
                    scaled_w = int(w * x_scale)
                    scaled_h = int(h * y_scale)

                    # Use different color for YOLO-detected monsters
                    rect_color = (255, 0, 255) if monster_info.get('yolo', False) else (0, 0, 255)

                    # Draw rectangle around monster
                    cv2.rectangle(display_frame, (scaled_x, scaled_y),
                                 (scaled_x + scaled_w, scaled_y + scaled_h),
                                 rect_color, 2)

                    # Draw crosshair at center of monster
                    center_x = scaled_x + scaled_w // 2
                    center_y = scaled_y + scaled_h // 2
                    cv2.drawMarker(display_frame, (center_x, center_y),
                                  (0, 255, 255), markerType=cv2.MARKER_CROSS,
                                  markerSize=20, thickness=2)

                    # Draw line from bottom center to monster center (aim line)
                    cv2.line(display_frame, (display_frame.shape[1]//2, display_frame.shape[0]),
                            (center_x, center_y), (0, 255, 255), 1)

                    # Show detection score
                    if 'confidence' in monster_info:
                        score = monster_info['confidence']
                    elif 'score' in monster_info:  # For backward compatibility
                        score = monster_info['score']
                    else:
                        score = 0.0

                    # If YOLO detection, display class name
                    if monster_info.get('yolo', False) and 'class_name' in monster_info:
                        score_text = f"{monster_info['class_name']}: {score:.2f}"
                    else:
                        score_text = f"{score:.2f}"

                    cv2.putText(display_frame, score_text,
                               (center_x + 15, center_y - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 1)

                    # Draw contour if available (shows exact shape detected)
                    if 'contour' in monster_info and monster_info['contour'] is not None:
                        contour = monster_info['contour'].copy()
                        # Scale contour to match display frame
                        contour = contour.astype(np.float32)
                        contour[:, 0, 0] *= x_scale
                        contour[:, 0, 1] *= y_scale
                        contour = contour.astype(np.int32)
                        cv2.drawContours(display_frame, [contour], 0, (0, 165, 255), 2)

        # Draw aiming guide - a center crosshair in the middle of screen
        # This helps player aim and see if monster is centered
        screen_center_x = display_frame.shape[1] // 2
        screen_center_y = display_frame.shape[0] // 2

        # Draw larger, more visible crosshair
        cv2.line(display_frame,
                (screen_center_x - 10, screen_center_y),
                (screen_center_x + 10, screen_center_y),
                (0, 255, 0), 1)
        cv2.line(display_frame,
                (screen_center_x, screen_center_y - 10),
                (screen_center_x, screen_center_y + 10),
                (0, 255, 0), 1)

        # Draw central aim circle with shooting threshold indicators
        cv2.circle(display_frame,
                  (screen_center_x, screen_center_y),
                  5, (0, 255, 0), 1)

        # Draw aim threshold indicators (where monster is considered "centered")
        # Use 6 pixels for YOLO-enhanced precision aiming (reduced from 8)
        threshold_x = int(6 * x_scale)
        cv2.line(display_frame,
                (screen_center_x - threshold_x, screen_center_y + 5),
                (screen_center_x - threshold_x, screen_center_y - 5),
                (0, 165, 255), 1)
        cv2.line(display_frame,
                (screen_center_x + threshold_x, screen_center_y + 5),
                (screen_center_x + threshold_x, screen_center_y - 5),
                (0, 165, 255), 1)

        # Get CA patterns
        ca_patterns = agent_state.get('ca_patterns', {})

        # Set text properties to match screenshot exactly
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        left_margin = 10

        # Move text position higher up for better visibility (at position visible in screenshot)
        text_y = 340  # Shifted even higher

        # Draw threat in red (BGR format: blue=0, green=0, red=255)
        if 'threat' in ca_patterns:
            cv2.putText(display_frame, f"threat: {ca_patterns['threat']:.2f}",
                       (left_margin, text_y), font, font_scale, (0, 0, 255), thickness)
            text_y += 20  # Line spacing

        # Draw safe in green
        if 'safe' in ca_patterns:
            cv2.putText(display_frame, f"safe: {ca_patterns['safe']:.2f}",
                       (left_margin, text_y), font, font_scale, (0, 255, 0), thickness)
            text_y += 20  # Line spacing

        # Draw explore in white
        if 'explore' in ca_patterns:
            cv2.putText(display_frame, f"explore: {ca_patterns['explore']:.2f}",
                       (left_margin, text_y), font, font_scale, (255, 255, 255), thickness)
            text_y += 20  # Line spacing

        # Draw action in cyan
        actions = agent_state.get('action_history', [])
        if actions:
            action_names = ['LEFT', 'RIGHT', 'SHOOT', 'FORWARD', 'TURN_L', 'TURN_R']
            last_action = action_names[actions[-1]] if actions[-1] < len(action_names) else 'UNKNOWN'
            cv2.putText(display_frame, f"Action: {last_action}",
                       (left_margin, text_y), font, font_scale, (0, 255, 255), thickness)
            text_y += 20  # Line spacing

        # Draw monster info if available
        if 'detected_monster' in agent_state:
            monster_info = agent_state['detected_monster']
            if monster_info.get('detected', False):
                # Show distance from center (aiming offset)
                offset = monster_info.get('offset', 0)
                cv2.putText(display_frame, f"Target offset: {offset:+.1f}",
                           (left_margin, text_y), font, font_scale, (0, 255, 255), thickness)
                text_y += 20

                # Show area of monster (size/distance indicator)
                area = monster_info.get('area', 0)
                cv2.putText(display_frame, f"Target size: {area}",
                           (left_margin, text_y), font, font_scale, (0, 255, 255), thickness)
                text_y += 20

                # Show classifier type (YOLO vs traditional)
                detector_type = "YOLO" if monster_info.get('yolo', False) else "Traditional"
                cv2.putText(display_frame, f"Detector: {detector_type}",
                           (left_margin, text_y), font, font_scale, (0, 255, 255), thickness)
                text_y += 20

        # Draw reward in purple or red
        rewards = agent_state.get('reward_history', [])
        if rewards:
            # Use purple for negative rewards as in screenshot
            reward_color = (128, 0, 128) if rewards[-1] < 0 else (0, 0, 255)
            cv2.putText(display_frame, f"Reward: {rewards[-1]}",
                       (left_margin, text_y), font, font_scale, reward_color, thickness)

        return display_frame

def setup_recording(filename="doom_agent"):
    """Setup video recording for Colab"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("/content/recordings", exist_ok=True)
    output_path = f"/content/recordings/{filename}_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))  # Will receive BGR frames
    return out

def display_frame(frame):
    """Display a frame in Colab"""
    # Try using IPython display method which works best in Colab
    try:
        # Make sure frame is in right format for display
        if len(frame.shape) == 2:
            # Convert grayscale to RGB
            display_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            # Convert BGR to RGB for display
            display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for display
        from PIL import Image
        img = Image.fromarray(display_rgb)

        # Use IPython display which works reliably in Colab
        # Scale down for display (but keep original resolution for recording)
        img = img.resize((640, 480))
        display(img)
        clear_output(wait=True)
        return
    except Exception as e:
        print(f"IPython display error, falling back to matplotlib: {e}")

    # Fallback to matplotlib if IPython display fails
    try:
        plt.figure(figsize=(12, 8))

        # Convert BGR to RGB for matplotlib display
        if len(frame.shape) == 2:
            # If grayscale
            plt.imshow(frame, cmap='gray')
        else:
            # Always convert BGR to RGB for matplotlib
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        plt.axis('off')
        plt.show()
        clear_output(wait=True)
        plt.close()  # Explicitly close figure to free memory
    except Exception as e:
        print(f"Display completely failed: {e}")

def correct_colors(frame):
    """Correct color balance to reduce blue tint and enhance brown tones"""
    if len(frame.shape) < 3:
        return frame  # Grayscale, no correction needed

    # Convert BGR to RGB for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Adjust color balance - reduce blue, enhance red/yellow
    # Uses weighted multiplication of RGB channels
    # These values help make walls appear more brown than blue
    rgb_frame = rgb_frame.astype(np.float32)
    rgb_frame[:,:,0] = np.clip(rgb_frame[:,:,0] * 1.2, 0, 255)  # Boost red
    rgb_frame[:,:,2] = np.clip(rgb_frame[:,:,2] * 0.85, 0, 255)  # Reduce blue
    rgb_frame = rgb_frame.astype(np.uint8)

    # Convert back to BGR for OpenCV
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

def test_agent(agent, n_episodes=3, record=True, memory_optimized=False):
    """
    BEHAVIORAL EXPRESSION AND PERFORMANCE ASSESSMENT

    Test the trained agent, analogous to:
    - Procedural memory expression in the basal ganglia
    - Habitual behavior via striatal circuits
    - Adaptive behavior through prefrontal-striatal loops
    - Performance monitoring in anterior cingulate

    This represents the brain's execution of learned skills,
    where training has created stable patterns that guide behavior.
    """
    # Initialize VizDoom environment
    game = vzd.DoomGame()

    # Get the correct path for the WAD file using the helper function
    try:
        wad_file = get_wad_file("defend_the_center.wad")
        print(f"Using WAD file: {wad_file}")
        game.set_doom_scenario_path(wad_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Trying one more approach - direct GitHub download...")
        # One last attempt directly in this function
        wad_url = "https://github.com/mwydmuch/ViZDoom/raw/master/scenarios/defend_the_center.wad"
        wad_file = "defend_the_center.wad"
        try:
            urllib.request.urlretrieve(wad_url, wad_file)
            print(f"Successfully downloaded {wad_file}")
            game.set_doom_scenario_path(wad_file)
        except Exception as download_error:
            raise Exception(f"Failed to download WAD file. Please download manually from {wad_url}")

    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)

    # Available buttons
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.ATTACK)
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.TURN_LEFT)
    game.add_available_button(vzd.Button.TURN_RIGHT)
    game.add_available_button(vzd.Button.MOVE_BACKWARD)
    game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)

    # Game variables for agent input
    game.add_available_game_variable(vzd.GameVariable.AMMO2)
    game.add_available_game_variable(vzd.GameVariable.HEALTH)
    game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)

    # Settings
    game.set_episode_timeout(2100)  # ~60 seconds
    game.set_episode_start_time(5)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_living_reward(-0.01)
    game.set_mode(vzd.Mode.PLAYER)

    # Initialize the game with proper error handling
    try:
        game.init()
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg and "defend_the_center.wad" in error_msg:
            # Try to download the WAD file directly from VizDoom GitHub repository
            print("Attempting one last download for WAD file from VizDoom GitHub repository...")
            wad_url = "https://github.com/mwydmuch/ViZDoom/raw/master/scenarios/defend_the_center.wad"
            wad_file = "defend_the_center.wad"
            try:
                urllib.request.urlretrieve(wad_url, wad_file)
                print(f"Successfully downloaded {wad_file}")
                # Update the path and try initializing again
                game.set_doom_scenario_path(wad_file)
                game.init()
            except Exception as download_error:
                raise Exception(f"Failed to download WAD file: {download_error}. Please download manually from {wad_url}")
        else:
            # Re-raise original error if it's not about missing WAD
            raise e

    # Initialize visualizer
    visualizer = Visualizer()

    # Create recording directory
    if record:
        recording_dir = '/content/recordings' if IN_COLAB else './recordings'
        os.makedirs(recording_dir, exist_ok=True)

    # Track test rewards
    test_rewards = []

    # Define frame processor with color correction
    def correct_colors(frame):
        """Apply color correction to reduce blue tint and enhance brown colors"""
        if frame is None:
            return None

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert BGR to RGB for processing (OpenCV uses BGR)
            # Adjust color balance to reduce blue and enhance red
            frame = frame.copy()  # Make a copy to avoid modifying the original
            # Reduce blue channel
            frame[:, :, 2] = np.clip(frame[:, :, 2] * 0.85, 0, 255).astype(np.uint8)
            # Enhance red channel
            frame[:, :, 0] = np.clip(frame[:, :, 0] * 1.15, 0, 255).astype(np.uint8)
        return frame

    def process_frame(frame):
        """Process frame for visualization with color correction"""
        # Apply color correction
        frame = correct_colors(frame)
        return frame

    # Test in each episode
    for episode in range(n_episodes):
        game.new_episode()
        episode_reward = 0

        # Create video writer for this episode
        video_writer = None
        if record:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(recording_dir, f'doom_test_episode_{episode}_{timestamp}.mp4')

            # Use lower resolution and framerate in memory-optimized mode
            if memory_optimized:
                frame_width, frame_height = 320, 240
                fps = 15
            else:
                frame_width, frame_height = 640, 480
                fps = 30

            # Initialize video writer
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
            except Exception as e:
                print(f"Warning: Could not create video writer: {e}")
                record = False  # Disable recording

        print(f"\nTesting Episode {episode + 1}/{n_episodes}")

        # For memory optimization in Colab, force garbage collection at the start
        if memory_optimized and IN_COLAB:
            import gc
            gc.collect()

            # Also check if YOLO detector needs to be reinitialized
            if hasattr(agent, 'using_yolo') and agent.using_yolo:
                try:
                    # Test YOLO detector health
                    if not agent.yolo.is_initialized:
                        print("Reinitializing YOLO detector for testing...")
                        agent.yolo = YOLODetector(confidence_threshold=0.3)
                        agent.using_yolo = agent.yolo is not None and agent.yolo.is_initialized
                except Exception as e:
                    print(f"YOLO reinitialization failed: {e}")
                    # Continue without YOLO
                    agent.using_yolo = False

        # Keep track of the previous observation for the update method
        previous_frame = None

        # Track frame count for sampling in memory-optimized mode
        frame_count = 0

        while not game.is_episode_finished():
            # Get state
            state = game.get_state()

            # Process frame
            frame = state.screen_buffer
            processed_frame = process_frame(frame)

            # Skip some frames in memory-optimized mode to save resources
            if memory_optimized and frame_count % 2 != 0:
                # Only do minimum processing on skipped frames
                action = agent.select_action(processed_frame)
                action_array = np.zeros(game.get_available_buttons_size())
                action_array[action] = 1
                reward = game.make_action(action_array)
                episode_reward += reward
                frame_count += 1

                # Still need to update previous_frame
                previous_frame = processed_frame
                continue

            # Select action
            action = agent.select_action(processed_frame)

            # Create action array
            action_array = np.zeros(game.get_available_buttons_size())
            action_array[action] = 1

            # Execute action
            reward = game.make_action(action_array)
            episode_reward += reward

            # Get next observation for the update method
            next_observation = None
            if not game.is_episode_finished():
                next_state = game.get_state()
                if next_state is not None:
                    next_frame = next_state.screen_buffer
                    next_observation = process_frame(next_frame)

            # Update agent using the previous observation as current and current as next
            if previous_frame is not None:
                agent.update(previous_frame, action, reward, processed_frame)

            # Update previous frame for next iteration
            previous_frame = processed_frame

            # Prepare agent state for visualization
            agent_state = {
                'agent': agent,
                'action_history': agent.action_history,
                'reward_history': agent.reward_history,
                'ca_patterns': agent.ca.get_pattern_activations()
            }

            if hasattr(agent, 'detected_monster'):
                agent_state['detected_monster'] = agent.detected_monster

            # Create visualization frame
            display_frame = visualizer.draw_agent_state(processed_frame, agent_state, processed_frame)

            # Convert to RGB for display
            if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
                display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            else:
                display_rgb = display_frame

            # Record video with memory optimization if needed
            if record and video_writer is not None:
                try:
                    # Resize if using memory optimization
                    if memory_optimized:
                        display_frame_resized = cv2.resize(display_frame, (frame_width, frame_height))
                    else:
                        display_frame_resized = display_frame

                    video_writer.write(display_frame_resized)
                except Exception as e:
                    print(f"Error writing video frame: {e}")

            # Display in notebook
            if IN_COLAB:
                # Colab-specific display with memory optimization
                clear_output(wait=True)

                # Use lower resolution for Colab to prevent memory issues
                if memory_optimized:
                    display_rgb = cv2.resize(display_rgb, (320, 240))

                # Convert to JPEG to reduce memory usage
                _, buffer = cv2.imencode('.jpg', display_rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                image_bytes = buffer.tobytes()

                # Display as HTML image
                display(HTML(f'<img src="data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}" />'))
                # Display additional info as text to save memory
                display(HTML(f'<p>Test Episode: {episode+1}/{n_episodes}, Reward: {episode_reward:.2f}</p>'))
            else:
                # Standard display for non-Colab environments
                plt.figure(figsize=(12, 9))
                plt.imshow(display_rgb)
                plt.axis('off')
                plt.title(f"Test Episode {episode+1}/{n_episodes}, Reward: {episode_reward:.2f}")
                plt.draw()
                plt.pause(0.001)
                plt.close()

            # For memory optimization in Colab, periodically force garbage collection
            frame_count += 1
            if memory_optimized and frame_count % 30 == 0:
                import gc
                gc.collect()

        test_rewards.append(episode_reward)
        print(f"Test Episode {episode+1} finished with reward {episode_reward}")

        # Close video writer
        if record and video_writer is not None:
            try:
                video_writer.release()
            except:
                pass

        # Force garbage collection between episodes
        if memory_optimized:
            import gc
            gc.collect()

        # Process final observation if needed
        if previous_frame is not None and not game.is_episode_finished():
            next_state = game.get_state()
            if next_state is not None:
                next_frame = next_state.screen_buffer
                next_observation = process_frame(next_frame)
                agent.update(previous_frame, action, reward, next_observation)

    # Clean up
    game.close()
    print(f"Testing completed. Average reward: {np.mean(test_rewards):.2f}")

    return test_rewards

class ScenarioManager:
    """Manages different DOOM game scenarios and their configurations"""

    # Define available scenarios with their configurations
    SCENARIO_CONFIGS = {
        "defend_center": {
            "wad_file": "defend_the_center.wad",
            "buttons": ["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
            "game_variables": ["AMMO2", "HEALTH"],
            "episode_timeout": 2100,
            "episode_start_time": 14,
            "living_reward": 0,
            "agent_params": {
                "hd_dim": 600,
                "learning_rate": 0.01,
                "ca_width": 20,
                "ca_height": 15
            },
            "skills": ["target acquisition", "timing", "weapon management"]
        },
        "health_gathering": {
            "wad_file": "health_gathering.wad",
            "buttons": ["MOVE_LEFT", "MOVE_RIGHT", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"],
            "game_variables": ["HEALTH"],
            "episode_timeout": 2100,
            "episode_start_time": 14,
            "living_reward": 1,
            "agent_params": {
                "hd_dim": 800,
                "learning_rate": 0.005,
                "ca_width": 25,
                "ca_height": 20
            },
            "skills": ["navigation", "resource management", "spatial awareness"]
        },
        "basic": {
            "wad_file": "basic.wad",
            "buttons": ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
            "game_variables": ["AMMO2"],
            "episode_timeout": 300,
            "episode_start_time": 14,
            "living_reward": 0,
            "agent_params": {
                "hd_dim": 500,
                "learning_rate": 0.02,
                "ca_width": 15,
                "ca_height": 10
            },
            "skills": ["aiming", "basic movement", "shooting"]
        },
        "defend_line": {
            "wad_file": "defend_the_line.wad",
            "buttons": ["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
            "game_variables": ["AMMO2", "HEALTH"],
            "episode_timeout": 2100,
            "episode_start_time": 14,
            "living_reward": 0,
            "agent_params": {
                "hd_dim": 700,
                "learning_rate": 0.008,
                "ca_width": 22,
                "ca_height": 18
            },
            "skills": ["advanced aiming", "target prioritization", "resource management"]
        },
        "deadly_corridor": {
            "wad_file": "deadly_corridor.wad",
            "buttons": ["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "ATTACK"],
            "game_variables": ["AMMO2", "HEALTH"],
            "episode_timeout": 2100,
            "episode_start_time": 14,
            "living_reward": 0,
            "agent_params": {
                "hd_dim": 1000,
                "learning_rate": 0.005,
                "ca_width": 30,
                "ca_height": 25
            },
            "skills": ["complex navigation", "combat in corridors", "strategic movement"]
        }
    }

    def __init__(self):
        """Initialize the scenario manager"""
        # Import ViZDoom here to handle import errors gracefully
        try:
            import vizdoom as vzd
            self.vzd = vzd
        except ImportError:
            print("Warning: vizdoom package not found. Scenario manager will have limited functionality.")
            self.vzd = None

    def setup_game_for_scenario(self, scenario_name):
        """
        Set up a VizDoom game instance for a specific scenario

        Args:
            scenario_name: Name of the scenario to configure

        Returns:
            Configured VizDoom game instance
        """
        if self.vzd is None:
            raise ImportError("vizdoom package is required for setting up game scenarios")

        if scenario_name not in self.SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        # Get configuration for this scenario
        config = self.SCENARIO_CONFIGS[scenario_name]

        # Create and configure the game
        game = self.vzd.DoomGame()

        # Set WAD file
        wad_file = get_wad_file(config["wad_file"])
        game.set_doom_scenario_path(wad_file)

        # Set screen format and resolution
        game.set_screen_format(self.vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(self.vzd.ScreenResolution.RES_160X120)

        # Set available buttons
        buttons = [getattr(self.vzd.Button, button) for button in config["buttons"]]
        game.set_available_buttons(buttons)

        # Set game variables
        game_vars = [getattr(self.vzd.GameVariable, var) for var in config["game_variables"]]
        game.set_available_game_variables(game_vars)

        # Set other parameters
        game.set_episode_timeout(config["episode_timeout"])
        game.set_episode_start_time(config["episode_start_time"])
        game.set_window_visible(False)
        game.set_living_reward(config["living_reward"])
        game.set_render_hud(False)
        game.set_render_minimal_hud(False)

        return game

    def get_optimal_agent_params(self, scenario_name):
        """
        Get the optimal agent parameters for a specific scenario

        Args:
            scenario_name: Name of the scenario

        Returns:
            Dictionary of optimal agent parameters
        """
        if scenario_name not in self.SCENARIO_CONFIGS:
            print(f"Warning: Unknown scenario: {scenario_name}, using default parameters")
            return {"hd_dim": 600, "learning_rate": 0.01, "ca_width": 20, "ca_height": 15}

        return self.SCENARIO_CONFIGS[scenario_name].get("agent_params", {"hd_dim": 600, "learning_rate": 0.01, "ca_width": 20, "ca_height": 15})

    def list_scenarios(self):
        """
        List all available scenarios and their key properties
        """
        print("\nAvailable Scenarios:")
        print("=" * 80)
        for name, config in self.SCENARIO_CONFIGS.items():
            skills = config.get("skills", ["N/A"])
            params = config.get("agent_params", {})
            print(f"- {name.upper()}: {', '.join(skills)}")
            print(f"  WAD file: {config['wad_file']}")
            print(f"  Parameters: HD dim={params.get('hd_dim', 'N/A')}, " +
                  f"Buttons={len(config['buttons'])}, " +
                  f"Timeout={config['episode_timeout']}")
            print("-" * 80)

    def download_scenario(self, scenario_name):
        """
        Download the WAD file for a scenario if not already present

        Args:
            scenario_name: Name of the scenario to download

        Returns:
            Path to the downloaded WAD file
        """
        if scenario_name not in self.SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        config = self.SCENARIO_CONFIGS[scenario_name]
        wad_filename = config["wad_file"]

        try:
            # Try to get the WAD file using the helper function
            return get_wad_file(wad_filename)
        except FileNotFoundError:
            # If not found, download it
            print(f"WAD file {wad_filename} not found. Downloading...")

            # Base URL for ViZDoom scenarios
            base_url = "https://github.com/mwydmuch/ViZDoom/raw/master/scenarios/"
            wad_url = f"{base_url}{wad_filename}"

            # Determine where to save the file
            save_dir = "./scenarios"
            if IN_COLAB:
                save_dir = "/content/scenarios"

            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, wad_filename)

            # Download the file
            try:
                import urllib.request
                urllib.request.urlretrieve(wad_url, save_path)
                print(f"Successfully downloaded {wad_filename}")
                return save_path
            except Exception as e:
                print(f"Error downloading WAD file: {e}")
                print(f"Please download manually from: {wad_url}")
                raise

    def get_scenario_info(self, scenario_name):
        """
        Get detailed information about a scenario

        Args:
            scenario_name: Name of the scenario

        Returns:
            Dictionary with detailed scenario information
        """
        if scenario_name not in self.SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        # Create a copy of the config to avoid modifying the original
        info = dict(self.SCENARIO_CONFIGS[scenario_name])

        # Add additional information
        info["name"] = scenario_name

        # Default skills if not defined
        if "skills" not in info:
            info["skills"] = ["basic gameplay", "survival"]

        return info

def setup_doom_env(scenario="defend_center"):
    """
    Setup the ViZDoom environment with the specified scenario

    Args:
        scenario: Name of the scenario to use (from ScenarioManager.SCENARIO_CONFIGS)

    Returns:
        game: Configured VizDoom game instance
    """
    try:
        import vizdoom as vzd
    except ImportError:
        print("Error: vizdoom package not found. Please install it with: pip install vizdoom")
        raise

    # Create scenario manager if not already created
    if not hasattr(setup_doom_env, 'scenario_manager'):
        setup_doom_env.scenario_manager = ScenarioManager()

    scenario_manager = setup_doom_env.scenario_manager

    # Check if the specified scenario exists
    available_scenarios = list(ScenarioManager.SCENARIO_CONFIGS.keys())
    if scenario not in available_scenarios:
        print(f"Warning: Scenario '{scenario}' not recognized. Available scenarios: {available_scenarios}")
        print(f"Defaulting to 'defend_center'")
        scenario = "defend_center"

    # Setup the game for the specified scenario
    try:
        game = scenario_manager.setup_game_for_scenario(scenario)
        return game
    except Exception as e:
        print(f"Error setting up the game: {e}")
        traceback.print_exc()
        print("Using default defend_the_center configuration as fallback")

        # Fallback to manual configuration if scenario setup fails
        game = vzd.DoomGame()

        # Try to get WAD file
        wad_file = get_wad_file("defend_the_center.wad")
        game.set_doom_scenario_path(wad_file)

        # Set screen format
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)

        # Set available buttons
        game.set_available_buttons([
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.ATTACK
        ])

        # Set game variables
        game.set_available_game_variables([
            vzd.GameVariable.AMMO2,
            vzd.GameVariable.HEALTH
        ])

        # Set other parameters
        game.set_episode_timeout(2100)
        game.set_episode_start_time(14)
        game.set_window_visible(False)
        game.set_living_reward(0)
        game.set_render_hud(False)
        game.set_render_minimal_hud(False)

        return game

def train_agent(n_episodes=50, visualize=False, memory_optimized=False, scenario="defend_center", save_model=True):
    """
    Train the agent using a specified scenario

    Args:
        n_episodes: Number of episodes to train for
        visualize: Whether to display visualization during training
        memory_optimized: Whether to use memory optimization techniques
        scenario: Which DOOM scenario to use for training
        save_model: Whether to save model checkpoints
    """
    # Print training info header
    print(f"Starting training for {n_episodes} episodes on scenario '{scenario}'")
    print("="*60)

    # Setup display for visualization
    if visualize:
        if IN_COLAB:
            # Virtual display for Colab
            display = setup_virtual_display()

    # Get appropriate game environment
    try:
        game = setup_doom_env(scenario)
    except Exception as e:
        print(f"Error setting up environment: {e}")
        return

    # Get optimal parameters for this scenario
    scenario_manager = setup_doom_env.scenario_manager
    params = scenario_manager.get_optimal_agent_params(scenario)

    # Initialize agent with scenario-specific parameters
    agent = DoomHDCSNNAgent(
        n_actions=game.get_available_buttons_size(),
        hd_dim=params.get("hd_dim", 600),
    )

    # Create a Visualizer only if needed
    if visualize:
        visualizer = Visualizer(640, 480)

    # Setup directories for saving
    save_dir = f"./models/{scenario}"
    os.makedirs(save_dir, exist_ok=True)

    # Game initialization
    game.init()

    # Training variables
    start_time = time.time()
    total_reward = 0
    kills = 0
    deaths = 0

    # For frame processing
    def correct_colors(frame):
        """Correct color channels if needed"""
        if frame is None:
            return None
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return frame  # Already RGB
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return frame

    def process_frame(frame):
        """Process the frame for agent input"""
        if frame is None:
            return np.zeros((120, 160, 3), dtype=np.uint8)

        frame = correct_colors(frame)

        # Memory optimization if requested
        if memory_optimized:
            # Downscale to save memory
            frame = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)

        return frame

    # Tracking metrics
    rewards_history = []
    kills_history = []

    try:
        # Training loop
        for episode in range(n_episodes):
            # Track episode stats
            episode_reward = 0
            episode_kills = 0
            step = 0

            # New episode
            game.new_episode()

            # Get initial frame
            prev_frame = None

            # Episode loop
            while not game.is_episode_finished():
                # Get state
                state = game.get_state()
                frame = state.screen_buffer

                # Process frame
                processed_frame = process_frame(frame)

                # Select action
                action = agent.select_action(processed_frame)

                # Execute action
                reward = game.make_action(action)
                episode_reward += reward

                # Check if episode is done
                done = game.is_episode_finished()

                # Get next frame if not done
                if not done:
                    next_state = game.get_state()
                    next_frame = next_state.screen_buffer
                    next_frame = process_frame(next_frame)

                    # Update agent with both current and next observation
                    agent.update(processed_frame, action, reward, next_frame)

                    # Track kills
                    if hasattr(next_state, 'game_variables'):
                        # In some scenarios, kill count is tracked in game variables
                        if len(next_state.game_variables) > 0:
                            current_kills = next_state.game_variables[0]
                            if step > 0 and current_kills > episode_kills:
                                episode_kills = current_kills
                                if visualize:
                                    print(f"Kill at step {step}! Total kills: {episode_kills}")
                else:
                    # Just update with current observation
                    agent.update(processed_frame, action, reward)

                # Display progress
                if step % 10 == 0:
                    print(f"Episode {episode+1}/{n_episodes} - Step {step} - Reward: {episode_reward:.2f}", end="\r")

                # Visualization
                if visualize and step % 5 == 0:  # Reduce visualization frequency to improve performance
                    # Display current state
                    agent_state = agent.get_state()
                    vis_frame = visualizer.draw_agent_state(processed_frame, agent_state, processed_frame)

                    # Display the frame
                    display_frame(vis_frame)

                # Save previous frame
                prev_frame = processed_frame
                step += 1

            # Episode finished
            total_reward += episode_reward
            kills += episode_kills

            # Check if the agent died (in most scenarios, negative reward at end means death)
            if reward < 0:
                deaths += 1

            # Save history for plotting
            rewards_history.append(episode_reward)
            kills_history.append(episode_kills)

            # Print episode results
            print(f"Episode {episode+1}/{n_episodes} finished - Reward: {episode_reward:.2f} - Kills: {episode_kills} - Steps: {step}      ")

            # Save model checkpoint every 10 episodes
            if save_model and (episode + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f"agent_checkpoint_ep{episode+1}.pkl")
                save_agent(agent, checkpoint_path)
                print(f"Model saved to {checkpoint_path}")

        # Training completed
        elapsed_time = time.time() - start_time
        print("="*60)
        print(f"Training completed in {elapsed_time:.2f} seconds")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Total kills: {kills}")
        print(f"Total deaths: {deaths}")

        # Save the final model
        if save_model:
            final_model_path = os.path.join(save_dir, "agent_final.pkl")
            save_agent(agent, final_model_path)
            print(f"Final model saved to {final_model_path}")

        # Plot training progress
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(rewards_history)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.subplot(1, 2, 2)
        plt.plot(kills_history)
        plt.title('Episode Kills')
        plt.xlabel('Episode')
        plt.ylabel('Kills')

        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(save_dir, "training_progress.png"))
        if visualize:
            plt.show()

        return rewards_history, agent

    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        return None
    finally:
        # Close the game
        game.close()

def test_agent(agent=None, n_episodes=3, record=True, memory_optimized=False, scenario="defend_center"):
    """
    Test a trained agent

    Args:
        agent: Pre-trained agent to test (if None, a new agent will be created)
        n_episodes: Number of episodes to test for
        record: Whether to record a video of the agent's performance
        memory_optimized: Whether to use memory optimization
        scenario: Which scenario to test on
    """
    # Print test info
    print(f"Testing agent on scenario '{scenario}' for {n_episodes} episodes")
    print("="*60)

    # Setup video recording if needed
    video_writer = None
    record_dir = f"./recordings/{scenario}"

    if record:
        os.makedirs(record_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_path = os.path.join(record_dir, f"doom_agent_test_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        print(f"Recording video to {video_path}")

    # Setup game environment
    game = setup_doom_env(scenario)

    # Create a new agent if none provided
    if agent is None:
        # Get optimal parameters for this scenario
        scenario_manager = setup_doom_env.scenario_manager
        params = scenario_manager.get_optimal_agent_params(scenario)

        # Create a new agent with scenario-specific parameters
        agent = DoomHDCSNNAgent(
            n_actions=game.get_available_buttons_size(),
            hd_dim=params.get("hd_dim", 600),
        )

        # Try to load the latest model for this scenario
        save_dir = f"./models/{scenario}"
        if os.path.exists(save_dir):
            model_files = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]
            if model_files:
                # Find the latest model
                latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(save_dir, x)))
                model_path = os.path.join(save_dir, latest_model)
                print(f"Loading pre-trained model from {model_path}")
                try:
                    load_agent(agent, model_path)
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Continuing with untrained agent")

    # Create visualizer for display
    visualizer = Visualizer(640, 480)

    # Initialize game
    game.init()
    game.set_window_visible(True)  # Show window during testing

    # Game variable indices
    ammo_index = 0
    health_index = 1

    # Metrics tracking across all episodes
    all_survival_times = []
    all_kills = []
    all_rewards = []
    all_action_entropies = []
    all_sparsities = []
    metrics_data = []  # For CSV export

    # Frame processing functions
    def correct_colors(frame):
        """Ensure correct color channels"""
        if frame is None:
            return None
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return frame
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return frame

    def process_frame(frame):
        """Process frame for agent input"""
        if frame is None:
            return np.zeros((120, 160, 3), dtype=np.uint8)

        frame = correct_colors(frame)

        if memory_optimized:
            frame = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)

        return frame

    try:
        # Test loop
        for episode in range(n_episodes):
            # Start a new episode
            game.new_episode()

            # Episode stats
            episode_reward = 0
            episode_kills = 0
            step = 0
            episode_actions = []  # Track actions for entropy calculation

            # Initialize previous frame
            previous_frame = None

            # Episode loop
            while not game.is_episode_finished():
                # Get state
                if game.is_player_dead():
                    # Handle player death
                    print("Player died!")
                    break

                state = game.get_state()
                frame = state.screen_buffer

                # Process frame
                processed_frame = process_frame(frame)

                # Select action based on processed frame
                action = agent.select_action(processed_frame)
                episode_actions.append(action)  # Record action for entropy calculation

                # Execute action
                reward = game.make_action(action)
                episode_reward += reward

                # Check if episode is done
                done = game.is_episode_finished()

                # Update previous_frame after the first step
                if previous_frame is None:
                    previous_frame = processed_frame

                # Get next frame if not done
                if not done:
                    next_state = game.get_state()
                    next_frame = process_frame(next_state.screen_buffer)

                    # Update agent with current and next observation
                    agent.update(processed_frame, action, reward, next_frame)

                    # Track kills
                    if hasattr(next_state, 'game_variables') and len(next_state.game_variables) > 0:
                        current_kills = next_state.game_variables[ammo_index]
                        if step > 0 and current_kills > episode_kills:
                            episode_kills = current_kills
                            print(f"Kill at step {step}! Total kills: {episode_kills}")

                    # Update previous frame
                    previous_frame = processed_frame
                else:
                    # Final update with just the current observation
                    agent.update(processed_frame, action, reward)

                # Get agent state for visualization
                agent_state = agent.get_state()

                # Create visualization frame
                vis_frame = visualizer.draw_agent_state(processed_frame, agent_state, processed_frame)

                # Record video if enabled
                if record and video_writer is not None:
                    video_writer.write(vis_frame)

                # Display progress
                if step % 10 == 0:
                    print(f"Episode {episode+1}/{n_episodes} - Step {step} - Reward: {episode_reward:.2f}", end="\r")

                step += 1

            # Episode end stats
            print(f"Episode {episode+1}/{n_episodes} finished - Reward: {episode_reward:.2f} - Kills: {episode_kills} - Steps: {step}      ")
            
            # Calculate episode metrics
            all_survival_times.append(step)
            all_kills.append(episode_kills)
            all_rewards.append(episode_reward)
            
            # Calculate memory activation sparsity
            sparsity = 0
            if hasattr(agent, 'ca') and hasattr(agent.ca, 'grid'):
                ca_grid = agent.ca.grid
                inactive_cells = np.sum(ca_grid == 0)
                total_cells = ca_grid.size
                sparsity = (inactive_cells / total_cells) * 100
            all_sparsities.append(sparsity)
            
            # Calculate action entropy
            entropy = 0
            normalized_entropy = 0
            if episode_actions:
                actions = np.array(episode_actions)
                unique_actions, counts = np.unique(actions, return_counts=True)
                probabilities = counts / counts.sum()
                entropy = -np.sum(probabilities * np.log2(probabilities))
                max_entropy = np.log2(len(unique_actions))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            all_action_entropies.append(normalized_entropy)
            
            # Store metrics for CSV export
            metrics_data.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'survival_time': step,
                'kills': episode_kills,
                'memory_sparsity': sparsity,
                'action_entropy': entropy,
                'normalized_entropy': normalized_entropy
            })
            
            # Display metrics after the last episode
            if episode == n_episodes - 1:
                print("\nEvaluation Metrics Summary:")
                print("=" * 60)
                
                # 1. Survival time
                avg_survival = sum(all_survival_times) / len(all_survival_times)
                max_survival = max(all_survival_times)
                print(f"Survival time: avg={avg_survival:.2f}, max={max_survival} steps")
                
                # 2. Enemy elimination count
                avg_kills = sum(all_kills) / len(all_kills)
                total_kills = sum(all_kills)
                print(f"Enemy eliminations: avg={avg_kills:.2f}, total={total_kills}")
                
                # 3. Memory activation sparsity
                avg_sparsity = sum(all_sparsities) / len(all_sparsities)
                print(f"Memory activation sparsity: avg={avg_sparsity:.2f}%")
                
                # 4. Action entropy
                avg_entropy = sum(all_action_entropies) / len(all_action_entropies)
                print(f"Action entropy (normalized): avg={avg_entropy:.4f}")

    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        if record and video_writer is not None:
            video_writer.release()
            
        # Export metrics to CSV
        if metrics_data:
            import csv
            csv_file = f"./metrics/{scenario}_evaluation_metrics.csv"
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
                writer.writeheader()
                writer.writerows(metrics_data)
            print(f"Evaluation metrics exported to {csv_file}")
            
            # Plot metrics visualization
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Create plot directory
                plot_dir = f"./metrics/plots/{scenario}"
                os.makedirs(plot_dir, exist_ok=True)
                
                # Create a figure with subplots
                fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                episodes = [data['episode'] for data in metrics_data]
                
                # Plot 1: Survival Time and Rewards
                ax1 = axs[0, 0]
                ax1.plot(episodes, [data['survival_time'] for data in metrics_data], 'b-', label='Survival Time')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Steps', color='b')
                ax1.tick_params(axis='y', labelcolor='b')
                
                ax1b = ax1.twinx()
                ax1b.plot(episodes, [data['reward'] for data in metrics_data], 'r-', label='Reward')
                ax1b.set_ylabel('Reward', color='r')
                ax1b.tick_params(axis='y', labelcolor='r')
                ax1.set_title('Survival Time and Rewards')
                
                # Create combined legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1b.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                # Plot 2: Enemy Eliminations
                ax2 = axs[0, 1]
                ax2.bar(episodes, [data['kills'] for data in metrics_data], color='green')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Enemy Eliminations')
                ax2.set_title('Enemy Eliminations per Episode')
                
                # Plot 3: Memory Activation Sparsity
                ax3 = axs[1, 0]
                ax3.plot(episodes, [data['memory_sparsity'] for data in metrics_data], 'purple')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Memory Sparsity (%)')
                ax3.set_title('Memory Activation Sparsity')
                
                # Plot 4: Action Entropy
                ax4 = axs[1, 1]
                ax4.plot(episodes, [data['normalized_entropy'] for data in metrics_data], 'orange')
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Normalized Entropy')
                ax4.set_title('Action Entropy (Normalized)')
                
                # Adjust layout and save
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/evaluation_metrics.png", dpi=300)
                
                # Additionally create a radar chart for the last episode metrics
                fig_radar = plt.figure(figsize=(8, 8))
                ax_radar = fig_radar.add_subplot(111, polar=True)
                
                # Prepare data - normalize all metrics to 0-1 scale
                last_data = metrics_data[-1]
                
                # Calculate max values for normalization
                max_survival = max([data['survival_time'] for data in metrics_data])
                max_kills = max([data['kills'] for data in metrics_data]) if max([data['kills'] for data in metrics_data]) > 0 else 1
                max_reward = max([data['reward'] for data in metrics_data]) if max([data['reward'] for data in metrics_data]) > 0 else 1
                
                # Normalized values (range 0-1)
                stats = [
                    last_data['survival_time'] / max_survival,
                    last_data['kills'] / max_kills,
                    last_data['memory_sparsity'] / 100,  # Already in percentage
                    last_data['normalized_entropy'],  # Already normalized
                    last_data['reward'] / max_reward
                ]
                
                # Labels
                labels = ['Survival Time', 'Eliminations', 'Memory Sparsity', 'Action Entropy', 'Reward']
                
                # Number of variables
                N = len(labels)
                
                # Compute angle for each axis
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Add the data
                stats = np.concatenate((stats, [stats[0]]))  # Close the loop
                
                # Draw the radar chart
                ax_radar.plot(angles, stats, 'o-', linewidth=2)
                ax_radar.fill(angles, stats, alpha=0.25)
                
                # Set labels and title
                ax_radar.set_thetagrids(np.degrees(angles[:-1]), labels)
                ax_radar.set_title("Agent Performance Profile", size=15)
                
                # Adjust radar chart
                ax_radar.grid(True)
                plt.savefig(f"{plot_dir}/performance_radar.png", dpi=300)
                
                print(f"Metrics visualization saved to {plot_dir}")
                
                # Display plots if in interactive environment
                try:
                    plt.show()
                except:
                    pass
                    
                # Close figures to free memory
                plt.close('all')
                
            except Exception as e:
                print(f"Warning: Could not generate metrics plots: {e}")
                traceback.print_exc()

        game.close()
        print("Testing completed")
        
        # Return metrics for further analysis
        return metrics_data

# Modified main execution
if __name__ == "__main__":
    import gc  # Import garbage collector
    import urllib.request
    import os

    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("Running in Google Colab environment")

        # Apply Colab-specific optimizations
        # 1. Configure TensorFlow to prevent memory conflicts with PyTorch
        try:
            import tensorflow as tf

            # Limit TensorFlow memory growth to prevent consuming all GPU memory
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("TensorFlow memory growth enabled")

            # Limit TensorFlow to use only a portion of available memory
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
                )
                print("TensorFlow memory limited to prevent conflicts")
            except:
                print("Could not limit TensorFlow memory (non-critical)")
        except Exception as e:
            print(f"TensorFlow configuration error (non-critical): {e}")

        # 2. Set up virtual display if in Colab
        try:
            from pyvirtualdisplay import Display
            virtual_display = Display(visible=0, size=(1400, 900))
            virtual_display.start()
            print("Virtual display started")
        except Exception as e:
            print(f"Could not initialize virtual display: {e}")

        # 3. Set environment variables for stability
        import os
        if TORCH_AVAILABLE:
            # Force PyTorch to release memory after operations
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            print("Set PyTorch memory allocation limits for stability")

    except:
        IN_COLAB = False
        print("Not running in Google Colab environment")

    # Download necessary ViZDoom files if they don't exist
    print("Checking for necessary ViZDoom WAD files...")
    wad_file = "defend_the_center.wad"
    wad_url = "https://github.com/mwydmuch/ViZDoom/raw/master/scenarios/defend_the_center.wad"

    if not os.path.exists(wad_file):
        print(f"WAD file not found. Downloading {wad_file}...")
        try:
            urllib.request.urlretrieve(wad_url, wad_file)
            print(f"Successfully downloaded {wad_file}")
        except Exception as e:
            print(f"Error downloading WAD file: {e}")
            print("Please download manually from:", wad_url)
            import sys
            sys.exit(1)  # Exit if we can't get the necessary files
    else:
        print(f"WAD file {wad_file} found.")

    # Check if YOLO dependencies are available with extra safeguards for Colab
    if TORCH_AVAILABLE:
        try:
            # Test if we can load the model safely
            if IN_COLAB:
                print("Testing YOLO model loading in Colab environment...")
                # Use CPU only in Colab to avoid library conflicts
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
                # Force reload to fix cache issues in Colab
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
                print("YOLO model loaded successfully on CPU!")
            else:
                # Normal loading for non-Colab environments
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                print("YOLO model loaded successfully!")

            # Free memory
            del model
            if torch.cuda.is_available() and not IN_COLAB:
                torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
            print("Will use traditional detection methods instead")
            TORCH_AVAILABLE = False

    # Set lower episode counts, especially for Colab to prevent timeouts
    if IN_COLAB:
        n_train_episodes = 10  # Even fewer episodes for Colab
        n_test_episodes = 3
    else:
        n_train_episodes = 20  # Reduced number of episodes
        n_test_episodes = 5

    # Clean up memory before starting
    gc.collect()

    # Train the agent
    print("Training agent...")
    try:
        # Start training with memory optimization
        if IN_COLAB:
            print("Using memory-optimized training for Colab...")
            # Use a smaller replay buffer and batch size in Colab
            rewards, trained_agent = train_agent(n_episodes=n_train_episodes, visualize=True, memory_optimized=True)
        else:
            rewards, trained_agent = train_agent(n_episodes=n_train_episodes, visualize=True)

        # Plot training results (with lower DPI for memory savings)
        plt.figure(figsize=(6, 3), dpi=80)  # Reduced figure size and DPI
        plt.plot(rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.tight_layout()  # Ensure everything fits
        plt.show()
        plt.close()  # Explicitly close figure

        # Force garbage collection before testing
        gc.collect()

        # Test the agent
        print("\nTesting agent...")
        test_rewards = test_agent(trained_agent, n_episodes=n_test_episodes, record=True)

        # Plot test results (with lower DPI)
        if test_rewards:
            plt.figure(figsize=(6, 3), dpi=80)  # Reduced figure size and DPI
            plt.plot(test_rewards)
            plt.title('Test Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.tight_layout()
            plt.show()
            plt.close()  # Explicitly close figure

        print("\nDone! Videos saved in /content/recordings/")

        # Free memory before zipping
        gc.collect()

        # Download the recorded videos but with compression
        if IN_COLAB:
            try:
                from google.colab import files

                # Create a zip file with the recordings
                with zipfile.ZipFile('/content/doom_recordings.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files_list in os.walk('/content/recordings/'):
                        for file in files_list:
                            zipf.write(os.path.join(root, file))

                # Download the zip file
                files.download('/content/doom_recordings.zip')
                print("Zip file ready for download in Colab")
            except Exception as e:
                print(f"Error during file download: {e}")
                print("You can manually download the file from /content/doom_recordings.zip")
        else:
            print("Not running in Colab. Zip file saved to ./doom_recordings.zip")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for better debugging

def demonstrate_multi_scenario_learning(scenarios=None, n_episodes_per_scenario=30, visualize=False):
    """
    Train and test the agent across multiple scenarios to demonstrate
    cognitive flexibility and transfer learning capabilities.

    This function simulates how the brain can generalize skills across different
    environments and tasks, adapting existing knowledge to new challenges.

    Args:
        scenarios: List of scenarios to train on (defaults to a progression of difficulty)
        n_episodes_per_scenario: How many episodes to train for in each scenario
        visualize: Whether to visualize the training process
    """
    # Default scenarios in order of increasing difficulty/complexity
    if scenarios is None:
        scenarios = [
            "basic",              # Simple targeting
            "defend_center",      # Turning and shooting
            "defend_line",        # More complex shooting
            "health_gathering",   # Navigation for resources
            "deadly_corridor"     # Complex navigation and combat
        ]

    print("=" * 80)
    print("MULTI-SCENARIO LEARNING DEMONSTRATION")
    print("=" * 80)
    print(f"Training progression: {' → '.join(scenarios)}")
    print(f"Episodes per scenario: {n_episodes_per_scenario}")
    print("=" * 80)

    # Initialize scenario manager and list available scenarios
    scenario_manager = ScenarioManager()
    scenario_manager.list_scenarios()

    # Pre-download all needed scenarios
    print("\nPre-downloading required scenarios...")
    for scenario in scenarios:
        try:
            scenario_manager.download_scenario(scenario)
        except Exception as e:
            print(f"Error downloading {scenario}: {e}")
            print(f"Removing {scenario} from the training progression.")
            scenarios.remove(scenario)

    # Check if we still have scenarios to train on
    if not scenarios:
        print("No valid scenarios remain. Aborting.")
        return

    # Initialize the agent with parameters optimized for the most complex scenario
    most_complex = scenarios[-1]
    params = scenario_manager.get_optimal_agent_params(most_complex)

    print(f"\nInitializing agent with parameters optimized for {most_complex}:")
    print(f"- HD dimension: {params['hd_dim']}")
    print(f"- Learning rate: {params['learning_rate']}")
    print(f"- CA dimensions: {params['ca_width']}x{params['ca_height']}")

    # Get number of actions from the most complex scenario to ensure compatibility
    game = scenario_manager.setup_game_for_scenario(most_complex)
    n_actions = game.get_available_buttons_size()
    game.close()

    agent = DoomHDCSNNAgent(
        n_actions=n_actions,
        hd_dim=params["hd_dim"]
    )

    # Track performance metrics across scenarios
    scenario_metrics = {}

    # Train across scenarios in sequence
    for i, scenario in enumerate(scenarios):
        print("\n" + "=" * 80)
        print(f"SCENARIO {i+1}/{len(scenarios)}: {scenario.upper()}")
        print(f"Skills targeted: {', '.join(scenario_manager.get_scenario_info(scenario)['skills'])}")
        print("=" * 80)

        # Train agent on this scenario
        print(f"\nTraining on {scenario} for {n_episodes_per_scenario} episodes...")
        training_result = train_agent(
            n_episodes=n_episodes_per_scenario,
            visualize=visualize,
            memory_optimized=True,
            scenario=scenario,
            save_model=True
        )

        # If training was successful, unpack the result
        if training_result is not None:
            rewards_history, trained_agent = training_result
        else:
            print(f"Training on {scenario} failed. Skipping to next scenario.")
            continue

        # Agent should be returned from train_agent, but if not, use our existing agent
        if trained_agent is not None:
            agent = trained_agent

        # Test the agent to measure performance
        print(f"\nTesting {scenario} performance...")
        test_agent(
            agent=agent,
            n_episodes=3,
            record=True,
            scenario=scenario
        )

        # Save the agent specifically for this scenario
        save_dir = f"./models/{scenario}"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"agent_after_progressive_training.pkl")
        save_agent(agent, model_path)
        print(f"Agent saved to {model_path}")

    # Final cross-scenario evaluation to measure knowledge transfer
    print("\n" + "=" * 80)
    print("FINAL EVALUATION - TESTING TRANSFER LEARNING")
    print("=" * 80)

    # Test on each scenario to measure transfer learning effects
    for scenario in scenarios:
        print(f"\nTesting on {scenario} with the final agent:")
        test_agent(
            agent=agent,
            n_episodes=3,
            record=True,
            scenario=scenario
        )

    print("\n" + "=" * 80)
    print("MULTI-SCENARIO LEARNING COMPLETED")
    print("=" * 80)
    print("The agent has been trained across a progression of scenarios,")
    print("demonstrating how brain-inspired agents can transfer knowledge")
    print("between different environments and tasks.")

    return agent

def save_agent(agent, filepath):
    """Save agent to disk using pickle"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(agent, f)
        return True
    except Exception as e:
        print(f"Error saving agent: {e}")
        return False

def load_agent(agent, filepath):
    """Load agent state from disk"""
    try:
        with open(filepath, 'rb') as f:
            loaded_agent = pickle.load(f)

        # Copy over the learned parameters
        agent.hd_encoder = loaded_agent.hd_encoder
        agent.snn = loaded_agent.snn
        agent.ca = loaded_agent.ca

        # Copy over history and memories if they exist
        if hasattr(loaded_agent, 'episodic_memory'):
            agent.episodic_memory = loaded_agent.episodic_memory
        if hasattr(loaded_agent, 'action_history'):
            agent.action_history = loaded_agent.action_history
        if hasattr(loaded_agent, 'reward_history'):
            agent.reward_history = loaded_agent.reward_history

        return True
    except Exception as e:
        print(f"Error loading agent: {e}")
        return False
