"""
Hyperdimensional Computing (HDC) Encoder

Implements a brain-inspired encoding mechanism for visual and state information.
Functions similar to the visual processing stream from V1-V5:
- Edge detection (analogous to V1 simple cells)
- Position encoding (analogous to spatial maps in parietal cortex)
- Motion processing (analogous to MT/V5 motion selective neurons)
- Feature binding (analogous to integration in higher visual areas)
"""

import numpy as np
import cv2

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
        self._initialize_base_vectors()

    def _initialize_base_vectors(self):
        """Initialize random bipolar vectors for basic features"""
        self.base_vectors = {}
        
        # Create random bipolar vectors for basic features
        features = ['edge', 'motion', 'shape', 'position', 'action']
        for feature in features:
            self.base_vectors[feature] = np.random.choice([-1, 1], size=self.D)

    def create_position_vector(self, x, y, resolution=(120, 160)):
        """Create position vector for a given x, y coordinate"""
        # Normalize coordinates to [0, 1]
        x_norm = x / resolution[1]
        y_norm = y / resolution[0]
        
        # Create x and y vectors using continuous mapping
        x_vec = self._continuous_to_hd(x_norm)
        y_vec = self._continuous_to_hd(y_norm)
        
        # Bind x and y vectors to create position vector
        return self._bind(x_vec, y_vec)

    def _continuous_to_hd(self, value):
        """Map a continuous value to an HD vector"""
        # Create a vector that smoothly varies with the input value
        phase = 2 * np.pi * value
        return np.sign(np.sin(np.arange(self.D) * phase / self.D))

    def _bind(self, vec1, vec2):
        """Bind two vectors using element-wise multiplication"""
        return vec1 * vec2

    def _bundle(self, vectors):
        """Combine multiple vectors by addition and binarization"""
        if not vectors:
            return np.zeros(self.D)
        
        # Sum all vectors
        result = np.sum(vectors, axis=0)
        
        # Binarize result
        return np.sign(result)

    def encode_observation(self, frame, motion=None):
        """
        Encode a frame and optional motion information into an HD vector.
        
        Args:
            frame: The current observation frame
            motion: Optional motion frame (difference between frames)
            
        Returns:
            HD vector representing the observation
        """
        if frame is None:
            return None
            
        # Ensure frame is grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
            
        # Detect edges
        edges = cv2.Canny(gray_frame, 100, 200)
        
        # Container for all feature vectors to bundle
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
                    # Get average motion in block
                    block = motion[y:y+16, x:x+16]
                    if block.size > 0:
                        motion_val = np.mean(block)
                        if motion_val > 10:  # Only consider significant motion
                            pos_vec = self.create_position_vector(x+8, y+8)  # Center of block
                            motion_vectors.append(self._bind(
                                self.base_vectors['motion'], 
                                self._bind(pos_vec, self._continuous_to_hd(motion_val/255))
                            ))
        
        # Combine all feature vectors
        all_vectors = edge_vectors + motion_vectors
        
        # If no features detected, return a random vector
        if not all_vectors:
            return np.random.choice([-1, 1], size=self.D)
            
        # Bundle all vectors
        return self._bundle(all_vectors)

    def encode_action(self, action):
        """Encode an action as an HD vector"""
        # Convert action to integer if it's a one-hot vector
        if isinstance(action, (list, np.ndarray)) and len(action) > 1:
            action = np.argmax(action)
            
        # Create action vector by combining with position
        if action not in self.item_memory:
            # Create a new random vector for this action
            base_vec = np.random.choice([-1, 1], size=self.D)
            self.item_memory[action] = base_vec
        
        return self.item_memory[action]

    def similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) 