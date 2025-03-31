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

# Import YOLO detector (with error handling for missing dependencies)
try:
    from brain.perception.yolo_detector import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

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

    def __init__(self, D=1000, use_yolo=False):  # D is the dimensionality of HD vectors
        self.D = D
        self.item_memory = {}
        
        # Initialize YOLO detector if requested and available
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.yolo_detector = None
        
        # Now initialize base vectors after setting use_yolo
        self._initialize_base_vectors()
        
        if self.use_yolo:
            if not YOLO_AVAILABLE:
                print("Warning: YOLO detection requested but dependencies not available")
                print("Install with: pip install torch torchvision ultralytics")
            else:
                self.yolo_detector = YOLODetector(model_size='n')
                print(f"YOLO detector initialized using {self.yolo_detector.device}")

    def _initialize_base_vectors(self):
        """Initialize random bipolar vectors for basic features"""
        self.base_vectors = {}
        
        # Create random bipolar vectors for basic features
        features = ['edge', 'motion', 'shape', 'position', 'action']
        for feature in features:
            self.base_vectors[feature] = np.random.choice([-1, 1], size=self.D)
            
        # Add YOLO-related features if needed
        if self.use_yolo:
            yolo_features = ['person', 'enemy', 'weapon', 'health', 'ammo', 'door']
            for feature in yolo_features:
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
        
        # Convert to grayscale for edge detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
            
        # Edge detection (V1-like processing)
        edges = cv2.Canny(gray, 50, 150)
        edge_positions = np.where(edges > 0)
        
        # Create vectors for edge positions
        edge_vectors = []
        for y, x in zip(edge_positions[0], edge_positions[1]):
            pos_vector = self.create_position_vector(x, y, (gray.shape[0], gray.shape[1]))
            edge_vector = self._bind(pos_vector, self.base_vectors['edge'])
            edge_vectors.append(edge_vector)
            
        # Combine edge vectors
        if edge_vectors:
            edges_hd = self._bundle(edge_vectors)
        else:
            edges_hd = np.zeros(self.D)
            
        # Process motion information if available
        if motion is not None:
            motion_vectors = []
            # Find areas with significant motion
            motion_points = np.where(motion > 30)
            for y, x in zip(motion_points[0], motion_points[1]):
                pos_vector = self.create_position_vector(x, y, (motion.shape[0], motion.shape[1]))
                motion_vector = self._bind(pos_vector, self.base_vectors['motion'])
                motion_vectors.append(motion_vector)
                
            if motion_vectors:
                motion_hd = self._bundle(motion_vectors)
                # Combine with edge information
                hd_vector = self._bind(edges_hd, motion_hd)
            else:
                hd_vector = edges_hd
        else:
            hd_vector = edges_hd
            
        # Add YOLO-based object detection if enabled
        if self.use_yolo and self.yolo_detector is not None:
            try:
                # Detect objects in the frame
                detections = self.yolo_detector.detect(frame)
                
                # Create object vectors for each detection
                object_vectors = []
                for det in detections:
                    # Position encoding for object center
                    cx, cy = det['center']
                    pos_vector = self.create_position_vector(
                        cx, cy, (frame.shape[0], frame.shape[1])
                    )
                    
                    # Class encoding
                    class_name = det['class_name']
                    if class_name not in self.item_memory:
                        self.item_memory[class_name] = np.random.choice([-1, 1], size=self.D)
                    class_vector = self.item_memory[class_name]
                    
                    # Size encoding
                    size = max(det['size']) / max(frame.shape[:2])  # normalize size
                    size_vector = self._continuous_to_hd(size)
                    
                    # Combine properties with binding
                    object_vector = self._bind(pos_vector, class_vector)
                    object_vector = self._bind(object_vector, size_vector)
                    
                    # Scale by confidence
                    object_vector = object_vector * det['confidence']
                    
                    object_vectors.append(object_vector)
                    
                # Bundle all object vectors
                if object_vectors:
                    object_hd = self._bundle(object_vectors)
                    
                    # Combine with basic encoding
                    hd_vector = self._bind(hd_vector, object_hd)
            except Exception as e:
                print(f"Error in YOLO processing: {e}")
                
        return hd_vector

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