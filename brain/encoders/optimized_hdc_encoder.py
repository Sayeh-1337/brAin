"""
Optimized Hyperdimensional Computing (HDC) Encoder

Implements an optimized brain-inspired encoding mechanism for visual and state information:
- Uses efficient HDVectorSpace for operations
- GPU acceleration for encoding operations
- Efficient batch processing and caching
- YOLO object detection integration
"""

import torch
import numpy as np
import cv2

from brain.encoders.hd_vector_space import HDVectorSpace
from brain.utils.config import config

# Import YOLO detector (with error handling for missing dependencies)
try:
    from brain.perception.yolo_detector import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class OptimizedHDCEncoder:
    """
    VISUAL CORTEX ANALOG
    
    Optimized hyperdimensional computing encoder for visual and state information.
    Functions similar to the visual processing stream from V1-V5:
    - Edge detection (analogous to V1 simple cells)
    - Position encoding (analogous to spatial maps in parietal cortex)
    - Motion processing (analogous to MT/V5 motion selective neurons)
    - Feature binding (analogous to integration in higher visual areas)
    
    Features:
    - GPU acceleration with PyTorch
    - Efficient batch processing
    - Vector caching for frequently used items
    - YOLO object detection integration
    """
    
    def __init__(self, dimension=10000, binary=True, use_yolo=False, device=None):
        """
        Initialize the HDC encoder
        
        Args:
            dimension: Dimensionality of HD vectors
            binary: Whether to use binary (-1/1) or continuous vectors
            use_yolo: Whether to use YOLO object detection
            device: Computation device ('cpu', 'cuda')
        """
        self.dimension = dimension
        self.binary = binary
        self.device = device if device is not None else config.device
        
        # Create efficient HDVectorSpace for operations
        self.hd_space = HDVectorSpace(dimension=dimension, binary=binary, device=self.device)
        
        # Initialize YOLO detector if requested and available
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.yolo_detector = None
        
        # Cache for previously encoded items - initialize before using in _initialize_base_vectors
        self.cache = {}
        
        # Initialize base vectors
        self._initialize_base_vectors()
        
        # Set up YOLO detector if requested
        if self.use_yolo:
            if not YOLO_AVAILABLE:
                print("Warning: YOLO detection requested but dependencies not available")
                print("Install with: pip install torch torchvision ultralytics")
            else:
                self.yolo_detector = YOLODetector(model_size='n')
                print(f"YOLO detector initialized using {self.yolo_detector.device}")
        
    def _initialize_base_vectors(self):
        """Initialize random vectors for basic features"""
        # Create random vectors for basic features
        features = ['edge', 'motion', 'shape', 'position', 'action', 
                   'texture', 'color', 'orientation', 'frequency']
        
        for feature in features:
            self.hd_space.item_memory[feature] = self.hd_space.random(1).squeeze(0)
            
        # Add YOLO-related features if needed
        if self.use_yolo:
            yolo_features = ['person', 'enemy', 'weapon', 'health', 'ammo', 'door',
                           'monster', 'pickup', 'obstacle', 'key']
            for feature in yolo_features:
                self.hd_space.item_memory[feature] = self.hd_space.random(1).squeeze(0)
                
        # Precompute position vectors for common coordinates
        # This greatly speeds up encoding of edges and features
        resolution = (120, 160)  # Standard observation size
        positions = {}
        
        # Only precompute a subset of positions to save memory
        step = 4  # Compute every 4th pixel
        for y in range(0, resolution[0], step):
            for x in range(0, resolution[1], step):
                positions[(x, y)] = self.create_position_vector(x, y, resolution)
                
        self.position_vectors = positions
        
    def create_position_vector(self, x, y, resolution=(120, 160)):
        """
        Create position vector for a given x, y coordinate
        
        Args:
            x: X coordinate
            y: Y coordinate
            resolution: Image resolution
            
        Returns:
            HD vector encoding the position
        """
        # Check cache first
        cache_key = f"pos_{x}_{y}_{resolution[0]}_{resolution[1]}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Normalize coordinates to [0, 1]
        x_norm = x / resolution[1]
        y_norm = y / resolution[0]
        
        # Create separate vectors for x and y
        x_vec = self.hd_space.encode_scalar(x_norm, 0, 1)
        y_vec = self.hd_space.encode_scalar(y_norm, 0, 1)
        
        # Bind x and y vectors
        pos_vector = self.hd_space.bind(x_vec, y_vec)
        
        # Cache result
        self.cache[cache_key] = pos_vector
        
        return pos_vector
    
    def find_nearest_position(self, x, y, resolution=(120, 160)):
        """
        Find the nearest precomputed position vector
        
        Args:
            x: X coordinate
            y: Y coordinate
            resolution: Image resolution
            
        Returns:
            Closest precomputed position vector
        """
        # If positions empty, create vector directly
        if not self.position_vectors:
            return self.create_position_vector(x, y, resolution)
            
        # Find nearest precomputed position
        min_dist = float('inf')
        nearest_pos = None
        
        for (px, py), vec in self.position_vectors.items():
            dist = (x - px)**2 + (y - py)**2
            if dist < min_dist:
                min_dist = dist
                nearest_pos = vec
                
        return nearest_pos
    
    def encode_observation(self, frame, motion=None, batch_process=True):
        """
        Encode a frame and optional motion information into an HD vector
        
        Args:
            frame: The current observation frame
            motion: Optional motion frame (difference between frames)
            batch_process: Whether to use batch processing for efficiency
            
        Returns:
            HD vector representing the observation
        """
        if frame is None:
            return None
            
        # Convert numpy array to torch tensor if needed
        if isinstance(frame, np.ndarray):
            frame_tensor = torch.from_numpy(frame).to(self.device)
        else:
            frame_tensor = frame.to(self.device)
            
        # Convert to grayscale for edge detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
            
        # Edge detection (V1-like processing)
        edges = cv2.Canny(gray, 50, 150)
        edge_positions = np.where(edges > 0)
        
        # Use efficient batch processing if enabled
        if batch_process and config.batch_process and len(edge_positions[0]) > 0:
            # Batch process edge vectors
            edge_vectors = self._batch_process_edges(edge_positions, gray.shape)
            
            # We now have a tensor
            edges_hd = edge_vectors.mean(dim=0)
            # Apply thresholding if using binary vectors
            if self.binary:
                edges_hd = torch.sign(edges_hd)
        else:
            # Create vectors for edge positions (sequential)
            edge_vectors = []
            for y, x in zip(edge_positions[0], edge_positions[1]):
                pos_vector = self.find_nearest_position(x, y, (gray.shape[0], gray.shape[1]))
                edge_vector = self.hd_space.bind(pos_vector, self.hd_space.item_memory['edge'])
                edge_vectors.append(edge_vector)
                
            # Combine edge vectors if we have any
            if len(edge_vectors) > 0:
                edges_hd = self.hd_space.bundle(edge_vectors)
            else:
                edges_hd = torch.zeros(self.dimension, device=torch.device(self.device))
            
        # Process motion information if available
        if motion is not None:
            if batch_process and config.batch_process:
                # Batch process motion vectors
                motion_hd = self._batch_process_motion(motion)
                if motion_hd is not None:
                    # Combine with edge information
                    edges_hd = self.hd_space.bind(edges_hd, motion_hd)
            else:
                motion_vectors = []
                # Find areas with significant motion
                motion_points = np.where(motion > 30)
                for y, x in zip(motion_points[0], motion_points[1]):
                    pos_vector = self.find_nearest_position(x, y, (motion.shape[0], motion.shape[1]))
                    motion_vector = self.hd_space.bind(pos_vector, self.hd_space.item_memory['motion'])
                    motion_vectors.append(motion_vector)
                    
                if len(motion_vectors) > 0:
                    motion_hd = self.hd_space.bundle(motion_vectors)
                    # Combine with edge information
                    edges_hd = self.hd_space.bind(edges_hd, motion_hd)
                    
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
                    pos_vector = self.find_nearest_position(
                        cx, cy, (frame.shape[0], frame.shape[1])
                    )
                    
                    # Class encoding
                    class_name = det['class_name']
                    if class_name in self.hd_space.item_memory:
                        class_vector = self.hd_space.item_memory[class_name]
                    else:
                        # Create new random vector for this class
                        class_vector = self.hd_space.random(1).squeeze(0)
                        self.hd_space.item_memory[class_name] = class_vector
                    
                    # Size encoding
                    size = max(det['size']) / max(frame.shape[:2])  # normalize size
                    size_vector = self.hd_space.encode_scalar(size, 0, 1)
                    
                    # Combine properties with binding
                    object_vector = self.hd_space.bind(pos_vector, class_vector)
                    object_vector = self.hd_space.bind(object_vector, size_vector)
                    
                    # Scale by confidence
                    object_vector = object_vector * det['confidence']
                    
                    object_vectors.append(object_vector)
                    
                # Bundle all object vectors
                if len(object_vectors) > 0:
                    object_hd = self.hd_space.bundle(object_vectors)
                    
                    # Combine with basic encoding
                    edges_hd = self.hd_space.bind(edges_hd, object_hd)
            except Exception as e:
                print(f"Error in YOLO processing: {e}")
                
        return edges_hd
    
    def _batch_process_edges(self, edge_positions, shape):
        """
        Process edge vectors in batch for efficiency
        
        Args:
            edge_positions: Tuple of arrays with edge y,x coordinates
            shape: Shape of the image
            
        Returns:
            Tensor of edge vectors or bundled edge vector
        """
        num_edges = len(edge_positions[0])
        if num_edges == 0:
            return torch.zeros(self.dimension, device=torch.device(self.device))
            
        # Create position vectors efficiently
        y_coords = torch.tensor(edge_positions[0], dtype=torch.float32, device=torch.device(self.device))
        x_coords = torch.tensor(edge_positions[1], dtype=torch.float32, device=torch.device(self.device))
        
        # Normalize coordinates
        y_norm = y_coords / shape[0]
        x_norm = x_coords / shape[1]
        
        # Create phase values
        y_phase = 2 * np.pi * y_norm
        x_phase = 2 * np.pi * x_norm
        
        # Create position vectors using broadcasting
        indices = torch.arange(self.dimension, device=torch.device(self.device)).view(1, -1)
        y_vecs = torch.sin(y_phase.view(-1, 1) * indices / self.dimension)
        x_vecs = torch.sin(x_phase.view(-1, 1) * indices / self.dimension)
        
        if self.binary:
            y_vecs = torch.sign(y_vecs)
            x_vecs = torch.sign(x_vecs)
            
        # Bind x and y for each position
        pos_vecs = y_vecs * x_vecs
        
        # Bind with edge vector (broadcasting)
        edge_vectors = pos_vecs * self.hd_space.item_memory['edge'].view(1, -1)
        
        return edge_vectors
    
    def _batch_process_motion(self, motion):
        """
        Process motion vectors in batch for efficiency
        
        Args:
            motion: Motion frame
            
        Returns:
            Motion HD vector
        """
        # Find areas with significant motion
        motion_points = np.where(motion > 30)
        num_points = len(motion_points[0])
        
        if num_points == 0:
            return None
            
        # Create position vectors efficiently (similar to edge processing)
        y_coords = torch.tensor(motion_points[0], dtype=torch.float32, device=torch.device(self.device))
        x_coords = torch.tensor(motion_points[1], dtype=torch.float32, device=torch.device(self.device))
        
        # Normalize coordinates
        y_norm = y_coords / motion.shape[0]
        x_norm = x_coords / motion.shape[1]
        
        # Create phase values
        y_phase = 2 * np.pi * y_norm
        x_phase = 2 * np.pi * x_norm
        
        # Create position vectors using broadcasting
        indices = torch.arange(self.dimension, device=torch.device(self.device)).view(1, -1)
        y_vecs = torch.sin(y_phase.view(-1, 1) * indices / self.dimension)
        x_vecs = torch.sin(x_phase.view(-1, 1) * indices / self.dimension)
        
        if self.binary:
            y_vecs = torch.sign(y_vecs)
            x_vecs = torch.sign(x_vecs)
            
        # Bind x and y for each position
        pos_vecs = y_vecs * x_vecs
        
        # Bind with motion vector (broadcasting)
        motion_vectors = pos_vecs * self.hd_space.item_memory['motion'].view(1, -1)
        
        # Bundle all motion vectors
        motion_hd = motion_vectors.mean(dim=0)
        if self.binary:
            motion_hd = torch.sign(motion_hd)
            
        return motion_hd
        
    def encode_action(self, action):
        """
        Encode an action as an HD vector
        
        Args:
            action: Action to encode (integer or one-hot vector)
            
        Returns:
            HD vector representing the action
        """
        # Convert action to integer if it's a one-hot vector
        if isinstance(action, (list, np.ndarray, torch.Tensor)) and len(action) > 1:
            if isinstance(action, torch.Tensor):
                action = torch.argmax(action).item()
            else:
                action = np.argmax(action)
                
        # Create action vector
        action_key = f"action_{action}"
        if action_key not in self.hd_space.item_memory:
            # Create a new random vector for this action
            self.hd_space.item_memory[action_key] = self.hd_space.random(1).squeeze(0)
        
        return self.hd_space.item_memory[action_key]
    
    def similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        return self.hd_space.similarity(vec1, vec2)
    
    def cleanup_memory(self, query, memory_items, threshold=0.3):
        """
        Find the closest matching item in memory
        
        Args:
            query: Query vector
            memory_items: List of memory vectors
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (closest item, similarity)
        """
        if not memory_items:
            return None, 0.0
            
        # Find most similar item and its similarity
        closest, similarity = self.hd_space.cleanup_memory(query, memory_items)
        
        # Only return if similarity exceeds threshold
        if similarity > threshold:
            return closest, similarity
        else:
            return None, 0.0 