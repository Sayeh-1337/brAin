"""
YOLO Object Detection module

Implements object detection using YOLO (You Only Look Once)
for identifying enemies, items, and environmental features.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

class YOLODetector:
    """
    VISUAL OBJECT RECOGNITION ANALOG
    
    YOLO-based object detector that mimics the ventral stream's
    object recognition capabilities in the visual cortex.
    """
    
    def __init__(self, model_size='n', device=None):
        """
        Initialize the YOLO detector
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            device: Compute device ('cpu', 'cuda', etc.)
        """
        # Auto-select device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load YOLO model
        self.model = YOLO(f'yolov8{model_size}.pt')
        
        # For fine-tuning on Doom-specific objects
        self.is_fine_tuned = False
        
        # Classes of interest in Doom
        self.doom_classes = {
            'enemy': 0,
            'health': 1, 
            'weapon': 2,
            'ammo': 3,
            'door': 4
        }
        
    def detect(self, frame, conf_threshold=0.25):
        """
        Detect objects in the given frame
        
        Args:
            frame: Input image frame
            conf_threshold: Confidence threshold
            
        Returns:
            List of detected objects with bounding boxes and classes
        """
        # Run inference
        results = self.model(frame, verbose=False)[0]
        
        # Process results
        detections = []
        
        for i, (box, score, cls) in enumerate(zip(results.boxes.xyxy, 
                                                 results.boxes.conf, 
                                                 results.boxes.cls)):
            if score >= conf_threshold:
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                class_id = int(cls.item())
                class_name = results.names[class_id]
                
                detections.append({
                    'id': i,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': float(score),
                    'bbox': [x1, y1, x2, y2],
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'size': [(x2 - x1), (y2 - y1)]
                })
                
        return detections
    
    def fine_tune(self, doom_dataset_path, epochs=10):
        """
        Fine-tune YOLO on Doom-specific dataset
        
        Args:
            doom_dataset_path: Path to Doom dataset in YOLO format
            epochs: Number of training epochs
        """
        # Train the model on custom dataset
        self.model.train(
            data=doom_dataset_path,
            epochs=epochs,
            imgsz=640,
            batch=8,
            name='doom_yolo'
        )
        
        self.is_fine_tuned = True
        
    def visualize_detections(self, frame, detections):
        """
        Visualize detected objects on the frame
        
        Args:
            frame: Input image frame
            detections: List of detections
            
        Returns:
            Frame with visualized detections
        """
        img = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return img
    
    def create_attention_map(self, frame, detections, width=160, height=120):
        """
        Create an attention map based on detections
        
        Args:
            frame: Input frame
            detections: List of detections
            width: Output width
            height: Output height
            
        Returns:
            Attention heatmap highlighting important objects
        """
        # Create empty attention map
        attention = np.zeros((height, width), dtype=np.float32)
        
        # Generate gaussian kernels for each detection
        for det in detections:
            # Normalize bbox to output dimensions
            x1, y1, x2, y2 = det['bbox']
            frame_h, frame_w = frame.shape[:2]
            
            # Scale coordinates to attention map size
            x1 = int(x1 * width / frame_w)
            y1 = int(y1 * height / frame_h)
            x2 = int(x2 * width / frame_w)
            y2 = int(y2 * height / frame_h)
            
            # Center and radius
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = max((x2 - x1) // 2, (y2 - y1) // 2, 3)
            
            # Weight by importance of class and confidence
            weight = det['confidence']
            
            # Apply gaussian kernel around object center
            y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
            mask = x*x + y*y <= radius*radius
            attention[mask] += weight
            
        # Normalize attention map
        if np.max(attention) > 0:
            attention = attention / np.max(attention)
            
        return attention 