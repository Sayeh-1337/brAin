"""
Perception module - Visual and sensory processing systems

This module contains components for vision processing and object recognition,
including YOLO-based object detection and visual processing.
"""

# Import components for easy access
try:
    from brain.perception.yolo_detector import YOLODetector
except ImportError:
    # If YOLO dependencies aren't installed, provide a placeholder
    class YOLODetector:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "YOLODetector requires torch, torchvision, and ultralytics. "
                "Install with 'pip install torch torchvision ultralytics'"
            ) 