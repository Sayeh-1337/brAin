"""
Optimized Hyperdimensional Vector Space

Implements optimized hyperdimensional computing operations using:
- PyTorch for GPU acceleration
- JIT compilation for performance
- Batched operations for efficiency
- Memory caching for frequently used vectors
"""

import torch
import torch.jit as jit
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from brain.utils.config import config

class HDVectorSpace:
    """
    Optimized implementation of hyperdimensional vector operations
    
    Features:
    - GPU acceleration with PyTorch tensors
    - JIT compilation for core operations
    - Efficient batch processing
    - Vector caching for frequently used vectors
    """
    
    def __init__(self, dimension: int = 10000, binary: bool = True, device: Optional[str] = None):
        """
        Initialize the HDC vector space
        
        Args:
            dimension: Dimensionality of hypervectors
            binary: Whether to use binary (-1/1) or continuous vectors
            device: Device to use for computations ('cpu', 'cuda')
        """
        self.dimension = dimension
        self.binary = binary
        self.device = device if device is not None else config.device
        self.item_memory: Dict[str, torch.Tensor] = {}  # Cache for frequently used vectors
        
    def random(self, num_vectors: int = 1) -> torch.Tensor:
        """
        Generate random hypervectors efficiently
        
        Args:
            num_vectors: Number of vectors to generate
            
        Returns:
            Tensor of random hypervectors
        """
        device = torch.device(self.device)
        if self.binary:
            # Binary vectors (-1/1) are more efficient for HDC operations
            return torch.randint(0, 2, (num_vectors, self.dimension), device=device) * 2 - 1
        else:
            # Continuous vectors
            return torch.randn(num_vectors, self.dimension, device=device)
    
    @staticmethod
    @torch.jit.script
    def _bind_jit(hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """JIT-compiled binding operation for speed"""
        return hv1 * hv2
    
    @staticmethod
    def _bind_normal(hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """Standard binding operation"""
        return hv1 * hv2
    
    def bind(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """
        Bind two hypervectors (element-wise multiplication)
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Bound hypervector
        """
        # Convert numpy arrays to torch tensors if needed
        if isinstance(hv1, np.ndarray):
            hv1 = torch.from_numpy(hv1).to(self.device)
        if isinstance(hv2, np.ndarray):
            hv2 = torch.from_numpy(hv2).to(self.device)
            
        if config.use_jit_compile:
            return self._bind_jit(hv1, hv2)
        return self._bind_normal(hv1, hv2)
    
    @staticmethod
    @torch.jit.script
    def _bundle_jit(hvs: List[torch.Tensor], threshold: bool = True) -> torch.Tensor:
        """JIT-compiled bundling operation"""
        bundled = torch.stack(hvs).sum(dim=0)
        if threshold:
            return torch.sign(bundled)
        return bundled
    
    def bundle(self, hvs: List[torch.Tensor], threshold: bool = True) -> torch.Tensor:
        """
        Bundle multiple hypervectors (element-wise addition with optional thresholding)
        
        Args:
            hvs: List of hypervectors to bundle
            threshold: Whether to threshold the result to binary values
            
        Returns:
            Bundled hypervector
        """
        if not hvs:
            return torch.zeros(self.dimension, device=torch.device(self.device))
            
        # Convert any numpy arrays to torch tensors
        torch_hvs = []
        for hv in hvs:
            if isinstance(hv, np.ndarray):
                torch_hvs.append(torch.from_numpy(hv).to(self.device))
            else:
                torch_hvs.append(hv.to(self.device))
            
        if config.use_jit_compile and len(torch_hvs) > 0:
            return self._bundle_jit(torch_hvs, threshold)
            
        # Standard implementation
        bundled = torch.stack(torch_hvs).sum(dim=0)
        if threshold:
            return torch.sign(bundled)
        return bundled
    
    @staticmethod
    @torch.jit.script
    def _permute_jit(hv: torch.Tensor, shifts: int = 1) -> torch.Tensor:
        """JIT-compiled permutation"""
        return torch.roll(hv, shifts=shifts, dims=-1)
    
    def permute(self, hv: torch.Tensor, shifts: int = 1) -> torch.Tensor:
        """
        Permute a hypervector (cyclic shift) to represent sequences
        
        Args:
            hv: Hypervector to permute
            shifts: Number of positions to shift
            
        Returns:
            Permuted hypervector
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(hv, np.ndarray):
            hv = torch.from_numpy(hv).to(self.device)
            
        if config.use_jit_compile:
            return self._permute_jit(hv, shifts)
        return torch.roll(hv, shifts=shifts, dims=-1)
    
    @staticmethod
    @torch.jit.script
    def _similarity_jit(hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """JIT-compiled cosine similarity"""
        # Handle 1D and 2D tensors appropriately
        if hv1.dim() == 1:
            hv1 = hv1.unsqueeze(0)
        if hv2.dim() == 1:
            hv2 = hv2.unsqueeze(0)
            
        # Normalize vectors
        hv1_norm = hv1 / (torch.norm(hv1, dim=1, keepdim=True) + 1e-8)
        hv2_norm = hv2 / (torch.norm(hv2, dim=1, keepdim=True) + 1e-8)
        
        # Compute similarity
        return torch.matmul(hv1_norm, hv2_norm.T).squeeze()
    
    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between hypervectors
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Cosine similarity between vectors
        """
        # Convert numpy arrays to torch tensors if needed
        if isinstance(hv1, np.ndarray):
            hv1 = torch.from_numpy(hv1).to(self.device)
        if isinstance(hv2, np.ndarray):
            hv2 = torch.from_numpy(hv2).to(self.device)
            
        if config.use_jit_compile:
            return self._similarity_jit(hv1, hv2)
            
        # Standard implementation (fallback)
        if hv1.dim() == 1:
            hv1 = hv1.unsqueeze(0)
        if hv2.dim() == 1:
            hv2 = hv2.unsqueeze(0)
            
        hv1_norm = hv1 / (torch.norm(hv1, dim=1, keepdim=True) + 1e-8)
        hv2_norm = hv2 / (torch.norm(hv2, dim=1, keepdim=True) + 1e-8)
        
        return torch.matmul(hv1_norm, hv2_norm.T).squeeze()
    
    def cleanup_memory(self, query_hv: torch.Tensor, memory_bank: List[torch.Tensor]) -> Tuple[Optional[torch.Tensor], float]:
        """
        Find the closest matching hypervector in memory
        
        Args:
            query_hv: Query hypervector
            memory_bank: List of memory hypervectors to search
            
        Returns:
            Tuple of (closest vector, similarity score)
        """
        if not memory_bank:
            return None, -1.0
            
        # Convert query to torch tensor if needed
        if isinstance(query_hv, np.ndarray):
            query_hv = torch.from_numpy(query_hv).to(self.device)
            
        # Convert memory bank items to torch tensors if needed
        torch_bank = []
        for hv in memory_bank:
            if isinstance(hv, np.ndarray):
                torch_bank.append(torch.from_numpy(hv).to(self.device))
            else:
                torch_bank.append(hv.to(self.device))
            
        # Compute similarities to all memory items efficiently
        similarities = self.similarity(query_hv, torch.stack(torch_bank))
        
        # Get the most similar item and its similarity score
        max_idx = torch.argmax(similarities)
        max_sim = similarities[max_idx].item()
        
        return memory_bank[max_idx], max_sim
        
    def encode_scalar(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
        """
        Encode a scalar value as a hypervector
        
        Args:
            value: Scalar value to encode
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Hypervector encoding of the scalar value
        """
        # Normalize value to 0-1 range
        norm_value = (value - min_val) / (max_val - min_val)
        norm_value = max(0.0, min(1.0, norm_value))  # Clamp to 0-1
        
        # Convert to phase for smooth encoding
        phase = 2 * np.pi * norm_value
        
        # Create vector with elements varying by phase
        indices = torch.arange(self.dimension, device=torch.device(self.device))
        continuous_vec = torch.sin(indices * phase / self.dimension)
        
        # Binarize if using binary vectors
        if self.binary:
            return torch.sign(continuous_vec)
        return continuous_vec
        
    def encode_index(self, index: int, total_indices: int) -> torch.Tensor:
        """
        Encode an index or position as a hypervector
        
        Args:
            index: Index to encode
            total_indices: Total number of possible indices
            
        Returns:
            Hypervector encoding of the index
        """
        # Check if already in item memory
        key = f"index_{index}_{total_indices}"
        if key in self.item_memory:
            return self.item_memory[key]
            
        # Create orthogonal vectors for indices
        if index >= total_indices:
            raise ValueError(f"Index {index} exceeds total indices {total_indices}")
            
        # If total indices is small, create orthogonal vectors
        if total_indices <= self.dimension:
            # Create a one-hot vector and project to high dimensions
            vec = torch.zeros(self.dimension, device=torch.device(self.device))
            vec[index % self.dimension] = 1.0
            # Add small noise to remaining dimensions for better distribution
            mask = torch.ones(self.dimension, device=torch.device(self.device))
            mask[index % self.dimension] = 0.0
            vec += torch.randn(self.dimension, device=torch.device(self.device)) * 0.01 * mask
        else:
            # For larger sets, use smooth encoding with phase
            normalized_idx = index / total_indices
            vec = self.encode_scalar(normalized_idx, 0, 1)
            
        # Store in item memory
        self.item_memory[key] = vec
        return vec
        
    def to_numpy(self, hv: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array"""
        return hv.cpu().numpy()
        
    def from_numpy(self, hv: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor"""
        return torch.from_numpy(hv).to(self.device) 