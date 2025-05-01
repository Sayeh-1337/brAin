"""
Optimized Episodic Memory implementation

Implements an optimized brain-inspired episodic memory system using:
- GPU-accelerated vector operations for storage and retrieval
- Efficient similarity search with batch processing
- Priority-based experience replay
- Memory consolidation with temporal sequence learning
"""

import torch
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Optional, Union, Any
from brain.utils.config import config

class OptimizedEpisodicMemory:
    """
    HIPPOCAMPUS ANALOG

    Optimized brain-inspired episodic memory system that mimics the hippocampus,
    particularly the CA3 region's ability to store and recall episodes.
    
    Features:
    - GPU-accelerated memory operations
    - Pattern completion (retrieval based on partial cues)
    - Experience storage and replay
    - Temporal sequence learning and prediction
    - Prioritized sampling based on importance
    - Memory consolidation to prevent forgetting
    """

    def __init__(self, capacity: int = 1000, similarity_threshold: float = 0.7, device=None):
        """
        Initialize episodic memory system
        
        Args:
            capacity: Maximum number of experiences to store
            similarity_threshold: Minimum similarity for retrieval
            device: Computation device ('cpu', 'cuda')
        """
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.device = device if device is not None else config.device
        
        # Memory structures
        self.states = []  # State vectors
        self.actions = []  # Action vectors/indices
        self.rewards = []  # Reward values
        self.next_states = []  # Next state vectors
        self.dones = []  # Episode termination flags
        self.priorities = torch.zeros(0, device=torch.device(self.device))  # Priority values
        self.timestamps = torch.zeros(0, dtype=torch.long, device=torch.device(self.device))  # Insertion timestamps
        
        # Counter for insertion order
        self.insert_count = 0
        
        # Index mappings for efficient retrieval
        self.state_similarity_cache = {}  # Cache for state similarities
        
    def store(self, state, action, reward, next_state, done, priority=None):
        """
        Store an experience tuple in memory
        
        Args:
            state: Current state vector
            action: Action taken
            reward: Reward received
            next_state: Next state vector
            done: Whether episode ended
            priority: Optional priority value for experience
        """
        # Convert inputs to torch tensors if needed
        if state is not None and not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=torch.device(self.device))
        if next_state is not None and not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, device=torch.device(self.device))
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, device=torch.device(self.device))
            
        # If priority not provided, use reward magnitude
        if priority is None:
            priority = abs(reward.item())
            
        # Add to memory
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        # Update priorities
        new_priorities = torch.cat([
            self.priorities,
            torch.tensor([priority], device=torch.device(self.device))
        ])
        self.priorities = new_priorities
        
        # Update timestamps
        new_timestamps = torch.cat([
            self.timestamps,
            torch.tensor([self.insert_count], device=torch.device(self.device))
        ])
        self.timestamps = new_timestamps
        self.insert_count += 1
        
        # Check if memory is full
        if len(self.states) > self.capacity:
            # Find least important memory to remove (lowest priority)
            if config.batch_process:
                # Use GPU for faster operation
                oldest_idx = torch.argmin(self.priorities).item()
            else:
                # Find oldest memory with low priority
                priorities_np = self.priorities.cpu().numpy()
                # Only consider memories with below-average priority for removal
                threshold = np.mean(priorities_np)
                candidates = np.where(priorities_np < threshold)[0]
                if len(candidates) > 0:
                    oldest_idx = candidates[0]
                else:
                    oldest_idx = 0
                
            # Remove memory at index
            self._remove_memory(oldest_idx)
            
        # Clear similarity cache when new memory added
        self.state_similarity_cache = {}
        
    def _remove_memory(self, index):
        """
        Remove memory at specific index
        
        Args:
            index: Memory index to remove
        """
        self.states.pop(index)
        self.actions.pop(index)
        self.rewards.pop(index)
        self.next_states.pop(index)
        self.dones.pop(index)
        
        # Update priorities and timestamps tensors
        mask = torch.ones(len(self.priorities), dtype=torch.bool, device=self.device)
        mask[index] = False
        self.priorities = self.priorities[mask]
        self.timestamps = self.timestamps[mask]
        
    def sample(self, batch_size):
        """
        Sample a random batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Batch of experiences
        """
        if len(self.states) == 0:
            return []
            
        indices = random.sample(range(len(self.states)), min(batch_size, len(self.states)))
        
        batch = []
        for i in indices:
            experience = (
                self.states[i], 
                self.actions[i], 
                self.rewards[i], 
                self.next_states[i], 
                self.dones[i]
            )
            batch.append(experience)
            
        return batch
        
    def retrieve_similar(self, query_state, max_results=5):
        """
        Retrieve experiences with similar states to the query state
        
        Args:
            query_state: The state to find similar experiences for
            max_results: Maximum number of results to return
            
        Returns:
            List of similar experiences sorted by similarity
        """
        if len(self.states) == 0 or query_state is None:
            return []
            
        # Convert query to tensor if needed
        if not isinstance(query_state, torch.Tensor):
            query_state = torch.tensor(query_state, device=torch.device(self.device))
            
        # Check cache first
        cache_key = str(query_state.cpu().numpy().data.tobytes())
        if cache_key in self.state_similarity_cache:
            cached_indices = self.state_similarity_cache[cache_key]
            results = []
            for idx in cached_indices[:max_results]:
                similarity = self._compute_similarity(query_state, self.states[idx])
                if similarity >= self.similarity_threshold:
                    experience = (
                        self.states[idx], 
                        self.actions[idx], 
                        self.rewards[idx], 
                        self.next_states[idx], 
                        self.dones[idx]
                    )
                    results.append((similarity, experience))
            return [exp for sim, exp in sorted(results, key=lambda x: x[0], reverse=True)]
            
        # Compute similarities efficiently
        if config.batch_process:
            # Batch compute similarities (faster on GPU)
            similarities = self._batch_compute_similarities(query_state)
            
            # Get top indices
            if len(similarities) > 0:
                top_indices = torch.argsort(similarities, descending=True)[:max_results].cpu().numpy()
                
                # Filter by threshold and create results
                results = []
                for idx in top_indices:
                    similarity = similarities[idx].item()
                    if similarity >= self.similarity_threshold:
                        experience = (
                            self.states[idx], 
                            self.actions[idx], 
                            self.rewards[idx], 
                            self.next_states[idx], 
                            self.dones[idx]
                        )
                        results.append((similarity, experience))
                
                # Cache results
                self.state_similarity_cache[cache_key] = top_indices
                
                return [exp for sim, exp in results]
            return []
        else:
            # Sequential computation (for smaller memories or CPU)
            similarities = []
            for i, state in enumerate(self.states):
                if state is not None and query_state is not None:
                    similarity = self._compute_similarity(query_state, state)
                    similarities.append((similarity, i))
                    
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Filter by threshold and create results
            results = []
            top_indices = []
            for similarity, idx in similarities[:max_results]:
                if similarity >= self.similarity_threshold:
                    experience = (
                        self.states[idx], 
                        self.actions[idx], 
                        self.rewards[idx], 
                        self.next_states[idx], 
                        self.dones[idx]
                    )
                    results.append(experience)
                    top_indices.append(idx)
            
            # Cache results
            self.state_similarity_cache[cache_key] = top_indices
            
            return results
            
    def _compute_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity value
        """
        if vec1 is None or vec2 is None:
            return 0.0
            
        # Convert to torch tensors if needed
        if not isinstance(vec1, torch.Tensor):
            vec1 = torch.tensor(vec1, device=torch.device(self.device))
        if not isinstance(vec2, torch.Tensor):
            vec2 = torch.tensor(vec2, device=torch.device(self.device))
            
        # Compute cosine similarity
        dot_product = torch.sum(vec1 * vec2)
        norm1 = torch.sqrt(torch.sum(vec1 * vec1))
        norm2 = torch.sqrt(torch.sum(vec2 * vec2))
        similarity = dot_product / (norm1 * norm2 + 1e-8)  # Add epsilon to prevent division by zero
        
        return similarity.item()
        
    def _batch_compute_similarities(self, query):
        """
        Compute similarities between query and all states in batch
        
        Args:
            query: Query vector
            
        Returns:
            Tensor of similarity values
        """
        if len(self.states) == 0:
            return torch.zeros(0, device=torch.device(self.device))
            
        # Filter out None states
        valid_indices = [i for i, s in enumerate(self.states) if s is not None]
        
        if not valid_indices:
            return torch.zeros(len(self.states), device=torch.device(self.device))
            
        # Stack valid states into tensor
        valid_states = torch.stack([self.states[i] for i in valid_indices])
        
        # Ensure query has batch dimension
        if query.dim() == 1:
            query = query.unsqueeze(0)
            
        # Compute L2 norms
        query_norm = torch.sqrt(torch.sum(query * query, dim=1, keepdim=True))
        states_norm = torch.sqrt(torch.sum(valid_states * valid_states, dim=1, keepdim=True))
        
        # Normalize vectors
        query_normalized = query / (query_norm + 1e-8)
        states_normalized = valid_states / (states_norm + 1e-8)
        
        # Compute cosine similarities
        similarities = torch.mm(query_normalized, states_normalized.t()).squeeze()
        
        # Create result tensor with similarities at valid indices
        result = torch.zeros(len(self.states), device=torch.device(self.device))
        if similarities.dim() == 0:  # Handle single result case
            result[valid_indices[0]] = similarities
        else:
            for i, idx in enumerate(valid_indices):
                result[idx] = similarities[i]
                
        return result
        
    def prioritized_sample(self, batch_size, temperature=1.0):
        """
        Sample experiences based on priority
        
        Args:
            batch_size: Number of samples to retrieve
            temperature: Controls randomness (higher = more uniform)
            
        Returns:
            Batch of experiences sampled by priority
        """
        if len(self.states) == 0:
            return []
            
        # Calculate sampling probabilities using priorities
        priorities = self.priorities.cpu().numpy()
        
        # Apply temperature to control randomness
        if temperature > 0:
            priorities = np.power(priorities, 1.0 / temperature)
            
        # Add small constant to ensure non-zero probabilities
        priorities = priorities + 1e-5
        
        # Normalize to probabilities
        probs = priorities / np.sum(priorities)
        
        # Sample based on priorities
        indices = np.random.choice(
            len(self.states), 
            size=min(batch_size, len(self.states)), 
            replace=False, 
            p=probs
        )
        
        batch = []
        for i in indices:
            experience = (
                self.states[i], 
                self.actions[i], 
                self.rewards[i], 
                self.next_states[i], 
                self.dones[i]
            )
            batch.append(experience)
            
        return batch
        
    def clear(self):
        """Clear all stored memories"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.priorities = torch.zeros(0, device=torch.device(self.device))
        self.timestamps = torch.zeros(0, dtype=torch.long, device=torch.device(self.device))
        self.insert_count = 0
        self.state_similarity_cache = {}
        
    def replay_episode(self, starting_state, max_length=10):
        """
        Replay a sequence of experiences starting from a similar state
        
        Args:
            starting_state: The state to start replay from
            max_length: Maximum length of the episode to replay
            
        Returns:
            Sequence of experiences forming an episode
        """
        # Find the most similar state to start with
        similar_experiences = self.retrieve_similar(starting_state, max_results=1)
        
        if not similar_experiences:
            return []
            
        episode = [similar_experiences[0]]
        current_state = similar_experiences[0][3]  # next_state of the first experience
        
        # Build episode by finding sequential experiences
        for _ in range(max_length - 1):
            if current_state is None or episode[-1][4]:  # Stop if terminal state reached
                break
                
            next_experiences = self.retrieve_similar(current_state, max_results=1)
            
            if not next_experiences:
                break
                
            episode.append(next_experiences[0])
            current_state = next_experiences[0][3]
            
        return episode
        
    def consolidate_memories(self, similarity_threshold=0.85, max_consolidations=10):
        """
        Consolidate similar memories to prevent redundancy
        
        Args:
            similarity_threshold: Similarity threshold for consolidation
            max_consolidations: Maximum number of consolidations to perform
        """
        if len(self.states) < 2:
            return
            
        # Find similar memories to consolidate
        consolidation_candidates = []
        
        # Use batch processing if enabled
        if config.batch_process and len(self.states) <= 1000:  # Limit for memory usage
            # Compute pairwise similarities (memory intensive)
            valid_indices = [i for i, s in enumerate(self.states) if s is not None]
            if len(valid_indices) < 2:
                return
                
            valid_states = torch.stack([self.states[i] for i in valid_indices])
            
            # Compute norms
            norms = torch.sqrt(torch.sum(valid_states * valid_states, dim=1, keepdim=True))
            
            # Normalize
            normalized_states = valid_states / (norms + 1e-8)
            
            # Compute pairwise similarities
            similarities = torch.mm(normalized_states, normalized_states.t())
            
            # Find high similarity pairs (excluding self-similarities)
            mask = torch.triu(torch.ones_like(similarities), diagonal=1)
            pairs = torch.nonzero(mask * (similarities > similarity_threshold))
            
            # Convert to candidate list
            for pair in pairs[:max_consolidations]:
                i, j = pair
                i_idx = valid_indices[i]
                j_idx = valid_indices[j]
                similarity = similarities[i, j].item()
                consolidation_candidates.append((i_idx, j_idx, similarity))
        else:
            # Sequential approach for larger memories
            # Check a random subset to avoid O(n²) complexity
            num_checks = min(100, len(self.states))
            indices = random.sample(range(len(self.states)), num_checks)
            
            for i in indices:
                if self.states[i] is None:
                    continue
                    
                # Compare with other states
                for j in range(len(self.states)):
                    if i == j or self.states[j] is None:
                        continue
                        
                    similarity = self._compute_similarity(self.states[i], self.states[j])
                    if similarity > similarity_threshold:
                        consolidation_candidates.append((i, j, similarity))
                        
                    if len(consolidation_candidates) >= max_consolidations:
                        break
                        
                if len(consolidation_candidates) >= max_consolidations:
                    break
        
        # Sort by similarity (highest first)
        consolidation_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Consolidate similar memories
        consolidated_indices = set()
        for i, j, _ in consolidation_candidates:
            # Skip if either memory has already been consolidated
            if i in consolidated_indices or j in consolidated_indices:
                continue
                
            # Use the memory with higher priority
            if self.priorities[i] >= self.priorities[j]:
                to_keep, to_remove = i, j
            else:
                to_keep, to_remove = j, i
                
            # Boost priority of kept memory
            self.priorities[to_keep] = max(
                self.priorities[to_keep], 
                1.2 * self.priorities[to_remove]
            )
            
            # Remove the other memory
            self._remove_memory(to_remove)
            
            # Track consolidated indices
            consolidated_indices.add(to_remove)
            
            # Update indices that shifted due to removal
            for k in range(len(consolidation_candidates)):
                ci, cj, cs = consolidation_candidates[k]
                if ci > to_remove:
                    consolidation_candidates[k] = (ci-1, cj, cs)
                if cj > to_remove:
                    consolidation_candidates[k] = (ci, cj-1, cs)
        
        # Clear similarity cache after consolidation
        self.state_similarity_cache = {}
        
    def get_stats(self):
        """Get memory statistics"""
        if len(self.states) == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "avg_priority": 0.0,
                "max_priority": 0.0,
                "min_priority": 0.0
            }
            
        priorities = self.priorities.cpu().numpy()
        
        return {
            "size": len(self.states),
            "capacity": self.capacity,
            "avg_priority": float(np.mean(priorities)),
            "max_priority": float(np.max(priorities)),
            "min_priority": float(np.min(priorities)),
            "full": len(self.states) >= self.capacity
        }
        
    def __len__(self):
        """Return the current size of memory"""
        return len(self.states) 