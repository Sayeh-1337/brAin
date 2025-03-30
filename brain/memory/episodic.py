"""
Episodic Memory implementation

Implements a brain-inspired episodic memory system using:
- Vector-based memory storage
- Similarity-based retrieval
- Experience replay mechanisms
"""

import numpy as np
import random
from collections import deque

class EpisodicMemory:
    """
    HIPPOCAMPUS ANALOG

    Brain-inspired episodic memory system that mimics some functions of the hippocampus,
    particularly the CA3 region's ability to store and recall sequences/episodes.
    Features:
    - Pattern completion (retrieval based on partial cues)
    - Experience storage and replay
    - Temporal sequence learning
    
    This system supports recall of previous experiences which enables
    episodic learning similar to the hippocampal system.
    """

    def __init__(self, capacity=1000, similarity_threshold=0.7):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.similarity_threshold = similarity_threshold

    def store(self, state, action, reward, next_state, done):
        """Store an experience tuple in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of experiences"""
        return random.sample(list(self.memory), min(batch_size, len(self.memory)))

    def retrieve_similar(self, query_state, max_results=5):
        """
        Retrieve experiences with similar states to the query state
        
        Args:
            query_state: The state to find similar experiences for
            max_results: Maximum number of results to return
            
        Returns:
            List of similar experiences sorted by similarity
        """
        if len(self.memory) == 0:
            return []
            
        similarities = []
        
        for experience in self.memory:
            state = experience[0]
            if state is not None and query_state is not None:
                # Calculate cosine similarity
                similarity = np.dot(state, query_state) / (np.linalg.norm(state) * np.linalg.norm(query_state))
                similarities.append((similarity, experience))
                
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Filter by threshold and return the experiences
        return [exp for sim, exp in similarities[:max_results] if sim >= self.similarity_threshold]

    def prioritized_sample(self, batch_size, priorities=None):
        """
        Sample experiences based on priority (e.g., reward magnitude)
        
        Args:
            batch_size: Number of samples to retrieve
            priorities: Function to calculate priority from experience
            
        Returns:
            Batch of experiences sampled by priority
        """
        if len(self.memory) == 0:
            return []
            
        if priorities is None:
            # Default priority: absolute reward value
            priorities = lambda exp: abs(exp[2])
            
        # Calculate priorities
        samples = list(self.memory)
        weights = np.array([priorities(exp) for exp in samples])
        
        # Add small constant to ensure non-zero probabilities
        weights = weights + 1e-5
        
        # Normalize to probabilities
        probs = weights / np.sum(weights)
        
        # Sample based on priorities
        indices = np.random.choice(
            len(samples), 
            size=min(batch_size, len(samples)), 
            replace=False, 
            p=probs
        )
        
        return [samples[idx] for idx in indices]

    def clear(self):
        """Clear all stored memories"""
        self.memory.clear()

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

    def __len__(self):
        """Return the current size of memory"""
        return len(self.memory) 