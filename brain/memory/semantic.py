"""
Semantic Memory implementation

Implements a brain-inspired semantic memory system that stores:
- Context associations
- Semantic relationships
- Action-outcome patterns
"""

import numpy as np
from collections import defaultdict

class SemanticMemory:
    """
    NEOCORTEX ANALOG

    Brain-inspired semantic memory system that mimics functions of the neocortex,
    particularly for storing associations and semantic knowledge.
    Features:
    - Association networks
    - Hierarchical concept representation
    - Semantic similarity metrics
    
    This system supports formation of semantic knowledge similar to
    how the neocortex consolidates information from episodic memory.
    """

    def __init__(self, vector_dim=1000):
        self.vector_dim = vector_dim
        
        # Main knowledge store: state -> action -> outcome
        self.state_action_outcomes = defaultdict(lambda: defaultdict(list))
        
        # Context vectors for similarity calculation
        self.context_vectors = {}
        
        # Success statistics
        self.action_success_counts = defaultdict(lambda: defaultdict(int))
        self.action_total_counts = defaultdict(lambda: defaultdict(int))
        
        # Custom associations
        self.associations = defaultdict(list)
        
    def _hash_vector(self, vector):
        """Convert a vector to a hashable representation"""
        if vector is None:
            return None
        # Round values and convert to tuple for hashing
        return tuple((vector * 100).astype(int))
        
    def store_experience(self, state_vector, action, reward, outcome_vector):
        """
        Store an experience in semantic memory
        
        Args:
            state_vector: Vector representing the state
            action: Action taken
            reward: Reward received
            outcome_vector: Vector representing the outcome state
        """
        # Hash vectors for dictionary keys
        state_key = self._hash_vector(state_vector)
        if state_key is None:
            return
            
        # Store outcome
        self.state_action_outcomes[state_key][action].append((outcome_vector, reward))
        
        # Update success statistics
        self.action_total_counts[state_key][action] += 1
        if reward > 0:
            self.action_success_counts[state_key][action] += 1
            
        # Store context vector for this state if not already present
        if state_key not in self.context_vectors:
            self.context_vectors[state_key] = state_vector
            
    def add_association(self, concept_a, concept_b, strength=1.0):
        """
        Add an associative link between two concepts
        
        Args:
            concept_a: First concept (vector or key)
            concept_b: Second concept (vector or key)
            strength: Association strength
        """
        # Convert vectors to keys if needed
        key_a = concept_a if isinstance(concept_a, tuple) else self._hash_vector(concept_a)
        key_b = concept_b if isinstance(concept_b, tuple) else self._hash_vector(concept_b)
        
        if key_a is None or key_b is None:
            return
            
        # Store bidirectional association
        self.associations[key_a].append((key_b, strength))
        self.associations[key_b].append((key_a, strength))
        
    def get_best_action(self, state_vector, threshold=0.8):
        """
        Get the most successful action for a state
        
        Args:
            state_vector: Vector representing the current state
            threshold: Similarity threshold for state matching
            
        Returns:
            Best action based on past success rate, or None if no match
        """
        # Find the most similar stored state
        state_key, similarity = self._find_similar_state(state_vector, threshold)
        
        if state_key is None:
            return None
            
        # Get all actions for this state with their success rates
        actions = self.action_total_counts[state_key].keys()
        if not actions:
            return None
            
        # Calculate success rates
        success_rates = {}
        for action in actions:
            total = self.action_total_counts[state_key][action]
            success = self.action_success_counts[state_key][action]
            success_rates[action] = success / total if total > 0 else 0
            
        # Return the action with the highest success rate
        return max(success_rates.items(), key=lambda x: x[1])[0] if success_rates else None
        
    def _find_similar_state(self, query_vector, threshold=0.8):
        """Find the most similar state to the query vector"""
        if query_vector is None:
            return None, 0
            
        best_match = None
        best_similarity = 0
        
        for state_key, context_vector in self.context_vectors.items():
            # Calculate cosine similarity
            similarity = np.dot(query_vector, context_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(context_vector))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = state_key
                
        # Return the best match if it exceeds the threshold
        if best_similarity >= threshold:
            return best_match, best_similarity
        return None, 0
        
    def get_associations(self, concept, min_strength=0.5):
        """
        Get all concepts associated with the given concept
        
        Args:
            concept: Concept vector or key
            min_strength: Minimum association strength
            
        Returns:
            List of associated concepts and strengths
        """
        key = concept if isinstance(concept, tuple) else self._hash_vector(concept)
        
        if key is None or key not in self.associations:
            return []
            
        # Filter by minimum strength
        return [(self.context_vectors.get(k), s) for k, s in self.associations[key] if s >= min_strength]
        
    def get_predicted_outcome(self, state_vector, action):
        """
        Predict the outcome of taking an action in a state
        
        Args:
            state_vector: Vector representing the state
            action: Action to predict outcome for
            
        Returns:
            (outcome_vector, expected_reward) or (None, 0) if no data
        """
        state_key, _ = self._find_similar_state(state_vector)
        
        if state_key is None or action not in self.state_action_outcomes[state_key]:
            return None, 0
            
        # Get all outcomes for this state-action pair
        outcomes = self.state_action_outcomes[state_key][action]
        
        if not outcomes:
            return None, 0
            
        # Average the outcomes and rewards
        avg_outcome = np.mean([outcome for outcome, _ in outcomes], axis=0)
        avg_reward = np.mean([reward for _, reward in outcomes])
        
        return avg_outcome, avg_reward 