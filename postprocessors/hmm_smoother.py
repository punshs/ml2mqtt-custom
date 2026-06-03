from collections import deque
import numpy as np
from typing import Dict, Any, Optional, Tuple, ClassVar, List
from .base import BasePostprocessor

class HMMSmootherPostprocessor(BasePostprocessor):
    """Postprocessor that decodes the most likely room sequence using the Viterbi algorithm on a Hidden Markov Model."""
    
    type: ClassVar[str] = "hmm_smoother"
    description: ClassVar[str] = "Trajectory smoothing via Hidden Markov Model (HMM) Viterbi decoding"
    
    config_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "window_size": {
                "type": "integer",
                "description": "History window length for Viterbi decoding",
                "minimum": 3,
                "default": 10
            },
            "transition_probability": {
                "type": "number",
                "description": "Probability of changing rooms between consecutive seconds",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.02
            }
        },
        "required": ["window_size", "transition_probability"]
    }
    
    def __init__(self, window_size: int = 10, transition_probability: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.p_trans = transition_probability
        self.history = deque(maxlen=window_size)
    
    def process(self, observation: Dict[str, Any], label: Any, confidence: Any) -> Tuple[Dict[str, Any], Optional[Any]]:
        """Applies Viterbi decoding over the rolling history of predictions.
        
        Args:
            observation: Dictionary of entity values
            label: The current predicted room label
            confidence: The prediction confidence float
            
        Returns:
            Tuple of (observation, decoded label for the current step)
        """
        if not label:
            return observation, None
            
        # Append (label, confidence) to rolling queue
        self.history.append((label, float(confidence)))
        
        # If history is too short, return the current label directly
        if len(self.history) < 3:
            return observation, label
            
        # Get unique states in current history
        states = list(set([item[0] for item in self.history]))
        K = len(states)
        T = len(self.history)
        
        if K == 1:
            return observation, states[0]
            
        # 1. Initialize Viterbi tables
        # dp[t][i] = max log probability of path ending in state i at time t
        dp = np.zeros((T, K))
        backpointer = np.zeros((T, K), dtype=int)
        
        # 2. Emission probability helper
        # P(obs | state)
        def get_log_emission(state_idx: int, obs_lbl: str, obs_conf: float) -> float:
            target_lbl = states[state_idx]
            if target_lbl == obs_lbl:
                # High probability if state matches prediction
                p = max(obs_conf, 0.01)
            else:
                # Distribute remaining probability among other states
                p = max((1.0 - obs_conf) / (K - 1), 1e-4)
            return float(np.log(p))

        # 3. Transition probability helper
        # P(state_j | state_i)
        def get_log_transition(from_idx: int, to_idx: int) -> float:
            if from_idx == to_idx:
                # Stay in same room
                p = 1.0 - self.p_trans
            else:
                # Transition to another room
                p = self.p_trans / (K - 1)
            return float(np.log(p))

        # 4. Initialization (t = 0)
        # Uniform initial state distribution
        init_log_prob = np.log(1.0 / K)
        first_lbl, first_conf = self.history[0]
        for i in range(K):
            dp[0][i] = init_log_prob + get_log_emission(i, first_lbl, first_conf)
            
        # 5. Recursion (t = 1 to T-1)
        for t in range(1, T):
            lbl, conf = self.history[t]
            for j in range(K):
                # Find max transition from previous states
                best_prob = -np.inf
                best_idx = 0
                for i in range(K):
                    prob = dp[t-1][i] + get_log_transition(i, j)
                    if prob > best_prob:
                        best_prob = prob
                        best_idx = i
                
                dp[t][j] = best_prob + get_log_emission(j, lbl, conf)
                backpointer[t][j] = best_idx
                
        # 6. Path Backtracking
        best_last_idx = int(np.argmax(dp[-1]))
        
        # We only need the decoded state for the current time step (t = T-1)
        decoded_label = states[best_last_idx]
        
        return observation, decoded_label

    def configToString(self) -> str:
        return f"I will smooth room transitions using Viterbi decoding over a window of {self.window_size} steps"
