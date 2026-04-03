import math
from typing import List, Dict

def calculate_predictive_entropy(token_distributions: List[Dict[str, float]]) -> float:
    """
    Calculates the Average Predictive Entropy (uncertainty) of the sequence.
    
    It looks at the Top-K options for every single token.
    - If the model splits probability between many words (e.g., "The cat is [white/black/brown]"), Entropy is HIGH.
    - If the model puts all probability on one word (e.g., "United States of [America]"), Entropy is LOW.
    """
    if not token_distributions:
        return 0.0
        
    total_entropy = 0.0
    count = 0
    
    for dist in token_distributions:
        # dist is {token_str: log_prob}, e.g., {'Paris': -0.1, 'London': -2.5}
        
        # 1. Convert log_probs to linear probabilities: p = exp(log_p)
        probs = [math.exp(lp) for lp in dist.values()]
        
        # 2. Normalize (Because we only have Top-K, sum might not be exactly 1.0)
        # We re-normalize so the math works for the local set of choices.
        sum_p = sum(probs)
        if sum_p == 0: 
            continue
        
        normalized_probs = [p / sum_p for p in probs]
        
        # 3. Calculate Shannon Entropy for this step: H = -sum(p * log(p))
        step_entropy = -sum(p * math.log(p) for p in normalized_probs if p > 0)
        
        total_entropy += step_entropy
        count += 1
        
    # Return average entropy over the whole sentence
    return total_entropy / count if count > 0 else 0.0
