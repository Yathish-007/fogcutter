import math
from typing import List

def calculate_perplexity(logprobs: List[float]) -> float:
    if not logprobs:
        return 0.0
    
    n = len(logprobs)
    sum_logprobs = sum(logprobs)
    avg_nll = -sum_logprobs / n
    
    if avg_nll > 50: 
        return float('inf')
        
    return math.exp(avg_nll)
