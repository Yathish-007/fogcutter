from typing import List

# Global variable for the model to ensure we only load it once
_SIMILARITY_MODEL = None

def _get_model():
    """
    Lazy load the local CrossEncoder model.
    This runs locally on your machine, not via an API.
    """
    global _SIMILARITY_MODEL
    if _SIMILARITY_MODEL is None:
        from sentence_transformers import CrossEncoder
        # Downloads model to local cache (~450MB) and runs offline
        _SIMILARITY_MODEL = CrossEncoder('cross-encoder/stsb-roberta-base')
    return _SIMILARITY_MODEL

def _are_equivalent(text1: str, text2: str, threshold: float) -> bool:
    """Check if two texts are semantically equivalent using local AI."""
    # Optimization: skip AI if strings are identical
    if text1 == text2:
        return True
        
    model = _get_model()
    # Predict returns a score 0.0 to 1.0
    score = model.predict([(text1, text2)])[0]
    return score > threshold

def semantic_consistency_score(answers: List[str], threshold: float = 0.85) -> float:
    """
    Calculate consistency using local semantic clustering.
    
    Groups equivalent answers (e.g., "Paris" and "The city of Paris") 
    before calculating the consistency score.
    """
    if not answers:
        return 0.0
        
    # Greedy Clustering (O(N^2))
    clusters = []
    
    for ans in answers:
        found_cluster = False
        for cluster in clusters:
            # Compare current answer with the first item in the cluster
            representative = cluster[0]
            if _are_equivalent(ans, representative, threshold):
                cluster.append(ans)
                found_cluster = True
                break
        
        if not found_cluster:
            clusters.append([ans])
            
    if not clusters:
        return 0.0
        
    # Score = size of largest cluster / total answers
    largest_cluster_size = max(len(c) for c in clusters)
    return largest_cluster_size / len(answers)
