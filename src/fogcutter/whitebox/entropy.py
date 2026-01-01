import torch
import torch.nn.functional as F

def token_entropy(logits: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Calculate the entropy of the token distribution.
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        normalize: If True, normalize by log(vocab_size) to scale between [0, 1].
    
    Returns:
        Entropy tensor of shape (batch_size, seq_len)
    """
    # 1. Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # 2. Calculate Entropy: H(x) = - sum(p * log(p))
    # Add 1e-9 to avoid log(0) error
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    
    # 3. Normalize (Optional)
    if normalize:
        vocab_size = logits.shape[-1]
        entropy = entropy / torch.log(torch.tensor(vocab_size))
        
    return entropy
