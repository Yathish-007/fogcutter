import torch
import pytest
from fogcutter.whitebox import token_entropy

def test_token_entropy_perfect_uncertainty():
    """
    Test: If logits are all equal (uniform distribution), entropy should be 1.0 (normalized).
    """
    # 1. Create fake logits for 1 batch, 1 token, 4 vocabulary words
    # Equal logits = Equal probability (25% each) -> Max Uncertainty
    logits = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]]) 
    
    # 2. Run your function
    entropy = token_entropy(logits, normalize=True)
    
    # 3. Assert (Check if result is close to 1.0)
    # 25% prob each means normalized entropy should be exactly 1.0
    assert torch.isclose(entropy, torch.tensor(1.0), atol=1e-6)

def test_token_entropy_zero_uncertainty():
    """
    Test: If one logit is huge and others are tiny, entropy should be ~0.0.
    """
    # One word is extremely likely (logit 100 vs 0)
    logits = torch.tensor([[[100.0, 0.0, 0.0, 0.0]]])
    
    entropy = token_entropy(logits, normalize=True)
    
    # Should be very close to 0
    assert torch.isclose(entropy, torch.tensor(0.0), atol=1e-4)
