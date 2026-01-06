from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class LogitsProvider(ABC):
    """Interface for models that can return token-level logits."""

    @abstractmethod
    def get_logits(self, prompt: str, **kwargs: Any) -> Any:
        """
        Return logits for the given prompt.

        Should return a tensor/array of shape (batch, seq_len, vocab_size),
        or something easily convertible to that by the caller.
        """
        raise NotImplementedError
    @abstractmethod
    async def sample_with_logprobs_async(
        self, 
        prompt: str, 
        n: int = 1, 
        top_k_logprobs: int = 5, 
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Generate answers AND return token-level log probabilities.
        
        Must return a List of Dictionaries with this structure:
        [
            {
                "text": "The generated answer...",
                "logprobs": [-0.1, -0.4, ...],  # Sequence of log-probs for chosen tokens (for Perplexity)
                "token_distributions": [        # List of dicts per token (for Entropy)
                    {"token_A": -0.1, "token_B": -2.5}, 
                    ...
                ]
            },
            ...
        ]
        """
        raise NotImplementedError


class SamplerProvider(ABC):
    """Interface for models that can generate multiple samples."""

    @abstractmethod
    def sample(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """
        Generate n completions for the given prompt.

        Returns:
            A list of generated strings.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def sample_async(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """
        Asynchronous generation (Required for High-Performance Pipeline).
        Should return a list of strings (answers).
        """
        raise NotImplementedError
