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
