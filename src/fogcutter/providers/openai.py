from __future__ import annotations

from typing import Any, List, Optional

from openai import OpenAI

from .base import LogitsProvider, SamplerProvider


class OpenAIProvider(LogitsProvider, SamplerProvider):
    """
    Provider wrapper for OpenAI-compatible chat models.
    
    Supports:
    - OpenAI models (default)
    - Google Gemini via OpenAI-compatible endpoint
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    # Black-box: Generate samples for self-consistency, semantic entropy, etc.
    def sample(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """Generate n completions for the prompt."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            **kwargs,
        )
        return [choice.message.content for choice in resp.choices]

    # White-box: Logits/logprobs access (placeholder for future)
    def get_logits(self, prompt: str, **kwargs: Any) -> Any:
        """Placeholder for token-level logits/logprobs."""
        raise NotImplementedError(
            "Logits access not implemented yet. Use logprobs=True in kwargs when available."
        )
