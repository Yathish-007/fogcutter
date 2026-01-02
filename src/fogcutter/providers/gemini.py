from __future__ import annotations
import asyncio
from typing import Any, List, Optional, Dict
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from .base import LogitsProvider, SamplerProvider

class GeminiProvider(LogitsProvider, SamplerProvider):
    def __init__(self, model: str, api_key: Optional[str] = None, vertex_ai: bool = False, project_id: Optional[str] = None, location: str = "us-central1") -> None:
        self.model_name = model
        self.vertex_ai = vertex_ai
        
        if vertex_ai:
            import vertexai
            vertexai.init(project=project_id, location=location)
            from vertexai.generative_models import GenerativeModel
            self._model = GenerativeModel(model)
        else:
            if api_key:
                genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model)

    # --- 1. Async Parallel Method (The one we use) ---
    async def sample_async(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """Generate n completions in parallel."""
        config_args = {}
        if 'temperature' in kwargs:
            config_args['temperature'] = kwargs.pop('temperature')
        config = GenerationConfig(**config_args)

        async def generate_single():
            try:
                response = await self._model.generate_content_async(prompt, generation_config=config, **kwargs)
                return response.text if response.text else ""
            except Exception as e:
                print(f"Error during async generation: {e}")
                return ""

        tasks = [generate_single() for _ in range(n)]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]

    # --- 2. Sync Method (Required by Interface) ---
    def sample(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """Sync version (not used by pipeline currently but required by interface)."""
        # We can implement it simply by running the async version if needed, 
        # or just keep the old loop logic.
        config_args = {}
        if 'temperature' in kwargs:
            config_args['temperature'] = kwargs.pop('temperature')
        config = GenerationConfig(**config_args)
        
        outputs = []
        for _ in range(n):
            try:
                response = self._model.generate_content(prompt, generation_config=config, **kwargs)
                if response.text:
                    outputs.append(response.text)
            except Exception as e:
                print(f"Error: {e}")
        return outputs

    # --- 3. Logits Method (Required by LogitsProvider) ---
    def get_logits(self, prompt: str, logprobs: int = 5, **kwargs: Any) -> Dict[str, Any]:
        """Get logprobs for the generated response (Vertex AI only)."""
        if not self.vertex_ai:
            raise RuntimeError("Logprobs require vertex_ai=True")

        config = GenerationConfig(
            response_logprobs=True,
            logprobs=logprobs,
            **kwargs
        )

        response = self._model.generate_content(prompt, generation_config=config)
        
        if not response.candidates:
             raise ValueError("No candidates returned")
             
        candidate = response.candidates[0]
        # Check for logprobs property safely
        if not hasattr(candidate, 'logprobs_result') or not candidate.logprobs_result:
             raise ValueError("No logprobs returned")

        return {
            "logprobs_result": candidate.logprobs_result,
            "avg_logprobs": getattr(candidate, "avg_logprobs", 0.0), 
            "response": response.text,
        }
