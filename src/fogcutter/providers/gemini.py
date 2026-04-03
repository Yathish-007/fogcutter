from __future__ import annotations
import asyncio
from typing import Any, List, Optional, Dict
import google.generativeai as genai
from google.generativeai.types import GenerationConfig as StandardGenerationConfig
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

    # --- 1. Async Parallel Method ---
    async def sample_async(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """Generate n completions in parallel."""
        
        # 1. Extract arguments meant for GenerationConfig
        config_args = {}
        
        # Temperature
        if 'temperature' in kwargs:
            config_args['temperature'] = kwargs.pop('temperature')
            
        # Response Schema (Structured Output)
        if 'response_schema' in kwargs:
            config_args['response_schema'] = kwargs.pop('response_schema')
            config_args['response_mime_type'] = "application/json"

        # Create the config object with all parameters
        
        if self.vertex_ai:
            from vertexai.generative_models import GenerationConfig as VertexGenerationConfig
            config = VertexGenerationConfig(**config_args)
        else:
            config = StandardGenerationConfig(**config_args)

        # Define single generation task
        async def generate_single():
            try:
                # Pass ONLY config and prompt. 'kwargs' contains other random args if any.
                response = await self._model.generate_content_async(
                    prompt, 
                    generation_config=config, 
                    **kwargs
                )
                return response.text if response.text else ""
            except Exception as e:
                print(f"Error during async generation: {e}")
                return ""

        # Run tasks
        tasks = [generate_single() for _ in range(n)]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]

    # --- 2. Sync Method (Required by Interface) ---
    def sample(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """Sync version (not used by pipeline currently but required by interface)."""
        config_args = {}
        if 'temperature' in kwargs:
            config_args['temperature'] = kwargs.pop('temperature')
        if 'response_schema' in kwargs:
            config_args['response_schema'] = kwargs.pop('response_schema')
            config_args['response_mime_type'] = "application/json"
        
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
        
        from vertexai.generative_models import GenerationConfig as VertexGenerationConfig
        config = VertexGenerationConfig(
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
     # --- 4. Parallel Whitebox Method (New) ---
    async def sample_with_logprobs_async(self, prompt: str, n: int = 1, top_k_logprobs: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Generates n answers in parallel with log probabilities for Whitebox analysis.
        Returns: List of dicts {'text': str, 'logprobs': List[float], 'token_distributions': List[Dict]}
        """
        if not self.vertex_ai:
            print("Warning: Logprobs generally require Vertex AI (vertex_ai=True) to work correctly.")

        # Setup Config specifically for Logprobs
        config_args = {}
        if 'temperature' in kwargs:
            config_args['temperature'] = kwargs.pop('temperature')
            
        # FORCE Logprobs on
        config_args['response_logprobs'] = True
        config_args['logprobs'] = top_k_logprobs

        if self.vertex_ai:
            from vertexai.generative_models import GenerationConfig as VertexGenerationConfig
            config = VertexGenerationConfig(**config_args)
        else:
            config = StandardGenerationConfig(**config_args)

        async def generate_single_whitebox():
            try:
                # Vertex AI Async Generation
                response = await self._model.generate_content_async(
                    prompt, 
                    generation_config=config, 
                    **kwargs
                )
                
                result = {"text": "", "logprobs": [], "token_distributions": []}
                
                if response.text:
                    result["text"] = response.text
                    
                # Parsing Complex Vertex Logprobs Structure
                if response.candidates:
                    candidate = response.candidates[0]
                    # Note: SDK attributes vary. We check 'logprobs_result' which is standard for Vertex.
                    if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                        
                        # top_candidates is a list of TopCandidates objects (one per token step)
                        for token_step in candidate.logprobs_result.top_candidates:
                            if token_step.candidates:
                                # 1. The Chosen Token (First in the list usually matches the generated text)
                                chosen = token_step.candidates[0]
                                result["logprobs"].append(chosen.log_probability)
                                
                                # 2. The Distribution (All candidates at this step for Entropy)
                                # Map {token_string: log_probability}
                                step_dist = {c.token: c.log_probability for c in token_step.candidates}
                                result["token_distributions"].append(step_dist)
                                
                return result

            except Exception as e:
                print(f"Error during whitebox generation: {e}")
                return None

        # Execute Parallel Tasks
        tasks = [generate_single_whitebox() for _ in range(n)]
        results = await asyncio.gather(*tasks)
        
        # Filter out failed calls
        return [r for r in results if r is not None]
