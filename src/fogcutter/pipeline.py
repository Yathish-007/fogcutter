# src/fogcutter/pipeline.py
import asyncio
from typing import Dict, Any, Optional
from src.fogcutter.config import settings
from src.fogcutter.providers.gemini import GeminiProvider
from src.fogcutter.blackbox.consistency import semantic_consistency_score
from src.fogcutter.blackbox.reflection import self_reflection_score
from src.fogcutter.providers.base import LogitsProvider
from src.fogcutter.whitebox.perplexity import calculate_perplexity
from src.fogcutter.whitebox.entropy import calculate_predictive_entropy

class FogCutterPipeline:
    def __init__(self):
        self.provider = GeminiProvider(
            model=settings.gemini_model,
            vertex_ai=settings.gemini_vertex_ai,
            project_id=settings.gemini_project_id
        )

    # Changed to 'async def'
    async def run(self, query: str, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Runs the pipeline with both Blackbox (Consistency/Reflection) 
        and Whitebox (Perplexity/Entropy) metrics.
        """
        final_n = num_samples if num_samples is not None else settings.n_samples
        temperature = settings.temperature
        
        print(f"Generating {final_n} answers for: '{query}' (Parallel + Whitebox)")
        
        # 1. GENERATE (Using the new method that gets Logprobs + Text)
        # We replaced 'sample_async' with 'sample_with_logprobs_async'
        if isinstance(self.provider, LogitsProvider):
            raw_results = await self.provider.sample_with_logprobs_async(
                prompt=query, 
                n=final_n, 
                temperature=temperature,
                top_k_logprobs=5
            )
        else:
            # Fallback (Just in case)
            print("Warning: Provider lacks whitebox support. Using standard sampling.")
            texts = await self.provider.sample_async(query, n=final_n, temperature=temperature)
            raw_results = [{"text": t, "logprobs": [], "token_distributions": []} for t in texts]

        if not raw_results:
            return {"error": "No answers generated"}

        # 2. EXTRACT TEXT (Restoring the variable your old code used)
        answers = [r["text"] for r in raw_results]
        best_answer = answers[0]

        # 3. CALCULATE WHITEBOX METRICS (New Logic)
        best_result = raw_results[0]
        perplexity_score = 0.0
        entropy_score = 0.0
        
        if best_result.get("logprobs"):
            perplexity_score = calculate_perplexity(best_result["logprobs"])
            
        if best_result.get("token_distributions"):
            entropy_score = calculate_predictive_entropy(best_result["token_distributions"])

        # 4. CALCULATE BLACKBOX METRICS (Old Logic - Kept)
        consistency_score = semantic_consistency_score(answers)
        reflection_score = await self_reflection_score(self.provider, query, best_answer)
        
        # 5. DETERMINE STATUS (Updated Logic)
        # We now require the model to be Consistent AND Confident (Low Perplexity)
        is_confident = (
            consistency_score > 0.8 and 
            reflection_score > 0.5 and
            (perplexity_score < 10.0 or perplexity_score == 0.0)
        )
        
        status = "CONFIDENT" if is_confident else "UNCERTAIN"

        # 6. RETURN (Updated Structure with new metrics)
        return {
            "query": query,
            "status": status,
            "best_answer": best_answer,
            "answers": answers,
            # Grouping metrics nicely
            "metrics": {
                "consistency_score": round(consistency_score, 4),
                "reflection_score": round(reflection_score, 4),
                "perplexity": round(perplexity_score, 4),
                "entropy": round(entropy_score, 4)
            }
        }