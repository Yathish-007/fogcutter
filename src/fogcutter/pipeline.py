# src/fogcutter/pipeline.py
import asyncio
from typing import Dict, Any, Optional
from src.fogcutter.config import settings
from src.fogcutter.providers.gemini import GeminiProvider
from src.fogcutter.blackbox.consistency import semantic_consistency_score

class FogCutterPipeline:
    def __init__(self):
        self.provider = GeminiProvider(
            model=settings.gemini_model,
            vertex_ai=settings.gemini_vertex_ai,
            project_id=settings.gemini_project_id
        )

    # Changed to 'async def'
    async def run(self, query: str, num_samples: Optional[int] = None) -> Dict[str, Any]:
        final_n = num_samples if num_samples is not None else settings.n_samples
        temperature = settings.temperature
        
        print(f"Generating {final_n} answers for: '{query}' (Parallel)")
        
        # AWAIT the async sample method
        answers = await self.provider.sample_async(
            prompt=query, 
            n=final_n, 
            temperature=temperature
        )
        
        if not answers:
            return {"error": "No answers generated"}

        # Consistency check is fast and local, so it can stay sync
        # (or you could wrap it in asyncio.to_thread if it blocks too long)
        consistency_score = semantic_consistency_score(answers)

        return {
            "query": query,
            "consistency_score": consistency_score,
            "status": "CONFIDENT" if consistency_score > 0.8 else "UNCERTAIN",
            "answers": answers,
            "best_answer": answers[0] if answers else ""
        }
