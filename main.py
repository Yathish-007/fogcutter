# main.py
import asyncio  # <--- Import asyncio
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.fogcutter.pipeline import FogCutterPipeline

# Changed main to be async
async def main():
    load_dotenv()
    
    # ... (Credentials check code) ...

    print("Initializing Pipeline...")
    pipeline = FogCutterPipeline()

    queries = [
        "What is the capital of France?",
        "Explain the meaning of life in one sentence."
    ]

    for query in queries:
        print(f"\n[QUERY] {query}")
        try:
            # AWAIT the pipeline run
            result = await pipeline.run(query) 
            
            metrics = result.get("metrics", {})
            
            print(f"  -> Consistency: {metrics.get('consistency_score', 0.0):.4f}")
            print(f"  -> Reflection:  {metrics.get('reflection_score', 0.0):.4f}")
            print(f"  -> Perplexity:  {metrics.get('perplexity', 0.0):.4f} (Lower is better)")
            print(f"  -> Entropy:     {metrics.get('entropy', 0.0):.4f}")
            print(f"  -> STATUS:      {result['status']}")
            print(f"  -> Answer:      {result['best_answer'][:100]}...")
            
            
        except Exception as e:
            print(f"  -> Error: {e}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
