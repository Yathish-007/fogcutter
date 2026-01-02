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
            
            print(f"  -> Score:  {result['consistency_score']:.4f} ({result['status']})")
            print(f"  -> Answer: {result['best_answer'][:100]}...")
            
        except Exception as e:
            print(f"  -> Error: {e}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
