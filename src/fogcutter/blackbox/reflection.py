import json
from src.fogcutter.providers.gemini import GeminiProvider
from src.fogcutter.models import ReflectionResponse

async def self_reflection_score(provider: GeminiProvider, query: str, answer: str) -> float:
    
    prompt = f"""
    Review the following Q&A pair.
    
    Question: {query}
    Proposed Answer: {answer}
    
    Task:
    1. Analyze if the answer is factually correct.
    2. Provide a confidence score (0.0 to 1.0).
    """
    
    try:
        # Pass the raw schema to Gemini (It supports Python types/dicts in newer SDKs)
        # Note: We pass the Pydantic model class; the SDK often handles the conversion 
        # or we pass the JSON schema dict.
        
        response_list = await provider.sample_async(
            prompt, 
            n=1, 
            temperature=0.0,
            response_schema=ReflectionResponse # Pass the Class itself (supported in newer Google SDKs)
        )
        
        if not response_list:
            return 0.5 
        print("hitting here",response_list)
        raw_text = response_list[0].strip()
        
        # Now raw_text is GUARANTEED to be JSON (no markdown backticks usually)
        # We can parse it directly
        reflection = ReflectionResponse.model_validate_json(raw_text)
        
        return reflection.score
            
    except Exception as e:
        print(f"Reflection error: {e}")
        return 0.0
