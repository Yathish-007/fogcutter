import json
from src.fogcutter.providers.gemini import SamplerProvider
from src.fogcutter.models import REFLECTION_SCHEMA

async def self_reflection_score(provider: SamplerProvider, query: str, answer: str) -> float:
    
    """
    Asks the model to grade its own answer on a scale of 0.0 to 1.0.
    """
    prompt = f"""
    Rate the following answer to the user's question on a scale from 0.0 to 1.0.
    
    Question: {query}
    Answer: {answer}
    
    Return strictly JSON with 'score' (float) and 'reason' (string).
    Example: {{"score": 0.9, "reason": "Accurate and complete"}}
    """
    
    try:
        # Pass the raw schema to Gemini (It supports Python types/dicts in newer SDKs)
        # Note: We pass the Pydantic model class; the SDK often handles the conversion 
        # or we pass the JSON schema dict.
        
        response_list = await provider.sample_async(
            prompt, 
            n=1, 
            temperature=0.0,
            response_schema=REFLECTION_SCHEMA # Pass the Class itself (supported in newer Google SDKs)
        )
        
        if not response_list:
            return 0.5 
        first_result = response_list[0]
        if isinstance(first_result, dict):
            raw_text = first_result.get("text", "").strip()
        else:
            raw_text = str(first_result).strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").replace("json", "").strip()

        data = json.loads(raw_text)
        return float(data.get("score", 0.0))

    except Exception as e:
        print(f"Reflection error: {e}")
        return 0.0
        