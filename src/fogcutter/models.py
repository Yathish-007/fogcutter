from pydantic import BaseModel, Field

class ReflectionResponse(BaseModel):
    """
    Schema for self-reflection output.
    """
    reasoning: str = Field(..., description="Short analysis of factual correctness")
    # REMOVED ge=0.0 and le=1.0 to avoid "Unknown field: maximum" error
    score: float = Field(..., description="Confidence score between 0.0 and 1.0")
