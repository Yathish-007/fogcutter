# src/fogcutter/models.py

# This must be a simple dictionary now
REFLECTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "NUMBER"},
        "reason": {"type": "STRING"}
    },
    "required": ["score", "reason"]
}
