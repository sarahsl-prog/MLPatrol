# Model Name Fix - November 13, 2025

## Issue
Application was using incorrect model name `claude-sonnet-4` which doesn't exist in Anthropic's API.

## Error
```
ERROR - Query classification failed: Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model: claude-sonnet-4'}}
```

## Root Cause
The model name format changed. Anthropic now uses dated model IDs like:
- `claude-sonnet-4-20250514` (not `claude-sonnet-4`)
- `claude-sonnet-4-5-20250929` (newest Sonnet 4.5)
- `claude-opus-4-20250514` (not `claude-opus-4`)

## Fix Applied

### Files Changed:
1. **`src/agent/reasoning_chain.py`** - Updated default model name and documentation
2. **`app.py`** - Updated model selection in AgentState

### Changes:

**Before:**
```python
model = "claude-sonnet-4"  # ❌ Invalid
```

**After:**
```python
model = "claude-sonnet-4-20250514"  # ✅ Valid
```

## Available Models (as of Nov 2025)

### Claude Models (Anthropic)
- `claude-haiku-4-5-20251001` - Fastest, cheapest
- `claude-sonnet-4-5-20250929` - **Newest Sonnet** (recommended for best quality)
- `claude-sonnet-4-20250514` - **Default** (balanced)
- `claude-opus-4-1-20250805` - Most capable
- `claude-opus-4-20250514` - Alternative Opus
- `claude-3-7-sonnet-20250219` - Legacy
- `claude-3-5-haiku-20241022` - Legacy
- `claude-3-haiku-20240307` - Legacy
- `claude-3-opus-20240229` - Legacy

### OpenAI Models
- `gpt-4` - Standard GPT-4
- `gpt-4-turbo` - Faster GPT-4
- `gpt-4o` - Multimodal GPT-4

## How to Change Model

### Option 1: Use Default (Recommended)
No changes needed - will use `claude-sonnet-4-20250514` automatically.

### Option 2: Set in Code
Edit `app.py` line 116:
```python
model = "claude-sonnet-4-5-20250929"  # Use newest Sonnet 4.5
```

### Option 3: Use create_mlpatrol_agent()
```python
from src.agent.reasoning_chain import create_mlpatrol_agent

agent = create_mlpatrol_agent(
    api_key="your-key",
    model="claude-sonnet-4-5-20250929",  # Specify model
    verbose=True
)
```

## Testing

Verified the fix works:
```bash
$ python -c "from app import AgentState; ..."
INFO - Initialized Claude LLM: claude-sonnet-4-20250514
INFO - LangGraph agent configured successfully
INFO - Agent initialized successfully
Result: Success ✅
```

## Recommendation

**For Production:** Use `claude-sonnet-4-20250514` (current default)
- Stable, tested
- Good balance of speed/quality
- Lower cost than Opus

**For Best Quality:** Use `claude-sonnet-4-5-20250929`
- Latest Sonnet 4.5
- Improved reasoning
- Worth testing if quality is critical

**For Cost Savings:** Use `claude-haiku-4-5-20251001`
- Much faster
- 10x cheaper
- Good for simple tasks

## Future-Proofing

To avoid this issue in future, consider:

1. **Add model validation** on startup:
```python
def validate_model(model: str):
    """Validate model exists before using."""
    from anthropic import Anthropic
    client = Anthropic()
    available = [m.id for m in client.models.list().data]
    if model not in available:
        raise ValueError(f"Model {model} not found. Available: {available[:5]}")
```

2. **Use environment variable** for model selection:
```python
model = os.getenv("MLPATROL_MODEL", "claude-sonnet-4-20250514")
```

3. **Add to .env.example**:
```bash
# Model Selection (optional)
# Options: claude-sonnet-4-20250514, claude-sonnet-4-5-20250929, gpt-4
# MLPATROL_MODEL=claude-sonnet-4-20250514
```

## Related Documentation

- [Anthropic Models](https://docs.anthropic.com/en/docs/about-claude/models)
- [OpenAI Models](https://platform.openai.com/docs/models)
- [MLPatrol README](README.md)
- [Development Guide](docs/DEVELOPMENT.md)
