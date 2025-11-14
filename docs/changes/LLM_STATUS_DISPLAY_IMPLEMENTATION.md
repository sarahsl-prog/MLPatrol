# LLM Status Display - Implementation Guide

**Date:** 2025-11-14
**Type:** UI Enhancement
**Status:** 📋 Design Ready for Implementation

---

## Overview

Add a status indicator in the MLPatrol UI to show users which LLM is being used (Cloud vs Local) and the specific model name. This improves transparency and helps users understand their configuration.

---

## Analysis

### Current UI Structure

**Location:** [app.py:847-861](../../app.py#L847-L861)

```python
# Header
gr.Markdown(f"# {APP_TITLE}")
gr.Markdown(APP_DESCRIPTION)

# Check agent initialization
agent_error = AgentState.get_error()
if agent_error:
    gr.Markdown(f"""
    ### ⚠️ Configuration Required

    {agent_error}

    Please set your API key and restart the application.
    """)
```

### Current AgentState Class

**Location:** [app.py:85-177](../../app.py#L85-L177)

The `AgentState` class stores:
- `_instance`: Agent instance
- `_initialized`: Initialization flag
- `_error`: Error message (if any)

**Missing:** Model information (model name, provider type, URL)

---

## Proposed Solution

### Option 1: Status Badge Below Header (Recommended ⭐)

Add a visual status badge right after the app description:

```
┌─────────────────────────────────────────────┐
│  # MLPatrol 🛡️                              │
│  AI-Powered Security Agent for ML Systems   │
│                                             │
│  🟢 Using: Claude Sonnet 4 (Cloud)          │
│  or                                         │
│  🔵 Using: llama3.1:8b (Local - Ollama)     │
└─────────────────────────────────────────────┘
```

**Pros:**
- Highly visible
- Always present
- Clear status indication
- Doesn't clutter the interface

### Option 2: Info Banner

Add an expandable info section:

```
┌─────────────────────────────────────────────┐
│  ℹ️ LLM Status: Click to expand             │
│  └─> Model: claude-sonnet-4-0              │
│      Provider: Anthropic (Cloud)            │
│      Status: Connected ✅                    │
└─────────────────────────────────────────────┘
```

**Pros:**
- Collapsible (less screen space)
- Can show more details
- Professional look

### Option 3: Settings Tab

Add a dedicated settings/info tab showing full configuration.

**Pros:**
- Complete configuration view
- Doesn't affect main UI

**Cons:**
- Hidden - users might not see it
- Extra click required

---

## Recommended Implementation (Option 1)

### Step 1: Add LLM Info Storage to AgentState

**File:** `app.py`
**Location:** Lines 85-177 (AgentState class)

```python
class AgentState:
    """Singleton class to manage the MLPatrol agent instance."""
    _instance: Optional[MLPatrolAgent] = None
    _initialized: bool = False
    _error: Optional[str] = None
    # NEW: Store LLM configuration info
    _llm_info: Optional[Dict[str, str]] = None

    @classmethod
    def get_llm_info(cls) -> Optional[Dict[str, str]]:
        """Get LLM configuration information.

        Returns:
            Dictionary with LLM info or None if not initialized
            {
                'provider': 'cloud' | 'local',
                'model': str,
                'type': 'anthropic' | 'openai' | 'ollama',
                'display_name': str,  # Friendly name for UI
                'status': 'connected' | 'error'
            }
        """
        if not cls._initialized:
            cls._initialize_agent()
        return cls._llm_info
```

### Step 2: Update _initialize_agent() to Store LLM Info

**File:** `app.py`
**Location:** Lines 108-169

**Add after successful initialization:**

```python
@classmethod
def _initialize_agent(cls) -> None:
    """Initialize the MLPatrol agent with API keys from environment."""
    try:
        logger.info("Initializing MLPatrol agent...")

        # Check for local LLM configuration first
        use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

        if use_local:
            # Local LLM via Ollama
            model = os.getenv("LOCAL_LLM_MODEL", "ollama/llama3.1:8b")
            base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
            api_key = None

            logger.info(f"Using local LLM: {model}")

            cls._instance = create_mlpatrol_agent(
                model=model,
                base_url=base_url,
                verbose=True,
                max_iterations=10,
                max_execution_time=180
            )

            logger.info(f"Agent initialized successfully with local model: {model}")
            cls._error = None

            # NEW: Store LLM info
            cls._llm_info = {
                'provider': 'local',
                'model': model.replace('ollama/', ''),  # Clean model name
                'type': 'ollama',
                'display_name': f"{model.replace('ollama/', '')} (Local - Ollama)",
                'url': base_url,
                'status': 'connected'
            }

        else:
            # Cloud LLMs (existing logic)
            api_key = os.getenv("ANTHROPIC_API_KEY")
            model = "claude-sonnet-4-0"

            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                model = "gpt-4"

            if not api_key:
                cls._error = (
                    "No API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                    "environment variable, or set USE_LOCAL_LLM=true for local models."
                )
                logger.error(cls._error)
                cls._initialized = True
                cls._llm_info = None  # NEW
                return

            cls._instance = create_mlpatrol_agent(
                api_key=api_key,
                model=model,
                verbose=True,
                max_iterations=10,
                max_execution_time=180
            )

            logger.info(f"Agent initialized successfully with model: {model}")
            cls._error = None

            # NEW: Store LLM info
            if "claude" in model.lower():
                provider_name = "Anthropic"
                llm_type = "anthropic"
            else:
                provider_name = "OpenAI"
                llm_type = "openai"

            cls._llm_info = {
                'provider': 'cloud',
                'model': model,
                'type': llm_type,
                'display_name': f"{model} (Cloud - {provider_name})",
                'status': 'connected'
            }

    except Exception as e:
        cls._error = f"Failed to initialize agent: {str(e)}"
        logger.error(f"Agent initialization failed: {e}", exc_info=True)
        # NEW: Set error status
        cls._llm_info = {
            'provider': 'unknown',
            'model': 'unknown',
            'type': 'unknown',
            'display_name': 'Not Connected',
            'status': 'error'
        }
    finally:
        cls._initialized = True
```

### Step 3: Add LLM Status Display to UI

**File:** `app.py`
**Location:** After line 850 (after APP_DESCRIPTION)

```python
# Header
gr.Markdown(f"# {APP_TITLE}")
gr.Markdown(APP_DESCRIPTION)

# NEW: Display LLM Status
llm_info = AgentState.get_llm_info()
if llm_info and llm_info['status'] == 'connected':
    if llm_info['provider'] == 'local':
        status_icon = "🔵"  # Blue for local
        status_color = "#3b82f6"  # Blue
    else:
        status_icon = "🟢"  # Green for cloud
        status_color = "#22c55e"  # Green

    gr.Markdown(f"""
    <div style="background-color: #f0f9ff; border-left: 4px solid {status_color}; padding: 12px 16px; margin: 10px 0; border-radius: 4px;">
        <strong>{status_icon} LLM Status:</strong> Using <code>{llm_info['display_name']}</code>
    </div>
    """)
elif llm_info and llm_info['status'] == 'error':
    gr.Markdown("""
    <div style="background-color: #fef2f2; border-left: 4px solid #dc2626; padding: 12px 16px; margin: 10px 0; border-radius: 4px;">
        <strong>🔴 LLM Status:</strong> Not Connected - Check configuration
    </div>
    """)

# Check agent initialization (existing code)
agent_error = AgentState.get_error()
if agent_error:
    gr.Markdown(f"""
    ### ⚠️ Configuration Required

    {agent_error}

    Please set your API key and restart the application.
    """)
```

---

## Complete Implementation Code

### Full Code Changes

#### Change 1: AgentState Class Updates

**File:** `app.py`
**Lines:** 85-177

```python
class AgentState:
    """Singleton class to manage the MLPatrol agent instance.

    This ensures we only initialize the agent once and reuse it across
    all user interactions for better performance.
    """
    _instance: Optional[MLPatrolAgent] = None
    _initialized: bool = False
    _error: Optional[str] = None
    _llm_info: Optional[Dict[str, str]] = None  # NEW

    @classmethod
    def get_agent(cls) -> Optional[MLPatrolAgent]:
        """Get or create the agent instance.

        Returns:
            MLPatrolAgent instance or None if initialization failed
        """
        if not cls._initialized:
            cls._initialize_agent()
        return cls._instance

    @classmethod
    def _initialize_agent(cls) -> None:
        """Initialize the MLPatrol agent with API keys from environment."""
        try:
            logger.info("Initializing MLPatrol agent...")

            # Check for local LLM configuration first
            use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

            if use_local:
                # Local LLM via Ollama
                model = os.getenv("LOCAL_LLM_MODEL", "ollama/llama3.1:8b")
                base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
                api_key = None

                logger.info(f"Using local LLM: {model}")

                cls._instance = create_mlpatrol_agent(
                    model=model,
                    base_url=base_url,
                    verbose=True,
                    max_iterations=10,
                    max_execution_time=180
                )

                logger.info(f"Agent initialized successfully with local model: {model}")
                cls._error = None

                # NEW: Store LLM info
                cls._llm_info = {
                    'provider': 'local',
                    'model': model.replace('ollama/', ''),
                    'type': 'ollama',
                    'display_name': f"{model.replace('ollama/', '')} (Local - Ollama)",
                    'url': base_url,
                    'status': 'connected'
                }

            else:
                # Cloud LLMs (existing logic)
                api_key = os.getenv("ANTHROPIC_API_KEY")
                model = "claude-sonnet-4-0"

                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY")
                    model = "gpt-4"

                if not api_key:
                    cls._error = (
                        "No API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                        "environment variable, or set USE_LOCAL_LLM=true for local models."
                    )
                    logger.error(cls._error)
                    cls._initialized = True
                    cls._llm_info = None  # NEW
                    return

                cls._instance = create_mlpatrol_agent(
                    api_key=api_key,
                    model=model,
                    verbose=True,
                    max_iterations=10,
                    max_execution_time=180
                )

                logger.info(f"Agent initialized successfully with model: {model}")
                cls._error = None

                # NEW: Store LLM info
                if "claude" in model.lower():
                    provider_name = "Anthropic"
                    llm_type = "anthropic"
                else:
                    provider_name = "OpenAI"
                    llm_type = "openai"

                cls._llm_info = {
                    'provider': 'cloud',
                    'model': model,
                    'type': llm_type,
                    'display_name': f"{model} (Cloud - {provider_name})",
                    'status': 'connected'
                }

        except Exception as e:
            cls._error = f"Failed to initialize agent: {str(e)}"
            logger.error(f"Agent initialization failed: {e}", exc_info=True)
            # NEW: Set error status
            cls._llm_info = {
                'provider': 'unknown',
                'model': 'unknown',
                'type': 'unknown',
                'display_name': 'Not Connected',
                'status': 'error'
            }
        finally:
            cls._initialized = True

    @classmethod
    def get_error(cls) -> Optional[str]:
        """Get initialization error if any."""
        if not cls._initialized:
            cls._initialize_agent()
        return cls._error

    @classmethod
    def get_llm_info(cls) -> Optional[Dict[str, str]]:
        """Get LLM configuration information.

        Returns:
            Dictionary with LLM info or None if not initialized
        """
        if not cls._initialized:
            cls._initialize_agent()
        return cls._llm_info
```

#### Change 2: UI Header Update

**File:** `app.py`
**Lines:** 847-861 (insert after line 849)

```python
# Header
gr.Markdown(f"# {APP_TITLE}")
gr.Markdown(APP_DESCRIPTION)

# NEW: Display LLM Status
llm_info = AgentState.get_llm_info()
if llm_info and llm_info['status'] == 'connected':
    if llm_info['provider'] == 'local':
        status_icon = "🔵"  # Blue for local
        status_color = "#3b82f6"  # Blue
        privacy_note = " • 100% Private"
    else:
        status_icon = "🟢"  # Green for cloud
        status_color = "#22c55e"  # Green
        privacy_note = ""

    gr.Markdown(f"""
    <div style="background-color: #f0f9ff; border-left: 4px solid {status_color}; padding: 12px 16px; margin: 10px 0; border-radius: 4px;">
        <strong>{status_icon} LLM Status:</strong> Using <code>{llm_info['display_name']}</code>{privacy_note}
    </div>
    """)
elif llm_info and llm_info['status'] == 'error':
    gr.Markdown("""
    <div style="background-color: #fef2f2; border-left: 4px solid #dc2626; padding: 12px 16px; margin: 10px 0; border-radius: 4px;">
        <strong>🔴 LLM Status:</strong> Not Connected - Check configuration
    </div>
    """)

# Check agent initialization
agent_error = AgentState.get_error()
# ... rest of existing code
```

---

## Visual Design

### Cloud LLM Status (Green)

```
┌────────────────────────────────────────────────────────────┐
│  🟢 LLM Status: Using claude-sonnet-4-0 (Cloud - Anthropic)│
└────────────────────────────────────────────────────────────┘
```

**Color:** Green (#22c55e)
**Background:** Light blue (#f0f9ff)
**Border:** Green left border

### Local LLM Status (Blue)

```
┌────────────────────────────────────────────────────────────┐
│  🔵 LLM Status: Using llama3.1:8b (Local - Ollama)         │
│               • 100% Private                               │
└────────────────────────────────────────────────────────────┘
```

**Color:** Blue (#3b82f6)
**Background:** Light blue (#f0f9ff)
**Border:** Blue left border

### Error Status (Red)

```
┌────────────────────────────────────────────────────────────┐
│  🔴 LLM Status: Not Connected - Check configuration        │
└────────────────────────────────────────────────────────────┘
```

**Color:** Red (#dc2626)
**Background:** Light red (#fef2f2)
**Border:** Red left border

---

## Alternative: Compact Version

For a more compact display:

```python
# Compact version (single line)
llm_info = AgentState.get_llm_info()
if llm_info and llm_info['status'] == 'connected':
    icon = "🔵" if llm_info['provider'] == 'local' else "🟢"
    gr.Markdown(f"""
    <div style="text-align: right; color: #6b7280; font-size: 0.9em; margin: -10px 0 10px 0;">
        {icon} {llm_info['display_name']}
    </div>
    """)
```

This shows in the top-right corner:
```
                                    🔵 llama3.1:8b (Local - Ollama)
```

---

## Testing Plan

### Test Cases

1. **Cloud - Anthropic:**
   - Set `ANTHROPIC_API_KEY`
   - Verify shows: `🟢 claude-sonnet-4-0 (Cloud - Anthropic)`

2. **Cloud - OpenAI:**
   - Set `OPENAI_API_KEY`
   - Verify shows: `🟢 gpt-4 (Cloud - OpenAI)`

3. **Local - Ollama:**
   - Set `USE_LOCAL_LLM=true`
   - Set `LOCAL_LLM_MODEL=ollama/llama3.1:8b`
   - Verify shows: `🔵 llama3.1:8b (Local - Ollama) • 100% Private`

4. **Error State:**
   - Remove all API keys
   - Set `USE_LOCAL_LLM=false`
   - Verify shows: `🔴 Not Connected - Check configuration`

5. **Different Models:**
   - Test with `llama3.1:70b`
   - Test with `mistral-small:3.1`
   - Test with `claude-opus-4-0`

---

## Summary of Changes

### Files Modified: 1
- **app.py:** AgentState class + UI header

### Lines Added: ~80
- AgentState: +40 lines (info storage + getter)
- UI: +40 lines (status display)

### Breaking Changes: None
- Fully backward compatible
- Existing functionality unchanged
- Only adds new UI element

---

## Benefits

✅ **Transparency:** Users see exactly which LLM is running
✅ **Debugging:** Easier to troubleshoot configuration issues
✅ **Privacy Awareness:** Clear indication when using local LLMs
✅ **Professional Look:** Polished UI with status indicators
✅ **No Clutter:** Compact, non-intrusive display

---

## Next Steps

1. Implement AgentState changes
2. Add UI status display
3. Test with all LLM types
4. Verify visual appearance
5. Update documentation if needed

---

**Author:** Claude Code
**Status:** Ready for Implementation
