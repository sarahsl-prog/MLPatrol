# MLPatrol Development Guide

Complete guide for setting up, running, and extending MLPatrol.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Locally](#running-locally)
- [Project Structure](#project-structure)
- [Code Quality](#code-quality)
- [Adding New Features](#adding-new-features)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required
- **Python 3.10+** (3.11 or 3.12 recommended for best performance)
- **pip** or **conda** for package management
- **API Key** for LLM provider:
  - **Claude Sonnet 4** (recommended): Get from [console.anthropic.com](https://console.anthropic.com)
  - **GPT-4** (alternative): Get from [platform.openai.com](https://platform.openai.com)

**Note:** Python 3.10+ is required for modern LangGraph and Pydantic v2 features.

### Optional
- **Git** for version control
- **VS Code** or **PyCharm** for IDE
- **Docker** for containerized deployment (future)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sarahsl-prog/MLPatrol.git
cd MLPatrol
```

### 2. Create Virtual Environment
We strongly recommend using a virtual environment to avoid dependency conflicts.

**Using venv (built-in):**
```bash
python -m venv venv

# Activate on Linux/Mac:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n mlpatrol python=3.10
conda activate mlpatrol
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- **Gradio 5.0+**: Web interface
- **LangGraph 1.0+**: Modern agent framework (graph-based)
- **LangChain 1.0+**: Core agent utilities
- **Pydantic 2.5+**: Fast data validation
- **NumPy, Pandas, scikit-learn, SciPy**: Data analysis
- **Plotly 5.18+**: Interactive visualizations
- **Anthropic/OpenAI SDKs**: LLM providers
- **Additional utilities**: requests, httpx, python-dotenv

### 4. Verify Installation
```bash
python -c "import gradio; print(f'Gradio {gradio.__version__}')"
python -c "import langgraph; print(f'LangGraph {langgraph.__version__}')"
python -c "import pydantic; print(f'Pydantic {pydantic.__version__}')"
```

Expected output:
```
Gradio 5.x.x
LangGraph 1.x.x
Pydantic 2.x.x
```

## Configuration

### 1. Create Environment File
```bash
cp .env.example .env
```

### 2. Add API Keys
Edit `.env` and add your API key:

**For Claude (recommended):**
```bash
ANTHROPIC_API_KEY=sk-ant-api03-...your-key-here...
```

**For OpenAI (alternative):**
```bash
OPENAI_API_KEY=sk-...your-key-here...
```

**Important:**
- Never commit `.env` to version control
- `.gitignore` is already configured to exclude `.env`
- The app will automatically choose Claude if available, otherwise GPT-4

### 3. Verify Configuration
```bash
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('OPENAI_API_KEY')
print('✅ API key configured!' if key else '❌ No API key found')
"
```

## Running Locally

### Basic Usage
```bash
python app.py
```

The app will:
1. Load environment variables
2. Initialize the MLPatrol agent (2-3 seconds)
3. Start Gradio server on port 7860
4. Open browser automatically (or manually visit http://localhost:7860)

### Expected Output
```
2025-11-13 10:30:00 - root - INFO - Starting MLPatrol application...
2025-11-13 10:30:00 - root - INFO - Initializing agent...
2025-11-13 10:30:02 - root - INFO - Agent initialized successfully with model: claude-sonnet-4
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

### Advanced Options

**Change port:**
```python
# In app.py, modify launch():
interface.launch(server_port=8080)
```

**Create public link:**
```python
interface.launch(share=True)
```
This generates a temporary public URL for sharing.

**Disable auto-open browser:**
```bash
GRADIO_SERVER_NAME="0.0.0.0" GRADIO_SERVER_PORT=7860 python app.py
```

## Project Structure

```
MLPatrol/
├── app.py                          # Main Gradio application (1,107 lines)
│   ├── AgentState                  # Singleton for agent management
│   ├── validate/sanitize functions # Input security
│   ├── Visualization functions     # Plotly charts
│   ├── Handler functions           # Tab logic
│   └── create_interface()          # Gradio UI layout
│
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── reasoning_chain.py      # MLPatrolAgent class (802 lines)
│   │   ├── prompts.py              # Agent prompts
│   │   └── tools.py                # Tool definitions (940 lines)
│   │
│   ├── security/                   # CVE monitoring (placeholder)
│   │   ├── __init__.py
│   │   ├── cve_monitor.py
│   │   ├── code_generator.py
│   │   └── threat_intel.py
│   │
│   ├── dataset/                    # Dataset analysis (placeholder)
│   │   ├── __init__.py
│   │   ├── poisoning_detector.py
│   │   ├── bias_analyzer.py
│   │   └── statistical_tests.py
│   │
│   ├── mcp/                        # MCP integration (placeholder)
│   │   ├── __init__.py
│   │   └── connectors.py
│   │
│   └── utils/                      # Utilities (placeholder)
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
│
├── tests/                          # Unit tests
│   ├── __init__.py
│   ├── test_agent.py               # Agent tests
│   ├── test_security.py            # Security tests
│   └── test_dataset.py             # Dataset tests
│
├── docs/                           # Documentation
│   ├── ARCHITECTURE.md             # System architecture
│   ├── AGENT_REASONING.md          # Agent reasoning guide
│   ├── DEVELOPMENT.md              # This file
│   └── DEMO_GUIDE.md               # Demo video guide
│
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── .env                            # Your API keys (git-ignored)
├── .gitignore                      # Git ignore rules
├── README.md                       # Main README
└── mlpatrol.log                    # Application logs (auto-generated)
```

## Code Quality

MLPatrol follows strict code quality standards:

### Type Hints
All functions must have complete type annotations:
```python
from typing import List, Dict, Optional, Tuple

def process_data(
    data: List[Dict[str, Any]],
    threshold: float = 0.5
) -> Tuple[bool, Optional[str]]:
    """Process data with threshold."""
    # Implementation
    return True, None
```

### Docstrings
Use Google-style docstrings for all classes and functions:
```python
def analyze_dataset(file_path: str) -> Dict[str, Any]:
    """Analyze dataset for security issues.

    Args:
        file_path: Path to CSV dataset file

    Returns:
        Dictionary containing analysis results with keys:
        - outliers: List of outlier indices
        - quality_score: Quality score (0-10)
        - recommendations: List of recommended actions

    Raises:
        FileNotFoundError: If file_path doesn't exist
        ValueError: If file is not valid CSV

    Example:
        >>> result = analyze_dataset("data.csv")
        >>> print(f"Quality: {result['quality_score']}")
    """
    # Implementation
```

### PEP 8 Compliance
```bash
# Check style
flake8 src/ app.py --max-line-length=100

# Auto-format
black src/ app.py --line-length=100
```

### Logging
Use Python's `logging` module, never `print()`:
```python
import logging

logger = logging.getLogger(__name__)

# Usage
logger.info("Starting process")
logger.warning("Validation failed")
logger.error("Critical error", exc_info=True)
```

### Error Handling
Always use specific exception types:
```python
try:
    result = process_data(data)
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
except ValueError as e:
    logger.warning(f"Invalid data: {e}")
    return default_value
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

## Adding New Features

### Adding a New Tab

1. **Define handler function in `app.py`:**
```python
def handle_new_feature(
    param1: str,
    param2: int,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Handle new feature request.

    Args:
        param1: Description
        param2: Description
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message, results_html)
    """
    try:
        logger.info(f"New feature request: {param1}")
        progress(0, desc="Starting...")

        # Get agent
        agent = AgentState.get_agent()
        if agent is None:
            return "Agent not initialized", ""

        # Create query
        query = f"Process {param1} with parameter {param2}"

        progress(0.5, desc="Processing...")

        # Run agent
        result = agent.run(query)

        # Format results
        html = f"<p>{result.answer}</p>"
        status = f"✅ Complete | Confidence: {result.confidence:.0%}"

        progress(1.0, desc="Done!")

        return status, html

    except Exception as e:
        logger.error(f"Feature failed: {e}", exc_info=True)
        return f"❌ Error: {e}", ""
```

2. **Add tab in `create_interface()` function:**
```python
with gr.Tab("🆕 New Feature"):
    gr.Markdown("### Description of new feature")

    with gr.Row():
        with gr.Column():
            input1 = gr.Textbox(label="Parameter 1")
            input2 = gr.Slider(1, 100, label="Parameter 2")
            submit_btn = gr.Button("Process", variant="primary")

        with gr.Column():
            status = gr.Textbox(label="Status")
            results = gr.HTML(label="Results")

    # Connect handler
    submit_btn.click(
        fn=handle_new_feature,
        inputs=[input1, input2],
        outputs=[status, results]
    )
```

### Adding a New Tool

1. **Define tool implementation in `tools.py`:**
```python
def new_tool_impl(param: str, option: int = 10) -> str:
    """Implementation of new tool.

    Args:
        param: Description
        option: Optional parameter

    Returns:
        JSON string with results
    """
    try:
        logger.info(f"New tool called with: {param}")

        # Tool logic here
        result = {"status": "success", "data": {...}}

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Tool failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)
        })
```

2. **Define input schema (Pydantic v2):**
```python
from pydantic import BaseModel, Field, field_validator

class NewToolInput(BaseModel):
    """Input schema for new tool."""

    param: str = Field(description="Description of param")
    option: int = Field(
        default=10,
        description="Description of option",
        ge=1,  # Constraints properly enforced in Pydantic v2
        le=100
    )

    @field_validator("param")
    @classmethod
    def validate_param(cls, v):
        """Validate param using Pydantic v2 syntax."""
        if not v:
            raise ValueError("param cannot be empty")
        return v.strip()
```

3. **Add to tool list in `create_mlpatrol_tools()`:**
```python
tools = [
    # Existing tools...
    StructuredTool.from_function(
        func=new_tool_impl,
        name="new_tool",
        description="Description for LLM - when to use this tool",
        args_schema=NewToolInput,
        return_direct=False,
    ),
]
```

### Adding a New Visualization

```python
def create_custom_chart(data: Dict[str, Any]) -> go.Figure:
    """Create custom Plotly chart.

    Args:
        data: Data dictionary with keys for chart

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(
        x=data.get('x', []),
        y=data.get('y', []),
        mode='lines+markers',
        name='Series 1'
    ))

    # Update layout
    fig.update_layout(
        title="Custom Chart Title",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        template="plotly_white",
        height=400,
    )

    return fig
```

## Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_agent.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run with verbose output
pytest tests/ -v
```

### Writing Tests
```python
# tests/test_new_feature.py
import pytest
from src.agent.tools import new_tool_impl

def test_new_tool_success():
    """Test new tool with valid input."""
    result = new_tool_impl("test_param", option=5)
    data = json.loads(result)

    assert data["status"] == "success"
    assert "data" in data

def test_new_tool_invalid_input():
    """Test new tool with invalid input."""
    with pytest.raises(ValueError):
        new_tool_impl("", option=-1)
```

### Manual Testing

1. **Start the app:**
```bash
python app.py
```

2. **Test each tab:**
   - CVE: Search for "numpy" with 90 days
   - Dataset: Upload a sample CSV
   - Code: Generate validation script
   - Chat: Ask a security question

3. **Check logs:**
```bash
tail -f mlpatrol.log
```

## Troubleshooting

### "No API key found" Error

**Cause:** Missing or incorrect environment variable

**Solution:**
```bash
# Check if .env exists
ls -la .env

# Verify contents (don't show full key!)
grep "API_KEY" .env

# Test loading
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('ANTHROPIC_API_KEY'))"
```

### "Agent initialization failed" Error

**Cause:** Missing LangChain dependencies

**Solution:**
```bash
pip install langgraph langchain-anthropic  # For Claude with LangGraph
# OR
pip install langgraph langchain-openai     # For GPT-4 with LangGraph

# Ensure you have the modern stack
pip install "langchain>=1.0" "langgraph>=1.0" "pydantic>=2.5"
```

### File Upload Fails

**Cause:** Invalid CSV format or size

**Solution:**
- Ensure file is valid CSV with headers
- Check file size < 10MB
- Verify CSV loads in pandas:
```python
import pandas as pd
df = pd.read_csv("your_file.csv")
print(df.head())
```

### Import Errors

**Cause:** Missing dependencies

**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Or install missing package
pip install <missing-package>
```

### Port Already in Use

**Cause:** Another process using port 7860

**Solution:**
```bash
# Find process using port
lsof -i :7860  # Mac/Linux
netstat -ano | findstr :7860  # Windows

# Kill process or change port in app.py
interface.launch(server_port=8080)
```

### Slow Agent Responses

**Possible causes:**
- Cold start (first request takes 2-3s for LLM init)
- Large dataset analysis
- Network issues with APIs

**Solutions:**
- Singleton pattern already optimizes subsequent requests
- Reduce dataset size for testing
- Check internet connection

## Development Workflow

### Recommended Setup

1. **IDE Configuration:**
   - VS Code: Install Python extension, Pylance for type checking
   - PyCharm: Enable type checking in settings

2. **Pre-commit hooks:**
```bash
# Install pre-commit
pip install pre-commit

# Setup hooks (future)
pre-commit install
```

3. **Type checking:**
```bash
# Install mypy
pip install mypy

# Check types
mypy src/ app.py
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes
# ... code ...

# Add and commit
git add .
git commit -m "Add new feature: description"

# Push and create PR
git push origin feature/new-feature
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Read [AGENT_REASONING.md](AGENT_REASONING.md) for agent internals
- Check [README.md](../README.md) for usage examples
- Join discussions on [GitHub Issues](https://github.com/sarahsl-prog/MLPatrol/issues)
