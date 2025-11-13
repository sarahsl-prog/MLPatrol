# MLPatrol Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                       │
│                  (Gradio 6 - app.py)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │   CVE    │ │ Dataset  │ │   Code   │ │   Chat   │  │
│  │   Tab    │ │   Tab    │ │   Tab    │ │   Tab    │  │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘  │
└────────┼────────────┼────────────┼────────────┼────────┘
         │            │            │            │
         └────────────┴────────────┴────────────┘
                      │
         ┌────────────▼───────────────┐
         │      AgentState             │
         │   (Singleton Pattern)       │
         │   - Single agent instance   │
         │   - Shared across requests  │
         └────────────┬───────────────┘
                      │
         ┌────────────▼───────────────────┐
         │      MLPatrolAgent             │
         │   (reasoning_chain.py)         │
         │                                │
         │  • Query classification        │
         │  • Multi-step planning         │
         │  • Tool selection & execution  │
         │  • Result synthesis            │
         │  • Confidence scoring          │
         └────────┬───────────────────────┘
                  │
         ┌────────▼──────────┐
         │  LangChain Core   │
         │  • Agent Executor │
         │  • Tool Binding   │
         │  • Prompt Manager │
         └────────┬──────────┘
                  │
         ┌────────▼──────────────────────────┐
         │         LLM Backend               │
         │  Claude Sonnet 4 / GPT-4          │
         │  • Function calling               │
         │  • Structured outputs             │
         │  • Context management             │
         └────────┬──────────────────────────┘
                  │
    ┌─────────────┼─────────────┬─────────────┐
    │             │             │             │
┌───▼───┐   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
│  CVE  │   │ Dataset │   │  Code   │   │   Web   │
│Search │   │Analysis │   │Generator│   │ Search  │
└───┬───┘   └────┬────┘   └────┬────┘   └────┬────┘
    │            │             │             │
    ▼            ▼             ▼             ▼
  NVD API    NumPy/Pandas   Templates    MCP Tools
  Security   Statistical    Jinja2       HF Datasets
  Papers     Tests          AST          Web APIs
```

## Agent Reasoning Flow

### 1. Query Classification (reasoning_chain.py:280-336)
```python
QueryType = Enum:
  - CVE_MONITORING
  - DATASET_ANALYSIS
  - CODE_GENERATION
  - GENERAL_SECURITY
```
- LLM analyzes user query to determine intent
- Extracts context (file uploads, specific libraries, etc.)
- Routes to appropriate tool set

### 2. Multi-Step Planning (reasoning_chain.py:464-595)
```python
Agent Steps:
1. Validate query (length, suspicious patterns)
2. Classify query type
3. Plan tool execution sequence
4. Execute tools with intermediate steps
5. Synthesize final answer
6. Calculate confidence score
```

### 3. Tool Execution (tools.py)
Each tool returns structured JSON:
```python
{
  "status": "success" | "error",
  "data": {...},
  "error": "optional error message"
}
```

### 4. Result Formatting (app.py:349-467)
- Parse tool outputs
- Generate visualizations (Plotly)
- Format as HTML with styling
- Extract reasoning steps for transparency

### 5. User Display
- Status updates via gr.Progress()
- Results in formatted HTML
- Interactive charts (CVE severity, quality gauges)
- Collapsible reasoning steps accordion

## Component Deep Dive

### 1. Gradio Interface (app.py - 1,107 lines)

**AgentState Class (lines 87-153)**
- Singleton pattern for agent management
- Lazy initialization on first request
- Error handling and recovery
- Thread-safe for Gradio's multi-request handling

**Security Functions (lines 159-219)**
- `validate_file_upload()`: CSV validation, size limits, format checks
- `sanitize_input()`: XSS prevention, length limits, pattern filtering

**Visualization Functions (lines 225-343)**
- `create_cve_severity_chart()`: Plotly bar chart with color coding
- `create_class_distribution_chart()`: Pie chart for dataset balance
- `create_quality_gauge()`: Gauge chart with score thresholds

**Handler Functions (lines 473-774)**
- `handle_cve_search()`: CVE monitoring workflow
- `handle_dataset_analysis()`: Dataset security analysis workflow
- `handle_code_generation()`: Security code generation workflow
- `handle_chat()`: General security Q&A workflow

Each handler follows this pattern:
```python
1. Input validation
2. Agent retrieval from singleton
3. Query construction with context
4. Agent execution with progress tracking
5. Result parsing from reasoning steps
6. Visualization generation
7. HTML formatting
8. Return (status, html, charts, reasoning)
```

### 2. Agent Engine (reasoning_chain.py - 802 lines)

**MLPatrolAgent Class (lines 157-627)**
- Orchestrates LangChain agent executor
- Manages chat history for multi-turn conversations
- Tracks reasoning steps with timestamps
- Calculates confidence scores

**Key Methods:**
- `__init__()`: Initialize with LLM and tools
- `analyze_query()`: Classify query type
- `run()`: Main execution loop
- `_validate_query()`: Input security checks
- `_extract_reasoning_steps()`: Parse intermediate steps
- `_calculate_confidence()`: Score based on tools used

### 3. Tool Definitions (tools.py - 940 lines)

**Five Main Tools:**

1. **cve_search** (lines 216-337)
   - Queries NVD API for vulnerabilities
   - Filters by library and date range
   - Returns CVEResult objects with CVSS scores

2. **web_search** (lines 339-395)
   - Searches for security information
   - Returns relevant articles and papers
   - (Placeholder - integrate with Google/Bing API)

3. **analyze_dataset** (lines 397-544)
   - Loads CSV/JSON datasets
   - Statistical outlier detection (Z-scores)
   - Class distribution analysis
   - Bias scoring
   - Quality assessment (0-10)

4. **generate_security_code** (lines 546-752)
   - Template-based code generation
   - CVE-specific validation scripts
   - Includes error handling and documentation

5. **huggingface_search** (lines 754-795)
   - Search HF Hub for datasets/models
   - (Placeholder - integrate with HF API)

### 4. Security Module (security/*)
Placeholder modules for future expansion:
- `cve_monitor.py`: Advanced CVE tracking
- `code_generator.py`: Enhanced code generation
- `threat_intel.py`: Threat intelligence feeds

### 5. Dataset Module (dataset/*)
Placeholder modules for future expansion:
- `poisoning_detector.py`: Advanced poisoning detection
- `bias_analyzer.py`: Bias analysis algorithms
- `statistical_tests.py`: Statistical test suite

### 6. MCP Integration (mcp/*)
Placeholder for Model Context Protocol tools:
- Web search integration
- HuggingFace dataset access
- Notion workspace integration

## Technology Stack

### Frontend
- **Gradio 6.0.0**: Web interface framework
  - Blocks API for custom layouts
  - Progress tracking for long operations
  - File upload with validation
  - Interactive charts (Plotly integration)

### Agent Framework
- **LangChain 0.3.0**: Agent orchestration
  - OpenAI function calling format
  - Tool binding and execution
  - Structured outputs
  - Error handling and retries

### LLM Backend
- **Claude Sonnet 4** (Primary)
  - 200K context window
  - Function calling support
  - Fast inference (~2s per request)

- **GPT-4** (Fallback)
  - 128K context window
  - Function calling support
  - Alternative if Claude unavailable

### Data Analysis
- **NumPy 1.26.0**: Numerical computing
- **Pandas 2.1.0**: Data manipulation
- **scikit-learn 1.3.0**: Machine learning utilities
- **SciPy 1.11.0**: Statistical functions

### Visualization
- **Plotly 5.18.0**: Interactive charts
  - Bar charts for CVE severity
  - Pie charts for class distribution
  - Gauge charts for quality scores

### Utilities
- **requests 2.31.0**: HTTP requests
- **httpx 0.25.0**: Async HTTP client
- **pydantic 2.5.0**: Data validation
- **python-dotenv 1.0.0**: Environment variables

## Data Flow Examples

### CVE Search Flow
```
1. User selects library "numpy" + 90 days
2. Gradio calls handle_cve_search()
3. Query: "Search for CVEs in numpy from last 90 days"
4. Agent classifies as CVE_MONITORING
5. Agent calls cve_search tool with params
6. Tool queries NVD API
7. Returns JSON with CVE list
8. Agent synthesizes answer
9. UI parses CVEs from reasoning steps
10. Creates severity chart
11. Formats HTML table
12. Displays to user
```

### Dataset Analysis Flow
```
1. User uploads dataset.csv (5MB)
2. Gradio validates: CSV format, size < 10MB
3. Calls handle_dataset_analysis()
4. Query: "Analyze dataset for security issues"
5. Agent classifies as DATASET_ANALYSIS
6. Agent calls analyze_dataset tool with file path
7. Tool loads CSV with pandas
8. Calculates Z-scores for outliers
9. Analyzes class distribution
10. Computes bias score
11. Generates quality score (0-10)
12. Returns JSON with findings
13. UI creates gauge + pie chart
14. Formats HTML report
15. Displays to user
```

## Performance Considerations

### Agent Initialization
- **Cold start**: 2-3 seconds (LLM initialization)
- **Subsequent requests**: <100ms (singleton reuse)

### Query Execution
- **CVE search**: 5-10 seconds (API call + agent reasoning)
- **Dataset analysis**: 10-30 seconds (depends on dataset size)
- **Code generation**: 5-15 seconds (template processing + LLM)
- **Chat**: 3-8 seconds (simpler queries)

### Optimizations
- Singleton pattern for agent reuse
- Lazy initialization on first request
- Timeout limits (180s max execution)
- Max iterations limit (10 steps)
- Result caching (future enhancement)

## Security Architecture

### Input Validation
```python
# Query validation (reasoning_chain.py:337-366)
- Length limits (5-10000 chars)
- Suspicious pattern detection
- XSS prevention

# File validation (app.py:159-196)
- Whitelist file types (.csv only)
- Size limits (10MB max)
- Format validation (pandas CSV read)
```

### Safe Code Generation
- Generated code is displayed, not executed
- Templates include security warnings
- No eval() or exec() calls
- User must manually review and run

### API Key Security
- Loaded from environment variables only
- Never logged or displayed
- Separate keys for different LLM providers
- Graceful failure if missing

## Extensibility

### Adding New Tools
```python
# In tools.py
def new_tool_impl(param: str) -> str:
    # Implementation
    return json.dumps(result)

# Add to tool list
tools.append(StructuredTool.from_function(
    func=new_tool_impl,
    name="new_tool",
    description="...",
    args_schema=NewToolInput
))
```

### Adding New Tabs
```python
# In app.py create_interface()
with gr.Tab("🆕 New Feature"):
    # Define UI components
    # Create handler function
    # Connect button to handler
```

### Adding New Visualizations
```python
# In app.py
def create_new_chart(data: Dict) -> go.Figure:
    fig = go.Figure(...)
    return fig
```

## Deployment Considerations

### Local Development
```bash
python app.py  # Runs on localhost:7860
```

### HuggingFace Spaces
```python
# app.py already configured
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False  # Set True for public link
)
```

### Docker (Future)
```dockerfile
FROM python:3.10
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

## Monitoring & Debugging

### Logging
All operations logged to:
- Console (stdout)
- File (mlpatrol.log)

Log levels:
- INFO: Normal operations
- WARNING: Validation failures, API timeouts
- ERROR: Tool failures, agent errors

### Error Recovery
- Agent initialization failure → Display error message, UI still loads
- Tool execution failure → Agent continues with partial results
- File upload failure → Clear error message to user
- API timeout → Graceful fallback message
