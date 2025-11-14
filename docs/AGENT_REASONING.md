# Agent Reasoning System

## How MLPatrol Thinks

MLPatrol uses **LangGraph**, a modern graph-based agent framework, to orchestrate multi-step reasoning and tool execution. This document explains the agent's internal reasoning process.

## Architecture Overview

### LangGraph ReAct Pattern

MLPatrol implements the **ReAct (Reasoning + Acting)** pattern:

```
┌─────────────────────────────────────────┐
│         User Query Input                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Query Classification (LLM)           │
│  CVE | Dataset | Code | General         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      LangGraph ReAct Agent              │
│  ┌──────────────────────────────────┐   │
│  │  Reasoning Loop (Graph Nodes)    │   │
│  │  ┌──────────────────────┐        │   │
│  │  │ 1. Think             │        │   │
│  │  │    What do I know?   │        │   │
│  │  │    What do I need?   │        │   │
│  │  └───────┬──────────────┘        │   │
│  │          │                       │   │
│  │          ▼                       │   │
│  │  ┌──────────────────────┐        │   │
│  │  │ 2. Act               │        │   │
│  │  │    Call tool         │        │   │
│  │  │    Get results       │        │   │
│  │  └───────┬──────────────┘        │   │
│  │          │                       │   │
│  │          ▼                       │   │
│  │  ┌──────────────────────┐        │   │
│  │  │ 3. Observe           │        │   │
│  │  │    Process results   │        │   │
│  │  │    Update knowledge  │        │   │
│  │  └───────┬──────────────┘        │   │
│  │          │                       │   │
│  │          ▼                       │   │
│  │  ┌──────────────────────┐        │   │
│  │  │ 4. Decide            │        │   │
│  │  │    Done? → Answer    │        │   │
│  │  │    Not done? → Think │        │   │
│  │  └───────┬──────────────┘        │   │
│  │          │                       │   │
│  │          └───────────┐           │   │
│  └────────────────────┐ │           │   │
│                       │ │           │   │
│                   Loop back         │   │
└───────────────────────┼─────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │ Final Answer  │
                └───────────────┘
```

## Reasoning Chain Example

### Example 1: CVE Monitoring

**User Query:** "Check my environment for numpy vulnerabilities"

**Agent Reasoning (LangGraph State Transitions):**

```
Node 1 - Query Analysis:
  Thought: User wants CVE information for numpy library
  Decision: Classify as CVE_MONITORING
  Next: Select tool

Node 2 - Tool Selection:
  Thought: Need to search for numpy CVEs
  Action: call cve_search(library="numpy", days_back=30)
  Next: Execute tool

Node 3 - Tool Execution:
  Action: Query NVD API for numpy vulnerabilities
  Observation: Found 2 CVEs (CVE-2024-12345, CVE-2024-12346)
  Next: Process results

Node 4 - Analysis:
  Thought: CVE-2024-12345 is critical (CVSS 9.1), CVE-2024-12346 is medium (CVSS 5.3)
  Thought: Need more context on exploitability
  Action: call web_search("CVE-2024-12345 exploitation")
  Next: Execute tool

Node 5 - Enhanced Context:
  Observation: Found proof-of-concept exploits available
  Thought: This is urgent - publicly exploitable
  Next: Generate recommendations

Node 6 - Code Generation:
  Thought: User should check their environment
  Action: call generate_security_code(cve_id="CVE-2024-12345")
  Observation: Generated validation script
  Next: Synthesize answer

Node 7 - Synthesis:
  Thought: Have all information needed
  Decision: DONE
  Action: Format comprehensive answer with:
    - CVE details
    - Severity analysis
    - Validation code
    - Remediation steps
```

**Final Output:**
```markdown
Found 2 vulnerabilities in numpy:

1. **CVE-2024-12345** (CVSS 9.1 - Critical)
   - Buffer overflow in numpy.core
   - Affects: numpy 1.24.0-1.25.2
   - Status: Publicly exploitable (PoC available)
   - Risk: Remote code execution

2. **CVE-2024-12346** (CVSS 5.3 - Medium)
   - Integer overflow in array operations
   - Affects: numpy 1.24.0-1.24.4
   - Status: No public exploits

**Validation Script:**
[Generated Python code to check version]

**Recommendations:**
1. Upgrade to numpy >= 1.25.3 immediately
2. Run validation script in your environment
3. Review dependencies that use numpy
```

### Example 2: Dataset Analysis

**User Query:** "Analyze this dataset for security issues" [uploads dataset.csv]

**Agent Reasoning:**

```
Node 1 - Classification:
  Input: User query + file upload context
  Thought: This is dataset security analysis
  Decision: DATASET_ANALYSIS query type

Node 2 - Dataset Profiling:
  Action: call analyze_dataset(file_path="dataset.csv")
  Observation:
    - 10,000 rows, 15 features
    - 23 statistical outliers (Z-score > 3)
    - Class imbalance: 89% class_0, 11% class_1
    - 5 suspected label flips
  Next: Analyze patterns

Node 3 - Pattern Analysis:
  Thought: Outliers + label flips could indicate poisoning
  Thought: Need to understand poisoning attack patterns
  Action: call web_search("label flipping attack detection")
  Observation: Found academic papers on clean-label backdoor attacks

Node 4 - Correlation Analysis:
  Thought: Compare observed patterns with known attack signatures
  Analysis: Patterns match clean-label attack methodology
  Confidence: Medium-High (70%)

Node 5 - Synthesis:
  Thought: Have sufficient evidence to provide assessment
  Decision: DONE
  Action: Generate comprehensive security report
```

**Final Output:**
```markdown
**Dataset Security Analysis**

Quality Score: 6.5/10 ⚠️

**Findings:**

1. **Statistical Anomalies** (23 samples)
   - Z-scores > 3 in features [col_3, col_7, col_12]
   - Recommendation: Manual review required

2. **Class Imbalance** (89%/11% split)
   - Severity: Moderate
   - Impact: May bias model training
   - Recommendation: Consider SMOTE/oversampling

3. **Suspected Label Poisoning** (5 samples)
   - Pattern: Consistent with clean-label backdoor attacks
   - Confidence: 70%
   - Samples: [idx 234, 1823, 4521, 7234, 8901]
   - Recommendation: Investigate these samples carefully

**Visualizations:**
[Class distribution chart]
[Outlier detection plot]

**Next Steps:**
1. Review flagged samples manually
2. Consider robust training techniques
3. Implement label smoothing
4. Use adversarial training
```

## Reasoning Patterns by Query Type

### 1. CVE Monitoring Pattern

**Tools Used:** `cve_search`, `web_search`, `generate_security_code`

**Typical Flow:**
```
1. Search CVE database
2. If CVEs found → Gather exploit information
3. Generate validation code
4. Provide remediation guidance
```

**Decision Logic:**
- Always search last 30-90 days by default
- Prioritize CVSS score >= 7.0 (High/Critical)
- Include exploit availability in severity assessment
- Generate code only for actionable CVEs

### 2. Dataset Analysis Pattern

**Tools Used:** `analyze_dataset`, `web_search` (for context)

**Typical Flow:**
```
1. Load and profile dataset
2. Statistical analysis (outliers, distribution)
3. Bias detection
4. Pattern correlation with known attacks
5. Generate quality score + recommendations
```

**Decision Logic:**
- Outliers: Z-score > 3 flagged
- Class imbalance: Warn if ratio > 70/30
- Bias: Check for protected attribute correlations
- Quality score: Weighted combination of metrics

### 3. Code Generation Pattern

**Tools Used:** `generate_security_code`, optionally `cve_search` for context

**Typical Flow:**
```
1. Understand security requirement
2. Gather relevant CVE/library info if needed
3. Select code template
4. Customize with specific parameters
5. Add error handling + documentation
```

**Decision Logic:**
- Template selection based on requirement type
- Include comprehensive error handling
- Add security warnings and comments
- Never auto-execute generated code

### 4. General Security Pattern

**Tools Used:** `web_search`, optionally others based on context

**Typical Flow:**
```
1. Understand security question
2. Search for best practices/papers
3. Synthesize information
4. Provide actionable recommendations
```

**Decision Logic:**
- Prioritize recent information (2023+)
- Include academic sources when available
- Provide code examples when relevant
- Link to official documentation

## Tool Selection Logic

### How the Agent Chooses Tools

The LangGraph agent uses the LLM's function calling capability to dynamically select tools based on:

1. **Query Classification:** Determines primary tool set
2. **Context Availability:** Adapts based on user-provided context
3. **Intermediate Results:** Tools can trigger additional tool calls
4. **Confidence Assessment:** Low confidence → More information gathering

**Example - Adaptive Tool Selection:**
```python
# Initial query: "Is my ML pipeline secure?"

# Agent thinks:
"Vague query - need to narrow down"

# Agent calls:
web_search("ML pipeline security best practices")

# Observes results mention:
- Dependency vulnerabilities
- Data poisoning
- Model stealing

# Agent asks user (via chat):
"I can help with several aspects:
1. Check library vulnerabilities
2. Analyze training data security
3. Review model deployment security

Which would you like me to focus on?"
```

## Confidence Scoring

The agent calculates confidence based on:

```python
confidence = base_confidence(0.5)
  + tools_used_bonus(0.1 per tool, max 0.3)
  + appropriate_tools_bonus(0.2 if relevant tools used)
  + reasoning_depth_bonus(0.1 if >= 2 steps)
  - error_penalty(0.1 per failed observation)
```

**Example Confidence Scores:**

- **90%+**: Multiple tools, all successful, comprehensive data
- **70-89%**: Good tool coverage, minor data gaps
- **50-69%**: Basic analysis, limited tools, or some errors
- **<50%**: Significant gaps, errors, or uncertain data

## State Management

### LangGraph State Schema

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]        # Conversation history
    query_type: Optional[str]          # Classified query type
    reasoning_steps: List[Dict]        # Steps taken
    tools_used: List[str]              # Tools invoked
    context: Optional[Dict]            # User context (files, etc.)
```

### State Transitions

```
User Input → Query Classification → Tool Selection
     ↓              ↓                      ↓
  Messages    Set query_type        Append to tools_used
     ↓              ↓                      ↓
Add reasoning step → Execute tool → Observe results
     ↓              ↓                      ↓
Update messages ← Process ← Check if done
     ↓                          ↓
     └──────← Loop or End ──────┘
```

## Error Handling & Recovery

### Tool Failure Recovery

```python
# If cve_search fails due to API timeout:
Agent thinks: "Primary data source unavailable"
Agent acts: call web_search("numpy vulnerabilities NVD")
Agent observes: Gets partial information from security blogs
Agent decides: Continue with lower confidence score
```

### Validation Errors

```python
# If user provides invalid input:
Agent validates: Check query length, patterns
If invalid: Return clear error message
If valid: Proceed with reasoning
```

### Graceful Degradation

```python
# If multiple tools fail:
Agent assesses: Can I still provide value?
If yes: Provide partial answer with caveats
If no: Explain what went wrong and suggest alternatives
```

## Advanced Reasoning Features

### Multi-Turn Conversations

LangGraph maintains state across turns:

```
Turn 1: "Check numpy for CVEs"
  → Finds CVE-2024-12345

Turn 2: "Show me how to fix it"
  → Remembers CVE-2024-12345 from context
  → Generates specific fix for that CVE

Turn 3: "What about TensorFlow?"
  → Understands new topic
  → Searches TensorFlow CVEs
  → Can still reference numpy from history
```

### Context Awareness

```python
# Agent tracks:
- Previous queries in session
- Files uploaded
- Tools already used
- Results from previous steps

# Uses this to:
- Avoid redundant searches
- Build on previous findings
- Provide more coherent multi-turn responses
```

## Performance Optimization

### Graph Execution

- **Parallel tool calls:** When multiple independent tools needed
- **Early termination:** Stops when high-confidence answer reached
- **Caching:** Reuses results within same session
- **Timeouts:** Max 180s execution, max 10 iterations

### LLM Efficiency

- **Structured outputs:** Uses Pydantic schemas for validation
- **Prompt optimization:** Concise system prompts
- **Temperature:** Low (0.1) for factual consistency
- **Token limits:** 4096 max tokens for responses

## Debugging & Transparency

### Reasoning Steps Exposed

Every reasoning step is logged and shown to users:

```json
{
  "step_number": 1,
  "thought": "Need to search for numpy CVEs",
  "action": "cve_search",
  "action_input": {"library": "numpy", "days_back": 30},
  "observation": "Found 2 CVEs...",
  "timestamp": 1699564234.567,
  "duration_ms": 2340.12
}
```

### Visualization

Users can see the agent's reasoning process in the UI via collapsible accordion showing all steps.

## Future Enhancements

### Planned Features

1. **Checkpointing:** Save/resume agent state
2. **Human-in-the-loop:** Interactive tool approval
3. **Streaming:** Real-time reasoning display
4. **Custom graphs:** User-defined reasoning flows
5. **Memory persistence:** Cross-session context

---

## Learn More

- **LangGraph Documentation:** https://langchain-ai.github.io/langgraph/
- **ReAct Paper:** https://arxiv.org/abs/2210.03629
- **MLPatrol Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
