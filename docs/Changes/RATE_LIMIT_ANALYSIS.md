# Rate Limit Error Analysis & Solutions

## Error Details

```
Error code: 429 - rate_limit_error
Message: This request would exceed the rate limit for your organization (7b84d153-bb20-4287-8f42-e86fb4818ec4) of 30,000 input tokens per minute.
```

## Root Cause

**Current Rate Limit:** 30,000 input tokens per minute (TPM)

**What's Happening:**
1. Query classification calls LLM with full system prompt
2. Main agent execution calls LLM again with same system prompt
3. System prompt in `src/agent/prompts.py` is **very long** (~3,000+ tokens)
4. Multiple rapid calls = exceeding 30k TPM limit

**Token Breakdown:**
- System prompt: ~3,000 tokens
- Query classification: 1 call = ~3,000 tokens
- Agent execution: Multiple calls (5-10) = ~15,000-30,000 tokens
- **Total per request:** ~18,000-33,000 tokens
- **If user makes 2 requests in <1 minute:** Exceeds limit ❌

## Problem Location

### 1. Query Classification ([reasoning_chain.py:290-336](reasoning_chain.py#L290-L336))

```python
def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryType:
    """Classify the type of security query."""
    # Uses classification_prompt + full system context
    classification_prompt = get_classification_prompt()
    messages = classification_prompt.format_messages(query=query, context=context_str)

    response = self.llm.invoke(messages)  # ⚠️ LLM call with long prompt
```

**Issue:** Calls LLM just to classify query type (CVE vs Dataset vs Code vs General)

### 2. Main Agent Execution ([reasoning_chain.py:455-595](reasoning_chain.py#L455-L595))

```python
def run(self, query: str, ...):
    # Step 1: Classify query (LLM call #1)
    query_type = self.analyze_query(query, context)

    # Step 2: Execute agent with LangGraph (LLM calls #2-10)
    result = self.agent_executor.invoke({"messages": messages_with_query})
```

**Issue:** Two separate LLM invocations = 2x the tokens

---

## Solutions (Ranked by Effort & Impact)

### ✅ **Solution 1: Skip Query Classification** (EASIEST - 5 minutes)

**Impact:** Reduces tokens by ~3,000 per request
**Effort:** Minimal code change

**Why it works:** Query classification is nice-to-have, not essential. The main agent can figure out query type during execution.

**Implementation:**

```python
# In reasoning_chain.py, run() method
def run(self, query: str, context: Optional[Dict[str, Any]] = None, ...):
    start_time = time.time()

    try:
        # Step 1: Validate query
        is_valid, error_msg = self._validate_query(query)
        if not is_valid:
            return AgentResult(answer=f"Invalid query: {error_msg}", error=error_msg)

        # Step 2: SKIP classification - let agent figure it out
        # query_type = self.analyze_query(query, context)  # ❌ REMOVE THIS
        query_type = QueryType.GENERAL_SECURITY  # ✅ Default, agent will adapt

        # Step 3: Execute agent (rest stays the same)
        messages_with_query = messages + [HumanMessage(content=full_query)]
        result = self.agent_executor.invoke({"messages": messages_with_query})

        # ... rest of code ...
```

**Pros:**
- Immediate fix
- No architecture changes
- Agent still works fine (LangGraph adapts to query type)

**Cons:**
- Lose explicit query type tracking (minor)
- Confidence calculation slightly less accurate

---

### ✅ **Solution 2: Use Simpler Classification** (EASY - 15 minutes)

**Impact:** Reduces classification tokens by 90%
**Effort:** Replace LLM classification with regex

**Why it works:** Query classification is predictable - can use pattern matching instead of LLM.

**Implementation:**

```python
def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryType:
    """Classify query type using pattern matching (no LLM call)."""
    query_lower = query.lower()

    # CVE/vulnerability keywords
    if any(word in query_lower for word in ["cve", "vulnerability", "vulnerabilities", "security", "exploit", "patch"]):
        # Check if library mentioned
        if any(lib in query_lower for lib in ["numpy", "pytorch", "tensorflow", "sklearn", "pandas", "scipy"]):
            logger.info("Query classified as CVE_MONITORING (pattern match)")
            return QueryType.CVE_MONITORING

    # Dataset analysis keywords
    if any(word in query_lower for word in ["dataset", "data", "poisoning", "bias", "outlier", "csv"]):
        if context and ("file_path" in context or "dataset" in context):
            logger.info("Query classified as DATASET_ANALYSIS (pattern match)")
            return QueryType.DATASET_ANALYSIS

    # Code generation keywords
    if any(word in query_lower for word in ["generate", "code", "script", "validation", "check"]):
        if any(word in query_lower for word in ["cve", "security", "validate"]):
            logger.info("Query classified as CODE_GENERATION (pattern match)")
            return QueryType.CODE_GENERATION

    # Default
    logger.info("Query classified as GENERAL_SECURITY (default)")
    return QueryType.GENERAL_SECURITY
```

**Pros:**
- Fast (no LLM call)
- 0 tokens used
- Deterministic classification

**Cons:**
- Less flexible than LLM classification
- May misclassify edge cases (rare)

---

### ✅ **Solution 3: Reduce System Prompt Size** (MEDIUM - 30 minutes)

**Impact:** Reduces every LLM call by 1,500-2,000 tokens
**Effort:** Edit prompts.py

**Why it works:** Current system prompt is very detailed (~3,000 tokens). Can condense to ~1,000 tokens.

**Current Prompt ([prompts.py](prompts.py)):**
- Has detailed examples
- Lists every tool with descriptions
- Multiple reasoning patterns

**Optimized Version:**

```python
# In src/agent/prompts.py

AGENT_SYSTEM_PROMPT = """You are MLPatrol, an AI security agent for ML systems.

**Your Mission:** Help ML practitioners defend against security threats.

**Core Capabilities:**
1. CVE monitoring in ML libraries (numpy, pytorch, tensorflow, etc.)
2. Dataset analysis (poisoning, bias, quality)
3. Security code generation
4. Security recommendations

**Available Tools:**
- cve_search: Query NVD for vulnerabilities
- analyze_dataset: Check data security
- generate_security_code: Create validation scripts
- web_search: Research security topics
- huggingface_search: Find datasets/models

**Process:**
1. Understand the user's security concern
2. Select appropriate tools
3. Execute and gather information
4. Synthesize actionable recommendations

**Guidelines:**
- Be precise with CVE IDs, CVSS scores, versions
- Include clear remediation steps
- Show your reasoning transparently
- Generate safe, well-documented code
- Provide confidence levels

Focus on being helpful, accurate, and actionable.
"""
```

**Token Reduction:**
- Before: ~3,000 tokens
- After: ~600 tokens
- **Saved:** ~2,400 tokens per request

**Pros:**
- Significant token savings
- Faster responses
- Still maintains core functionality

**Cons:**
- Less detailed guidance for agent
- May need fine-tuning to maintain quality

---

### ✅ **Solution 4: Implement Rate Limit Handling** (MEDIUM - 1 hour)

**Impact:** Prevents errors, auto-retries
**Effort:** Add retry logic with exponential backoff

**Why it works:** Catches 429 errors and waits before retrying.

**Implementation:**

```python
# In reasoning_chain.py

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from anthropic import RateLimitError

class MLPatrolAgent:

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
    )
    def _call_llm_with_retry(self, messages):
        """Call LLM with automatic retry on rate limit."""
        return self.llm.invoke(messages)

    def analyze_query(self, query: str, context: Optional[Dict] = None) -> QueryType:
        """Classify query with retry logic."""
        try:
            classification_prompt = get_classification_prompt()
            messages = classification_prompt.format_messages(query=query, context=context_str)

            # Use retry wrapper
            response = self._call_llm_with_retry(messages)
            # ... rest of logic ...

        except RateLimitError as e:
            logger.warning(f"Rate limit hit, using default classification: {e}")
            return QueryType.GENERAL_SECURITY  # Fallback
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            raise QueryClassificationError(f"Failed to classify: {e}")
```

**Add to requirements.txt:**
```
tenacity>=8.2.0
```

**Pros:**
- Automatic retry with backoff
- Graceful degradation (fallback to default)
- User doesn't see error

**Cons:**
- Adds latency (4-60 seconds delay)
- Requires new dependency

---

### ✅ **Solution 5: Implement Token Budget Tracking** (MEDIUM - 1-2 hours)

**Impact:** Prevents exceeding limits proactively
**Effort:** Add token counting and throttling

**Why it works:** Track tokens used and pause if approaching limit.

**Implementation:**

```python
# In reasoning_chain.py

import tiktoken
from collections import deque
from datetime import datetime, timedelta

class TokenBudgetTracker:
    """Track token usage to avoid rate limits."""

    def __init__(self, limit_tpm: int = 30000):
        self.limit_tpm = limit_tpm
        self.usage_history = deque()  # (timestamp, tokens) tuples
        self.encoder = tiktoken.encoding_for_model("gpt-4")  # Approximation

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))

    def add_usage(self, tokens: int):
        """Record token usage."""
        self.usage_history.append((datetime.now(), tokens))
        self._cleanup_old_entries()

    def _cleanup_old_entries(self):
        """Remove entries older than 1 minute."""
        cutoff = datetime.now() - timedelta(minutes=1)
        while self.usage_history and self.usage_history[0][0] < cutoff:
            self.usage_history.popleft()

    def get_current_usage(self) -> int:
        """Get tokens used in last minute."""
        self._cleanup_old_entries()
        return sum(tokens for _, tokens in self.usage_history)

    def can_make_request(self, estimated_tokens: int) -> bool:
        """Check if request would exceed limit."""
        current = self.get_current_usage()
        return (current + estimated_tokens) < self.limit_tpm

    def wait_if_needed(self, estimated_tokens: int):
        """Wait if request would exceed limit."""
        while not self.can_make_request(estimated_tokens):
            logger.info(f"Approaching rate limit, waiting 5s...")
            time.sleep(5)


class MLPatrolAgent:
    def __init__(self, llm, tools, verbose=True, ...):
        # ... existing code ...
        self.token_tracker = TokenBudgetTracker(limit_tpm=30000)

    def run(self, query: str, ...):
        # Estimate tokens for this request
        system_prompt_tokens = self.token_tracker.count_tokens(str(get_agent_prompt()))
        query_tokens = self.token_tracker.count_tokens(query)
        estimated_total = system_prompt_tokens + query_tokens + 5000  # Buffer for tools

        # Wait if needed
        self.token_tracker.wait_if_needed(estimated_total)

        # Make request
        result = self.agent_executor.invoke(...)

        # Record usage
        self.token_tracker.add_usage(estimated_total)

        return result
```

**Add to requirements.txt:**
```
tiktoken>=0.5.0
```

**Pros:**
- Proactive prevention
- Precise tracking
- No errors reach user

**Cons:**
- Adds complexity
- May slow down rapid requests
- Token counting is approximate

---

### ✅ **Solution 6: Use Cheaper/Faster Model for Classification** (ADVANCED - 2 hours)

**Impact:** Reduces cost and tokens for classification
**Effort:** Separate models for classification vs execution

**Why it works:** Use fast/cheap Haiku for classification, Sonnet for main work.

**Implementation:**

```python
class MLPatrolAgent:
    def __init__(self, llm, tools, verbose=True, ...):
        self.llm = llm  # Main LLM (Sonnet)

        # Create separate LLM for classification (Haiku - 10x cheaper, 5x faster)
        from langchain_anthropic import ChatAnthropic
        self.classification_llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",  # Fast + cheap
            temperature=0,
            max_tokens=100,  # Only need short classification
        )

    def analyze_query(self, query: str, context: Optional[Dict] = None) -> QueryType:
        """Classify query using cheaper Haiku model."""
        try:
            classification_prompt = get_classification_prompt()
            messages = classification_prompt.format_messages(query=query, context=context_str)

            # Use Haiku for classification
            response = self.classification_llm.invoke(messages)
            # ... parse response ...

        except Exception as e:
            logger.warning(f"Classification failed, using default: {e}")
            return QueryType.GENERAL_SECURITY
```

**Pros:**
- Much faster classification (<1s vs 2-3s)
- 90% cost reduction for classification
- Main agent still uses Sonnet for quality

**Cons:**
- Need to manage two API clients
- Slightly less accurate classification (Haiku vs Sonnet)

---

### ✅ **Solution 7: Cache Classification Results** (ADVANCED - 1 hour)

**Impact:** Eliminate repeated classifications
**Effort:** Add LRU cache for similar queries

**Why it works:** Similar queries get same classification without LLM call.

**Implementation:**

```python
from functools import lru_cache
import hashlib

class MLPatrolAgent:

    @lru_cache(maxsize=100)
    def _classify_query_cached(self, query_hash: str, query: str, context_str: str) -> str:
        """Cached classification (keyed by hash)."""
        # Actual LLM call
        classification_prompt = get_classification_prompt()
        messages = classification_prompt.format_messages(query=query, context=context_str)
        response = self.llm.invoke(messages)
        return response.content.strip()

    def analyze_query(self, query: str, context: Optional[Dict] = None) -> QueryType:
        """Classify query with caching."""
        # Create cache key
        context_str = str(context) if context else ""
        cache_key = hashlib.md5(f"{query}{context_str}".encode()).hexdigest()

        # Try cache first
        try:
            cached_result = self._classify_query_cached(cache_key, query, context_str)
            logger.info("Using cached classification")
            # Parse cached result to QueryType...

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return QueryType.GENERAL_SECURITY
```

**Pros:**
- Instant results for repeated queries
- 0 tokens for cached queries
- Works well for testing/demos

**Cons:**
- Uses memory
- May return stale classifications if context changes

---

## Recommended Implementation Strategy

### **Phase 1: Immediate Fix (Choose ONE)**

**Option A: Simplest** (5 minutes)
```python
# In run() method, replace:
query_type = self.analyze_query(query, context)
# With:
query_type = QueryType.GENERAL_SECURITY  # Agent adapts automatically
```

**Option B: Better** (15 minutes)
- Implement Solution 2 (Pattern-based classification)
- 0 tokens, instant, accurate enough

### **Phase 2: Robust Handling** (30 minutes)
- Implement Solution 4 (Retry logic)
- Add graceful degradation
- Catch and handle 429 errors

### **Phase 3: Optimization** (1 hour)
- Implement Solution 3 (Reduce prompt size)
- Cut system prompt from 3000→1000 tokens
- Still maintain quality

### **Phase 4: Advanced** (Optional, 2-4 hours)
- Implement Solution 5 (Token tracking)
- Implement Solution 6 (Haiku for classification)
- Maximum efficiency

---

## Quick Fix Code (Copy-Paste Ready)

**File: `src/agent/reasoning_chain.py`**

Replace `analyze_query` method with this:

```python
def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryType:
    """Classify query type using pattern matching (no LLM call).

    This avoids rate limits by using regex instead of LLM calls.
    """
    try:
        logger.info(f"Classifying query: {query[:100]}...")
        query_lower = query.lower()

        # CVE/vulnerability patterns
        cve_keywords = ["cve", "vulnerability", "vulnerabilities", "security", "exploit", "patch", "nvd"]
        library_keywords = ["numpy", "pytorch", "tensorflow", "sklearn", "scikit-learn", "pandas", "scipy", "keras", "xgboost"]

        if any(kw in query_lower for kw in cve_keywords):
            if any(lib in query_lower for lib in library_keywords):
                logger.info("Query classified as CVE_MONITORING")
                return QueryType.CVE_MONITORING

        # Dataset analysis patterns
        dataset_keywords = ["dataset", "data", "poisoning", "bias", "outlier", "csv", "analyze"]
        if any(kw in query_lower for kw in dataset_keywords):
            if context and ("file_path" in context or "dataset" in context):
                logger.info("Query classified as DATASET_ANALYSIS")
                return QueryType.DATASET_ANALYSIS

        # Code generation patterns
        code_keywords = ["generate", "code", "script", "validation", "check", "create"]
        if any(kw in query_lower for kw in code_keywords):
            if any(kw in query_lower for kw in ["security", "validate", "cve"]):
                logger.info("Query classified as CODE_GENERATION")
                return QueryType.CODE_GENERATION

        # Default
        logger.info("Query classified as GENERAL_SECURITY (default)")
        return QueryType.GENERAL_SECURITY

    except Exception as e:
        logger.error(f"Query classification failed: {e}", exc_info=True)
        # Fallback to GENERAL_SECURITY on any error
        return QueryType.GENERAL_SECURITY
```

**Test the fix:**
```bash
python app.py
# Try multiple chat queries in quick succession
# Should no longer get 429 errors
```

---

## Monitoring & Prevention

### Add Usage Logging

```python
def run(self, query: str, ...):
    start_time = time.time()

    # Log token estimate
    logger.info(f"Starting query processing (estimated tokens: ~{len(query) * 1.3})")

    # ... existing code ...

    # Log completion
    logger.info(f"Query completed in {time.time() - start_time:.2f}s")
```

### Add to .env for Rate Limit Awareness

```bash
# Rate Limit Configuration
ANTHROPIC_RATE_LIMIT_TPM=30000  # Tokens per minute
ANTHROPIC_RATE_LIMIT_RPM=50     # Requests per minute
```

---

## Summary

| Solution | Effort | Impact | Recommended |
|----------|--------|--------|-------------|
| 1. Skip Classification | 5 min | Medium | ✅ Quick fix |
| 2. Pattern Classification | 15 min | High | ✅ Best ROI |
| 3. Reduce Prompt Size | 30 min | High | ✅ Long-term |
| 4. Retry Logic | 1 hr | Medium | ✅ Robustness |
| 5. Token Tracking | 2 hrs | High | ⚠️ If needed |
| 6. Dual Models | 2 hrs | Medium | ⚠️ Advanced |
| 7. Caching | 1 hr | Low | ⚠️ Nice-to-have |

**Recommended Implementation:**
1. **Now:** Solution 2 (Pattern classification) - 15 minutes
2. **This week:** Solution 3 (Reduce prompt) + Solution 4 (Retry logic)
3. **Later:** Solution 5 (Token tracking) if still hitting limits

This will solve your immediate 429 error and prevent future rate limit issues! 🚀
