# Perplexity Integration Analysis for MLPatrol

**Date:** November 14, 2025
**Branch:** `claude/analyze-mlpatrol-updates-01MfDgYXm5BtxPZWaaKjKfic`
**Purpose:** Evaluate adding Perplexity API for web search functionality

---

## Executive Summary

MLPatrol currently has a **placeholder web search implementation** that needs to be replaced with a real search API. This analysis evaluates using Perplexity AI (via direct API or MCP server) to power the `web_search` tool, which is critical for finding security advisories, research papers, and threat intelligence.

**Recommendation:** Start with **Perplexity API** for simplicity and production readiness, with option to add MCP server later for advanced use cases.

---

## 1. Latest Repository Updates (Last 24 Hours)

### 🆕 Major Features Added

#### 1.1 Ollama Local LLM Support (16 minutes ago - commit `5a4bdae`)
- **What:** Full support for running 100% private local AI via Ollama
- **Impact:** Users can now run MLPatrol without any cloud API dependencies
- **Models Supported:** llama3.1:8b/70b, mistral-small:3.1, qwen2.5:14b
- **Configuration:** `USE_LOCAL_LLM=true` in `.env`
- **Files Changed:**
  - `reasoning_chain.py` - Added Ollama ChatOllama backend
  - `app.py` - Updated agent initialization logic
  - `requirements.txt` - Added `langchain-ollama>=0.2.0`
  - `docs/OLLAMA_QUICKSTART.md` - Comprehensive setup guide

**Significance for Perplexity:** This establishes the pattern for optional, configurable API integrations via `.env` - exactly what we need for Perplexity.

#### 1.2 LLM Status Display (3 hours ago - commit `ea1417a`)
- **What:** UI indicator showing which LLM is active
- **Display:** "🟢 Claude Sonnet 4 (Cloud)" or "🔵 llama3.1:8b (Local - Ollama)"
- **Implementation:** `get_llm_info()` method in `AgentState` class
- **Files Changed:** `app.py` lines 80-85, 145-159

**Significance for Perplexity:** Could extend this pattern to show "Web Search: Perplexity" status.

#### 1.3 Agent Answer Formatting Enhancement (3 hours ago)
- **What:** Better markdown rendering in agent responses
- **Impact:** Improved readability of code blocks, tables, and lists
- **Files Changed:** `app.py` - CSS styling updates

#### 1.4 Rate Limit Fix (11 hours ago - commit `a7b7fb9`)
- **What:** Moved from LLM-based query classification to regex patterns
- **Impact:** Faster classification, avoids unnecessary API calls
- **Files Changed:** `reasoning_chain.py` - Classification logic
- **Migration:** Updated to LangGraph 1.0+ (deprecated API fixes)

---

## 2. Current Web Search Implementation

### 2.1 Location and Status
**File:** `/home/user/MLPatrol/src/agent/tools.py` (lines 343-399)

**Status:** ⚠️ **PLACEHOLDER ONLY - NOT FUNCTIONAL**

```python
def web_search_impl(query: str, max_results: int = 5) -> str:
    """Search the web for security information."""

    # Mock search results structure (replace with actual API call)
    results = {
        "status": "success",
        "query": query_clean,
        "results": [
            {
                "title": f"Search results for: {query_clean}",
                "url": "https://example.com",
                "snippet": "In production, this would contain actual web search results...",
                "relevance_score": 0.95
            }
        ],
        "note": "This is a placeholder. Integrate with Google Custom Search API or similar service.",
        "recommendations": [...]
    }
    return json.dumps(results, indent=2)
```

### 2.2 Current Tool Definition
**File:** `/home/user/MLPatrol/src/agent/tools.py` (lines 809-817)

```python
web_search_tool = StructuredTool.from_function(
    func=web_search_impl,
    name="web_search",
    description="""Search the web for ML security information, research papers,
                   blog posts, vulnerability reports, and best practices...""",
    args_schema=WebSearchInput,
)
```

**Input Schema:**
```python
class WebSearchInput(BaseModel):
    query: str = Field(description="Search query for finding security information")
    max_results: int = Field(default=5, description="Maximum number of results")
```

### 2.3 When Web Search Is Called
The agent autonomously decides to call `web_search` when:
1. User asks about security best practices
2. Looking for recent vulnerability reports not in NVD
3. Finding research papers on ML security topics
4. Getting community recommendations
5. General security consultation questions

**Example Queries That Would Trigger Web Search:**
- "What are the latest best practices for securing PyTorch models?"
- "Find research papers on adversarial attacks against transformers"
- "What does the security community recommend for model serving?"

---

## 3. What Needs to Be Updated for Perplexity Integration

### 3.1 Environment Configuration (.env)

**Add to `.env.example`:**
```bash
# ============================================================================
# Web Search Configuration
# ============================================================================

# Enable/disable web search functionality
ENABLE_WEB_SEARCH=true  # Set to 'false' to disable web search entirely

# Web Search Provider (options: perplexity_api, perplexity_mcp, disabled)
WEB_SEARCH_PROVIDER=perplexity_api

# Perplexity API Settings (only used if WEB_SEARCH_PROVIDER=perplexity_api)
PERPLEXITY_API_KEY=your_perplexity_api_key_here
PERPLEXITY_MODEL=llama-3.1-sonar-small-128k-online  # Options: sonar-small, sonar-medium, sonar-large
PERPLEXITY_MAX_RESULTS=5  # Number of search results to return

# MCP Server Settings (only used if WEB_SEARCH_PROVIDER=perplexity_mcp)
PERPLEXITY_MCP_SERVER_URL=http://localhost:3000
PERPLEXITY_MCP_TIMEOUT=30  # Timeout in seconds
```

### 3.2 Code Changes Required

#### A. Update `tools.py` - Replace Placeholder Implementation

**File:** `/home/user/MLPatrol/src/agent/tools.py`

**Changes Needed:**
1. **Add imports** (top of file):
```python
import os
from typing import Literal
import httpx  # Already in requirements.txt
```

2. **Replace `web_search_impl()` function** (lines 343-399):
```python
def web_search_impl(query: str, max_results: int = 5) -> str:
    """Search the web for security information using Perplexity AI."""

    # Check if web search is enabled
    if os.getenv("ENABLE_WEB_SEARCH", "true").lower() != "true":
        return json.dumps({
            "status": "disabled",
            "message": "Web search is disabled. Enable it in .env with ENABLE_WEB_SEARCH=true"
        })

    provider = os.getenv("WEB_SEARCH_PROVIDER", "perplexity_api")

    if provider == "perplexity_api":
        return _perplexity_api_search(query, max_results)
    elif provider == "perplexity_mcp":
        return _perplexity_mcp_search(query, max_results)
    elif provider == "disabled":
        return json.dumps({"status": "disabled", "message": "Web search provider is disabled"})
    else:
        return json.dumps({"status": "error", "message": f"Unknown provider: {provider}"})
```

3. **Add Perplexity API implementation:**
```python
def _perplexity_api_search(query: str, max_results: int = 5) -> str:
    """Search using Perplexity API directly."""
    try:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            return json.dumps({
                "status": "error",
                "message": "PERPLEXITY_API_KEY not set in .env file"
            })

        model = os.getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online")

        # Sanitize query
        query_clean = re.sub(r"[^\w\s-]", "", query)[:200]

        # Call Perplexity API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a security research assistant. Provide factual, well-sourced information about ML security topics. Include URLs when possible."
                },
                {
                    "role": "user",
                    "content": f"Search for: {query_clean}. Provide top {max_results} most relevant results with sources."
                }
            ],
            "return_citations": True,
            "return_images": False,
            "search_recency_filter": "month"  # Focus on recent security info
        }

        response = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()

        data = response.json()

        # Extract answer and citations
        answer = data["choices"][0]["message"]["content"]
        citations = data.get("citations", [])

        # Format results
        results = {
            "status": "success",
            "query": query_clean,
            "provider": "perplexity",
            "model": model,
            "summary": answer,
            "citations": citations[:max_results],
            "results": [
                {
                    "title": f"Citation {i+1}",
                    "url": url,
                    "snippet": answer[max(0, answer.find(url)-100):answer.find(url)+100],
                    "relevance_score": 1.0 - (i * 0.1)
                }
                for i, url in enumerate(citations[:max_results])
            ]
        }

        logger.info(f"Perplexity API search completed: {len(citations)} citations")
        return json.dumps(results, indent=2)

    except httpx.TimeoutException:
        logger.error("Perplexity API timeout")
        return json.dumps({"status": "error", "message": "Search request timed out"})
    except httpx.HTTPStatusError as e:
        logger.error(f"Perplexity API HTTP error: {e.response.status_code}")
        return json.dumps({"status": "error", "message": f"API error: {e.response.status_code}"})
    except Exception as e:
        logger.error(f"Perplexity API search failed: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)})
```

4. **Add MCP server implementation:**
```python
def _perplexity_mcp_search(query: str, max_results: int = 5) -> str:
    """Search using Perplexity MCP server."""
    try:
        mcp_url = os.getenv("PERPLEXITY_MCP_SERVER_URL", "http://localhost:3000")
        timeout = int(os.getenv("PERPLEXITY_MCP_TIMEOUT", "30"))

        # Sanitize query
        query_clean = re.sub(r"[^\w\s-]", "", query)[:200]

        # Call MCP server
        response = httpx.post(
            f"{mcp_url}/search",
            json={
                "query": query_clean,
                "max_results": max_results
            },
            timeout=float(timeout)
        )
        response.raise_for_status()

        data = response.json()

        results = {
            "status": "success",
            "query": query_clean,
            "provider": "perplexity_mcp",
            "results": data.get("results", [])
        }

        logger.info(f"Perplexity MCP search completed: {len(results['results'])} results")
        return json.dumps(results, indent=2)

    except Exception as e:
        logger.error(f"Perplexity MCP search failed: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)})
```

#### B. Update `app.py` - Add Status Display (Optional)

**File:** `/home/user/MLPatrol/app.py`

Add web search status to the UI (similar to LLM status):

```python
# In get_llm_info() or create new get_web_search_info()
def get_web_search_status() -> str:
    """Get current web search configuration status."""
    enabled = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
    if not enabled:
        return "⚫ Web Search: Disabled"

    provider = os.getenv("WEB_SEARCH_PROVIDER", "perplexity_api")
    if provider == "perplexity_api":
        return "🟢 Web Search: Perplexity API"
    elif provider == "perplexity_mcp":
        return "🔵 Web Search: Perplexity MCP"
    else:
        return "⚫ Web Search: Not Configured"
```

#### C. Update `src/mcp/connectors.py` (Optional - Only if Using MCP)

**File:** `/home/user/MLPatrol/src/mcp/connectors.py`

Currently empty. Would need:
```python
"""MCP Server connectors for external integrations."""

import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PerplexityMCPConnector:
    """Connector for Perplexity MCP server."""

    def __init__(self, server_url: str, timeout: int = 30):
        self.server_url = server_url
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Perform search via MCP server."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.server_url}/search",
                json={"query": query, "max_results": max_results},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
```

#### D. Update Documentation

**Files to Update:**
1. `README.md` - Add Perplexity to features and configuration
2. `docs/ARCHITECTURE.md` - Update web search section
3. `.env.example` - Add Perplexity configuration

---

## 4. Perplexity API vs MCP Server: Detailed Comparison

### 4.1 Perplexity API (Direct Integration)

#### ✅ **Pros:**

1. **Simplicity**
   - Single HTTP POST request
   - No additional infrastructure needed
   - Straightforward error handling
   - Clear documentation from Perplexity

2. **Production Ready**
   - Perplexity's API is stable and well-maintained
   - Built-in rate limiting and error handling
   - 99.9% uptime SLA
   - No dependency on local MCP server availability

3. **Performance**
   - Direct connection = lower latency
   - No intermediate proxy/server overhead
   - Predictable response times (~2-5 seconds)

4. **Features**
   - **Citations:** Returns source URLs automatically
   - **Search Recency:** Can filter by day/week/month/year
   - **Model Selection:** Choose speed vs quality (sonar-small/medium/large)
   - **Domain Filtering:** Can restrict to specific domains
   - **Image Results:** Optional (not needed for MLPatrol)

5. **Cost Control**
   - Pay per API call (transparent pricing)
   - No infrastructure costs
   - Easy to budget and monitor usage

6. **Security**
   - Direct HTTPS connection to Perplexity
   - API key stored securely in `.env`
   - No additional attack surface

7. **Maintenance**
   - Zero infrastructure to maintain
   - Automatic updates from Perplexity
   - No server deployment/monitoring needed

#### ❌ **Cons:**

1. **API Key Dependency**
   - Requires Perplexity API key (paid service)
   - Key must be kept secure
   - If key expires/revoked, feature breaks

2. **Rate Limits**
   - Subject to Perplexity's rate limits
   - Tier-based limits (20-1000 requests/minute depending on plan)
   - May hit limits during heavy usage

3. **Cost**
   - Per-request pricing: ~$0.005-0.02 per search
   - Can add up with heavy usage
   - No free tier beyond trial

4. **Vendor Lock-in**
   - Tied to Perplexity's API structure
   - Switching providers requires code changes
   - Dependent on Perplexity's service availability

5. **Limited Customization**
   - Can't modify search algorithm
   - Limited control over result ranking
   - Can't add custom preprocessing/postprocessing on server side

6. **Internet Required**
   - Must have internet connection for each search
   - Can't work offline
   - Latency depends on network

#### 💰 **Perplexity API Pricing (Nov 2024):**
- **Sonar Small (online):** $0.005 per 1K tokens (~$0.01 per search)
- **Sonar Medium (online):** $0.01 per 1K tokens (~$0.02 per search)
- **Sonar Large (online):** $0.02 per 1K tokens (~$0.04 per search)

**Estimated Monthly Cost for MLPatrol:**
- Light usage (50 searches/day): ~$15-30/month
- Medium usage (200 searches/day): ~$60-120/month
- Heavy usage (1000 searches/day): ~$300-600/month

---

### 4.2 MCP Server (Model Context Protocol)

#### ✅ **Pros:**

1. **Flexibility**
   - Can aggregate multiple search sources (Perplexity + Google + Bing)
   - Custom preprocessing of queries
   - Custom postprocessing of results
   - Add caching layer for repeated queries

2. **Extensibility**
   - Easy to add new search providers
   - Can integrate with other MCP tools (HuggingFace, Notion, etc.)
   - Centralized tool orchestration
   - Shared across multiple agents/applications

3. **Caching/Optimization**
   - Cache search results locally
   - Deduplicate similar queries
   - Batch multiple searches
   - Reduce API costs through smart caching

4. **Monitoring**
   - Centralized logging of all searches
   - Usage analytics across tools
   - Performance metrics
   - Error tracking

5. **Security Features**
   - Can add custom authentication layer
   - Rate limiting at server level
   - Query sanitization/filtering
   - Audit logging

6. **Cost Optimization**
   - Implement intelligent caching to reduce API calls
   - Use cheaper providers for certain query types
   - Fallback to free search APIs when possible

7. **Future-Proofing**
   - MCP is emerging standard for tool integration
   - Easier to migrate between providers
   - Growing ecosystem of MCP tools
   - Better for long-term scalability

#### ❌ **Cons:**

1. **Complexity**
   - Requires separate MCP server setup
   - More moving parts = more failure points
   - Need to manage server lifecycle
   - More code to maintain

2. **Infrastructure**
   - Need to run MCP server (Node.js/Python)
   - Requires port management (e.g., 3000)
   - Need to handle server startup/shutdown
   - Resource overhead (memory, CPU)

3. **Development Time**
   - Takes longer to implement
   - Need to build/configure MCP server
   - More testing required
   - Learning curve for MCP protocol

4. **Deployment Challenges**
   - HuggingFace Spaces may not support persistent MCP servers
   - Docker container needed for production
   - Port exposure complexity
   - Process management (pm2, systemd, etc.)

5. **Debugging**
   - Network issues between agent and MCP server
   - Additional layer to debug
   - Need to monitor server health
   - Log management across two systems

6. **Single Point of Failure**
   - If MCP server crashes, all tools fail
   - Need health checks and restart logic
   - Server downtime affects entire agent

7. **Latency**
   - Additional network hop (agent → MCP → Perplexity)
   - Slightly higher response times
   - Potential timeout issues

8. **Not Production Ready for MLPatrol Yet**
   - `src/mcp/connectors.py` is currently empty (1 line)
   - Would need significant development
   - No existing MCP infrastructure in codebase
   - Testing/hardening required

---

### 4.3 Side-by-Side Comparison

| Feature | Perplexity API | MCP Server |
|---------|---------------|------------|
| **Setup Time** | 30 minutes | 4-8 hours |
| **Code Complexity** | Low (~100 lines) | High (~500+ lines) |
| **Infrastructure** | None | MCP server required |
| **Latency** | 2-5 seconds | 3-7 seconds |
| **Reliability** | Very High (99.9%) | Medium (depends on server) |
| **Cost (medium usage)** | ~$60-120/month | ~$60-120/month + server costs |
| **Maintenance** | Minimal | Moderate-High |
| **Customization** | Limited | Extensive |
| **Caching** | No | Yes (can implement) |
| **Multi-Provider** | No | Yes |
| **Production Ready** | ✅ Yes | ⚠️ Needs development |
| **HuggingFace Spaces** | ✅ Works great | ⚠️ May have issues |
| **Offline Support** | ❌ No | ❌ No (still calls external APIs) |
| **Best For** | Quick implementation, reliability | Long-term scalability, flexibility |

---

## 5. Recommendations

### 5.1 Recommended Approach: **Start with Perplexity API**

**Rationale:**
1. **Immediate Value:** Get web search working in 1-2 hours vs 1-2 days
2. **Hackathon Timeline:** API integration fits ~2 week timeframe better
3. **Reliability:** Production-ready, no infrastructure concerns for demo/launch
4. **MLPatrol's Current State:** MCP infrastructure is not built yet (`connectors.py` is empty)
5. **Deployment Target:** HuggingFace Spaces works better with direct API calls

### 5.2 Implementation Phases

#### **Phase 1: Perplexity API Integration (Recommended for Hackathon)**

**Timeline:** 2-4 hours

**Tasks:**
1. ✅ Add Perplexity config to `.env.example`
2. ✅ Implement `_perplexity_api_search()` in `tools.py`
3. ✅ Update `web_search_impl()` to call Perplexity
4. ✅ Add enable/disable toggle via `ENABLE_WEB_SEARCH`
5. ✅ Test with real queries
6. ✅ Update documentation

**Deliverable:** Fully functional web search using Perplexity API

---

#### **Phase 2: MCP Server (Post-Hackathon Enhancement)**

**Timeline:** 1-2 days

**Tasks:**
1. ✅ Build MCP server (Node.js or Python)
2. ✅ Implement MCP client in `connectors.py`
3. ✅ Add caching layer
4. ✅ Add multiple provider support (Perplexity + Brave + Google)
5. ✅ Deployment configuration (Docker, docker-compose)
6. ✅ Monitoring and logging

**Deliverable:** Scalable, multi-provider search with caching

---

### 5.3 Hybrid Approach (Best Long-Term Solution)

**Configuration:**
```bash
# .env
WEB_SEARCH_PROVIDER=perplexity_api  # Options: perplexity_api, perplexity_mcp, disabled
```

- **Default:** Perplexity API (simple, reliable)
- **Advanced Users:** Can switch to MCP server if they want customization
- **Fallback:** If MCP server unavailable, fall back to direct API

This gives users choice while keeping default simple.

---

## 6. Specific Recommendations for MLPatrol

### 6.1 Why Perplexity API is Better for MLPatrol Right Now

1. **Security Focus:** MLPatrol deals with security research where uptime/reliability matters
2. **Demo-Friendly:** For hackathon demo, direct API = fewer variables, less debugging
3. **HuggingFace Spaces:** Easier to deploy (no MCP server process management)
4. **Development Velocity:** Can implement and test in one session
5. **User Experience:** "Just works" with API key, no server setup

### 6.2 Why MCP Could Be Better Later

1. **Cost Optimization:** If MLPatrol gains users, caching could save significant money
2. **Multi-Tool Integration:** MCP aligns with project vision (README mentions MCP tools)
3. **Community Contribution:** MCP server could be shared with other security tools
4. **Extensibility:** Easier to add threat intelligence feeds, CVE databases, etc.

### 6.3 Recommended Configuration

```bash
# .env.example
# ============================================================================
# Web Search Configuration
# ============================================================================

# Enable web search (set to 'false' to disable entirely)
ENABLE_WEB_SEARCH=true

# Web Search Provider Options:
#   - perplexity_api: Direct API calls (recommended, easiest setup)
#   - perplexity_mcp: Via MCP server (advanced, requires server setup)
#   - disabled: Disable web search
WEB_SEARCH_PROVIDER=perplexity_api

# Perplexity API Settings (only used if WEB_SEARCH_PROVIDER=perplexity_api)
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Model options (sonar-small = fast/cheap, sonar-large = best quality)
#   - llama-3.1-sonar-small-128k-online (recommended: fast, good quality)
#   - llama-3.1-sonar-medium-128k-online (balanced)
#   - llama-3.1-sonar-large-128k-online (highest quality, slower)
PERPLEXITY_MODEL=llama-3.1-sonar-small-128k-online

# Search settings
PERPLEXITY_SEARCH_RECENCY=month  # Options: day, week, month, year
PERPLEXITY_MAX_RESULTS=5
PERPLEXITY_TIMEOUT=30  # seconds

# MCP Server Settings (only used if WEB_SEARCH_PROVIDER=perplexity_mcp)
PERPLEXITY_MCP_SERVER_URL=http://localhost:3000
PERPLEXITY_MCP_TIMEOUT=30
```

---

## 7. Implementation Checklist

### For Perplexity API Integration:

- [ ] Create Perplexity API account and get API key
- [ ] Update `.env.example` with Perplexity configuration
- [ ] Implement `_perplexity_api_search()` in `tools.py`
- [ ] Update `web_search_impl()` to check `ENABLE_WEB_SEARCH` flag
- [ ] Add provider routing logic (API vs MCP vs disabled)
- [ ] Test with real security queries (e.g., "PyTorch security best practices")
- [ ] Update `README.md` with Perplexity setup instructions
- [ ] Update `docs/ARCHITECTURE.md` web search section
- [ ] Add web search status indicator to UI (optional)
- [ ] Test error handling (missing API key, timeout, rate limits)
- [ ] Commit and push changes

### For MCP Server (Future):

- [ ] Design MCP server architecture
- [ ] Choose implementation language (Node.js/Python)
- [ ] Build MCP server with Perplexity integration
- [ ] Add caching layer (Redis/in-memory)
- [ ] Implement MCP client in `src/mcp/connectors.py`
- [ ] Add health check endpoint
- [ ] Create Docker container for MCP server
- [ ] Update docker-compose with MCP service
- [ ] Add MCP monitoring/logging
- [ ] Test MCP failover to direct API
- [ ] Document MCP server setup

---

## 8. Code Example: Complete Implementation

Here's the complete implementation for `_perplexity_api_search()`:

```python
def _perplexity_api_search(query: str, max_results: int = 5) -> str:
    """
    Search using Perplexity AI API for ML security information.

    Perplexity provides real-time web search with citations, perfect for finding:
    - Latest security advisories and CVE reports
    - Research papers on ML security
    - Community best practices and recommendations
    - Blog posts from security researchers

    Args:
        query: Security-related search query
        max_results: Maximum number of results/citations to return

    Returns:
        JSON string with search results and citations

    Environment Variables Required:
        - PERPLEXITY_API_KEY: Your Perplexity API key
        - PERPLEXITY_MODEL: Model to use (default: llama-3.1-sonar-small-128k-online)
        - PERPLEXITY_SEARCH_RECENCY: Time filter (default: month)
    """
    try:
        # Get configuration from environment
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            logger.error("PERPLEXITY_API_KEY not set")
            return json.dumps({
                "status": "error",
                "message": "Perplexity API key not configured. Add PERPLEXITY_API_KEY to .env file."
            })

        model = os.getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online")
        recency = os.getenv("PERPLEXITY_SEARCH_RECENCY", "month")
        timeout = int(os.getenv("PERPLEXITY_TIMEOUT", "30"))

        # Sanitize query (basic security)
        query_clean = re.sub(r"[^\w\s\-\.\,\?\!]", "", query)[:300]

        logger.info(f"Perplexity search: {query_clean} (model: {model}, recency: {recency})")

        # Prepare API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cybersecurity research assistant specializing in ML security. "
                        "Provide factual, well-sourced information about machine learning security topics. "
                        "Focus on: CVEs, vulnerabilities, best practices, research papers, and threat intelligence. "
                        "Always cite your sources with URLs."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Research the following ML security topic and provide the top {max_results} "
                        f"most relevant and recent findings with sources:\n\n{query_clean}"
                    )
                }
            ],
            "return_citations": True,
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": recency,
            "temperature": 0.2,  # Lower temperature for factual responses
            "top_p": 0.9
        }

        # Make API request
        start_time = time.time()
        response = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=float(timeout)
        )
        response.raise_for_status()
        duration_ms = (time.time() - start_time) * 1000

        # Parse response
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        citations = data.get("citations", [])

        # Format results for MLPatrol agent
        results = {
            "status": "success",
            "query": query_clean,
            "provider": "perplexity",
            "model": model,
            "duration_ms": round(duration_ms, 2),
            "summary": answer,
            "citation_count": len(citations),
            "citations": citations[:max_results],
            "results": [
                {
                    "title": f"Source {i+1}",
                    "url": url,
                    "snippet": _extract_snippet(answer, url),
                    "relevance_score": round(1.0 - (i * 0.15), 2),  # Decreasing relevance
                    "source_type": _classify_source(url)
                }
                for i, url in enumerate(citations[:max_results])
            ],
            "metadata": {
                "search_recency": recency,
                "total_citations": len(citations),
                "response_truncated": len(citations) > max_results
            }
        }

        logger.info(
            f"Perplexity search completed in {duration_ms:.0f}ms: "
            f"{len(citations)} citations, {len(answer)} chars"
        )
        return json.dumps(results, indent=2)

    except httpx.TimeoutException:
        logger.error(f"Perplexity API timeout after {timeout}s")
        return json.dumps({
            "status": "error",
            "message": f"Search request timed out after {timeout} seconds. Try again or increase PERPLEXITY_TIMEOUT."
        })

    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        logger.error(f"Perplexity API HTTP error: {status_code}")

        error_messages = {
            401: "Invalid API key. Check PERPLEXITY_API_KEY in .env file.",
            403: "API key does not have permission. Check your Perplexity account.",
            429: "Rate limit exceeded. Wait a moment and try again.",
            500: "Perplexity API server error. Try again later.",
            503: "Perplexity API temporarily unavailable. Try again later."
        }

        return json.dumps({
            "status": "error",
            "message": error_messages.get(status_code, f"API error: {status_code}"),
            "error_code": status_code
        })

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Perplexity API response: {e}")
        return json.dumps({
            "status": "error",
            "message": "Invalid response from Perplexity API. Try again."
        })

    except Exception as e:
        logger.error(f"Perplexity search failed: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": f"Search failed: {str(e)}"
        })


def _extract_snippet(text: str, url: str, context_chars: int = 150) -> str:
    """Extract snippet around URL mention in text."""
    try:
        url_pos = text.find(url)
        if url_pos == -1:
            # URL not mentioned, return first part of text
            return text[:context_chars] + "..." if len(text) > context_chars else text

        start = max(0, url_pos - context_chars)
        end = min(len(text), url_pos + len(url) + context_chars)
        snippet = text[start:end]

        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet
    except:
        return text[:context_chars] + "..." if len(text) > context_chars else text


def _classify_source(url: str) -> str:
    """Classify source type based on URL."""
    url_lower = url.lower()

    if any(x in url_lower for x in ['github.com', 'gitlab.com']):
        return "code_repository"
    elif any(x in url_lower for x in ['arxiv.org', 'ieeexplore', 'acm.org', 'springer']):
        return "research_paper"
    elif any(x in url_lower for x in ['nvd.nist.gov', 'cve.org', 'cve.mitre.org']):
        return "vulnerability_database"
    elif any(x in url_lower for x in ['.gov', 'nist.gov', 'cert.org']):
        return "official_advisory"
    elif any(x in url_lower for x in ['stackoverflow.com', 'reddit.com']):
        return "community_discussion"
    elif any(x in url_lower for x in ['medium.com', 'blog', 'post']):
        return "blog_post"
    elif any(x in url_lower for x in ['pytorch.org', 'tensorflow.org', 'scikit-learn.org']):
        return "official_documentation"
    else:
        return "web_page"
```

---

## 9. Testing Plan

### Test Cases:

1. **Basic Search:**
   - Query: "PyTorch security vulnerabilities 2024"
   - Expected: Recent CVEs and security advisories

2. **Research Papers:**
   - Query: "adversarial attacks on transformers research"
   - Expected: arXiv papers and conference proceedings

3. **Best Practices:**
   - Query: "ML model serving security best practices"
   - Expected: Blog posts, documentation, guides

4. **Specific CVE:**
   - Query: "CVE-2024-12345 exploit details"
   - Expected: CVE database entries, writeups

5. **Error Handling:**
   - Test with missing API key
   - Test with invalid API key
   - Test with network timeout
   - Test with rate limit exceeded

---

## 10. Migration Path

If you later want to add MCP server:

```python
# In tools.py
def web_search_impl(query: str, max_results: int = 5) -> str:
    provider = os.getenv("WEB_SEARCH_PROVIDER", "perplexity_api")

    if provider == "perplexity_api":
        return _perplexity_api_search(query, max_results)
    elif provider == "perplexity_mcp":
        return _perplexity_mcp_search(query, max_results)
    else:
        return json.dumps({"status": "disabled"})
```

Users can switch by changing `.env`:
```bash
WEB_SEARCH_PROVIDER=perplexity_mcp  # Switch to MCP
WEB_SEARCH_PROVIDER=perplexity_api  # Switch back to API
WEB_SEARCH_PROVIDER=disabled         # Disable web search
```

---

## 11. Conclusion

**For the MLPatrol hackathon submission and initial launch:**
- ✅ **Use Perplexity API** - Fast implementation, reliable, production-ready
- ✅ **Make it optional** - `ENABLE_WEB_SEARCH` flag in `.env`
- ✅ **Keep it simple** - Direct API calls, clear error messages
- ✅ **Document well** - Clear setup instructions in README

**For future enhancement:**
- 🔮 **Add MCP server** - When you need multi-provider support or caching
- 🔮 **Hybrid approach** - Let users choose between API and MCP
- 🔮 **Cost optimization** - Implement smart caching when usage grows

The Perplexity API integration can be completed in 2-4 hours and will provide immediate value to MLPatrol users searching for ML security information.

---

**Next Steps:**
1. Get Perplexity API key from https://perplexity.ai
2. Implement `_perplexity_api_search()` in `tools.py`
3. Update `.env.example` with configuration
4. Test with real queries
5. Update documentation
6. Demo in hackathon submission!

---

*Document created: November 14, 2025*
*Author: Claude (AI Assistant)*
*For: MLPatrol - MCP 1st Birthday Hackathon*
