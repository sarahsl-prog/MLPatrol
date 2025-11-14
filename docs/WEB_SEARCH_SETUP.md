# Web Search Setup Guide for MLPatrol

**Last Updated:** November 14, 2025

This guide explains how to configure web search capabilities in MLPatrol using Tavily AI and/or Brave Search APIs.

---

## Overview

MLPatrol supports two web search providers:

1. **Tavily AI** - AI-native search optimized for LLM agents (recommended for general security Q&A)
2. **Brave Search** - Privacy-focused independent search (recommended for CVE monitoring)

You can use **one or both** providers. When both are enabled, MLPatrol intelligently routes queries:
- **CVE monitoring queries** → Brave Search (breaking news, real-time updates)
- **General security queries** → Tavily AI (AI-optimized summaries)

---

## Quick Start

### Option 1: Use Both Providers (Recommended)

**Cost:** ~$45-75/month for medium usage (200 searches/day)

```bash
# 1. Get API keys
#    - Tavily: https://tavily.com
#    - Brave: https://brave.com/search/api/

# 2. Edit .env file
ENABLE_WEB_SEARCH=true
USE_TAVILY_SEARCH=true
USE_BRAVE_SEARCH=true
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx
BRAVE_API_KEY=BSAxxxxxxxxxxxxx

# 3. Install dependencies (if not already installed)
pip install tavily-python

# 4. Restart MLPatrol
python app.py
```

### Option 2: Tavily Only (Best for AI Integration)

**Cost:** ~$45-60/month for medium usage

```bash
# Edit .env
ENABLE_WEB_SEARCH=true
USE_TAVILY_SEARCH=true
USE_BRAVE_SEARCH=false
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx
```

### Option 3: Brave Only (Most Cost-Effective)

**Cost:** ~$12-18/month for medium usage (75% cheaper!)

```bash
# Edit .env
ENABLE_WEB_SEARCH=true
USE_TAVILY_SEARCH=false
USE_BRAVE_SEARCH=true
BRAVE_API_KEY=BSAxxxxxxxxxxxxx
```

---

## Detailed Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# ============================================================================
# Web Search Configuration
# ============================================================================

# Enable/disable web search functionality
ENABLE_WEB_SEARCH=true  # Set to 'false' to disable entirely

# Web Search Providers (can enable one or both)
USE_TAVILY_SEARCH=true   # AI-native search
USE_BRAVE_SEARCH=true    # Privacy-focused search

# Tavily AI Settings
TAVILY_API_KEY=your_tavily_api_key_here
TAVILY_SEARCH_DEPTH=advanced  # Options: basic, advanced
TAVILY_MAX_RESULTS=5

# Brave Search API Settings
BRAVE_API_KEY=your_brave_api_key_here
BRAVE_SEARCH_FRESHNESS=pw  # Options: pd (day), pw (week), pm (month), py (year)
BRAVE_MAX_RESULTS=5

# Search Routing Strategy (when both providers enabled)
WEB_SEARCH_ROUTE_CVE_TO=brave      # Route CVE queries to: brave or tavily
WEB_SEARCH_ROUTE_GENERAL_TO=tavily # Route general queries to: brave or tavily
```

---

## Getting API Keys

### Tavily AI

1. Visit https://tavily.com
2. Sign up for an account
3. Navigate to API Keys section
4. Copy your API key (starts with `tvly-`)

**Free Tier:**
- 1,000 API credits/month
- Great for development and testing

**Paid Plans:**
- Project: $30/month (4,000 credits)
- Add-on: $100 one-time (8,000 credits, never expire)
- Enterprise: Custom pricing

### Brave Search API

1. Visit https://brave.com/search/api/
2. Sign up for an account
3. Subscribe to a plan (Free tier available!)
4. Copy your API key (starts with `BSA`)

**Free Tier:**
- 2,000 queries/month
- Perfect for development and low-volume use

**Paid Plans:**
- Base: $3 per 1,000 queries (CPM pricing)
- Pro: Custom volume pricing

---

## Smart Routing

When both providers are enabled, MLPatrol automatically routes queries based on type:

### CVE Monitoring Queries → Brave Search

**Triggers:**
- Contains CVE ID (e.g., "CVE-2024-1234")
- Contains "vulnerability" + year
- Phrases like "latest CVE", "recent vulnerability", "security advisory"

**Why Brave?**
- Breaking news coverage
- Fresh index (100M updates/day)
- Excellent for real-time threat intel

**Examples:**
- "Find CVE-2024-12345 details"
- "Latest vulnerabilities in PyTorch 2024"
- "Recent security advisory for TensorFlow"

### General Security Queries → Tavily AI

**Triggers:**
- All other security-related queries
- Best practices questions
- Explanatory queries

**Why Tavily?**
- AI-optimized summaries
- Fact-checked information
- Clean, structured output for LLM consumption

**Examples:**
- "What are PyTorch security best practices?"
- "Explain adversarial attacks on transformers"
- "How to secure ML model serving?"

### Override Routing

You can change the routing in `.env`:

```bash
# Route everything to Tavily
WEB_SEARCH_ROUTE_CVE_TO=tavily
WEB_SEARCH_ROUTE_GENERAL_TO=tavily

# Route everything to Brave
WEB_SEARCH_ROUTE_CVE_TO=brave
WEB_SEARCH_ROUTE_GENERAL_TO=brave

# Default (recommended)
WEB_SEARCH_ROUTE_CVE_TO=brave
WEB_SEARCH_ROUTE_GENERAL_TO=tavily
```

---

## Advanced Configuration

### Tavily Search Depth

**Basic:** Faster, top results only
```bash
TAVILY_SEARCH_DEPTH=basic
```

**Advanced:** Deeper crawl, multiple sources (recommended)
```bash
TAVILY_SEARCH_DEPTH=advanced
```

### Brave Search Freshness

Control how recent results should be:

```bash
BRAVE_SEARCH_FRESHNESS=pd   # Past day (breaking news)
BRAVE_SEARCH_FRESHNESS=pw   # Past week (recommended)
BRAVE_SEARCH_FRESHNESS=pm   # Past month
BRAVE_SEARCH_FRESHNESS=py   # Past year
```

### Result Limits

Control how many results each provider returns:

```bash
TAVILY_MAX_RESULTS=5   # Default: 5
BRAVE_MAX_RESULTS=5    # Default: 5
```

More results = better context but higher cost and token usage.

---

## Status Indicators

MLPatrol displays web search status on the main screen:

### 🔍 Active

**Green box:**
```
🔍 Web Search: Tavily AI + Brave Search • Privacy-focused
```

Both providers configured and ready.

### ⚠️ Not Configured

**Yellow box:**
```
⚠️ Web Search: Not configured - Add API keys to .env
```

Providers enabled but API keys missing or invalid.

### ⚫ Disabled

**Gray box:**
```
⚫ Web Search: Disabled
```

`ENABLE_WEB_SEARCH=false` in `.env`.

---

## Troubleshooting

### "Web search is disabled"

**Cause:** `ENABLE_WEB_SEARCH=false` or not set

**Fix:**
```bash
ENABLE_WEB_SEARCH=true
```

### "No web search providers enabled"

**Cause:** Both `USE_TAVILY_SEARCH` and `USE_BRAVE_SEARCH` are false

**Fix:** Enable at least one:
```bash
USE_TAVILY_SEARCH=true
# OR
USE_BRAVE_SEARCH=true
```

### "Tavily API key not configured"

**Cause:** API key missing or still placeholder value

**Fix:**
```bash
TAVILY_API_KEY=tvly-your-actual-key-here
```

### "Brave API error: 401"

**Cause:** Invalid API key

**Fix:**
1. Check API key in Brave dashboard
2. Ensure no extra spaces in `.env`
3. Restart app after updating `.env`

### "Rate limit exceeded"

**Cause:** Hit provider's rate limit

**Tavily:**
- Free tier: 1,000 credits/month
- Wait for monthly reset or upgrade plan

**Brave:**
- Free tier: 2,000 queries/month
- Paid tier: Up to 20 queries/second

**Fix:** Upgrade to paid plan or wait for rate limit reset

### Search returns empty results

**Cause:** Provider may be down or query too specific

**Fix:**
1. Check provider status pages
2. Try rephrasing query
3. Enable both providers for redundancy

---

## Cost Estimation

### Light Usage (50 searches/day = 1,500/month)

| Configuration | Monthly Cost |
|---------------|--------------|
| Tavily only | $30 (Project plan) |
| Brave only | FREE (under 2,000) |
| Both providers | ~$30 (mostly Tavily) |

### Medium Usage (200 searches/day = 6,000/month)

| Configuration | Monthly Cost |
|---------------|--------------|
| Tavily only | $45-60 |
| Brave only | $12-18 |
| Both (smart routing) | $45-75 |

**Recommended:** Both providers with smart routing gives best quality at reasonable cost.

### Heavy Usage (1,000 searches/day = 30,000/month)

| Configuration | Monthly Cost |
|---------------|--------------|
| Tavily only | Enterprise pricing |
| Brave only | $90 |
| Both (smart routing) | $120-180 |

**Recommended:** Brave only for cost optimization at scale.

---

## Testing Your Setup

### 1. Check Status Indicator

After configuring, open MLPatrol and look for the status box:

✅ **Good:**
```
🔍 Web Search: Tavily AI + Brave Search • Privacy-focused
```

### 2. Test CVE Query (Should route to Brave)

In Security Chat tab, ask:
```
"Find details about CVE-2024-1234"
```

Check logs for:
```
Routing to Brave (query type: cve_monitoring)
```

### 3. Test General Query (Should route to Tavily)

Ask:
```
"What are PyTorch security best practices?"
```

Check logs for:
```
Routing to Tavily (query type: general_security)
```

### 4. Check Response Quality

**Tavily responses should include:**
- Clean, structured summaries
- Multiple cited sources
- Fact-checked information

**Brave responses should include:**
- Fresh, recent results
- Direct URLs
- Published dates

---

## Performance Tips

### 1. Use Smart Routing

Enable both providers and let MLPatrol route intelligently:
```bash
USE_TAVILY_SEARCH=true
USE_BRAVE_SEARCH=true
```

Benefits:
- Best provider for each query type
- Fallback if one provider fails
- Optimized cost (use cheap Brave for high-volume CVE monitoring)

### 2. Adjust Search Depth

For faster responses, use basic depth:
```bash
TAVILY_SEARCH_DEPTH=basic
```

For better quality, use advanced:
```bash
TAVILY_SEARCH_DEPTH=advanced
```

### 3. Optimize Result Limits

Fewer results = faster + cheaper:
```bash
TAVILY_MAX_RESULTS=3
BRAVE_MAX_RESULTS=3
```

More results = better context:
```bash
TAVILY_MAX_RESULTS=10
BRAVE_MAX_RESULTS=10
```

Default (5) is a good balance.

### 4. Tune Freshness

For CVE monitoring, use recent results:
```bash
BRAVE_SEARCH_FRESHNESS=pd  # Past day
```

For general research, allow older content:
```bash
BRAVE_SEARCH_FRESHNESS=pm  # Past month
```

---

## Security & Privacy

### Tavily AI

- SOC 2 certified
- Zero data retention policy
- AI security layer (prevents prompt injection)
- HTTPS encrypted API calls

### Brave Search

- **No user tracking** (privacy-first mission)
- Independent index (not Google/Bing)
- No profiling or data collection
- HTTPS encrypted API calls

### MLPatrol Best Practices

1. **Never commit `.env` to git** (contains API keys)
2. **Rotate API keys regularly**
3. **Monitor usage** via provider dashboards
4. **Use environment-specific keys** (dev vs prod)

---

## Comparison: Which Provider to Use?

### Use Tavily When:

✅ You want AI-optimized search for agents
✅ LangChain integration is important
✅ Factual accuracy is critical
✅ You need clean, structured output
✅ Budget allows $30-60/month

### Use Brave When:

✅ Cost is primary concern (75% cheaper)
✅ Privacy is critical
✅ You need breaking news/CVE monitoring
✅ You want independent search index
✅ High search volume (20M queries/month available)

### Use Both When:

✅ You want best quality for all use cases
✅ You need redundancy/fallback
✅ Budget allows $45-75/month
✅ Different queries have different needs

**Recommendation:** Start with both, optimize based on usage patterns.

---

## Migration Guide

### From Placeholder to Tavily

**Before (placeholder):**
```python
# Returns mock data
web_search("pytorch security")
```

**After (Tavily):**
```bash
# In .env
USE_TAVILY_SEARCH=true
TAVILY_API_KEY=tvly-xxxxx
```

```python
# Returns real search results with citations
web_search("pytorch security")
```

### From Tavily to Both Providers

**Step 1:** Get Brave API key

**Step 2:** Update `.env`:
```bash
USE_BRAVE_SEARCH=true
BRAVE_API_KEY=BSAxxxxx
```

**Step 3:** Restart app - smart routing active!

### From Both to Brave Only (Cost Optimization)

If Tavily costs too high:

```bash
USE_TAVILY_SEARCH=false  # Disable
USE_BRAVE_SEARCH=true    # Keep
```

**Savings:** ~$30-45/month

---

## FAQ

### Q: Can I use web search without API keys during development?

**A:** No. The placeholder implementation was replaced with real API integrations. Use the free tiers:
- Tavily: 1,000 credits/month free
- Brave: 2,000 queries/month free

### Q: What happens if I hit rate limits?

**A:** The search will return an error message. Enable both providers for automatic fallback, or upgrade to paid tier.

### Q: Can I disable web search for specific users?

**A:** Yes, set `ENABLE_WEB_SEARCH=false` in `.env`. The agent will skip web search tool.

### Q: Which provider is better for hackathon demo?

**A:** Tavily - easier setup (LangChain integration), impressive AI-optimized results.

### Q: Which provider is better for production?

**A:** Both with smart routing - best quality, redundancy, and cost optimization.

### Q: How do I monitor usage?

**A:** Check provider dashboards:
- Tavily: https://tavily.com/dashboard
- Brave: https://brave.com/search/api/dashboard

### Q: Can I use custom search engines?

**A:** Yes, modify `_tavily_search()` or `_brave_search()` functions in `src/agent/tools.py` to integrate other providers.

---

## Support

### Documentation

- [Tavily API Docs](https://docs.tavily.com)
- [Brave Search API Docs](https://brave.com/search/api/)
- [MLPatrol Architecture](./ARCHITECTURE.md)
- [MLPatrol Development Guide](./DEVELOPMENT.md)

### Issues

If you encounter problems:

1. Check this guide's troubleshooting section
2. Review logs in `mlpatrol.log`
3. Test API keys directly via provider dashboards
4. Open issue on GitHub with logs and configuration (redact API keys!)

### Community

- [MLPatrol GitHub](https://github.com/sarahsl-prog/MLPatrol)
- [Discord: #agents-mcp-hackathon-winter25](https://discord.gg/fveShqytyh)

---

*Last updated: November 14, 2025*
*MLPatrol v1.0 - MCP 1st Birthday Hackathon*
