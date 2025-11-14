# CVE Search Enhancement Analysis

## Requested Features

1. **Allow user to add any search term** (custom library input)
2. **Allow searching for all libraries in the dropdown** ("Search All" feature)
3. **Add a list of newest CVEs in AI/ML space** (dashboard widget)

---

## Current Implementation Analysis

### Current UI Structure ([app.py:845-887](app.py#L845-L887))

```python
with gr.Tab("🔍 CVE Monitoring"):
    cve_library = gr.Dropdown(
        choices=SUPPORTED_LIBRARIES,  # Fixed list of 10 libraries
        value="numpy",
        label="Library",
    )
    cve_days = gr.Slider(7, 365, value=90)
    cve_search_btn = gr.Button("🔍 Search for CVEs")
```

**Supported Libraries** ([app.py:58-69](app.py#L58-L69)):
- numpy, scipy, pandas, scikit-learn, pytorch, tensorflow, keras, xgboost, lightgbm, transformers

**Handler Function** ([app.py:473-547](app.py#L473-L547)):
- Takes `library` (string) and `days_back` (int)
- Creates query: `f"Search for CVEs in {library} from last {days_back} days"`
- Calls agent with query
- Parses results and creates visualizations

### Current Tool Implementation ([tools.py:220-340](tools.py#L220-L340))

**`cve_search_impl(library: str, days_back: int)`**:
- Uses NVD API with keyword search: `keywordSearch: library`
- Supports any library name (not restricted to dropdown)
- Date range filtering with `pubStartDate` and `pubEndDate`
- Returns structured JSON with CVE list

**Pydantic Schema** ([tools.py:115-134](tools.py#L115-L134)):
```python
class CVESearchInput(BaseModel):
    library: str = Field(description="Library name")
    days_back: int = Field(default=90, ge=1, le=3650)

    @field_validator("library")
    @classmethod
    def validate_library_name(cls, v):
        # Allows alphanumeric + hyphens/underscores
        return v.lower()
```

---

## Feature 1: Custom Library Input

### Requirements
- Allow users to type any library name
- Keep the dropdown for quick access to common libraries
- Validate custom input

### Implementation Changes

#### **EASY** - Estimated Effort: **30 minutes**

**File: `app.py`**

**Change 1:** Update UI component (lines 853-858)
```python
# BEFORE:
cve_library = gr.Dropdown(
    choices=SUPPORTED_LIBRARIES,
    value="numpy",
    label="Library",
)

# AFTER:
cve_library = gr.Dropdown(
    choices=SUPPORTED_LIBRARIES,
    value="numpy",
    label="Library",
    allow_custom_value=True,  # ✨ NEW: Allows typing custom values
    info="Select or type any library name"
)
```

**That's it!** Gradio's `allow_custom_value=True` enables typing custom values while keeping the dropdown.

**Validation:**
- Already handled by Pydantic validator in [tools.py:128-134](tools.py#L128-L134)
- Validates format: alphanumeric, hyphens, underscores
- No additional code needed

**Testing:**
1. User types "fastapi" → Validates → Searches NVD
2. User types "invalid@lib!" → Pydantic raises error → Shows clear message
3. User selects "numpy" → Works as before

---

## Feature 2: Search All Libraries

### Requirements
- Add "Search All" option to search all libraries at once
- Display aggregated results
- Show breakdown by library

### Implementation Changes

#### **MEDIUM** - Estimated Effort: **2-3 hours**

**Change 1: UI** (`app.py`)

Add checkbox or special dropdown value:

```python
# Option A: Add checkbox
with gr.Row():
    cve_library = gr.Dropdown(
        choices=SUPPORTED_LIBRARIES,
        value="numpy",
        label="Library",
        allow_custom_value=True,
    )
    search_all_checkbox = gr.Checkbox(
        label="Search All Libraries",
        value=False,
        info="Search all supported libraries at once"
    )

# Option B: Add "All Libraries" to dropdown (simpler)
SUPPORTED_LIBRARIES_WITH_ALL = ["🔍 All Libraries"] + SUPPORTED_LIBRARIES

cve_library = gr.Dropdown(
    choices=SUPPORTED_LIBRARIES_WITH_ALL,
    value="numpy",
    label="Library",
    allow_custom_value=True,
)
```

**Recommendation:** Use Option B (simpler UX)

**Change 2: Handler Function** (`app.py:473-547`)

Update `handle_cve_search` to detect "All" and iterate:

```python
def handle_cve_search(
    library: str,
    days_back: int,
    progress=gr.Progress()
) -> Tuple[str, str, Any, str]:
    """Handle CVE search request."""
    try:
        # Detect "Search All" request
        if library == "🔍 All Libraries":
            return handle_cve_search_all(days_back, progress)

        # ... existing single library logic ...

def handle_cve_search_all(
    days_back: int,
    progress=gr.Progress()
) -> Tuple[str, str, Any, str]:
    """Handle search for all supported libraries."""
    try:
        agent = AgentState.get_agent()
        if agent is None:
            return "Agent not initialized", "", None, ""

        all_cves = []
        library_counts = {}

        total_libs = len(SUPPORTED_LIBRARIES)

        # Iterate through all libraries
        for idx, lib in enumerate(SUPPORTED_LIBRARIES):
            progress((idx + 1) / total_libs, desc=f"Searching {lib}...")

            # Create query for this library
            query = f"Search for CVEs in {lib} from last {days_back} days"
            result = agent.run(query)

            # Parse CVEs from this library
            for step in result.reasoning_steps:
                if step.action == "cve_search":
                    try:
                        data = json.loads(step.observation)
                        if data.get("status") == "success":
                            cves = data.get("cves", [])
                            all_cves.extend(cves)
                            library_counts[lib] = len(cves)
                    except:
                        pass

        # Create combined visualization
        chart = create_cve_severity_chart_by_library(all_cves, library_counts)

        # Format aggregated results
        results_html = f"""
        <div class='results-container'>
            <h2>CVE Search Results - All Libraries</h2>
            <p><strong>Time Range:</strong> Last {days_back} days</p>
            <p><strong>Total CVEs Found:</strong> {len(all_cves)}</p>
            <p><strong>Libraries Searched:</strong> {len(SUPPORTED_LIBRARIES)}</p>
            <hr>
            <h3>Breakdown by Library:</h3>
            <table>
                <tr><th>Library</th><th>CVEs Found</th></tr>
                {format_library_breakdown_table(library_counts)}
            </table>
            <hr>
            <h3>All CVEs:</h3>
            {format_cve_results(all_cves)}
        </div>
        """

        status = f"✅ Found {len(all_cves)} total CVE(s) across {len(SUPPORTED_LIBRARIES)} libraries"

        return status, results_html, chart, ""

    except Exception as e:
        logger.error(f"Search all failed: {e}", exc_info=True)
        return f"❌ Error: {e}", "", None, ""
```

**Change 3: New Visualization** (`app.py`)

Add function to create library breakdown chart:

```python
def create_cve_severity_chart_by_library(
    cves: List[Dict],
    library_counts: Dict[str, int]
) -> go.Figure:
    """Create stacked bar chart showing CVEs by library and severity."""

    # Group by library and severity
    data_by_lib = {}
    for cve in cves:
        lib = cve.get("library", "Unknown")
        severity = cve.get("severity", "UNKNOWN")

        if lib not in data_by_lib:
            data_by_lib[lib] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

        data_by_lib[lib][severity] = data_by_lib[lib].get(severity, 0) + 1

    # Create stacked bar chart
    fig = go.Figure()

    severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    colors = [THEME_COLORS["critical"], THEME_COLORS["high"],
              THEME_COLORS["medium"], THEME_COLORS["low"]]

    for severity, color in zip(severities, colors):
        values = [data_by_lib.get(lib, {}).get(severity, 0)
                  for lib in library_counts.keys()]

        fig.add_trace(go.Bar(
            name=severity,
            x=list(library_counts.keys()),
            y=values,
            marker_color=color
        ))

    fig.update_layout(
        barmode='stack',
        title="CVEs by Library and Severity",
        xaxis_title="Library",
        yaxis_title="Number of CVEs",
        template="plotly_white",
        height=400,
    )

    return fig

def format_library_breakdown_table(library_counts: Dict[str, int]) -> str:
    """Format library breakdown as HTML table rows."""
    rows = []
    for lib, count in sorted(library_counts.items(), key=lambda x: x[1], reverse=True):
        icon = "🔴" if count > 5 else "🟡" if count > 0 else "🟢"
        rows.append(f"<tr><td>{icon} {lib}</td><td><strong>{count}</strong></td></tr>")
    return "\n".join(rows)
```

**Change 4: Update Button Connection** (`app.py:883-887`)

```python
# Update to pass search_all parameter if using checkbox
cve_search_btn.click(
    fn=handle_cve_search,
    inputs=[cve_library, cve_days],  # Unchanged - handler detects "All"
    outputs=[cve_status, cve_results, cve_chart, cve_reasoning]
)
```

**Performance Considerations:**
- Searching 10 libraries sequentially: ~30-60 seconds
- Progress bar shows which library is being searched
- Consider timeout (current: 180s max, should be sufficient)

**Alternative: Parallel Search**

For better performance, use threading:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def handle_cve_search_all(days_back, progress):
    """Handle search for all libraries (parallel)."""
    agent = AgentState.get_agent()

    def search_library(lib):
        """Search single library."""
        query = f"Search for CVEs in {lib} from last {days_back} days"
        return lib, agent.run(query)

    all_cves = []
    library_counts = {}

    # Parallel execution (max 3 concurrent to avoid rate limits)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(search_library, lib): lib
                   for lib in SUPPORTED_LIBRARIES}

        for idx, future in enumerate(as_completed(futures)):
            lib, result = future.result()
            progress((idx + 1) / len(SUPPORTED_LIBRARIES),
                     desc=f"Completed {idx + 1}/{len(SUPPORTED_LIBRARIES)}")

            # Parse results...
            for step in result.reasoning_steps:
                if step.action == "cve_search":
                    # Extract CVEs...

    # ... rest of formatting ...
```

---

## Feature 3: AI/ML CVE Dashboard Widget

### Requirements
- Display newest CVEs in AI/ML space (not library-specific)
- Auto-refresh or load on startup
- Prominent placement in UI

### Implementation Changes

#### **MEDIUM** - Estimated Effort: **2-4 hours**

**Change 1: Add Dashboard Section** (`app.py`)

Add before the tabbed interface:

```python
def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(theme=gr.themes.Soft(), title="MLPatrol") as interface:
        # Header
        gr.Markdown(HEADER_MARKDOWN)

        # ========== NEW: AI/ML CVE Dashboard ==========
        gr.Markdown("## 🚨 Latest AI/ML Vulnerabilities")
        gr.Markdown("Real-time feed of the newest CVEs affecting AI/ML systems")

        with gr.Row():
            with gr.Column(scale=3):
                latest_cves_html = gr.HTML(
                    value="<p>Loading latest CVEs...</p>",
                    label="Latest CVEs"
                )

            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                gr.Markdown(f"""
                **Auto-updates:** Every 6 hours
                **Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
                """)

        gr.Markdown("---")
        # =============================================

        # Existing tabs...
        with gr.Tabs():
            # ... CVE Monitoring tab ...
            # ... Dataset Analysis tab ...
            # ... etc ...
```

**Change 2: Create Dashboard Data Fetcher** (`app.py`)

Add new function to fetch latest AI/ML CVEs:

```python
def fetch_latest_aiml_cves(max_results: int = 10) -> str:
    """Fetch latest CVEs in AI/ML space.

    Args:
        max_results: Maximum number of CVEs to display

    Returns:
        HTML string with formatted CVE list
    """
    try:
        logger.info("Fetching latest AI/ML CVEs...")

        # Search for AI/ML related CVEs from last 30 days
        ai_ml_keywords = [
            "machine learning",
            "artificial intelligence",
            "deep learning",
            "neural network",
            "model",
            "pytorch",
            "tensorflow",
            "huggingface",
            "transformers",
            "AI",
            "ML"
        ]

        # Query NVD API for recent CVEs with AI/ML keywords
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

        all_cves = []

        # Try a combined keyword search (NVD supports OR logic)
        # Note: In production, you may need to make multiple requests
        # or use more sophisticated filtering

        keyword_query = " OR ".join(ai_ml_keywords[:5])  # Limit to avoid too long URL

        params = {
            "keywordSearch": keyword_query,
            "pubStartDate": start_date.strftime("%Y-%m-%dT00:00:00.000"),
            "pubEndDate": end_date.strftime("%Y-%m-%dT23:59:59.000"),
            "resultsPerPage": max_results,
        }

        try:
            response = requests.get(
                base_url,
                params=params,
                headers={"User-Agent": "MLPatrol-SecurityAgent/1.0"},
                timeout=10
            )
            response.raise_for_status()
            nvd_data = response.json()

            # Parse vulnerabilities
            if "vulnerabilities" in nvd_data:
                for vuln in nvd_data["vulnerabilities"][:max_results]:
                    cve = vuln.get("cve", {})

                    # Extract data
                    cve_id = cve.get("id", "UNKNOWN")
                    descriptions = cve.get("descriptions", [])
                    description = descriptions[0].get("value", "") if descriptions else ""

                    # Extract CVSS
                    cvss_score = 0.0
                    severity = "UNKNOWN"
                    metrics = cve.get("metrics", {})
                    if "cvssMetricV31" in metrics and metrics["cvssMetricV31"]:
                        cvss_data = metrics["cvssMetricV31"][0]["cvssData"]
                        cvss_score = cvss_data.get("baseScore", 0.0)
                        severity = cvss_data.get("baseSeverity", "UNKNOWN")

                    published = cve.get("published", "Unknown")

                    all_cves.append({
                        "cve_id": cve_id,
                        "description": description,
                        "cvss_score": cvss_score,
                        "severity": severity,
                        "published": published,
                    })

        except requests.RequestException as e:
            logger.error(f"Failed to fetch AI/ML CVEs: {e}")
            return f"<p style='color: orange;'>⚠️ Unable to fetch latest CVEs. Please try refreshing.</p>"

        # Format as HTML
        if not all_cves:
            return "<p>No new AI/ML CVEs found in the last 30 days. ✅</p>"

        html = "<div class='dashboard-cves'>"
        html += f"<p><strong>{len(all_cves)} recent AI/ML vulnerabilities found</strong></p>"

        for cve in all_cves:
            severity_color = {
                "CRITICAL": THEME_COLORS["critical"],
                "HIGH": THEME_COLORS["high"],
                "MEDIUM": THEME_COLORS["medium"],
                "LOW": THEME_COLORS["low"],
            }.get(cve["severity"], "#6b7280")

            # Format published date
            try:
                pub_date = datetime.fromisoformat(cve["published"].replace("Z", "+00:00"))
                days_ago = (datetime.now(pub_date.tzinfo) - pub_date).days
                date_str = f"{days_ago} days ago" if days_ago > 0 else "Today"
            except:
                date_str = cve["published"]

            html += f"""
            <div style='border-left: 4px solid {severity_color}; padding: 12px; margin: 10px 0; background: #f9fafb;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <strong style='font-size: 16px;'>{cve['cve_id']}</strong>
                    <span style='background: {severity_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;'>
                        {cve['severity']} ({cve['cvss_score']})
                    </span>
                </div>
                <p style='margin: 8px 0; color: #4b5563; font-size: 14px;'>
                    {cve['description'][:200]}{'...' if len(cve['description']) > 200 else ''}
                </p>
                <small style='color: #9ca3af;'>Published: {date_str}</small>
            </div>
            """

        html += "</div>"
        return html

    except Exception as e:
        logger.error(f"Error in fetch_latest_aiml_cves: {e}", exc_info=True)
        return f"<p style='color: red;'>Error loading CVEs: {str(e)}</p>"
```

**Change 3: Auto-load on Startup** (`app.py`)

Add to interface creation:

```python
# Load latest CVEs when interface loads
interface.load(
    fn=fetch_latest_aiml_cves,
    inputs=[],
    outputs=[latest_cves_html]
)

# Connect refresh button
refresh_btn.click(
    fn=fetch_latest_aiml_cves,
    inputs=[],
    outputs=[latest_cves_html]
)
```

**Change 4: Optional - Periodic Auto-Refresh**

For auto-refresh every N hours, use Gradio's timer:

```python
# Add auto-refresh timer (every 6 hours = 21600 seconds)
gr.Timer(
    value=21600,  # 6 hours in seconds
    active=True
).tick(
    fn=fetch_latest_aiml_cves,
    inputs=[],
    outputs=[latest_cves_html]
)
```

**Alternative Placement:**

Instead of top dashboard, could add as a sidebar or separate tab:

```python
with gr.Tab("🚨 Latest AI/ML CVEs"):
    gr.Markdown("Real-time feed of newest vulnerabilities in AI/ML systems")
    refresh_btn = gr.Button("🔄 Refresh Feed")
    latest_cves_html = gr.HTML()

    # Auto-load
    interface.load(fn=fetch_latest_aiml_cves, outputs=[latest_cves_html])
    refresh_btn.click(fn=fetch_latest_aiml_cves, outputs=[latest_cves_html])
```

---

## Summary Table

| Feature | Complexity | Effort | Files Modified | New Functions | Testing Time |
|---------|-----------|--------|----------------|---------------|--------------|
| **1. Custom Input** | 🟢 Easy | 30 min | `app.py` (1 line) | 0 | 15 min |
| **2. Search All** | 🟡 Medium | 2-3 hours | `app.py` (~150 lines) | 3 | 30-45 min |
| **3. CVE Dashboard** | 🟡 Medium | 2-4 hours | `app.py` (~100 lines) | 1 | 30 min |
| **Total** | - | **5-8 hours** | `app.py` | 4 | **1.5-2 hours** |

---

## Implementation Priority

### Recommended Order:

1. **Feature 1 (Custom Input)** - Quick win, 30 minutes
2. **Feature 3 (CVE Dashboard)** - High visibility, useful
3. **Feature 2 (Search All)** - More complex, lower priority

### Rationale:
- Feature 1 is trivial (1 parameter change) - do first
- Feature 3 provides value to all users immediately
- Feature 2 is nice-to-have but less critical (users can search libraries individually)

---

## Testing Checklist

### Feature 1: Custom Input
- [ ] Type custom library name → Searches correctly
- [ ] Type invalid characters → Shows validation error
- [ ] Select from dropdown → Still works
- [ ] Empty input → Shows error

### Feature 2: Search All
- [ ] Select "All Libraries" → Searches all 10
- [ ] Progress bar updates correctly
- [ ] Results show breakdown by library
- [ ] Chart shows stacked bars correctly
- [ ] Handles errors gracefully (some libraries fail)
- [ ] Performance acceptable (<60s total)

### Feature 3: Dashboard
- [ ] Loads on startup
- [ ] Shows newest CVEs
- [ ] Refresh button works
- [ ] Severity colors correct
- [ ] Links work (if added)
- [ ] Handles NVD API errors gracefully
- [ ] Auto-refresh works (if enabled)

---

## Potential Issues & Solutions

### Issue 1: NVD API Rate Limiting
**Problem:** NVD has rate limits (without API key: 5 requests/30s)

**Solution:**
- Request free API key from NVD
- Add rate limiting logic
- Cache results (store in memory/disk for 6 hours)

**Code Addition:**
```python
import time
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_nvd_request(keyword, start_date, end_date):
    """Cache NVD requests for 6 hours."""
    time.sleep(0.6)  # Rate limit: max 5 req/30s = 1 req/6s
    response = requests.get(...)
    return response.json()
```

### Issue 2: "Search All" Takes Too Long
**Problem:** Sequential search of 10 libraries = 30-60 seconds

**Solution:**
- Use parallel ThreadPoolExecutor (shown above)
- Limit to 3 concurrent requests (NVD rate limit)
- Show progress for each library

### Issue 3: Dashboard Shows Too Many CVEs
**Problem:** 100+ CVEs overwhelming

**Solution:**
- Limit to top 10 most recent
- Add "Load More" button
- Add severity filter (show only CRITICAL/HIGH)

**Code:**
```python
# Add severity filter
with gr.Row():
    severity_filter = gr.CheckboxGroup(
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        value=["CRITICAL", "HIGH"],
        label="Show Severities"
    )

def fetch_latest_aiml_cves(severities):
    # Filter by selected severities
    filtered_cves = [cve for cve in all_cves
                     if cve['severity'] in severities]
    return format_cves(filtered_cves)
```

---

## Additional Enhancements (Optional)

### 1. Export Functionality
Add export button to download CVE results as CSV/JSON:

```python
export_btn = gr.Button("📥 Export Results")

def export_cves(cves):
    """Export CVEs to downloadable CSV."""
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["cve_id", "severity", "cvss_score", "description"])
    writer.writeheader()
    writer.writerows(cves)

    return output.getvalue()
```

### 2. Email Alerts
Allow users to subscribe to CVE alerts:

```python
email_input = gr.Textbox(label="Email for Alerts")
subscribe_btn = gr.Button("Subscribe to Alerts")

def subscribe_to_alerts(email, libraries):
    """Subscribe user to CVE alerts."""
    # Store in database
    # Send confirmation email
    # Schedule daily checks
```

### 3. CVE Severity Trends
Add chart showing CVE trends over time:

```python
def create_cve_trend_chart(cves):
    """Create line chart showing CVEs over time."""
    # Group by week
    # Count by severity
    # Plot trend
```

---

## Cost/Benefit Analysis

| Feature | User Value | Development Cost | Maintenance Cost |
|---------|-----------|------------------|------------------|
| Custom Input | 🟢🟢🟢 High | 🟢 Very Low | 🟢 None |
| Search All | 🟡🟡 Medium | 🟡 Medium | 🟡 Low |
| Dashboard | 🟢🟢🟢 High | 🟡 Medium | 🟡🟡 Medium |

**Recommendation:** Implement Features 1 & 3, defer Feature 2 unless requested.

---

## Files to Modify

```
MLPatrol/
├── app.py                    # Main changes (all 3 features)
│   ├── SUPPORTED_LIBRARIES   # Feature 2: Add "All"
│   ├── create_interface()    # All features
│   ├── handle_cve_search()   # Feature 1 & 2
│   └── NEW: handle_cve_search_all()           # Feature 2
│   └── NEW: fetch_latest_aiml_cves()          # Feature 3
│   └── NEW: create_cve_severity_chart_by_library()  # Feature 2
│
├── src/agent/tools.py        # No changes needed (already supports any library)
├── requirements.txt          # No new dependencies
└── docs/
    └── NEW: CVE_FEATURES.md  # Document new features
```

---

## Conclusion

All three features are **feasible and valuable**:

✅ **Feature 1 (Custom Input)**: Trivial to implement, high value
✅ **Feature 2 (Search All)**: Moderate effort, useful for power users
✅ **Feature 3 (Dashboard)**: Moderate effort, high visibility

**Total implementation time**: 5-8 hours for all three features.

**Recommended approach**: Implement incrementally, test each feature before moving to next.
