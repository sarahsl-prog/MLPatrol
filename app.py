"""MLPatrol - AI-Powered Security Agent for ML Systems.

This is the main Gradio application that provides an interactive interface for:
- CVE monitoring in ML libraries
- Dataset security analysis (poisoning, bias detection)
- Security code generation
- General ML security consultation

The app uses LangChain/LangGraph with Claude Sonnet or GPT-4 for multi-step reasoning.
"""

import json
import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Check Python version before importing any other modules
if sys.version_info < (3, 12):
    print(f"Error: MLPatrol requires Python 3.12 or higher.")
    print(
        f"Current version: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    print(f"\nPlease upgrade Python:")
    print(f"  - Windows: Download from https://www.python.org/downloads/")
    print(f"  - macOS: brew install python@3.12")
    print(f"  - Linux: sudo apt install python3.12")
    sys.exit(1)

import re
import time

import gradio as gr
import markdown
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import MLPatrol components (lazy-loaded below to avoid heavy imports at module import time)
# The real `create_mlpatrol_agent` and related classes are imported inside AgentState._initialize_agent
create_mlpatrol_agent = None
MLPatrolAgent = None
AgentResult = Any
from src.security.code_generator import build_cve_security_script
from src.security.cve_monitor import CVEMonitor
from src.utils.config_validator import validate_and_exit_on_error

# Load environment variables
load_dotenv()

# Validate configuration at startup
validate_and_exit_on_error()

# Configure logging with Loguru
from src.utils.logger import get_logger

logger = get_logger("main", log_dir="app")

# ============================================================================
# Constants and Configuration
# ============================================================================

APP_TITLE = "🛡️ MLPatrol - AI Security Agent for ML Systems"
APP_DESCRIPTION = """
**MLPatrol** is your intelligent security companion for machine learning systems.
It helps you monitor vulnerabilities, analyze datasets for security issues, and generate
validation code - all powered by advanced AI reasoning.
"""
LOGO_PATH = Path("mlpatrol_logo.png")

SUPPORTED_LIBRARIES = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "pytorch",
    "tensorflow",
    "keras",
    "xgboost",
    "lightgbm",
    "transformers",
]

MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_TYPES = [".csv"]

# UI Constants
MAX_ALERTS_DISPLAY = 200
MAX_AGENT_ITERATIONS = 10
LOGO_HEIGHT = 140
CHAT_INTERFACE_HEIGHT = 400
AGENT_STATUS_REFRESH_INTERVAL_SECONDS = 30

# CVE Search Constants
CVE_DAYS_MIN = 7
CVE_DAYS_MAX = 365
CVE_DAYS_DEFAULT = 90
CVE_DAYS_STEP = 7
CVE_DESCRIPTION_PREVIEW_LENGTH = 200

# Dataset Analysis Constants
DATASET_OUTLIER_ZSCORE_THRESHOLD = 3.0
DATASET_POISONING_THRESHOLD = 0.05
DATASET_HIGH_BIAS_THRESHOLD = 0.3
DATASET_QUALITY_SCORE_MAX = 10.0
DATASET_OUTLIER_PENALTY_FACTOR = 20
DATASET_BIAS_PENALTY_FACTOR = 3
DATASET_POISONING_PENALTY_FACTOR = 2
DATASET_OUTLIER_CONFIDENCE_FACTOR = 10

# Code Generation Constants
CODE_PURPOSE_MAX_LENGTH = 200
CODE_RESEARCH_SUMMARY_MAX_LENGTH = 400

# CVE Severity Threshold
CVE_HIGH_SEVERITY_THRESHOLD = 7.0

# Progress Bar Progress Points
PROGRESS_FORMATTING = 0.7

# Theme colors
THEME_COLORS = {
    "critical": "#dc2626",  # Red
    "high": "#ea580c",  # Orange
    "medium": "#eab308",  # Yellow
    "low": "#22c55e",  # Green
    "unknown": "#6b7280",  # Gray
}

# ============================================================================
# Global Agent State
# ============================================================================


class AgentState:
    """Thread-safe singleton to manage the MLPatrol agent instance.

    This class keeps initialization non-blocking, lazily imports the heavy
    agent factory, and provides thread-safe alert storage with on-disk
    persistence.
    """

    _instance: Optional[Any] = None
    _initialized: bool = False
    _error: Optional[str] = None
    _llm_info: Optional[Dict[str, str]] = None
    _lock: threading.Lock = threading.Lock()
    _alerts_lock: threading.Lock = threading.Lock()
    _alerts: List[Dict[str, Any]] = []

    @classmethod
    def get_agent(cls) -> Optional[Any]:
        """Return the agent instance, initializing it if necessary.

        Uses double-checked locking to avoid repeated initializations across
        threads while keeping the startup path quick.
        """
        if not cls._initialized:
            with cls._lock:
                if not cls._initialized:
                    cls._initialize_agent()
        return cls._instance

    @classmethod
    def _initialize_agent(cls) -> None:
        """Initialize the MLPatrol agent with environment configuration.

        This method performs a lazy import of the agent factory to avoid
        heavy imports during module import time.
        """
        try:
            logger.info("Initializing MLPatrol agent...")

            # Lazy import of heavy agent factory
            try:
                from src.agent.reasoning_chain import (
                    create_mlpatrol_agent as _create_mlpatrol_agent,
                )
            except Exception as e:
                cls._error = f"Failed to import agent factory: {e}"
                logger.error(cls._error, exc_info=True)
                cls._initialized = True
                return

            use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

            if use_local:
                model = os.getenv("LOCAL_LLM_MODEL", "ollama/llama3.1:8b")
                base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
                logger.info(f"Using local LLM: {model}")
                cls._instance = _create_mlpatrol_agent(
                    model=model,
                    base_url=base_url,
                    verbose=True,
                    max_iterations=MAX_AGENT_ITERATIONS,
                    max_execution_time=180,
                )
                cls._llm_info = {
                    "provider": "local",
                    "model": model.replace("ollama/", ""),
                    "type": "ollama",
                    "display_name": f"{model.replace('ollama/', '')} (Local - Ollama)",
                    "url": base_url,
                    "status": "connected",
                }
                cls._error = None
            else:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                model = "claude-sonnet-4-0"
                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY")
                    model = "gpt-4"

                if not api_key:
                    cls._error = (
                        "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
                        "or set USE_LOCAL_LLM=true for local models."
                    )
                    logger.error(cls._error)
                    cls._initialized = True
                    cls._llm_info = None
                    return

                cls._instance = _create_mlpatrol_agent(
                    api_key=api_key,
                    model=model,
                    verbose=True,
                    max_iterations=MAX_AGENT_ITERATIONS,
                    max_execution_time=180,
                )
                cls._llm_info = {
                    "provider": "cloud",
                    "model": model,
                    "type": "cloud",
                    "display_name": f"{model}",
                    "status": "connected",
                }

            try:
                cls._initialized_at = datetime.now(timezone.utc).isoformat()
                cls._instance_info = {"type": type(cls._instance).__name__}
                logger.debug("Agent instance info: %s", cls._instance_info)
            except Exception:
                logger.debug("Failed to record agent diagnostics", exc_info=True)

            cls._error = None

        except Exception as e:
            cls._error = f"Failed to initialize agent: {e}"
            logger.error(cls._error, exc_info=True)
            cls._llm_info = {
                "provider": "unknown",
                "model": "unknown",
                "type": "unknown",
                "display_name": "Not Connected",
                "status": "error",
            }
        finally:
            cls._initialized = True

    @classmethod
    def is_ready(cls) -> bool:
        return cls._initialized and cls._instance is not None and cls._error is None

    @classmethod
    def get_error(cls) -> Optional[str]:
        if not cls._initialized:
            return None
        return cls._error

    @classmethod
    def get_llm_info(cls) -> Optional[Dict[str, str]]:
        return cls._llm_info

    @classmethod
    def add_alert(cls, alert: Dict[str, Any]) -> None:
        try:
            with cls._alerts_lock:
                cls._alerts.insert(0, alert)
                cls._alerts = cls._alerts[:MAX_ALERTS_DISPLAY]
            # persist
            try:
                alerts_dir = Path("data")
                alerts_dir.mkdir(parents=True, exist_ok=True)
                alerts_file = alerts_dir / "alerts.json"
                alerts_file.write_text(json.dumps(cls._alerts, indent=2))
            except Exception:
                logger.debug("Failed to persist alerts", exc_info=True)
        except Exception:
            logger.debug("Failed to add alert", exc_info=True)

    @classmethod
    def get_alerts(cls) -> List[Dict[str, Any]]:
        try:
            with cls._alerts_lock:
                return list(cls._alerts)
        except Exception:
            return list(cls._alerts)

    @classmethod
    def load_alerts_from_disk(cls) -> None:
        try:
            alerts_file = Path("data") / "alerts.json"
            if alerts_file.exists():
                content = alerts_file.read_text()
                arr = json.loads(content)
                if isinstance(arr, list):
                    with cls._alerts_lock:
                        cls._alerts = arr[:MAX_ALERTS_DISPLAY]
        except Exception:
            logger.debug("Failed to load alerts from disk", exc_info=True)

    @staticmethod
    def get_web_search_info() -> Dict[str, Any]:
        enabled = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
        if not enabled:
            return {
                "enabled": False,
                "providers": [],
                "status": "disabled",
                "display_text": "Web Search: Disabled",
            }
        use_tavily = os.getenv("USE_TAVILY_SEARCH", "false").lower() == "true"
        use_brave = os.getenv("USE_BRAVE_SEARCH", "false").lower() == "true"
        providers = []
        if use_tavily:
            providers.append("Tavily AI")
        if use_brave:
            providers.append("Brave Search")
        if not providers:
            return {
                "enabled": True,
                "providers": [],
                "status": "not_configured",
                "display_text": "Web Search: Not Configured",
            }
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        brave_key = os.getenv("BRAVE_API_KEY", "")
        active_providers = []
        if use_tavily and tavily_key and tavily_key != "your_tavily_api_key_here":
            active_providers.append("Tavily AI")
        if use_brave and brave_key and brave_key != "your_brave_api_key_here":
            active_providers.append("Brave Search")
        if not active_providers:
            return {
                "enabled": True,
                "providers": providers,
                "status": "not_configured",
                "display_text": f"Web Search: {', '.join(providers)} (API keys needed)",
            }
        provider_text = " + ".join(active_providers)
        return {
            "enabled": True,
            "providers": active_providers,
            "status": "active",
            "display_text": f"Web Search: {provider_text}",
        }


# ============================================================================
# Validation and Security Functions
# ============================================================================


def validate_file_upload(file_path: str) -> Tuple[bool, Optional[str]]:
    """Validate uploaded file for security and format.

    Args:
        file_path: Path to uploaded file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        path = Path(file_path)

        # Check file exists
        if not path.exists():
            return False, "File does not exist"

        # Check file extension
        if path.suffix.lower() not in ALLOWED_FILE_TYPES:
            return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_FILE_TYPES)}"

        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return False, f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"

        # Try to read as CSV to validate format
        try:
            df = pd.read_csv(file_path, nrows=5)
            if len(df.columns) == 0:
                return False, "CSV file has no columns"
        except Exception as e:
            return False, f"Invalid CSV format: {str(e)}"

        return True, None

    except Exception as e:
        logger.error(f"File validation error: {e}", exc_info=True)
        return False, f"Validation error: {str(e)}"


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent injection attacks.

    Args:
        text: User input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Truncate to max length
    text = text[:max_length]

    # Remove potentially dangerous characters/patterns
    # (basic sanitization - the agent also validates)
    text = text.replace("<script>", "").replace("</script>", "")
    text = text.replace("javascript:", "")

    return text.strip()


def clear_agent_history_safe() -> None:
    """Clear agent history if agent is available (safe for UI callbacks)."""
    try:
        agent = AgentState.get_agent()
        if agent:
            agent.clear_history()
    except Exception:
        logger.debug("Failed to clear agent history", exc_info=True)


# ============================================================================
# Visualization Functions
# ============================================================================


def create_cve_severity_chart(cves: List[Dict[str, Any]]) -> go.Figure:
    """Create a bar chart showing CVE severity distribution.

    Args:
        cves: List of CVE dictionaries

    Returns:
        Plotly Figure object
    """
    if not cves:
        return go.Figure().update_layout(title="No CVEs found")

    # Count by severity
    severity_counts = {}
    for cve in cves:
        severity = cve.get("severity", "UNKNOWN")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    # Create bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(severity_counts.keys()),
                y=list(severity_counts.values()),
                marker_color=[
                    THEME_COLORS.get(s.lower(), THEME_COLORS["unknown"])
                    for s in severity_counts.keys()
                ],
                text=list(severity_counts.values()),
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="CVE Severity Distribution",
        xaxis_title="Severity",
        yaxis_title="Count",
        template="plotly_white",
        height=300,
    )

    return fig


def format_alerts_dashboard_html(alerts: List[Dict[str, Any]]) -> str:
    """Render alerts list into a compact HTML dashboard."""
    if not alerts:
        return "<div class='alerts-dashboard'><p><em>No active alerts</em></p></div>"

    html = "<div class='alerts-dashboard'>\n"
    html += "<h3>Active Alerts</h3>\n<ul>\n"
    for a in alerts:
        title = a.get("title", "Alert")
        severity = a.get("severity", "UNKNOWN")
        ts = a.get("timestamp", "n/a")
        details = a.get("details", "")
        html += f"<li><strong>{title}</strong> <em>({severity})</em> - <small>{ts}</small><div>{details}</div></li>\n"
    html += "</ul>\n</div>"
    return html


def create_class_distribution_chart(distribution: Dict[str, float]) -> go.Figure:
    """Create a pie chart showing class distribution.

    Args:
        distribution: Dictionary mapping class labels to proportions

    Returns:
        Plotly Figure object
    """
    if not distribution:
        return go.Figure().update_layout(title="No class distribution data")

    labels = list(distribution.keys())
    values = [distribution[k] * 100 for k in labels]  # Convert to percentages

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Class Distribution",
        template="plotly_white",
        height=300,
    )

    return fig


def create_quality_gauge(score: float) -> go.Figure:
    """Create a gauge chart showing dataset quality score.

    Args:
        score: Quality score from 0-10

    Returns:
        Plotly Figure object
    """
    # Determine color based on score
    if score >= 8:
        color = THEME_COLORS["low"]  # Green for good quality
    elif score >= 6:
        color = THEME_COLORS["medium"]  # Yellow for medium quality
    elif score >= 4:
        color = THEME_COLORS["high"]  # Orange for low quality
    else:
        color = THEME_COLORS["critical"]  # Red for poor quality

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Dataset Quality Score"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, DATASET_QUALITY_SCORE_MAX]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 4], "color": "rgba(220, 38, 38, 0.2)"},
                    {"range": [4, 6], "color": "rgba(234, 88, 12, 0.2)"},
                    {"range": [6, 8], "color": "rgba(234, 179, 8, 0.2)"},
                    {
                        "range": [8, DATASET_QUALITY_SCORE_MAX],
                        "color": "rgba(34, 197, 94, 0.2)",
                    },
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))

    return fig


# ============================================================================
# Result Formatting Functions
# ============================================================================


def format_reasoning_steps(agent_result: AgentResult) -> str:
    """Format reasoning steps as HTML for display.

    Args:
        agent_result: AgentResult object containing reasoning steps

    Returns:
        HTML string with formatted reasoning steps
    """
    if not agent_result.reasoning_steps:
        return "<p><em>No reasoning steps available</em></p>"

    html_parts = ["<div class='reasoning-steps'>"]

    for step in agent_result.reasoning_steps:
        html_parts.append(
            f"""
        <div class='reasoning-step'>
            <h4>Step {step.step_number}: {step.action}</h4>
            <p><strong>Input:</strong> <code>{json.dumps(step.action_input, indent=2)}</code></p>
            <p><strong>Output:</strong></p>
            <pre>{step.observation[:500]}{"..." if len(step.observation) > 500 else ""}</pre>
            <p><small>Duration: {step.duration_ms:.0f}ms</small></p>
        </div>
        """
        )

    html_parts.append("</div>")

    return "\n".join(html_parts)


def format_agent_answer(answer: str) -> str:
    """Format agent answer with markdown support and proper HTML structure.

    Args:
        answer: Raw agent answer text (may contain markdown)

    Returns:
        HTML string with formatted and styled content
    """
    if not answer or not answer.strip():
        return "<p><em>No response available</em></p>"

    # Convert markdown to HTML with extensions for better formatting
    html_content = markdown.markdown(
        answer,
        extensions=[
            "fenced_code",  # Support for ```code blocks```
            "tables",  # Support for markdown tables
            "nl2br",  # Convert newlines to <br>
            "sane_lists",  # Better list handling
        ],
    )

    # Wrap in a styled container
    formatted_html = f"""
    <div class='agent-answer'>
        {html_content}
    </div>
    """

    return formatted_html


def format_cve_results(cves: List[Dict[str, Any]]) -> str:
    """Format CVE results as HTML table.

    Args:
        cves: List of CVE dictionaries

    Returns:
        HTML string with formatted table
    """
    if not cves:
        return "<p><em>No CVEs found</em></p>"

    html = """
    <table style='width:100%; border-collapse: collapse;'>
        <thead>
            <tr style='background-color: #f3f4f6;'>
                <th style='padding: 8px; border: 1px solid #ddd;'>CVE ID</th>
                <th style='padding: 8px; border: 1px solid #ddd;'>Severity</th>
                <th style='padding: 8px; border: 1px solid #ddd;'>CVSS</th>
                <th style='padding: 8px; border: 1px solid #ddd;'>Description</th>
            </tr>
        </thead>
        <tbody>
    """

    for cve in cves:
        severity = cve.get("severity", "UNKNOWN")
        color = THEME_COLORS.get(severity.lower(), THEME_COLORS["unknown"])

        html += f"""
        <tr>
            <td style='padding: 8px; border: 1px solid #ddd;'><strong>{cve.get("cve_id", "N/A")}</strong></td>
            <td style='padding: 8px; border: 1px solid #ddd; background-color: {color}; color: white;'>
                <strong>{severity}</strong>
            </td>
            <td style='padding: 8px; border: 1px solid #ddd;'>{cve.get("cvss_score", "N/A")}</td>
            <td style='padding: 8px; border: 1px solid #ddd;'>{cve.get("description", "No description")[:CVE_DESCRIPTION_PREVIEW_LENGTH]}...</td>
        </tr>
        """

    html += """
        </tbody>
    </table>
    """

    return html


def format_dataset_analysis(analysis: Dict[str, Any]) -> str:
    """Format dataset analysis results as HTML.

    Args:
        analysis: Analysis dictionary

    Returns:
        HTML string with formatted analysis
    """
    html = f"""
    <div class='analysis-results'>
        <h3>Dataset Overview</h3>
        <ul>
            <li><strong>Rows:</strong> {analysis.get("num_rows", "N/A"):,}</li>
            <li><strong>Features:</strong> {analysis.get("num_features", "N/A")}</li>
            <li><strong>Outliers Detected:</strong> {analysis.get("outlier_count", "N/A")}</li>
        </ul>

        <h3>Security Assessment</h3>
        <ul>
            <li><strong>Suspected Poisoning:</strong>
                <span style='color: {"#dc2626" if analysis.get("suspected_poisoning") else "#22c55e"};'>
                    {"⚠️ YES" if analysis.get("suspected_poisoning") else "✓ NO"}
                </span>
            </li>
            <li><strong>Poisoning Confidence:</strong> {analysis.get("poisoning_confidence", 0) * 100:.1f}%</li>
            <li><strong>Bias Score:</strong> {analysis.get("bias_score", 0):.2f}</li>
            <li><strong>Quality Score:</strong> {analysis.get("quality_score", 0):.1f}/{DATASET_QUALITY_SCORE_MAX}</li>
        </ul>

        <h3>Recommendations</h3>
        <ul>
    """

    for rec in analysis.get("recommendations", []):
        html += f"<li>{rec}</li>"

    html += """
        </ul>
    </div>
    """

    return html


# ============================================================================
# Main Handler Functions
# ============================================================================


def handle_cve_search(
    library: str, days_back: int, progress=gr.Progress()
) -> Tuple[str, str, Any, str]:
    """Handle CVE search request.

    Args:
        library: Library name to search
        days_back: Number of days to look back
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message, results_html, chart, reasoning_html)
    """
    try:
        logger.info(f"CVE search request: {library}, {days_back} days")
        progress(0, desc="Initializing agent...")

        # Get agent
        agent = AgentState.get_agent()
        if agent is None:
            error = AgentState.get_error() or "Agent not initialized"
            return error, "", None, ""

        # Create query
        query = f"Search for CVEs and vulnerabilities in {library} from the last {days_back} days"

        progress(0.3, desc="Analyzing CVEs...")

        # Run agent
        result = agent.run(query)

        progress(PROGRESS_FORMATTING, desc="Formatting results...")

        # Parse CVE results from reasoning steps
        cves = []
        for step in result.reasoning_steps:
            if step.action == "cve_search":
                try:
                    data = json.loads(step.observation)
                    if data.get("status") == "success":
                        cves = data.get("cves", [])
                except Exception as e:
                    logger.debug(f"Failed to parse CVE tool output: {e}")

        # Create visualizations
        chart = create_cve_severity_chart(cves) if cves else None

        # Format results
        results_html = f"""
        <div class='results-container'>
            <h2>CVE Search Results for {library}</h2>
            <p><strong>Time Range:</strong> Last {days_back} days</p>
            <p><strong>CVEs Found:</strong> {len(cves)}</p>
            <hr>
            <h3>Agent Analysis:</h3>
            {format_agent_answer(result.answer)}
            <hr>
            {format_cve_results(cves) if cves else "<p><em>No CVEs found in the specified time range.</em></p>"}
        </div>
        """

        reasoning_html = format_reasoning_steps(result)

        status = f"✅ Found {len(cves)} CVE(s) for {library} | Confidence: {result.confidence:.0%} | Duration: {result.total_duration_ms:.0f}ms"

        progress(1.0, desc="Complete!")

        return status, results_html, chart, reasoning_html

    except Exception as e:
        logger.error(f"CVE search failed: {e}", exc_info=True)
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, f"<p style='color: red;'>{error_msg}</p>", None, ""


def handle_dataset_analysis(
    file, progress=gr.Progress()
) -> Tuple[str, str, Any, Any, str]:
    """Handle dataset analysis request.

    Args:
        file: Uploaded file object
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message, results_html, gauge_chart, distribution_chart, reasoning_html)
    """
    try:
        if file is None:
            return "❌ Please upload a dataset file", "", None, None, ""

        logger.info(f"Dataset analysis request: {file.name}")
        progress(0, desc="Validating file...")

        # Validate file
        is_valid, error_msg = validate_file_upload(file.name)
        if not is_valid:
            return (
                f"❌ {error_msg}",
                f"<p style='color: red;'>{error_msg}</p>",
                None,
                None,
                "",
            )

        progress(0.2, desc="Initializing agent...")

        # Get agent
        agent = AgentState.get_agent()
        if agent is None:
            error = AgentState.get_error() or "Agent not initialized"
            return error, "", None, None, ""

        # Create query with file context
        query = "Analyze this dataset for security issues including poisoning, bias, outliers, and data quality problems. Provide detailed findings and recommendations."

        progress(0.4, desc="Analyzing dataset...")

        # Run agent with dataset context
        context = {"file_path": file.name, "dataset": True}
        result = agent.run(query, context=context)

        progress(PROGRESS_FORMATTING, desc="Creating visualizations...")

        # Parse analysis results from reasoning steps
        analysis_data = None
        for step in result.reasoning_steps:
            if step.action == "analyze_dataset":
                try:
                    data = json.loads(step.observation)
                    if data.get("status") == "success":
                        analysis_data = data
                except Exception as e:
                    logger.debug(f"Failed to parse dataset tool output: {e}")

        # Create visualizations
        gauge_chart = None
        distribution_chart = None

        if analysis_data:
            quality_score = analysis_data.get("quality_score", 0)
            gauge_chart = create_quality_gauge(quality_score)

            class_dist = analysis_data.get("class_distribution", {})
            if class_dist:
                distribution_chart = create_class_distribution_chart(class_dist)

        # Format results
        results_html = f"""
        <div class='results-container'>
            <h2>Dataset Security Analysis</h2>
            <p><strong>File:</strong> {Path(file.name).name}</p>
            <hr>
            <h3>Agent Analysis:</h3>
            {format_agent_answer(result.answer)}
            <hr>
            {format_dataset_analysis(analysis_data) if analysis_data else "<p><em>Analysis data not available</em></p>"}
        </div>
        """

        reasoning_html = format_reasoning_steps(result)

        status = f"✅ Analysis complete | Confidence: {result.confidence:.0%} | Duration: {result.total_duration_ms:.0f}ms"

        progress(1.0, desc="Complete!")

        return status, results_html, gauge_chart, distribution_chart, reasoning_html

    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}", exc_info=True)
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, f"<p style='color: red;'>{error_msg}</p>", None, None, ""


def get_dashboard_html() -> str:
    """Return current alerts dashboard HTML for the UI refresh button."""
    alerts = AgentState.get_alerts()
    return format_alerts_dashboard_html(alerts)


def get_agent_status_html() -> str:
    """Return HTML summarizing the agent readiness and any errors."""
    try:
        if AgentState.is_ready():
            llm = AgentState.get_llm_info() or {}
            display = llm.get("display_name") if llm else "Connected"
            return f"<div style='background-color:#ecfdf5; border-left:4px solid #10b981; padding:8px; border-radius:4px;'><strong>🟢 Agent Ready:</strong> {display}</div>"
        # Not ready - show error if exists
        err = AgentState.get_error()
        if err:
            return f"<div style='background-color:#fff1f2; border-left:4px solid #ef4444; padding:8px; border-radius:4px;'><strong>🔴 Agent Error:</strong> {err}</div>"
        # Falling back to initializing
        return "<div style='background-color:#f8fafc; border-left:4px solid #f59e0b; padding:8px; border-radius:4px;'><strong>⚪ Agent:</strong> Initializing...</div>"
    except Exception as e:
        logger.debug(f"Failed to build agent status HTML: {e}", exc_info=True)
        return "<div><strong>Agent:</strong> Status unavailable</div>"


def run_background_coordinator_once() -> None:
    """Run a single CVE scan pass across supported libraries.

    This function is safe to call from a background thread or on-demand.
    It writes generated check scripts to `generated_checks/` and adds alerts
    to `AgentState` for UI consumption.
    """
    try:
        logger.info("Background coordinator running a single scan pass...")

        # determine since date (use file to persist last run)
        last_file = Path("data/cve_cache/last_check.txt")
        if last_file.exists():
            try:
                ts = datetime.fromisoformat(last_file.read_text().strip())
                # Ensure ts is timezone-aware; assume UTC if naive
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc) - pd.Timedelta(days=14)
        else:
            ts = datetime.now(timezone.utc) - pd.Timedelta(days=14)

        days_back = max(1, (datetime.now(timezone.utc) - ts).days)

        monitor = CVEMonitor(api_key=os.getenv("NVD_API_KEY"))

        any_alerts = []
        for lib in SUPPORTED_LIBRARIES:
            try:
                res = monitor.search_recent(lib, days_back)
            except Exception as e:
                logger.warning(f"CVE monitor failed for {lib}: {e}")
                continue

            if res.get("cve_count", 0) > 0:
                severity = (
                    "HIGH"
                    if any(
                        c.get("cvss_score", 0) >= CVE_HIGH_SEVERITY_THRESHOLD
                        for c in res.get("cves", [])
                    )
                    else "MEDIUM"
                )

                # Create an alert summary
                alert = {
                    "title": f"{res.get('cve_count', 0)} CVE(s) for {lib}",
                    "severity": severity,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "details": f"Found {res.get('cve_count', 0)} CVE(s) for {lib} in last {days_back} days.",
                    "cves": res.get("cves", []),
                }
                AgentState.add_alert(alert)
                any_alerts.append(alert)

                # For each CVE, attempt threat research (agent) and generate a check script
                agent = AgentState.get_agent()
                for cve in res.get("cves", []):
                    cve_id = cve.get("cve_id") or cve.get("id")
                    # Threat research via agent if available
                    research_summary = None
                    if agent:
                        try:
                            query = f"Research CVE {cve_id} affecting {lib}: summarize impact, exploits, and mitigations."
                            # Retry with exponential backoff
                            research_summary = None
                            for attempt in range(3):
                                try:
                                    r = agent.run(query)
                                    research_summary = r.answer
                                    break
                                except Exception as e:
                                    logger.debug(
                                        f"Agent research attempt {attempt + 1} failed for {cve_id}: {e}"
                                    )
                                    time.sleep(2**attempt)

                            AgentState.add_alert(
                                {
                                    "title": f"Research: {cve_id}",
                                    "severity": "LOW",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "details": (
                                        research_summary[
                                            :CODE_RESEARCH_SUMMARY_MAX_LENGTH
                                        ]
                                        if research_summary
                                        else ""
                                    ),
                                }
                            )
                        except Exception as e:
                            logger.debug(f"Threat research failed for {cve_id}: {e}")

                    # Generate a security check script for the CVE
                    try:
                        affected_versions = cve.get("affected_versions") or []
                        script = build_cve_security_script(
                            purpose=f"Check for {cve_id}",
                            library=lib,
                            cve_id=cve_id,
                            affected_versions=affected_versions,
                        )
                        out_dir = Path("generated_checks")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        # sanitize lib and cve_id for safe filenames
                        try:
                            safe_lib = re.sub(r"[^A-Za-z0-9_.-]", "_", str(lib))
                            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(cve_id))
                        except Exception:
                            safe_lib = str(lib).replace("/", "_")
                            safe_id = str(cve_id).replace("/", "_")
                        filename = out_dir / f"check_{safe_lib}_{safe_id}.py"
                        filename.write_text(script)
                        # attach script path to alert
                        AgentState.add_alert(
                            {
                                "title": f"Generated check for {cve_id}",
                                "severity": "LOW",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "details": f"Script written to {str(filename)}",
                                "script_path": str(filename),
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Failed to generate script for {cve_id}: {e}")

        # persist last run timestamp
        try:
            last_file.write_text(datetime.now(timezone.utc).isoformat())
        except Exception:
            pass

        if any_alerts:
            logger.info(f"Background coordinator found {len(any_alerts)} alerts")
        else:
            logger.info("Background coordinator found no alerts")
    except Exception as e:
        logger.error(f"Background coordinator failed: {e}", exc_info=True)


def _background_coordinator_loop() -> None:
    interval_min = int(os.getenv("CVE_POLL_INTERVAL_MIN", "60"))
    while True:
        try:
            run_background_coordinator_once()
        except Exception:
            logger.debug("Coordinator loop iteration failed", exc_info=True)
        # Sleep for the configured interval (in minutes)
        for _ in range(max(1, interval_min)):
            time.sleep(60)


def run_scan_now() -> str:
    """Trigger a single background scan asynchronously and return status text."""
    t = threading.Thread(target=run_background_coordinator_once, daemon=True)
    t.start()
    return "Scan started"


def handle_code_generation(
    purpose: str, library: str, cve_id: Optional[str], progress=gr.Progress()
) -> Tuple[str, str, str, str]:
    """Handle security code generation request.

    Args:
        purpose: Purpose of the security script
        library: Target library
        cve_id: Optional CVE ID
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message, code, filename, reasoning_html)
    """
    try:
        logger.info(f"Code generation request: {purpose} for {library}")
        progress(0, desc="Initializing agent...")

        # Validate inputs
        purpose = sanitize_input(purpose, max_length=CODE_PURPOSE_MAX_LENGTH)
        library = sanitize_input(library, max_length=50)
        cve_id = sanitize_input(cve_id, max_length=20) if cve_id else None

        if not purpose or not library:
            return "❌ Purpose and library are required", "", "", ""

        # Get agent
        agent = AgentState.get_agent()
        if agent is None:
            error = AgentState.get_error() or "Agent not initialized"
            return error, "", "", ""

        # Create query
        query = f"Generate Python security validation code for: {purpose}. Target library: {library}."
        if cve_id:
            query += f" CVE ID: {cve_id}."

        progress(0.3, desc="Generating code...")

        # Run agent
        result = agent.run(query)

        progress(PROGRESS_FORMATTING, desc="Formatting code...")

        # Parse generated code from reasoning steps
        generated_code = ""
        filename = f"mlpatrol_check_{library.replace('-', '_')}.py"

        for step in result.reasoning_steps:
            if step.action == "generate_security_code":
                try:
                    data = json.loads(step.observation)
                    if data.get("status") == "success":
                        generated_code = data.get("code", "")
                        filename = data.get("filename", filename)
                except Exception as e:
                    logger.debug(f"Failed to parse generated code output: {e}")

        reasoning_html = format_reasoning_steps(result)

        if not generated_code:
            # Fallback: use answer as code
            generated_code = result.answer

        status = f"✅ Code generated | Confidence: {result.confidence:.0%} | Duration: {result.total_duration_ms:.0f}ms"

        progress(1.0, desc="Complete!")

        return status, generated_code, filename, reasoning_html

    except Exception as e:
        logger.error(f"Code generation failed: {e}", exc_info=True)
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, "", "", ""


def handle_chat(
    message: str, history: List[Dict[str, str]], progress=gr.Progress()
) -> Tuple[List[Dict[str, str]], str, str]:
    """Handle general security chat.

    Args:
        message: User's message
        history: Chat history (list of message dicts with 'role' and 'content')
        progress: Gradio progress tracker

    Returns:
        Tuple of (updated_history, reasoning_html, status_message)
    """
    try:
        logger.info(f"Chat request: {message[:100]}...")
        progress(0, desc="Processing...")

        # Sanitize input
        message = sanitize_input(message, max_length=1000)

        if not message:
            return history, "", "❌ Please enter a message"

        # Get agent
        agent = AgentState.get_agent()
        if agent is None:
            error = AgentState.get_error() or "Agent not initialized"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"Error: {error}"})
            return history, "", f"❌ {error}"

        progress(0.3, desc="Thinking...")

        # Run agent
        result = agent.run(message)

        progress(PROGRESS_FORMATTING, desc="Formatting response...")

        # Update history with messages format
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": result.answer})

        reasoning_html = format_reasoning_steps(result)

        status = f"✅ Response generated | Confidence: {result.confidence:.0%} | Duration: {result.total_duration_ms:.0f}ms"

        progress(1.0, desc="Complete!")

        return history, reasoning_html, status

    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        error_msg = f"❌ Error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append(
            {"role": "assistant", "content": f"I encountered an error: {str(e)}"}
        )
        return history, "", error_msg


# ============================================================================
# Gradio Interface - Helper Functions
# ============================================================================


def get_interface_css() -> str:
    """Get the CSS styling for the Gradio interface.

    Returns:
        CSS string for the interface
    """
    return """
        .results-container {
            padding: 20px;
            background-color: #f9fafb;
            border-radius: 8px;
            margin: 10px 0;
            color: #0f172a; /* enforce dark text for readability */
        }
        .reasoning-steps {
            max-height: 600px;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 10px;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            background-color: #f9fafb;
            color: #0f172a;
        }
        .reasoning-step {
            background-color: white;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #3b82f6;
            border-radius: 4px;
            color: #0f172a;
        }
        .reasoning-step h4 {
            margin-top: 0;
            color: #1f2937;
        }
        .reasoning-step pre {
            background-color: #f3f4f6;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            max-height: 300px;
            overflow-y: auto;
            color: #0f172a;
        }
        .analysis-results h3 {
            color: #1f2937;
            margin-top: 15px;
        }
        .analysis-results ul {
            line-height: 1.8;
            color: #0f172a;
        }

        /* Agent Answer Markdown Formatting */
        .agent-answer {
            background-color: white;
            padding: 20px;
            border-radius: 6px;
            border: 1px solid #e5e7eb;
            margin: 15px 0;
            line-height: 1.7;
            color: #0f172a; /* enforce dark text */
        }
        .agent-answer p {
            margin: 12px 0;
            font-size: 15px;
        }
        .agent-answer h1, .agent-answer h2, .agent-answer h3 {
            color: #111827;
            margin-top: 20px;
            margin-bottom: 12px;
            font-weight: 600;
        }
        .agent-answer h1 {
            font-size: 24px;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 8px;
        }
        .agent-answer h2 {
            font-size: 20px;
        }
        .agent-answer h3 {
            font-size: 18px;
        }
        .agent-answer ul, .agent-answer ol {
            margin: 12px 0;
            padding-left: 28px;
        }
        .agent-answer li {
            margin: 8px 0;
            line-height: 1.6;
        }
        .agent-answer code {
            background-color: #f3f4f6;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 14px;
            color: #dc2626;
        }
        .agent-answer pre {
            background-color: #1f2937;
            color: #f9fafb;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 16px 0;
            border: 1px solid #374151;
        }
        .agent-answer pre code {
            background-color: transparent;
            padding: 0;
            color: #f9fafb;
            font-size: 13px;
        }
        /* Ensure tables and generic result areas use dark text for light backgrounds */
        .results-container table, .results-container th, .results-container td,
        .agent-answer table, .agent-answer th, .agent-answer td,
        .analysis-results, .analysis-results li {
            color: #0f172a;
        }
        .agent-answer blockquote {
            border-left: 4px solid #3b82f6;
            margin: 16px 0;
            padding: 12px 20px;
            background-color: #eff6ff;
            color: #1e40af;
        }
        .agent-answer table {
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
            font-size: 14px;
        }
        .agent-answer th, .agent-answer td {
            border: 1px solid #e5e7eb;
            padding: 10px 12px;
            text-align: left;
        }
        .agent-answer th {
            background-color: #f3f4f6;
            font-weight: 600;
            color: #111827;
        }
        .agent-answer tr:nth-child(even) {
            background-color: #f9fafb;
        }
        .agent-answer strong {
            font-weight: 600;
            color: #111827;
        }
        .agent-answer em {
            font-style: italic;
            color: #4b5563;
        }
        .agent-answer a {
            color: #2563eb;
            text-decoration: none;
        }
        .agent-answer a:hover {
            text-decoration: underline;
        }
        .agent-answer hr {
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 20px 0;
        }
        """


def create_dashboard_tab() -> None:
    """Create the Dashboard/Alerts tab."""
    gr.Markdown(
        """
    ### Alerts Dashboard
    This dashboard shows alerts discovered by the background CVE monitor.
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            dashboard_status = gr.Textbox(
                label="Status",
                interactive=False,
                show_label=True,
            )
            dashboard_refresh = gr.Button("Refresh Dashboard")
            dashboard_run_scan = gr.Button("Run CVE Scan Now", variant="primary")

        with gr.Column(scale=2):
            dashboard_html = gr.HTML(label="Alerts")

    # Connect handlers
    dashboard_refresh.click(
        fn=get_dashboard_html, inputs=None, outputs=[dashboard_html]
    )
    dashboard_run_scan.click(fn=run_scan_now, inputs=None, outputs=[dashboard_status])


def create_cve_monitoring_tab() -> None:
    """Create the CVE Monitoring tab."""
    gr.Markdown(
        """
    ### Search for vulnerabilities in ML libraries
    Monitor CVEs from the National Vulnerability Database for your ML dependencies.
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            cve_library = gr.Dropdown(
                choices=SUPPORTED_LIBRARIES,
                value="numpy",
                label="Library",
                info="Select the ML library to check",
            )
            cve_days = gr.Slider(
                minimum=CVE_DAYS_MIN,
                maximum=CVE_DAYS_MAX,
                value=CVE_DAYS_DEFAULT,
                step=CVE_DAYS_STEP,
                label="Days to look back",
                info="Search for CVEs published in the last N days",
            )
            cve_search_btn = gr.Button("🔍 Search for CVEs", variant="primary")

        with gr.Column(scale=2):
            cve_status = gr.Textbox(label="Status", interactive=False, show_label=True)
            cve_chart = gr.Plot(label="Severity Distribution")

    cve_results = gr.HTML(label="Results")

    with gr.Accordion("🧠 Agent Reasoning Steps", open=False):
        cve_reasoning = gr.HTML()

    # Connect handler
    cve_search_btn.click(
        fn=handle_cve_search,
        inputs=[cve_library, cve_days],
        outputs=[cve_status, cve_results, cve_chart, cve_reasoning],
    )


def create_dataset_analysis_tab() -> None:
    """Create the Dataset Analysis tab."""
    gr.Markdown(
        """
    ### Analyze datasets for security issues
    Detect poisoning attempts, statistical anomalies, bias, and quality issues.
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            dataset_file = gr.File(
                label="Upload Dataset (CSV)",
                file_types=[".csv"],
                type="filepath",
            )
            dataset_analyze_btn = gr.Button("🔬 Analyze Dataset", variant="primary")

            gr.Markdown(
                f"""
            **Requirements:**
            - File format: CSV
            - Max size: {MAX_FILE_SIZE_MB}MB
            - Should contain numerical features and labels
            """
            )

        with gr.Column(scale=2):
            dataset_status = gr.Textbox(
                label="Status", interactive=False, show_label=True
            )
            with gr.Row():
                dataset_gauge = gr.Plot(label="Quality Score")
                dataset_dist_chart = gr.Plot(label="Class Distribution")

    dataset_results = gr.HTML(label="Analysis Results")

    with gr.Accordion("🧠 Agent Reasoning Steps", open=False):
        dataset_reasoning = gr.HTML()

    # Connect handler
    dataset_analyze_btn.click(
        fn=handle_dataset_analysis,
        inputs=[dataset_file],
        outputs=[
            dataset_status,
            dataset_results,
            dataset_gauge,
            dataset_dist_chart,
            dataset_reasoning,
        ],
    )


def create_code_generation_tab() -> None:
    """Create the Code Generation tab."""
    gr.Markdown(
        """
    ### Generate security validation code
    Create Python scripts to check for vulnerabilities and validate your environment.
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            code_purpose = gr.Textbox(
                label="Purpose",
                placeholder="e.g., Check for CVE vulnerability, validate data integrity",
                lines=2,
                info="Describe what the security script should do",
            )
            code_library = gr.Textbox(
                label="Target Library",
                placeholder="e.g., numpy, pytorch, tensorflow",
                info="The library to validate",
            )
            code_cve_id = gr.Textbox(
                label="CVE ID (Optional)",
                placeholder="e.g., CVE-2021-34141",
                info="Specific CVE to check (optional)",
            )
            code_generate_btn = gr.Button("⚡ Generate Code", variant="primary")

        with gr.Column(scale=2):
            code_status = gr.Textbox(label="Status", interactive=False, show_label=True)
            code_filename = gr.Textbox(label="Suggested Filename", interactive=False)

    code_output = gr.Code(label="Generated Code", language="python", lines=20)

    code_download_btn = gr.DownloadButton(label="⬇️ Download Script", visible=False)

    with gr.Accordion("🧠 Agent Reasoning Steps", open=False):
        code_reasoning = gr.HTML()

    # Connect handler
    code_generate_btn.click(
        fn=handle_code_generation,
        inputs=[code_purpose, code_library, code_cve_id],
        outputs=[code_status, code_output, code_filename, code_reasoning],
    )


def create_security_chat_tab() -> None:
    """Create the Security Chat tab."""
    gr.Markdown(
        """
    ### Ask general ML security questions
    Get expert advice on ML security best practices, threats, and mitigations.
    """
    )

    chat_interface = gr.Chatbot(
        label="MLPatrol Security Assistant",
        height=CHAT_INTERFACE_HEIGHT,
        type="messages",
    )

    with gr.Row():
        chat_input = gr.Textbox(
            placeholder="Ask about ML security...",
            show_label=False,
            scale=4,
        )
        chat_submit = gr.Button("Send", variant="primary", scale=1)
        chat_clear = gr.Button("Clear", scale=1)

    chat_status = gr.Textbox(label="Status", interactive=False, show_label=True)

    with gr.Accordion("🧠 Agent Reasoning Steps", open=False):
        chat_reasoning = gr.HTML()

    # Connect handlers
    chat_submit.click(
        fn=handle_chat,
        inputs=[chat_input, chat_interface],
        outputs=[chat_interface, chat_reasoning, chat_status],
    ).then(lambda: "", outputs=[chat_input])

    chat_input.submit(
        fn=handle_chat,
        inputs=[chat_input, chat_interface],
        outputs=[chat_interface, chat_reasoning, chat_status],
    ).then(lambda: "", outputs=[chat_input])

    chat_clear.click(
        lambda: ([], "", ""),
        outputs=[chat_interface, chat_reasoning, chat_status],
    ).then(
        lambda: (
            AgentState.get_agent().clear_history() if AgentState.get_agent() else None
        )
    )


def create_header() -> None:
    """Create the interface header with logo and status indicators."""
    # Header
    if LOGO_PATH.exists():
        gr.Image(
            value=str(LOGO_PATH),
            show_label=False,
            height=LOGO_HEIGHT,
            interactive=False,
            elem_id="mlpatrol-logo",
        )
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(APP_DESCRIPTION)

    # Display LLM Status
    llm_info = AgentState.get_llm_info()
    if llm_info and llm_info["status"] == "connected":
        if llm_info["provider"] == "local":
            status_icon = "🔵"  # Blue for local
            status_color = "#3b82f6"  # Blue
            privacy_note = " • 100% Private"
        else:
            status_icon = "🟢"  # Green for cloud
            status_color = "#22c55e"  # Green
            privacy_note = ""

        gr.Markdown(
            f"""
<div style="background-color: #f0f9ff; border-left: 4px solid {status_color}; padding: 12px 16px; margin: 10px 0; border-radius: 4px;">
    <strong>{status_icon} LLM Status:</strong> Using <code>{llm_info["display_name"]}</code>{privacy_note}
</div>
        """
        )
    elif llm_info and llm_info["status"] == "error":
        gr.Markdown(
            """
<div style="background-color: #fef2f2; border-left: 4px solid #dc2626; padding: 12px 16px; margin: 10px 0; border-radius: 4px;">
    <strong>🔴 LLM Status:</strong> Not Connected - Check configuration
</div>
        """
        )

    # Display Web Search Status
    web_search_info = AgentState.get_web_search_info()
    if web_search_info["status"] == "active":
        # Active providers - show in green/blue
        providers_text = " + ".join(
            [f"<code>{p}</code>" for p in web_search_info["providers"]]
        )
        gr.Markdown(
            f"""
<div style="background-color: #f0fdf4; border-left: 4px solid #22c55e; padding: 12px 16px; margin: 10px 0; border-radius: 4px;">
    <strong>🔍 Web Search:</strong> {providers_text} • Privacy-focused
</div>
        """
        )
    elif web_search_info["status"] == "not_configured":
        # Enabled but no API keys
        gr.Markdown(
            """
<div style="background-color: #fffbeb; border-left: 4px solid #eab308; padding: 12px 16px; margin: 10px 0; border-radius: 4px;">
    <strong>⚠️ Web Search:</strong> Not configured - Add API keys to .env
</div>
        """
        )
    elif web_search_info["status"] == "disabled":
        # Disabled
        gr.Markdown(
            """
<div style="background-color: #f3f4f6; border-left: 4px solid #6b7280; padding: 12px 16px; margin: 10px 0; border-radius: 4px;">
    <strong>⚫ Web Search:</strong> Disabled
</div>
        """
        )

    # Agent status block (refreshable)
    agent_status_md = gr.Markdown(get_agent_status_html())
    agent_status_refresh = gr.Button("Refresh Agent Status")
    agent_status_refresh.click(
        fn=get_agent_status_html, inputs=None, outputs=[agent_status_md]
    )
    # Auto-refresh agent status every 30 seconds using gr.Timer if available
    try:
        status_timer = gr.Timer(
            value=AGENT_STATUS_REFRESH_INTERVAL_SECONDS, active=True, render=True
        )
        status_timer.tick(
            fn=get_agent_status_html, inputs=None, outputs=[agent_status_md]
        )
    except Exception:
        logger.debug("gr.Timer not available; skipping auto-refresh for agent status")


def create_footer() -> None:
    """Create the interface footer."""
    gr.Markdown(
        """
    ---
    ### About MLPatrol

    MLPatrol is an AI-powered security agent built for the MCP 1st Birthday Hackathon.
    It helps secure ML systems through intelligent CVE monitoring, dataset analysis, and code generation.

    **Powered by:** LangChain, Claude Sonnet 4, Gradio 6

    **⚠️ Security Notice:** Always review generated code before execution. This tool provides
    security analysis but should be used alongside manual security reviews and testing.
    """
    )


# ============================================================================
# Gradio Interface - Main
# ============================================================================


def create_interface() -> gr.Blocks:
    """Create the main Gradio interface.

    Returns:
        Gradio Blocks interface
    """
    # Detect a compatible Gradio theme if available (keeps compatibility across versions)
    theme_kw = {}
    try:
        if hasattr(gr, "themes") and hasattr(gr.themes, "Soft"):
            theme_kw["theme"] = gr.themes.Soft()  # type: ignore[attr-defined]
        elif hasattr(gr, "themes") and hasattr(gr.themes, "Default"):
            theme_kw["theme"] = gr.themes.Default()  # type: ignore[attr-defined]
    except Exception:
        theme_kw = {}

    with gr.Blocks(
        title="MLPatrol - ML Security Agent",
        **theme_kw,
        css=get_interface_css(),
    ) as interface:
        create_header()

        # Main tabs
        with gr.Tabs():
            with gr.Tab("⚠️ Dashboard"):
                create_dashboard_tab()

            with gr.Tab("🔍 CVE Monitoring"):
                create_cve_monitoring_tab()

            with gr.Tab("📊 Dataset Analysis"):
                create_dataset_analysis_tab()

            with gr.Tab("💻 Code Generation"):
                create_code_generation_tab()

            with gr.Tab("💬 Security Chat"):
                create_security_chat_tab()

        create_footer()

    return interface


# ============================================================================
# Main Entry Point
# ============================================================================


def _check_runtime_dependencies() -> list:
    """Check for commonly required runtime dependencies and return missing names.

    This is a light-weight check that logs helpful guidance instead of failing
    hard. It is intended to make developer onboarding and debugging easier.
    """
    missing = []
    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("gradio", "gradio"),
    ]
    for pkg_name, import_name in required:
        try:
            __import__(import_name)
        except Exception:
            missing.append(pkg_name)

    if missing:
        logger.info(
            "Runtime dependency check: missing packages: %s. Install with `pip install -r requirements.txt`.",
            ", ".join(missing),
        )

    return missing


def main():
    """Launch the MLPatrol Gradio application."""
    try:
        # Ensure data directories exist
        Path("data/cve_cache").mkdir(parents=True, exist_ok=True)
        Path("data/user_data").mkdir(parents=True, exist_ok=True)

        logger.info("Starting MLPatrol application...")

        # Runtime dependency check
        missing = _check_runtime_dependencies()
        if missing:
            logger.warning(
                "Missing optional dependencies: %s. Some features may not work as expected.",
                ", ".join(missing),
            )

        # Initialize agent in background (non-blocking) and load persisted alerts
        logger.info("Initializing agent in background thread...")
        try:
            AgentState.load_alerts_from_disk()
        except Exception:
            logger.debug("Failed to load persisted alerts", exc_info=True)
        threading.Thread(target=AgentState.get_agent, daemon=True).start()

        # Create and launch interface
        interface = create_interface()

        # Launch configuration
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  # Set to True if you want a public link
            show_error=True,
            show_api=False,
        )

    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        print(f"ERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
