"""MLPatrol - AI-Powered Security Agent for ML Systems.

This is the main Gradio application that provides an interactive interface for:
- CVE monitoring in ML libraries
- Dataset security analysis (poisoning, bias detection)
- Security code generation
- General ML security consultation

The app uses LangChain/LangGraph with Claude Sonnet or GPT-4 for multi-step reasoning.
"""

import os
import sys
import logging
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from functools import lru_cache

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

# Import MLPatrol components
from src.agent.reasoning_chain import MLPatrolAgent, create_mlpatrol_agent, AgentResult
from src.agent.tools import parse_cve_results, parse_dataset_analysis

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mlpatrol.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants and Configuration
# ============================================================================

APP_TITLE = "🛡️ MLPatrol - AI Security Agent for ML Systems"
APP_DESCRIPTION = """
**MLPatrol** is your intelligent security companion for machine learning systems.
It helps you monitor vulnerabilities, analyze datasets for security issues, and generate
validation code - all powered by advanced AI reasoning.
"""

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

# Theme colors
THEME_COLORS = {
    "critical": "#dc2626",  # Red
    "high": "#ea580c",      # Orange
    "medium": "#eab308",    # Yellow
    "low": "#22c55e",       # Green
    "unknown": "#6b7280",   # Gray
}

# ============================================================================
# Global Agent State
# ============================================================================

class AgentState:
    """Singleton class to manage the MLPatrol agent instance.

    This ensures we only initialize the agent once and reuse it across
    all user interactions for better performance.
    """
    _instance: Optional[MLPatrolAgent] = None
    _initialized: bool = False
    _error: Optional[str] = None

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

            # Try Anthropic first, then OpenAI
            api_key = os.getenv("ANTHROPIC_API_KEY")
            model = "claude-sonnet-4-20250514"

            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                model = "gpt-4"

            if not api_key:
                cls._error = (
                    "No API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                    "environment variable."
                )
                logger.error(cls._error)
                cls._initialized = True
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

        except Exception as e:
            cls._error = f"Failed to initialize agent: {str(e)}"
            logger.error(f"Agent initialization failed: {e}", exc_info=True)
        finally:
            cls._initialized = True

    @classmethod
    def get_error(cls) -> Optional[str]:
        """Get initialization error if any."""
        if not cls._initialized:
            cls._initialize_agent()
        return cls._error

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
    fig = go.Figure(data=[
        go.Bar(
            x=list(severity_counts.keys()),
            y=list(severity_counts.values()),
            marker_color=[THEME_COLORS.get(s.lower(), THEME_COLORS["unknown"])
                         for s in severity_counts.keys()],
            text=list(severity_counts.values()),
            textposition="auto",
        )
    ])

    fig.update_layout(
        title="CVE Severity Distribution",
        xaxis_title="Severity",
        yaxis_title="Count",
        template="plotly_white",
        height=300,
    )

    return fig

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

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>',
        )
    ])

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

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Dataset Quality Score"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 4], 'color': "rgba(220, 38, 38, 0.2)"},
                {'range': [4, 6], 'color': "rgba(234, 88, 12, 0.2)"},
                {'range': [6, 8], 'color': "rgba(234, 179, 8, 0.2)"},
                {'range': [8, 10], 'color': "rgba(34, 197, 94, 0.2)"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )

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
        html_parts.append(f"""
        <div class='reasoning-step'>
            <h4>Step {step.step_number}: {step.action}</h4>
            <p><strong>Input:</strong> <code>{json.dumps(step.action_input, indent=2)}</code></p>
            <p><strong>Output:</strong></p>
            <pre>{step.observation[:500]}{"..." if len(step.observation) > 500 else ""}</pre>
            <p><small>Duration: {step.duration_ms:.0f}ms</small></p>
        </div>
        """)

    html_parts.append("</div>")

    return "\n".join(html_parts)

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
            <td style='padding: 8px; border: 1px solid #ddd;'><strong>{cve.get('cve_id', 'N/A')}</strong></td>
            <td style='padding: 8px; border: 1px solid #ddd; background-color: {color}; color: white;'>
                <strong>{severity}</strong>
            </td>
            <td style='padding: 8px; border: 1px solid #ddd;'>{cve.get('cvss_score', 'N/A')}</td>
            <td style='padding: 8px; border: 1px solid #ddd;'>{cve.get('description', 'No description')[:200]}...</td>
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
            <li><strong>Rows:</strong> {analysis.get('num_rows', 'N/A'):,}</li>
            <li><strong>Features:</strong> {analysis.get('num_features', 'N/A')}</li>
            <li><strong>Outliers Detected:</strong> {analysis.get('outlier_count', 'N/A')}</li>
        </ul>

        <h3>Security Assessment</h3>
        <ul>
            <li><strong>Suspected Poisoning:</strong>
                <span style='color: {"#dc2626" if analysis.get("suspected_poisoning") else "#22c55e"};'>
                    {"⚠️ YES" if analysis.get("suspected_poisoning") else "✓ NO"}
                </span>
            </li>
            <li><strong>Poisoning Confidence:</strong> {analysis.get('poisoning_confidence', 0) * 100:.1f}%</li>
            <li><strong>Bias Score:</strong> {analysis.get('bias_score', 0):.2f}</li>
            <li><strong>Quality Score:</strong> {analysis.get('quality_score', 0):.1f}/10</li>
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
    library: str,
    days_back: int,
    progress=gr.Progress()
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

        progress(0.7, desc="Formatting results...")

        # Parse CVE results from reasoning steps
        cves = []
        for step in result.reasoning_steps:
            if step.action == "cve_search":
                try:
                    data = json.loads(step.observation)
                    if data.get("status") == "success":
                        cves = data.get("cves", [])
                except:
                    pass

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
            <p>{result.answer}</p>
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
    file,
    progress=gr.Progress()
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
            return f"❌ {error_msg}", f"<p style='color: red;'>{error_msg}</p>", None, None, ""

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

        progress(0.7, desc="Creating visualizations...")

        # Parse analysis results from reasoning steps
        analysis_data = None
        for step in result.reasoning_steps:
            if step.action == "analyze_dataset":
                try:
                    data = json.loads(step.observation)
                    if data.get("status") == "success":
                        analysis_data = data
                except:
                    pass

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
            <p>{result.answer}</p>
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

def handle_code_generation(
    purpose: str,
    library: str,
    cve_id: str,
    progress=gr.Progress()
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
        purpose = sanitize_input(purpose, max_length=200)
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

        progress(0.7, desc="Formatting code...")

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
                except:
                    pass

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
    message: str,
    history: List[List[str]],
    progress=gr.Progress()
) -> Tuple[List[List[str]], str, str]:
    """Handle general security chat.

    Args:
        message: User's message
        history: Chat history
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
            history.append([message, f"Error: {error}"])
            return history, "", f"❌ {error}"

        progress(0.3, desc="Thinking...")

        # Run agent
        result = agent.run(message)

        progress(0.7, desc="Formatting response...")

        # Update history
        history.append([message, result.answer])

        reasoning_html = format_reasoning_steps(result)

        status = f"✅ Response generated | Confidence: {result.confidence:.0%} | Duration: {result.total_duration_ms:.0f}ms"

        progress(1.0, desc="Complete!")

        return history, reasoning_html, status

    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        error_msg = f"❌ Error: {str(e)}"
        history.append([message, f"I encountered an error: {str(e)}"])
        return history, "", error_msg

# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface() -> gr.Blocks:
    """Create the main Gradio interface.

    Returns:
        Gradio Blocks interface
    """

    with gr.Blocks(
        title="MLPatrol - ML Security Agent",
        theme=gr.themes.Soft(),
        css="""
        .results-container {
            padding: 20px;
            background-color: #f9fafb;
            border-radius: 8px;
            margin: 10px 0;
        }
        .reasoning-step {
            background-color: white;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #3b82f6;
            border-radius: 4px;
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
        }
        .analysis-results h3 {
            color: #1f2937;
            margin-top: 15px;
        }
        .analysis-results ul {
            line-height: 1.8;
        }
        """
    ) as interface:

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

        # Main tabs
        with gr.Tabs() as tabs:

            # ================================================================
            # Tab 1: CVE Monitoring
            # ================================================================
            with gr.Tab("🔍 CVE Monitoring"):
                gr.Markdown("""
                ### Search for vulnerabilities in ML libraries
                Monitor CVEs from the National Vulnerability Database for your ML dependencies.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        cve_library = gr.Dropdown(
                            choices=SUPPORTED_LIBRARIES,
                            value="numpy",
                            label="Library",
                            info="Select the ML library to check"
                        )
                        cve_days = gr.Slider(
                            minimum=7,
                            maximum=365,
                            value=90,
                            step=7,
                            label="Days to look back",
                            info="Search for CVEs published in the last N days"
                        )
                        cve_search_btn = gr.Button("🔍 Search for CVEs", variant="primary")

                    with gr.Column(scale=2):
                        cve_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            show_label=True
                        )
                        cve_chart = gr.Plot(label="Severity Distribution")

                cve_results = gr.HTML(label="Results")

                with gr.Accordion("🧠 Agent Reasoning Steps", open=False):
                    cve_reasoning = gr.HTML()

                # Connect handler
                cve_search_btn.click(
                    fn=handle_cve_search,
                    inputs=[cve_library, cve_days],
                    outputs=[cve_status, cve_results, cve_chart, cve_reasoning]
                )

            # ================================================================
            # Tab 2: Dataset Analysis
            # ================================================================
            with gr.Tab("📊 Dataset Analysis"):
                gr.Markdown("""
                ### Analyze datasets for security issues
                Detect poisoning attempts, statistical anomalies, bias, and quality issues.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        dataset_file = gr.File(
                            label="Upload Dataset (CSV)",
                            file_types=[".csv"],
                            type="filepath"
                        )
                        dataset_analyze_btn = gr.Button("🔬 Analyze Dataset", variant="primary")

                        gr.Markdown(f"""
                        **Requirements:**
                        - File format: CSV
                        - Max size: {MAX_FILE_SIZE_MB}MB
                        - Should contain numerical features and labels
                        """)

                    with gr.Column(scale=2):
                        dataset_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            show_label=True
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
                    outputs=[dataset_status, dataset_results, dataset_gauge,
                            dataset_dist_chart, dataset_reasoning]
                )

            # ================================================================
            # Tab 3: Code Generation
            # ================================================================
            with gr.Tab("💻 Code Generation"):
                gr.Markdown("""
                ### Generate security validation code
                Create Python scripts to check for vulnerabilities and validate your environment.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        code_purpose = gr.Textbox(
                            label="Purpose",
                            placeholder="e.g., Check for CVE vulnerability, validate data integrity",
                            lines=2,
                            info="Describe what the security script should do"
                        )
                        code_library = gr.Textbox(
                            label="Target Library",
                            placeholder="e.g., numpy, pytorch, tensorflow",
                            info="The library to validate"
                        )
                        code_cve_id = gr.Textbox(
                            label="CVE ID (Optional)",
                            placeholder="e.g., CVE-2021-34141",
                            info="Specific CVE to check (optional)"
                        )
                        code_generate_btn = gr.Button("⚡ Generate Code", variant="primary")

                    with gr.Column(scale=2):
                        code_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            show_label=True
                        )
                        code_filename = gr.Textbox(
                            label="Suggested Filename",
                            interactive=False
                        )

                code_output = gr.Code(
                    label="Generated Code",
                    language="python",
                    lines=20
                )

                code_download_btn = gr.DownloadButton(
                    label="⬇️ Download Script",
                    visible=False
                )

                with gr.Accordion("🧠 Agent Reasoning Steps", open=False):
                    code_reasoning = gr.HTML()

                # Connect handler
                code_generate_btn.click(
                    fn=handle_code_generation,
                    inputs=[code_purpose, code_library, code_cve_id],
                    outputs=[code_status, code_output, code_filename, code_reasoning]
                )

            # ================================================================
            # Tab 4: Security Chat
            # ================================================================
            with gr.Tab("💬 Security Chat"):
                gr.Markdown("""
                ### Ask general ML security questions
                Get expert advice on ML security best practices, threats, and mitigations.
                """)

                chat_interface = gr.Chatbot(
                    label="MLPatrol Security Assistant",
                    height=400,
                    bubble_full_width=False
                )

                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask about ML security...",
                        show_label=False,
                        scale=4
                    )
                    chat_submit = gr.Button("Send", variant="primary", scale=1)
                    chat_clear = gr.Button("Clear", scale=1)

                chat_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    show_label=True
                )

                with gr.Accordion("🧠 Agent Reasoning Steps", open=False):
                    chat_reasoning = gr.HTML()

                # Connect handlers
                chat_submit.click(
                    fn=handle_chat,
                    inputs=[chat_input, chat_interface],
                    outputs=[chat_interface, chat_reasoning, chat_status]
                ).then(
                    lambda: "",
                    outputs=[chat_input]
                )

                chat_input.submit(
                    fn=handle_chat,
                    inputs=[chat_input, chat_interface],
                    outputs=[chat_interface, chat_reasoning, chat_status]
                ).then(
                    lambda: "",
                    outputs=[chat_input]
                )

                chat_clear.click(
                    lambda: ([], "", ""),
                    outputs=[chat_interface, chat_reasoning, chat_status]
                ).then(
                    lambda: AgentState.get_agent().clear_history() if AgentState.get_agent() else None
                )

        # Footer
        gr.Markdown("""
        ---
        ### About MLPatrol

        MLPatrol is an AI-powered security agent built for the MCP 1st Birthday Hackathon.
        It helps secure ML systems through intelligent CVE monitoring, dataset analysis, and code generation.

        **Powered by:** LangChain, Claude Sonnet 4, Gradio 6

        **⚠️ Security Notice:** Always review generated code before execution. This tool provides
        security analysis but should be used alongside manual security reviews and testing.
        """)

    return interface

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Launch the MLPatrol Gradio application."""
    try:
        logger.info("Starting MLPatrol application...")

        # Initialize agent in background
        logger.info("Initializing agent...")
        AgentState.get_agent()

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
