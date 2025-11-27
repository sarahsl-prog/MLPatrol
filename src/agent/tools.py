"""Tool definitions and wrappers for the MLPatrol agent.

This module provides LangChain-compatible tool wrappers for:
- CVE database searches
- Web searches for security information
- Dataset analysis
- Security code generation
- HuggingFace dataset searches
"""

import logging
import re
import os
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator
import pandas as pd

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled via dataset tests
    np = None

from src.dataset.bias_analyzer import analyze_bias
from src.dataset.poisoning_detector import detect_poisoning
from src.dataset.statistical_tests import (
    detect_outliers_zscore,
    dataset_summary,
    ks_test_between_columns,
    chi2_of_categorical,
)
from src.security.code_generator import (
    build_cve_security_script,
    build_general_security_script,
)
from src.security.cve_monitor import CVEMonitor

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models for Tool Inputs/Outputs
# ============================================================================


@dataclass
class CVEResult:
    """A CVE (Common Vulnerabilities and Exposures) result.

    Attributes:
        cve_id: The CVE identifier (e.g., CVE-2024-12345)
        description: Description of the vulnerability
        cvss_score: CVSS score (0-10)
        severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
        affected_versions: List of affected library versions
        published_date: When the CVE was published
        references: List of reference URLs
        library: The affected library name
    """

    cve_id: str
    description: str
    cvss_score: float
    severity: str
    affected_versions: List[str]
    published_date: str
    references: List[str]
    library: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "cve_id": self.cve_id,
            "description": self.description,
            "cvss_score": self.cvss_score,
            "severity": self.severity,
            "affected_versions": self.affected_versions,
            "published_date": self.published_date,
            "references": self.references,
            "library": self.library,
        }


@dataclass
class DatasetAnalysisResult:
    """Results from dataset security analysis.

    Attributes:
        num_rows: Number of rows in dataset
        num_features: Number of features/columns
        outliers: List of outlier indices
        outlier_count: Number of statistical outliers
        class_distribution: Distribution of classes/labels
        suspected_poisoning: Boolean indicating suspected poisoning
        poisoning_confidence: Confidence score for poisoning detection (0-1)
        bias_score: Bias assessment score (0-1, higher = more bias)
        quality_score: Overall data quality score (0-10)
        recommendations: List of recommended actions
    """

    num_rows: int
    num_features: int
    outliers: List[int]
    outlier_count: int
    class_distribution: Dict[str, float]
    suspected_poisoning: bool
    poisoning_confidence: float
    bias_score: float
    quality_score: float
    recommendations: List[str]
    # Optional statistical test summaries (KS tests, chi2 for categorical)
    stat_tests: Dict[str, Any] = None
    # Small dataset summary (numeric/categorical columns)
    dataset_summary: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_rows": self.num_rows,
            "num_features": self.num_features,
            "outlier_count": self.outlier_count,
            # Provide both a truncated sample and a fuller outliers list (truncated to reasonable size)
            "outliers_sample": self.outliers[:10],  # Only first 10 for brevity
            "outliers": self.outliers[:200],
            "class_distribution": self.class_distribution,
            "suspected_poisoning": self.suspected_poisoning,
            "poisoning_confidence": self.poisoning_confidence,
            "bias_score": self.bias_score,
            "quality_score": self.quality_score,
            "recommendations": self.recommendations,
            "stat_tests": self.stat_tests or {},
            "dataset_summary": self.dataset_summary or {},
        }


# ============================================================================
# Input Schemas for LangChain Tools
# ============================================================================


class CVESearchInput(BaseModel):
    """Input schema for CVE search tool."""

    library: str = Field(
        description="Name of the ML library to search (e.g., 'numpy', 'pytorch', 'tensorflow', 'scikit-learn')"
    )
    days_back: int = Field(
        default=90,
        description="Number of days to look back for CVEs (default: 90 days)",
        ge=1,
        le=3650,
    )

    @field_validator("library")
    @classmethod
    def validate_library_name(cls, v):
        """Validate library name format."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Library name must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v.lower()


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""

    query: str = Field(
        description="Search query for security information, papers, or blog posts"
    )
    max_results: int = Field(
        default=5, description="Maximum number of results to return", ge=1, le=10
    )


class DatasetAnalysisInput(BaseModel):
    """Input schema for dataset analysis tool."""

    data_path: Optional[str] = Field(
        default=None,
        description="Path to CSV file to analyze (optional if data_json is provided)",
    )
    data_json: Optional[str] = Field(
        default=None,
        description="JSON string of dataset data (optional if data_path is provided)",
    )

    @field_validator("data_json")
    @classmethod
    def validate_data_json(cls, v, info):
        """Ensure at least one data source is provided."""
        if v is None and info.data.get("data_path") is None:
            raise ValueError("Either data_path or data_json must be provided")
        return v


class CodeGenerationInput(BaseModel):
    """Input schema for security code generation tool."""

    purpose: str = Field(
        description="Purpose of the security script (e.g., 'check CVE vulnerability', 'validate data integrity')"
    )
    library: str = Field(description="Target library for the security check")
    cve_id: Optional[str] = Field(
        default=None, description="CVE ID if generating CVE-specific validation code"
    )
    affected_versions: Optional[List[str]] = Field(
        default=None, description="List of affected versions to check against"
    )

    @field_validator("cve_id")
    @classmethod
    def validate_cve_id(cls, v):
        """Validate CVE ID format if provided."""
        if v and not re.match(r"CVE-\d{4}-\d{4,7}", v):
            raise ValueError("CVE ID must match format CVE-YYYY-NNNNN")
        return v


class HuggingFaceSearchInput(BaseModel):
    """Input schema for HuggingFace search tool."""

    query: str = Field(description="Search query for HuggingFace datasets or models")
    search_type: str = Field(
        default="datasets", description="Type of search: 'datasets' or 'models'"
    )

    @field_validator("search_type")
    @classmethod
    def validate_search_type(cls, v):
        """Validate search type."""
        if v not in ["datasets", "models"]:
            raise ValueError("search_type must be 'datasets' or 'models'")
        return v


# ============================================================================
# Tool Implementation Functions
# ============================================================================


def cve_search_impl(library: str, days_back: int = 90) -> str:
    """Search the National Vulnerability Database for CVEs affecting a library.

    This function queries the NVD API for recent CVEs affecting the specified
    ML library. It includes proper error handling and rate limiting.

    Args:
        library: Name of the library to search (e.g., 'numpy', 'pytorch')
        days_back: Number of days to look back (default: 90)

    Returns:
        JSON string containing list of CVE results

    Raises:
        requests.RequestException: If the API request fails

    Example:
        >>> result = cve_search_impl("numpy", 30)
        >>> cves = json.loads(result)
        >>> print(f"Found {len(cves)} CVEs")
    """
    try:
        logger.info(f"Searching for CVEs in {library} from last {days_back} days")
        monitor = CVEMonitor(api_key=os.getenv("NVD_API_KEY"))
        search_result = monitor.search_recent(library, days_back)

        payload = {
            "status": "success",
            **search_result,
        }

        logger.info("Found %s CVEs for %s", payload["cve_count"], library)
        return json.dumps(payload, indent=2)

    except requests.Timeout:
        logger.warning("CVE search timed out for %s", library)
        return json.dumps(
            {
                "status": "timeout",
                "message": f"CVE search timed out for {library}. Try again or check with fewer days_back.",
                "library": library,
            }
        )
    except requests.RequestException as e:
        logger.error("CVE search failed: %s", e)
        return json.dumps(
            {
                "status": "error",
                "message": f"CVE database temporarily unavailable. Recommendation: Check {library} advisories at https://nvd.nist.gov",
                "library": library,
                "error": str(e),
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in cve_search: {e}", exc_info=True)
        return json.dumps(
            {
                "status": "error",
                "message": "Unexpected error during CVE search",
                "error": str(e),
            }
        )


def _classify_search_query(query: str) -> str:
    """Classify query type for smart routing.

    Args:
        query: Search query string

    Returns:
        Query type: 'cve_monitoring' or 'general_security'
    """
    cve_patterns = [
        r"CVE-\d{4}-\d+",  # CVE-2024-1234
        r"vulnerability.*\d{4}",  # "vulnerabilities 2024"
        r"latest.*CVE",  # "latest CVE in PyTorch"
        r"recent.*vulnerability",  # "recent vulnerabilities"
        r"security.*advisory",  # "security advisory"
        r"breaking.*threat",  # "breaking threat"
    ]

    query_lower = query.lower()
    for pattern in cve_patterns:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return "cve_monitoring"

    return "general_security"


def _tavily_search(query: str, max_results: int = 5) -> str:
    """Search using Tavily AI API.

    Tavily is optimized for AI agents and LLM consumption, providing
    clean, structured results with built-in content extraction.

    Args:
        query: Search query string
        max_results: Maximum number of results

    Returns:
        JSON string with search results
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.error("TAVILY_API_KEY not set")
            return json.dumps(
                {
                    "status": "error",
                    "provider": "tavily",
                    "message": "Tavily API key not configured. Add TAVILY_API_KEY to .env file.",
                }
            )
        api_key = os.getenv("TAVILY_API_KEY", "")
        # Check for missing or placeholder API key
        if not api_key or api_key == "your_tavily_api_key_here":
            logger.error("TAVILY_API_KEY not configured")
            return json.dumps({
                "status": "error",
                "provider": "tavily",
                "message": "Tavily API key not configured. Get your API key at https://tavily.com and add TAVILY_API_KEY to .env file."
            })
        if len(api_key) < 20:
            logger.error("TAVILY_API_KEY appears invalid (too short)")
            return json.dumps({
                "status": "error",
                "provider": "tavily",
                "message": "Tavily API key appears invalid. Please check your TAVILY_API_KEY in .env file."
            })

        # Get configuration
        search_depth = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")

        # Sanitize query
        query_clean = re.sub(r"[^\w\s\-\.\,\?\!]", "", query)[:300]

        logger.info(f"Tavily search: {query_clean} (depth: {search_depth})")

        # Use Tavily via LangChain integration if available
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            tool = TavilySearchResults(
                max_results=max_results,
                search_depth=search_depth,
                include_answer=True,
                include_raw_content=False,
                include_images=False,
                api_key=api_key,
            )

            start_time = time.time()
            results = tool.invoke({"query": query_clean})
            duration_ms = (time.time() - start_time) * 1000

            # Format results
            formatted_results = {
                "status": "success",
                "provider": "tavily",
                "query": query_clean,
                "duration_ms": round(duration_ms, 2),
                "results": results if isinstance(results, list) else [results],
                "count": len(results) if isinstance(results, list) else 1,
            }

            logger.info(f"Tavily search completed in {duration_ms:.0f}ms")
            return json.dumps(formatted_results, indent=2)

        except ImportError:
            # Fallback to direct API call if LangChain tool not available
            logger.warning("langchain-community not installed, using direct API")

            headers = {"Content-Type": "application/json"}

            payload = {
                "api_key": api_key,
                "query": query_clean,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False,
            }

            start_time = time.time()
            response = requests.post(
                "https://api.tavily.com/search",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            duration_ms = (time.time() - start_time) * 1000

            data = response.json()
            data["status"] = "success"
            data["provider"] = "tavily"
            data["duration_ms"] = round(duration_ms, 2)

            logger.info(f"Tavily search completed in {duration_ms:.0f}ms")
            return json.dumps(data, indent=2)

    except requests.exceptions.Timeout:
        logger.error("Tavily API timeout")
        return json.dumps(
            {
                "status": "error",
                "provider": "tavily",
                "message": "Search request timed out. Try again later.",
            }
        )
    except requests.exceptions.HTTPError as e:
        logger.error(f"Tavily API HTTP error: {e.response.status_code}")
        return json.dumps(
            {
                "status": "error",
                "provider": "tavily",
                "message": f"API error: {e.response.status_code}",
            }
        )
    except Exception as e:
        logger.error(f"Tavily search failed: {e}", exc_info=True)
        return json.dumps(
            {
                "status": "error",
                "provider": "tavily",
                "message": f"Search failed: {str(e)}",
            }
        )


def _brave_search(query: str, max_results: int = 5) -> str:
    """Search using Brave Search API.

    Brave provides privacy-focused search with an independent index,
    excellent for breaking news, CVE monitoring, and technical content.

    Args:
        query: Search query string
        max_results: Maximum number of results

    Returns:
        JSON string with search results
    """
    try:
        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key:
            logger.error("BRAVE_API_KEY not set")
            return json.dumps(
                {
                    "status": "error",
                    "provider": "brave",
                    "message": "Brave API key not configured. Add BRAVE_API_KEY to .env file.",
                }
            )
        api_key = os.getenv("BRAVE_API_KEY", "")
        # Check for missing or placeholder API key
        if not api_key or api_key == "your_brave_api_key_here":
            logger.error("BRAVE_API_KEY not configured")
            return json.dumps({
                "status": "error",
                "provider": "brave",
                "message": "Brave API key not configured. Get your API key at https://brave.com/search/api/ and add BRAVE_API_KEY to .env file."
            })
        if len(api_key) < 20:
            logger.error("BRAVE_API_KEY appears invalid (too short)")
            return json.dumps({
                "status": "error",
                "provider": "brave",
                "message": "Brave API key appears invalid. Please check your BRAVE_API_KEY in .env file."
            })

        # Get configuration
        freshness = os.getenv("BRAVE_SEARCH_FRESHNESS", "pw")  # past week

        # Sanitize query
        query_clean = re.sub(r"[^\w\s\-\.\,\?\!]", "", query)[:300]

        logger.info(f"Brave search: {query_clean} (freshness: {freshness})")

        headers = {"Accept": "application/json", "X-Subscription-Token": api_key}

        params = {
            "q": query_clean,
            "count": max_results,
            "search_lang": "en",
            "safesearch": "moderate",
            "freshness": freshness,
        }

        start_time = time.time()
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=30.0,
        )
        response.raise_for_status()
        duration_ms = (time.time() - start_time) * 1000

        data = response.json()

        # Extract web results
        web_results = data.get("web", {}).get("results", [])

        # Format results
        formatted_results = {
            "status": "success",
            "provider": "brave",
            "query": query_clean,
            "duration_ms": round(duration_ms, 2),
            "results": [
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("description", ""),
                    "published_date": result.get("age", ""),
                }
                for result in web_results[:max_results]
            ],
            "count": len(web_results),
        }

        logger.info(
            f"Brave search completed in {duration_ms:.0f}ms: {len(web_results)} results"
        )
        return json.dumps(formatted_results, indent=2)

    except requests.exceptions.Timeout:
        logger.error("Brave API timeout")
        return json.dumps(
            {
                "status": "error",
                "provider": "brave",
                "message": "Search request timed out. Try again later.",
            }
        )
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        logger.error(f"Brave API HTTP error: {status_code}")

        error_messages = {
            401: "Invalid API key. Check BRAVE_API_KEY in .env file.",
            403: "API key does not have permission.",
            429: "Rate limit exceeded. Wait a moment and try again.",
            500: "Brave API server error. Try again later.",
        }

        return json.dumps(
            {
                "status": "error",
                "provider": "brave",
                "message": error_messages.get(status_code, f"API error: {status_code}"),
            }
        )
    except Exception as e:
        logger.error(f"Brave search failed: {e}", exc_info=True)
        return json.dumps(
            {
                "status": "error",
                "provider": "brave",
                "message": f"Search failed: {str(e)}",
            }
        )


def web_search_impl(query: str, max_results: int = 5) -> str:
    """Search the web for security information using Tavily and/or Brave.

    This function performs intelligent web search routing between Tavily AI
    (AI-optimized) and Brave Search (privacy-focused, breaking news) based on
    query type and configuration.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        JSON string containing search results

    Example:
        >>> result = web_search_impl("pytorch security best practices")
        >>> results = json.loads(result)
    """
    try:
        # Check if web search is enabled
        if os.getenv("ENABLE_WEB_SEARCH", "true").lower() != "true":
            return json.dumps(
                {
                    "status": "disabled",
                    "message": "Web search is disabled. Enable it in .env with ENABLE_WEB_SEARCH=true",
                }
            )

        # Check which providers are enabled
        use_tavily = os.getenv("USE_TAVILY_SEARCH", "false").lower() == "true"
        use_brave = os.getenv("USE_BRAVE_SEARCH", "false").lower() == "true"

        if not use_tavily and not use_brave:
            return json.dumps(
                {
                    "status": "disabled",
                    "message": "No web search providers enabled. Set USE_TAVILY_SEARCH or USE_BRAVE_SEARCH to true in .env",
                }
            )

        # Classify query type for routing
        query_type = _classify_search_query(query)

        # Determine which provider to use based on configuration and query type
        if query_type == "cve_monitoring":
            # For CVE monitoring, prefer the configured CVE provider
            preferred = os.getenv("WEB_SEARCH_ROUTE_CVE_TO", "brave").lower()
        else:
            # For general security, prefer the configured general provider
            preferred = os.getenv("WEB_SEARCH_ROUTE_GENERAL_TO", "tavily").lower()

        # Route to preferred provider if enabled, otherwise fallback
        if preferred == "tavily" and use_tavily:
            logger.info(f"Routing to Tavily (query type: {query_type})")
            return _tavily_search(query, max_results)
        elif preferred == "brave" and use_brave:
            logger.info(f"Routing to Brave (query type: {query_type})")
            return _brave_search(query, max_results)
        elif use_tavily:
            logger.info(f"Fallback to Tavily (preferred {preferred} not available)")
            return _tavily_search(query, max_results)
        elif use_brave:
            logger.info(f"Fallback to Brave (preferred {preferred} not available)")
            return _brave_search(query, max_results)
        else:
            return json.dumps(
                {"status": "error", "message": "No web search provider available"}
            )

    except Exception as e:
        logger.error(f"Web search routing failed: {e}", exc_info=True)
        return json.dumps(
            {"status": "error", "message": "Web search unavailable", "error": str(e)}
        )


def analyze_dataset_impl(
    data_path: Optional[str] = None, data_json: Optional[str] = None
) -> str:
    """Analyze a dataset for security issues (poisoning, bias, anomalies).

    This function performs comprehensive statistical analysis on datasets to detect:
    - Statistical outliers (Z-score based)
    - Class imbalance and bias
    - Suspected poisoning patterns
    - Data quality issues

    Args:
        data_path: Path to CSV file to analyze
        data_json: JSON string representation of dataset

    Returns:
        JSON string containing analysis results

    Example:
        >>> result = analyze_dataset_impl(data_path="/path/to/data.csv")
        >>> analysis = json.loads(result)
        >>> print(f"Quality score: {analysis['quality_score']}")
    """
    try:
        logger.info("Starting dataset security analysis")

        # Load data
        if data_path:
            try:
                df = pd.read_csv(data_path)
            except Exception as e:
                return json.dumps(
                    {"status": "error", "message": f"Failed to load CSV file: {e}"}
                )
        elif data_json:
            try:
                data = json.loads(data_json)
                df = pd.DataFrame(data)
            except Exception as e:
                return json.dumps(
                    {"status": "error", "message": f"Failed to parse JSON data: {e}"}
                )
        else:
            return json.dumps({"status": "error", "message": "No data source provided"})

        if np is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Numpy is not installed. Cannot perform dataset analysis.",
                }
            )

        # Validate dataset
        num_rows, num_features = df.shape
        if num_rows < 10:
            return json.dumps({
                "status": "error",
                "message": f"Dataset too small for analysis. Need at least 10 rows, got {num_rows}"
            })
        if num_features < 1:
            return json.dumps({
                "status": "error",
                "message": "Dataset has no features to analyze"
            })

        # Use poisoning detector's outliers, but augment with z-score based detection
        outliers = poisoning_report.outlier_indices or []
        try:
            z_outliers = detect_outliers_zscore(df, threshold=3.0)
        except Exception:
            z_outliers = []
        # merge unique indices
        combined_outliers = sorted(set(outliers) | set(z_outliers))
        outlier_count = len(combined_outliers)
        outlier_ratio = outlier_count / max(len(df), 1)

        class_distribution = bias_report.class_distribution
        bias_score = bias_report.imbalance_score
        suspected_poisoning = poisoning_report.suspected_poisoning or (
            outlier_ratio > 0.05
        )
        poisoning_confidence = max(
            poisoning_report.confidence, min(outlier_ratio * 10, 1.0)
        )
        logger.info(f"Analyzing dataset: {num_rows} rows, {num_features} features")

        # Run bias analysis with error handling
        try:
            bias_report = analyze_bias(df)
            class_distribution = bias_report.class_distribution
            bias_score = bias_report.imbalance_score
        except Exception as e:
            logger.warning(f"Bias analysis failed: {e}", exc_info=True)
            # Use fallback values
            class_distribution = {}
            bias_score = 0.0
            bias_report = type('BiasReport', (), {
                'class_distribution': {},
                'imbalance_score': 0.0,
                'warnings': [f"Bias analysis failed: {str(e)}"]
            })()

        # Run poisoning detection with error handling
        try:
            poisoning_report = detect_poisoning(df)
            outliers = poisoning_report.outlier_indices
            outlier_count = len(outliers)
            outlier_ratio = poisoning_report.outlier_ratio
            suspected_poisoning = poisoning_report.suspected_poisoning
            poisoning_confidence = poisoning_report.confidence
        except Exception as e:
            logger.warning(f"Poisoning detection failed: {e}", exc_info=True)
            # Use fallback values
            outliers = []
            outlier_count = 0
            outlier_ratio = 0.0
            suspected_poisoning = False
            poisoning_confidence = 0.0
            if not hasattr(bias_report, 'warnings'):
                bias_report.warnings = []
            bias_report.warnings.append(f"Poisoning detection failed: {str(e)}")

        # Calculate quality score (0-10)
        quality_score = 10.0
        quality_score -= outlier_ratio * 20  # Penalize outliers
        quality_score -= bias_score * 3  # Penalize bias
        if suspected_poisoning:
            quality_score -= poisoning_confidence * 2
        quality_score = max(0.0, min(10.0, quality_score))

        # Generate recommendations
        recommendations = []
        if outlier_count > 0:
            recommendations.append(
                f"Manually review {outlier_count} statistical outliers"
            )
        if bias_score > 0.3:
            recommendations.append(
                "Address class imbalance through resampling or class weighting"
            )
        if suspected_poisoning:
            recommendations.append(
                "Implement robust training techniques (label smoothing, outlier removal)"
            )
        if quality_score < 7.0:
            recommendations.append(
                "Consider data cleaning and validation before training"
            )
        recommendations.extend(bias_report.warnings)

        # Statistical tests: KS between numeric columns (pairwise up to 10 columns), chi2 for categorical
        stat_tests: Dict[str, Any] = {}
        try:
            summary = dataset_summary(df)
            stat_tests["dataset_summary"] = summary

            numeric_cols = summary.get("numeric_columns", [])[:10]
            ks_results: Dict[str, Dict[str, float]] = {}
            # pairwise KS: compare each column to the first numeric column
            if len(numeric_cols) >= 2:
                ref = df[numeric_cols[0]]
                for col in numeric_cols[1:]:
                    stat, p = ks_test_between_columns(ref, df[col])
                    ks_results[f"{numeric_cols[0]}_vs_{col}"] = {
                        "stat": stat,
                        "pvalue": p,
                    }
            stat_tests["ks_tests"] = ks_results

            categorical_cols = summary.get("categorical_columns", [])
            chi2_results: Dict[str, Dict[str, float]] = {}
            for col in categorical_cols:
                try:
                    chi2, p = chi2_of_categorical(df[col])
                    chi2_results[col] = {"chi2": chi2, "pvalue": p}
                except Exception:
                    chi2_results[col] = {"chi2": 0.0, "pvalue": 1.0}
            stat_tests["chi2_tests"] = chi2_results
        except Exception as e:
            logger.debug(f"Statistical tests failed: {e}")
            stat_tests = {}

        result = DatasetAnalysisResult(
            num_rows=num_rows,
            num_features=num_features,
            outliers=combined_outliers,
            outlier_count=outlier_count,
            class_distribution=class_distribution,
            suspected_poisoning=suspected_poisoning,
            poisoning_confidence=poisoning_confidence,
            bias_score=bias_score,
            quality_score=quality_score,
            recommendations=recommendations,
            stat_tests=stat_tests,
            dataset_summary=summary if "summary" in locals() else {},
        )

        logger.info(f"Dataset analysis complete. Quality score: {quality_score:.1f}/10")

        return json.dumps({"status": "success", **result.to_dict()}, indent=2)

    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}", exc_info=True)
        return json.dumps(
            {"status": "error", "message": "Dataset analysis failed", "error": str(e)}
        )


def generate_security_code_impl(
    purpose: str,
    library: str,
    cve_id: Optional[str] = None,
    affected_versions: Optional[List[str]] = None,
) -> str:
    """Generate Python security validation code.

    Creates production-ready Python scripts for security validation tasks such as:
    - CVE vulnerability checks
    - Version validation
    - Data integrity verification

    Args:
        purpose: Purpose of the security script
        library: Target library for validation
        cve_id: Optional CVE ID for CVE-specific checks
        affected_versions: Optional list of affected versions

    Returns:
        JSON string containing generated Python code

    Example:
        >>> result = generate_security_code_impl(
        ...     purpose="check CVE vulnerability",
        ...     library="numpy",
        ...     cve_id="CVE-2021-34141"
        ... )
    """
    try:
        logger.info(f"Generating security code for {library}: {purpose}")

        if cve_id or "cve" in (purpose or "").lower():
            code = build_cve_security_script(
                purpose, library, cve_id, affected_versions
            )
        else:
            code = build_general_security_script(purpose, library, affected_versions)

        sanitized_name = library.replace("-", "_")

        result = {
            "status": "success",
            "purpose": purpose,
            "library": library,
            "cve_id": cve_id,
            "code": code,
            "filename": f"mlpatrol_check_{sanitized_name}.py",
            "usage": f"python mlpatrol_check_{sanitized_name}.py",
        }

        logger.info("Security code generated successfully")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Code generation failed: {e}", exc_info=True)
        return json.dumps(
            {"status": "error", "message": "Code generation failed", "error": str(e)}
        )


def huggingface_search_impl(query: str, search_type: str = "datasets") -> str:
    """Search HuggingFace for datasets or models.

    Args:
        query: Search query
        search_type: Type of search ('datasets' or 'models')

    Returns:
        JSON string containing search results
    """
    try:
        logger.info(f"Searching HuggingFace {search_type}: {query}")

        # In production, use HuggingFace Hub API
        # from huggingface_hub import list_datasets, list_models

        base_url = f"https://huggingface.co/{search_type}"

        result = {
            "status": "success",
            "query": query,
            "search_type": search_type,
            "results": [
                {
                    "name": f"Example {search_type} result for: {query}",
                    "url": f"{base_url}?search={query}",
                    "note": "In production, integrate with HuggingFace Hub API for real results",
                }
            ],
        }

        logger.info(f"HuggingFace search completed")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"HuggingFace search failed: {e}", exc_info=True)
        return json.dumps(
            {"status": "error", "message": "HuggingFace search failed", "error": str(e)}
        )


# ============================================================================
# LangChain Tool Definitions
# ============================================================================


def create_mlpatrol_tools() -> List[StructuredTool]:
    """Create all MLPatrol tools for the agent.

    Returns:
        List of LangChain Tool objects ready for use by the agent

    Example:
        >>> tools = create_mlpatrol_tools()
        >>> for tool in tools:
        ...     print(f"{tool.name}: {tool.description}")
    """
    tools = [
        StructuredTool.from_function(
            func=cve_search_impl,
            name="cve_search",
            description="Search the National Vulnerability Database (NVD) for CVEs affecting ML libraries. "
            "Use this when users ask about vulnerabilities, security issues, or CVEs in libraries like "
            "numpy, pytorch, tensorflow, scikit-learn, etc. Returns CVE IDs, CVSS scores, and details.",
            args_schema=CVESearchInput,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=web_search_impl,
            name="web_search",
            description="Search the web for ML security information, research papers, blog posts, and best practices. "
            "Use this for general security questions, exploitation techniques, or recent security trends. "
            "Returns relevant web resources and summaries.",
            args_schema=WebSearchInput,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=analyze_dataset_impl,
            name="analyze_dataset",
            description="Analyze a dataset for security issues including poisoning attempts, statistical outliers, "
            "bias, and data quality problems. Use when users upload datasets or ask about data security. "
            "Returns statistical analysis, outlier detection, and quality scores.",
            args_schema=DatasetAnalysisInput,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=generate_security_code_impl,
            name="generate_security_code",
            description="Generate Python security validation scripts for checking vulnerabilities, validating versions, "
            "or performing security checks. Use when users need code to validate their environment. "
            "Returns production-ready Python code with error handling.",
            args_schema=CodeGenerationInput,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=huggingface_search_impl,
            name="huggingface_search",
            description="Search HuggingFace Hub for datasets and models. Use when users ask about finding datasets, "
            "models, or resources on HuggingFace. Returns links and metadata.",
            args_schema=HuggingFaceSearchInput,
            return_direct=False,
        ),
    ]

    logger.info(f"Created {len(tools)} MLPatrol tools")
    return tools


# ============================================================================
# Tool Result Parsers
# ============================================================================


def parse_cve_results(tool_output: str) -> List[CVEResult]:
    """Parse CVE search tool output into structured results.

    Args:
        tool_output: JSON string from cve_search tool

    Returns:
        List of CVEResult objects
    """
    try:
        if not tool_output or not tool_output.strip():
            logger.warning("Empty tool output for CVE search")
            return []

        data = json.loads(tool_output)

        if not isinstance(data, dict):
            logger.error(f"Invalid CVE tool output format: expected dict, got {type(data).__name__}")
            return []

        if data.get("status") != "success":
            error_msg = data.get("message", "Unknown error")
            logger.warning(f"CVE search returned non-success status: {error_msg}")
            return []

        cves_data = data.get("cves", [])
        if not isinstance(cves_data, list):
            logger.error(f"Invalid CVEs format: expected list, got {type(cves_data).__name__}")
            return []

        cves = []
        for i, cve_dict in enumerate(cves_data):
            try:
                if not isinstance(cve_dict, dict):
                    logger.warning(f"Skipping CVE entry {i}: not a dict")
                    continue

                cve = CVEResult(
                    cve_id=cve_dict.get("cve_id", "UNKNOWN"),
                    description=cve_dict.get("description", ""),
                    cvss_score=float(cve_dict.get("cvss_score", 0.0)),
                    severity=cve_dict.get("severity", "UNKNOWN"),
                    affected_versions=cve_dict.get("affected_versions", []),
                    published_date=cve_dict.get("published_date", ""),
                    references=cve_dict.get("references", []),
                    library=cve_dict.get("library", ""),
                )
                cves.append(cve)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse CVE entry {i}: {e}")
                continue

        return cves

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse CVE results as JSON: {e}")
        logger.debug(f"Tool output was: {tool_output[:200]}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error parsing CVE results: {e}", exc_info=True)
        return []


def parse_dataset_analysis(tool_output: str) -> Optional[DatasetAnalysisResult]:
    """Parse dataset analysis tool output into structured result.

    Args:
        tool_output: JSON string from analyze_dataset tool

    Returns:
        DatasetAnalysisResult object or None if parsing fails
    """
    try:
        if not tool_output or not tool_output.strip():
            logger.warning("Empty tool output for dataset analysis")
            return None

        data = json.loads(tool_output)

        if not isinstance(data, dict):
            logger.error(f"Invalid dataset analysis output format: expected dict, got {type(data).__name__}")
            return None

        if data.get("status") != "success":
            error_msg = data.get("message", "Unknown error")
            logger.warning(f"Dataset analysis returned non-success status: {error_msg}")
            return None

        # Ensure class_distribution is a dict with proper types
        class_dist = data.get("class_distribution", {})
        if not isinstance(class_dist, dict):
            logger.warning(f"Invalid class_distribution type: {type(class_dist).__name__}, using empty dict")
            class_dist = {}

        return DatasetAnalysisResult(
            num_rows=int(data.get("num_rows", 0)),
            num_features=int(data.get("num_features", 0)),
            outliers=data.get("outliers", data.get("outliers_sample", [])),
            outlier_count=int(data.get("outlier_count", 0)),
            class_distribution=class_dist,
            suspected_poisoning=bool(data.get("suspected_poisoning", False)),
            poisoning_confidence=float(data.get("poisoning_confidence", 0.0)),
            bias_score=float(data.get("bias_score", 0.0)),
            quality_score=float(data.get("quality_score", 0.0)),
            recommendations=data.get("recommendations", []),
            stat_tests=data.get("stat_tests", {}),
            dataset_summary=data.get("dataset_summary", {}),
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse dataset analysis as JSON: {e}")
        logger.debug(f"Tool output was: {tool_output[:200]}")
        return None
    except (ValueError, TypeError) as e:
        logger.error(f"Type conversion error in dataset analysis: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing dataset analysis: {e}", exc_info=True)
        return None
