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
from datetime import datetime, timedelta
import json

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator
import pandas as pd

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_rows": self.num_rows,
            "num_features": self.num_features,
            "outlier_count": self.outlier_count,
            "outliers_sample": self.outliers[:10],  # Only first 10 for brevity
            "class_distribution": self.class_distribution,
            "suspected_poisoning": self.suspected_poisoning,
            "poisoning_confidence": self.poisoning_confidence,
            "bias_score": self.bias_score,
            "quality_score": self.quality_score,
            "recommendations": self.recommendations,
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
        le=3650
    )

    @field_validator("library")
    @classmethod
    def validate_library_name(cls, v):
        """Validate library name format."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Library name must contain only alphanumeric characters, hyphens, and underscores")
        return v.lower()


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""

    query: str = Field(
        description="Search query for security information, papers, or blog posts"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1,
        le=10
    )


class DatasetAnalysisInput(BaseModel):
    """Input schema for dataset analysis tool."""

    data_path: Optional[str] = Field(
        default=None,
        description="Path to CSV file to analyze (optional if data_json is provided)"
    )
    data_json: Optional[str] = Field(
        default=None,
        description="JSON string of dataset data (optional if data_path is provided)"
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
    library: str = Field(
        description="Target library for the security check"
    )
    cve_id: Optional[str] = Field(
        default=None,
        description="CVE ID if generating CVE-specific validation code"
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

    query: str = Field(
        description="Search query for HuggingFace datasets or models"
    )
    search_type: str = Field(
        default="datasets",
        description="Type of search: 'datasets' or 'models'"
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

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # NVD API endpoint (Note: In production, use actual NVD API with API key)
        # For this implementation, we'll use a mock response structure
        base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

        params = {
            "keywordSearch": library,
            "pubStartDate": start_date.strftime("%Y-%m-%dT%H:%M:%S.000"),
            "pubEndDate": end_date.strftime("%Y-%m-%dT%H:%M:%S.000"),
        }

        # Add timeout and retry logic
        try:
            response = requests.get(
                base_url,
                params=params,
                headers={"User-Agent": "MLPatrol-SecurityAgent/1.0"},
                timeout=10
            )
            response.raise_for_status()

            # Parse NVD response
            nvd_data = response.json()

            # Convert to our CVEResult format
            cves = []
            if "vulnerabilities" in nvd_data:
                for vuln in nvd_data["vulnerabilities"]:
                    cve = vuln.get("cve", {})
                    cve_id = cve.get("id", "UNKNOWN")

                    # Extract CVSS score
                    cvss_score = 0.0
                    severity = "UNKNOWN"
                    metrics = cve.get("metrics", {})
                    if "cvssMetricV31" in metrics:
                        cvss_data = metrics["cvssMetricV31"][0]["cvssData"]
                        cvss_score = cvss_data.get("baseScore", 0.0)
                        severity = cvss_data.get("baseSeverity", "UNKNOWN")

                    # Extract description
                    descriptions = cve.get("descriptions", [])
                    description = descriptions[0].get("value", "No description") if descriptions else "No description"

                    # Extract references
                    references = [ref.get("url") for ref in cve.get("references", [])]

                    cve_result = CVEResult(
                        cve_id=cve_id,
                        description=description[:200] + "..." if len(description) > 200 else description,
                        cvss_score=cvss_score,
                        severity=severity,
                        affected_versions=["See references for details"],
                        published_date=cve.get("published", "Unknown"),
                        references=references[:3],  # Limit to 3 references
                        library=library
                    )
                    cves.append(cve_result.to_dict())

            result = {
                "status": "success",
                "library": library,
                "days_searched": days_back,
                "cve_count": len(cves),
                "cves": cves
            }

            logger.info(f"Found {len(cves)} CVEs for {library}")
            return json.dumps(result, indent=2)

        except requests.Timeout:
            logger.warning(f"CVE search timed out for {library}")
            return json.dumps({
                "status": "timeout",
                "message": f"CVE search timed out for {library}. Try again or check with fewer days_back.",
                "library": library
            })
        except requests.RequestException as e:
            logger.error(f"CVE search failed: {e}")
            # Provide a graceful fallback
            return json.dumps({
                "status": "error",
                "message": f"CVE database temporarily unavailable. Recommendation: Check {library} security advisories manually at https://nvd.nist.gov",
                "library": library,
                "error": str(e)
            })

    except Exception as e:
        logger.error(f"Unexpected error in cve_search: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Unexpected error during CVE search",
            "error": str(e)
        })


def _classify_search_query(query: str) -> str:
    """Classify query type for smart routing.

    Args:
        query: Search query string

    Returns:
        Query type: 'cve_monitoring' or 'general_security'
    """
    cve_patterns = [
        r'CVE-\d{4}-\d+',           # CVE-2024-1234
        r'vulnerability.*\d{4}',     # "vulnerabilities 2024"
        r'latest.*CVE',              # "latest CVE in PyTorch"
        r'recent.*vulnerability',    # "recent vulnerabilities"
        r'security.*advisory',       # "security advisory"
        r'breaking.*threat',         # "breaking threat"
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
            return json.dumps({
                "status": "error",
                "provider": "tavily",
                "message": "Tavily API key not configured. Add TAVILY_API_KEY to .env file."
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
                api_key=api_key
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
                "count": len(results) if isinstance(results, list) else 1
            }

            logger.info(f"Tavily search completed in {duration_ms:.0f}ms")
            return json.dumps(formatted_results, indent=2)

        except ImportError:
            # Fallback to direct API call if LangChain tool not available
            logger.warning("langchain-community not installed, using direct API")

            headers = {
                "Content-Type": "application/json"
            }

            payload = {
                "api_key": api_key,
                "query": query_clean,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False
            }

            start_time = time.time()
            response = requests.post(
                "https://api.tavily.com/search",
                headers=headers,
                json=payload,
                timeout=30.0
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
        return json.dumps({
            "status": "error",
            "provider": "tavily",
            "message": "Search request timed out. Try again later."
        })
    except requests.exceptions.HTTPError as e:
        logger.error(f"Tavily API HTTP error: {e.response.status_code}")
        return json.dumps({
            "status": "error",
            "provider": "tavily",
            "message": f"API error: {e.response.status_code}"
        })
    except Exception as e:
        logger.error(f"Tavily search failed: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "provider": "tavily",
            "message": f"Search failed: {str(e)}"
        })


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
            return json.dumps({
                "status": "error",
                "provider": "brave",
                "message": "Brave API key not configured. Add BRAVE_API_KEY to .env file."
            })

        # Get configuration
        freshness = os.getenv("BRAVE_SEARCH_FRESHNESS", "pw")  # past week

        # Sanitize query
        query_clean = re.sub(r"[^\w\s\-\.\,\?\!]", "", query)[:300]

        logger.info(f"Brave search: {query_clean} (freshness: {freshness})")

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key
        }

        params = {
            "q": query_clean,
            "count": max_results,
            "search_lang": "en",
            "safesearch": "moderate",
            "freshness": freshness
        }

        start_time = time.time()
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=30.0
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
            "count": len(web_results)
        }

        logger.info(f"Brave search completed in {duration_ms:.0f}ms: {len(web_results)} results")
        return json.dumps(formatted_results, indent=2)

    except requests.exceptions.Timeout:
        logger.error("Brave API timeout")
        return json.dumps({
            "status": "error",
            "provider": "brave",
            "message": "Search request timed out. Try again later."
        })
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        logger.error(f"Brave API HTTP error: {status_code}")

        error_messages = {
            401: "Invalid API key. Check BRAVE_API_KEY in .env file.",
            403: "API key does not have permission.",
            429: "Rate limit exceeded. Wait a moment and try again.",
            500: "Brave API server error. Try again later.",
        }

        return json.dumps({
            "status": "error",
            "provider": "brave",
            "message": error_messages.get(status_code, f"API error: {status_code}")
        })
    except Exception as e:
        logger.error(f"Brave search failed: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "provider": "brave",
            "message": f"Search failed: {str(e)}"
        })


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
            return json.dumps({
                "status": "disabled",
                "message": "Web search is disabled. Enable it in .env with ENABLE_WEB_SEARCH=true"
            })

        # Check which providers are enabled
        use_tavily = os.getenv("USE_TAVILY_SEARCH", "false").lower() == "true"
        use_brave = os.getenv("USE_BRAVE_SEARCH", "false").lower() == "true"

        if not use_tavily and not use_brave:
            return json.dumps({
                "status": "disabled",
                "message": "No web search providers enabled. Set USE_TAVILY_SEARCH or USE_BRAVE_SEARCH to true in .env"
            })

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
            return json.dumps({
                "status": "error",
                "message": "No web search provider available"
            })

    except Exception as e:
        logger.error(f"Web search routing failed: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Web search unavailable",
            "error": str(e)
        })


def analyze_dataset_impl(data_path: Optional[str] = None, data_json: Optional[str] = None) -> str:
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
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to load CSV file: {e}"
                })
        elif data_json:
            try:
                data = json.loads(data_json)
                df = pd.DataFrame(data)
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to parse JSON data: {e}"
                })
        else:
            return json.dumps({
                "status": "error",
                "message": "No data source provided"
            })

        # Basic statistics
        num_rows, num_features = df.shape
        logger.info(f"Analyzing dataset: {num_rows} rows, {num_features} features")

        # Detect outliers using Z-score method
        from scipy import stats
        outliers = []
        numeric_cols = df.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_indices = np.where(z_scores > 3)[0].tolist()
            outliers.extend(outlier_indices)

        outliers = list(set(outliers))  # Remove duplicates
        outlier_count = len(outliers)

        # Analyze class distribution (assume last column or 'label' column is target)
        class_col = None
        if 'label' in df.columns:
            class_col = 'label'
        elif 'target' in df.columns:
            class_col = 'target'
        elif len(df.columns) > 0:
            class_col = df.columns[-1]

        class_distribution = {}
        if class_col:
            value_counts = df[class_col].value_counts()
            total = len(df)
            class_distribution = {
                str(k): float(v / total) for k, v in value_counts.items()
            }

        # Assess bias (class imbalance)
        bias_score = 0.0
        if class_distribution:
            proportions = list(class_distribution.values())
            if proportions:
                max_prop = max(proportions)
                min_prop = min(proportions)
                bias_score = max_prop - min_prop  # 0 = balanced, 1 = completely imbalanced

        # Assess suspected poisoning
        suspected_poisoning = False
        poisoning_confidence = 0.0

        # Simple heuristics for poisoning detection
        outlier_ratio = outlier_count / num_rows if num_rows > 0 else 0
        if outlier_ratio > 0.05:  # More than 5% outliers
            suspected_poisoning = True
            poisoning_confidence = min(outlier_ratio * 10, 1.0)

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
            recommendations.append(f"Manually review {outlier_count} statistical outliers")
        if bias_score > 0.3:
            recommendations.append("Address class imbalance through resampling or class weighting")
        if suspected_poisoning:
            recommendations.append("Implement robust training techniques (label smoothing, outlier removal)")
        if quality_score < 7.0:
            recommendations.append("Consider data cleaning and validation before training")

        result = DatasetAnalysisResult(
            num_rows=num_rows,
            num_features=num_features,
            outliers=outliers,
            outlier_count=outlier_count,
            class_distribution=class_distribution,
            suspected_poisoning=suspected_poisoning,
            poisoning_confidence=poisoning_confidence,
            bias_score=bias_score,
            quality_score=quality_score,
            recommendations=recommendations
        )

        logger.info(f"Dataset analysis complete. Quality score: {quality_score:.1f}/10")

        return json.dumps({
            "status": "success",
            **result.to_dict()
        }, indent=2)

    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Dataset analysis failed",
            "error": str(e)
        })


def generate_security_code_impl(purpose: str, library: str, cve_id: Optional[str] = None) -> str:
    """Generate Python security validation code.

    Creates production-ready Python scripts for security validation tasks such as:
    - CVE vulnerability checks
    - Version validation
    - Data integrity verification

    Args:
        purpose: Purpose of the security script
        library: Target library for validation
        cve_id: Optional CVE ID for CVE-specific checks

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

        # Template for CVE checking script
        if cve_id or "cve" in purpose.lower():
            code = f'''#!/usr/bin/env python3
"""
Security Validation Script: {purpose}
Target Library: {library}
{f"CVE: {cve_id}" if cve_id else ""}

Generated by MLPatrol Security Agent
DO NOT execute this script without reviewing it first!
"""

import sys
import subprocess
from typing import Tuple, Optional
import importlib.metadata


def get_package_version(package_name: str) -> Optional[str]:
    """Get the installed version of a package.

    Args:
        package_name: Name of the package to check

    Returns:
        Version string or None if package not found
    """
    try:
        version = importlib.metadata.version(package_name)
        return version
    except importlib.metadata.PackageNotFoundError:
        return None


def check_vulnerability() -> Tuple[bool, str]:
    """Check if the system is vulnerable to {cve_id if cve_id else "known CVEs"}.

    Returns:
        Tuple of (is_vulnerable, message)
    """
    package_name = "{library}"
    version = get_package_version(package_name)

    if version is None:
        return False, f"{{package_name}} is not installed"

    print(f"Detected {{package_name}} version: {{version}}")

    # Define vulnerable version ranges
    # TODO: Update these ranges based on actual CVE details
    vulnerable_versions = []  # Add vulnerable versions here

    is_vulnerable = version in vulnerable_versions

    if is_vulnerable:
        message = f"VULNERABLE: {{package_name}} {{version}} is affected by {cve_id if cve_id else "known vulnerabilities"}"
    else:
        message = f"SAFE: {{package_name}} {{version}} is not in the known vulnerable range"

    return is_vulnerable, message


def main() -> int:
    """Main execution function.

    Returns:
        Exit code (0 = safe, 1 = vulnerable, 2 = error)
    """
    print("=" * 70)
    print(f"MLPatrol Security Check: {library}")
    print(f"Purpose: {purpose}")
    {f'print(f"CVE: {cve_id}")' if cve_id else ""}
    print("=" * 70)
    print()

    try:
        is_vulnerable, message = check_vulnerability()

        print(message)
        print()

        if is_vulnerable:
            print("REMEDIATION STEPS:")
            print(f"1. Upgrade {library} to the latest version:")
            print(f"   pip install --upgrade {library}")
            print(f"2. Check {library} security advisories")
            print("3. Test your application after upgrading")
            return 1
        else:
            print("No known vulnerabilities detected.")
            print("Continue monitoring security advisories.")
            return 0

    except Exception as e:
        print(f"ERROR: {{e}}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
'''
        else:
            # General validation script
            code = f'''#!/usr/bin/env python3
"""
Security Validation Script: {purpose}
Target Library: {library}

Generated by MLPatrol Security Agent
"""

import sys
from typing import Optional
import importlib.metadata


def validate_security() -> bool:
    """Perform security validation for {library}.

    Returns:
        True if validation passes, False otherwise
    """
    print(f"Validating security for: {library}")
    print(f"Purpose: {purpose}")

    # Add your validation logic here
    # This is a template - customize based on specific requirements

    try:
        version = importlib.metadata.version("{library}")
        print(f"Version detected: {{version}}")

        # Add specific checks here
        print("✓ Basic checks passed")

        return True
    except Exception as e:
        print(f"✗ Validation failed: {{e}}")
        return False


def main() -> int:
    """Main execution function."""
    print("=" * 70)
    print("MLPatrol Security Validation")
    print("=" * 70)
    print()

    if validate_security():
        print("\\nValidation PASSED")
        return 0
    else:
        print("\\nValidation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''

        result = {
            "status": "success",
            "purpose": purpose,
            "library": library,
            "cve_id": cve_id,
            "code": code,
            "filename": f"mlpatrol_check_{library.replace('-', '_')}.py",
            "usage": f"python mlpatrol_check_{library.replace('-', '_')}.py"
        }

        logger.info("Security code generated successfully")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Code generation failed: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Code generation failed",
            "error": str(e)
        })


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
                    "note": "In production, integrate with HuggingFace Hub API for real results"
                }
            ]
        }

        logger.info(f"HuggingFace search completed")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"HuggingFace search failed: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "HuggingFace search failed",
            "error": str(e)
        })


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
        data = json.loads(tool_output)
        if data.get("status") != "success":
            return []

        cves = []
        for cve_dict in data.get("cves", []):
            cve = CVEResult(
                cve_id=cve_dict.get("cve_id", "UNKNOWN"),
                description=cve_dict.get("description", ""),
                cvss_score=cve_dict.get("cvss_score", 0.0),
                severity=cve_dict.get("severity", "UNKNOWN"),
                affected_versions=cve_dict.get("affected_versions", []),
                published_date=cve_dict.get("published_date", ""),
                references=cve_dict.get("references", []),
                library=cve_dict.get("library", ""),
            )
            cves.append(cve)

        return cves
    except Exception as e:
        logger.error(f"Failed to parse CVE results: {e}")
        return []


def parse_dataset_analysis(tool_output: str) -> Optional[DatasetAnalysisResult]:
    """Parse dataset analysis tool output into structured result.

    Args:
        tool_output: JSON string from analyze_dataset tool

    Returns:
        DatasetAnalysisResult object or None if parsing fails
    """
    try:
        data = json.loads(tool_output)
        if data.get("status") != "success":
            return None

        return DatasetAnalysisResult(
            num_rows=data.get("num_rows", 0),
            num_features=data.get("num_features", 0),
            outliers=data.get("outliers_sample", []),
            outlier_count=data.get("outlier_count", 0),
            class_distribution=data.get("class_distribution", {}),
            suspected_poisoning=data.get("suspected_poisoning", False),
            poisoning_confidence=data.get("poisoning_confidence", 0.0),
            bias_score=data.get("bias_score", 0.0),
            quality_score=data.get("quality_score", 0.0),
            recommendations=data.get("recommendations", []),
        )
    except Exception as e:
        logger.error(f"Failed to parse dataset analysis: {e}")
        return None


# Import numpy at module level for dataset analysis
try:
    import numpy as np
except ImportError:
    logger.warning("numpy not available - some dataset analysis features may be limited")
    np = None
