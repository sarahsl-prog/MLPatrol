"""Shared fixtures for E2E tests."""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_dataset_csv(tmp_path):
    """Create a sample CSV dataset for testing."""
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],  # 100.0 is an outlier
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "label": [0, 0, 0, 1, 1, 1],
        }
    )
    path = tmp_path / "test_dataset.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing agent workflows."""
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="Test response from LLM",
        tool_calls=[],
    )


@pytest.fixture
def mock_cve_data():
    """Mock CVE data for testing."""
    return {
        "library": "tensorflow",
        "days_searched": 90,
        "cve_count": 2,
        "cves": [
            {
                "cve_id": "CVE-2024-0001",
                "description": "Test CVE 1",
                "cvss_score": 7.5,
                "severity": "HIGH",
                "affected_versions": ["2.0.0"],
                "published_date": "2024-01-01",
                "references": ["https://example.com"],
                "library": "tensorflow",
            },
            {
                "cve_id": "CVE-2024-0002",
                "description": "Test CVE 2",
                "cvss_score": 5.0,
                "severity": "MEDIUM",
                "affected_versions": ["2.1.0"],
                "published_date": "2024-01-02",
                "references": ["https://example.com"],
                "library": "tensorflow",
            },
        ],
    }
