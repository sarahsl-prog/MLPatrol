"""Unit tests for dataset analysis tools.

This test suite covers:
- Dataset analysis implementation
- Numpy availability handling
- Statistical analysis correctness
"""

import json
import sys
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.agent.tools import analyze_dataset_impl


# Mock data for testing
@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 100.0],  # 100.0 is an outlier
            "feature2": [0.1, 0.2, 0.3, 0.4],
            "label": [0, 0, 1, 1],
        }
    )
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_json():
    """Create a sample JSON string for testing."""
    data = [
        {"feature1": 1.0, "feature2": 0.1, "label": 0},
        {"feature1": 2.0, "feature2": 0.2, "label": 0},
        {"feature1": 3.0, "feature2": 0.3, "label": 1},
    ]
    return json.dumps(data)


class TestDatasetAnalysis:
    """Tests for analyze_dataset_impl."""

    def test_analyze_csv_success(self, sample_csv):
        """Test successful analysis of a CSV file."""
        result_json = analyze_dataset_impl(data_path=sample_csv)
        result = json.loads(result_json)

        assert result["status"] == "success"
        assert result["num_rows"] == 4
        assert result["num_features"] == 3
        assert "outlier_count" in result
        assert "quality_score" in result

    def test_analyze_json_success(self, sample_json):
        """Test successful analysis of JSON data."""
        result_json = analyze_dataset_impl(data_json=sample_json)
        result = json.loads(result_json)

        assert result["status"] == "success"
        assert result["num_rows"] == 3
        assert result["num_features"] == 3

    def test_missing_numpy_handling(self, sample_csv):
        """Test handling when numpy is not available."""
        # Simulate numpy being None in tools module
        with patch("src.agent.tools.np", None):
            result_json = analyze_dataset_impl(data_path=sample_csv)
            result = json.loads(result_json)

            assert result["status"] == "error"
            assert "Numpy is not installed" in result["message"]

    def test_invalid_file_path(self):
        """Test handling of invalid file path."""
        result_json = analyze_dataset_impl(data_path="nonexistent.csv")
        result = json.loads(result_json)

        assert result["status"] == "error"
        assert "Failed to load CSV file" in result["message"]

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        result_json = analyze_dataset_impl(data_json="{invalid_json}")
        result = json.loads(result_json)

        assert result["status"] == "error"
        assert "Failed to parse JSON data" in result["message"]

    def test_no_input_provided(self):
        """Test handling when no input is provided."""
        result_json = analyze_dataset_impl()
        result = json.loads(result_json)

        assert result["status"] == "error"
        assert "No data source provided" in result["message"]
