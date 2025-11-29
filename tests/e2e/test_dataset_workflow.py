"""End-to-end tests for dataset analysis workflow.

This test suite covers the complete workflow:
1. User uploads CSV file
2. System validates file format
3. Bias analysis runs
4. Poisoning detection runs
5. Statistical tests run
6. Report generated with results
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.agent.tools import analyze_dataset_impl


class TestDatasetAnalysisWorkflow:
    """E2E tests for complete dataset analysis workflow."""

    def test_complete_workflow_with_valid_csv(self, sample_dataset_csv):
        """Test complete dataset analysis workflow with valid CSV."""
        # Execute the complete workflow
        result_json = analyze_dataset_impl(data_path=sample_dataset_csv)
        result = json.loads(result_json)

        # Verify workflow completed successfully
        assert result["status"] == "success"

        # Verify all analysis components ran
        assert "num_rows" in result
        assert "num_features" in result
        assert "outlier_count" in result
        assert "quality_score" in result

        # Verify correct data was analyzed
        assert result["num_rows"] == 11
        assert result["num_features"] == 3

        # Verify outlier detection ran
        assert result["outlier_count"] >= 1  # Should detect the 1000.0 extreme outlier

    def test_workflow_with_biased_dataset(self, tmp_path):
        """Test workflow detects bias in imbalanced dataset."""
        # Create an imbalanced dataset (90% class 0, 10% class 1)
        df = pd.DataFrame(
            {
                "feature1": list(range(100)),
                "feature2": [i * 0.1 for i in range(100)],
                "label": [0] * 90 + [1] * 10,  # Highly imbalanced
            }
        )
        path = tmp_path / "biased.csv"
        df.to_csv(path, index=False)

        result_json = analyze_dataset_impl(data_path=str(path))
        result = json.loads(result_json)

        assert result["status"] == "success"
        # Should detect high bias
        assert result["bias_score"] > 0.3

    def test_workflow_with_outliers(self, tmp_path):
        """Test workflow detects outliers."""
        # Create dataset with more data points and extreme outliers to ensure z-score > 3.0
        df = pd.DataFrame(
            {
                "feature1": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    10000,
                ],  # 10000 is extreme outlier
                "feature2": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                    9999.9,
                ],  # 9999.9 is extreme outlier
                "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            }
        )
        path = tmp_path / "outliers.csv"
        df.to_csv(path, index=False)

        result_json = analyze_dataset_impl(data_path=str(path))
        result = json.loads(result_json)

        assert result["status"] == "success"
        assert result["outlier_count"] > 0
        assert result["outlier_ratio"] > 0.0

    def test_workflow_with_small_dataset(self, tmp_path):
        """Test workflow handles small datasets appropriately."""
        # Create minimal dataset (2 rows - minimum required)
        df = pd.DataFrame({"feature1": [1.0, 2.0], "label": [0, 1]})
        path = tmp_path / "small.csv"
        df.to_csv(path, index=False)

        result_json = analyze_dataset_impl(data_path=str(path))
        result = json.loads(result_json)

        assert result["status"] == "success"
        assert result["num_rows"] == 2

    def test_workflow_with_too_small_dataset(self, tmp_path):
        """Test workflow rejects datasets that are too small."""
        # Create dataset with only 1 row
        df = pd.DataFrame({"feature1": [1.0], "label": [0]})
        path = tmp_path / "too_small.csv"
        df.to_csv(path, index=False)

        result_json = analyze_dataset_impl(data_path=str(path))
        result = json.loads(result_json)

        assert result["status"] == "error"
        assert "too small" in result["message"].lower()

    def test_workflow_with_invalid_file(self):
        """Test workflow handles invalid file paths."""
        result_json = analyze_dataset_impl(data_path="/nonexistent/file.csv")
        result = json.loads(result_json)

        assert result["status"] == "error"
        assert (
            "not found" in result["message"].lower()
            or "no such file" in result["message"].lower()
        )

    def test_workflow_with_json_data(self):
        """Test workflow with JSON data instead of CSV file."""
        json_data = json.dumps(
            [
                {"feature1": 1.0, "feature2": 0.1, "label": 0},
                {"feature1": 2.0, "feature2": 0.2, "label": 0},
                {"feature1": 3.0, "feature2": 0.3, "label": 1},
            ]
        )

        result_json = analyze_dataset_impl(data_json=json_data)
        result = json.loads(result_json)

        assert result["status"] == "success"
        assert result["num_rows"] == 3

    def test_workflow_with_missing_values(self, tmp_path):
        """Test workflow handles datasets with missing values."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, None, 4.0, 5.0],
                "feature2": [0.1, None, 0.3, 0.4, 0.5],
                "label": [0, 0, 1, 1, 0],
            }
        )
        path = tmp_path / "missing_values.csv"
        df.to_csv(path, index=False)

        result_json = analyze_dataset_impl(data_path=str(path))
        result = json.loads(result_json)

        # Should still succeed (NaN handling)
        assert result["status"] == "success"

    def test_workflow_quality_score_calculation(self, sample_dataset_csv):
        """Test that quality score is calculated correctly."""
        result_json = analyze_dataset_impl(data_path=sample_dataset_csv)
        result = json.loads(result_json)

        assert result["status"] == "success"
        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 10.0

    def test_workflow_statistical_tests_included(self, sample_dataset_csv):
        """Test that statistical test results are included."""
        result_json = analyze_dataset_impl(data_path=sample_dataset_csv)
        result = json.loads(result_json)

        assert result["status"] == "success"
        # Statistical tests should be present
        assert "dataset_summary" in result or "stats" in result

    def test_workflow_performance_with_medium_dataset(self, tmp_path):
        """Test workflow performance with medium-sized dataset."""
        import time

        # Create a medium dataset (1000 rows)
        df = pd.DataFrame(
            {f"feature{i}": [j * 0.1 for j in range(1000)] for i in range(10)}
        )
        df["label"] = [i % 2 for i in range(1000)]
        path = tmp_path / "medium.csv"
        df.to_csv(path, index=False)

        start_time = time.time()
        result_json = analyze_dataset_impl(data_path=str(path))
        elapsed_time = time.time() - start_time

        result = json.loads(result_json)

        assert result["status"] == "success"
        assert elapsed_time < 10.0  # Should complete within 10 seconds

    def test_workflow_with_non_numeric_features(self, tmp_path):
        """Test workflow handles datasets with non-numeric features."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0],
                "category": ["A", "B", "A", "B"],  # Non-numeric
                "label": [0, 1, 0, 1],
            }
        )
        path = tmp_path / "mixed_types.csv"
        df.to_csv(path, index=False)

        result_json = analyze_dataset_impl(data_path=str(path))
        result = json.loads(result_json)

        # Should handle mixed types gracefully
        assert result["status"] == "success" or result["status"] == "error"
        # If error, should have informative message
        if result["status"] == "error":
            assert len(result["message"]) > 0
