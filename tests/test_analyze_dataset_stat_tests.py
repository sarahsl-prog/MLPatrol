import json

import pandas as pd

from src.agent.tools import analyze_dataset_impl


def _to_json_records(df: pd.DataFrame) -> str:
    return json.dumps(df.to_dict(orient="records"))


def test_analyze_dataset_includes_stat_tests_and_dataset_summary():
    # Build a tiny dataset with numeric and categorical columns
    df = pd.DataFrame(
        {
            "num1": [1, 1, 1, 1, 100],
            "num2": [1, 2, 1, 2, 2],
            "cat": ["a", "a", "b", "b", "a"],
        }
    )

    result_json = analyze_dataset_impl(data_json=_to_json_records(df))
    data = json.loads(result_json)

    assert data.get("status") == "success"

    # stat_tests should be present and include ks_tests and chi2_tests
    stat_tests = data.get("stat_tests")
    assert isinstance(stat_tests, dict)

    # ks_tests may be empty if only one numeric column is found; ensure key exists
    assert "ks_tests" in stat_tests
    assert isinstance(stat_tests.get("ks_tests"), dict)

    # chi2_tests should include the categorical column 'cat'
    chi2_tests = stat_tests.get("chi2_tests", {})
    assert isinstance(chi2_tests, dict)
    assert "cat" in chi2_tests

    # dataset_summary should describe numeric/categorical columns
    summary = data.get("dataset_summary")
    assert isinstance(summary, dict)
    assert summary.get("num_rows") == 5
    assert "numeric_columns" in summary and "categorical_columns" in summary


def test_analyze_dataset_stat_tests_values_shape():
    # Ensure that ks_tests entries have stat and pvalue keys when present
    df = pd.DataFrame({"a": [0, 0, 0, 1], "b": [1, 1, 1, 2], "c": [2, 2, 2, 3]})
    result = analyze_dataset_impl(data_json=_to_json_records(df))
    data = json.loads(result)

    ks = data.get("stat_tests", {}).get("ks_tests", {})
    for _k, v in ks.items():
        assert isinstance(v, dict)
        assert "stat" in v and "pvalue" in v
