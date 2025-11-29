"""Statistical test helpers used by MLPatrol dataset analysis.

Provides small, dependency-light wrappers for common statistical checks
used by `analyze_dataset_impl` and unit tests.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_z_scores(series: pd.Series) -> pd.Series:
    """Compute absolute z-scores for a numeric pandas Series.

    Returns a Series of absolute z-scores aligned with the input index.
    """
    # Drop NaNs for stable computation, but preserve index alignment
    vals = series.dropna().astype(float)
    if vals.empty:
        return pd.Series([], dtype=float)

    z = np.abs(stats.zscore(vals))
    return pd.Series(z, index=vals.index)


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> List[int]:
    """Return a sorted list of unique row indices that are outliers by z-score
    across any numeric column in `df`.
    """
    numeric = df.select_dtypes(include=["number"])
    outliers = []
    for _, series in numeric.items():
        if series.dropna().empty:
            continue
        z = compute_z_scores(series)
        # z has only non-null indices
        out_idx = z.index[z > threshold].tolist()
        outliers.extend(out_idx)

    unique_outliers = sorted(set(int(i) for i in outliers))
    return unique_outliers


def ks_test_between_columns(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    """Perform a two-sample Kolmogorov-Smirnov test between two numeric Series.

    Returns (statistic, pvalue).
    """
    a_vals = a.dropna().astype(float)
    b_vals = b.dropna().astype(float)
    if a_vals.empty or b_vals.empty:
        return 0.0, 1.0
    stat, p = stats.ks_2samp(a_vals, b_vals)
    return float(stat), float(p)


def chi2_of_categorical(series: pd.Series) -> Tuple[float, float]:
    """Compute chi-squared statistic against uniform distribution for a categorical Series.

    Returns (chi2_stat, pvalue).
    """
    counts = series.fillna("__MISSING__").value_counts()
    if counts.size <= 1:
        return 0.0, 1.0
    observed = counts.values
    expected = np.full_like(observed, fill_value=observed.mean(), dtype=float)
    chi2, p = stats.chisquare(observed, f_exp=expected)
    return float(chi2), float(p)


def dataset_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Return a small summary used by higher-level analyses.

    Keys: num_rows, num_features, numeric_columns, categorical_columns
    """
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(exclude=["number"]).columns.tolist()
    return {
        "num_rows": int(df.shape[0]),
        "num_features": int(df.shape[1]),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
    }


# Statistical Tests
