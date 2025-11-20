"""Simple poisoning / anomaly detection heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PoisoningReport:
    """Results of heuristic poisoning detection."""

    outlier_indices: List[int]
    outlier_ratio: float
    suspected_poisoning: bool
    confidence: float


class PoisoningDetector:
    """Detect potential poisoning by looking for anomalous samples."""

    def __init__(self, z_threshold: float = 3.0) -> None:
        self.z_threshold = z_threshold

    def detect(self, df: pd.DataFrame) -> PoisoningReport:
        numeric_cols = df.select_dtypes(include=["number"])
        outliers: List[int] = []

        for _, series in numeric_cols.items():
            z_scores = np.abs(stats.zscore(series.dropna()))
            outlier_indices = np.where(z_scores > self.z_threshold)[0].tolist()
            outliers.extend(outlier_indices)

        unique_outliers = sorted(set(outliers))
        outlier_ratio = len(unique_outliers) / max(len(df), 1)
        suspected = outlier_ratio > 0.05
        confidence = min(outlier_ratio * 10, 1.0) if suspected else 0.0

        return PoisoningReport(
            outlier_indices=unique_outliers,
            outlier_ratio=outlier_ratio,
            suspected_poisoning=suspected,
            confidence=confidence,
        )


def detect_poisoning(df: pd.DataFrame, z_threshold: float = 3.0) -> PoisoningReport:
    """Functional wrapper for the PoisoningDetector class."""
    detector = PoisoningDetector(z_threshold=z_threshold)
    return detector.detect(df)