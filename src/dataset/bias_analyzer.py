"""Bias analysis utilities for ML datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BiasReport:
    """Structured view of dataset bias metrics."""

    label_column: Optional[str]
    class_distribution: Dict[str, float]
    imbalance_score: float
    entropy: float
    warnings: List[str]


class BiasAnalyzer:
    """Analyze class distribution and highlight imbalance concerns."""

    def __init__(self, imbalance_threshold: float = 0.3) -> None:
        self.imbalance_threshold = imbalance_threshold

    def detect_label_column(self, df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
        if preferred and preferred in df.columns:
            return preferred
        for candidate in ("label", "target", "y"):
            if candidate in df.columns:
                return candidate
        return df.columns[-1] if not df.columns.empty else None

    def analyze(self, df: pd.DataFrame, label_column: Optional[str] = None) -> BiasReport:
        """Compute class distribution, imbalance scores, and warnings."""
        target_col = self.detect_label_column(df, label_column)
        warnings: List[str] = []
        distribution: Dict[str, float] = {}
        imbalance_score = 0.0
        entropy = 0.0

        if target_col:
            counts = df[target_col].value_counts(dropna=False)
            total = float(counts.sum()) or 1.0
            distribution = {str(label): float(count / total) for label, count in counts.items()}

            if distribution:
                values = np.array(list(distribution.values()))
                imbalance_score = float(values.max() - values.min())
                entropy = float(-np.sum(values * np.log2(values + 1e-12)))

            if imbalance_score > self.imbalance_threshold:
                warnings.append(
                    f"Class imbalance detected (score={imbalance_score:.2f}). "
                    "Consider re-weighting or resampling."
                )

        if not distribution:
            warnings.append("Could not infer a label/target column for bias analysis.")

        return BiasReport(
            label_column=target_col,
            class_distribution=distribution,
            imbalance_score=imbalance_score,
            entropy=entropy,
            warnings=warnings,
        )


def analyze_bias(df: pd.DataFrame, label_column: Optional[str] = None) -> BiasReport:
    """Functional wrapper used by components that prefer functions over classes."""
    analyzer = BiasAnalyzer()
    return analyzer.analyze(df, label_column=label_column)