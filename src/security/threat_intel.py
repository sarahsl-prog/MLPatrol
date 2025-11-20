"""High-level helpers for fusing ML security intelligence signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class ThreatIntelInsight:
    """Normalized representation of a security insight."""

    source: str
    title: str
    severity: str
    details: str
    references: List[str]


class ThreatIntelAggregator:
    """Aggregate multiple insight sources into a prioritized list."""

    def __init__(self) -> None:
        self._insights: List[ThreatIntelInsight] = []

    def add_cve_summary(
        self,
        library: str,
        cve_count: int,
        severity: str,
        references: Optional[List[str]] = None,
    ) -> None:
        title = f"{cve_count} CVE(s) affecting {library}"
        details = (
            f"Recent CVE monitoring identified {cve_count} disclosure(s) for {library}. "
            "Prioritize dependency upgrades before shipping new models."
        )
        self._insights.append(
            ThreatIntelInsight(
                source="NVD",
                title=title,
                severity=severity.upper(),
                details=details,
                references=references or [],
            )
        )

    def add_dataset_findings(self, quality_score: float, suspected_poisoning: bool) -> None:
        severity = "HIGH" if suspected_poisoning else ("MEDIUM" if quality_score < 7 else "LOW")
        status = "Potential poisoning indicators detected." if suspected_poisoning else "Dataset quality degraded."
        details = f"{status} Quality score recorded at {quality_score:.1f}/10."
        self._insights.append(
            ThreatIntelInsight(
                source="DatasetAnalysis",
                title="Training data health check",
                severity=severity,
                details=details,
                references=[],
            )
        )

    def extend(self, insights: Iterable[ThreatIntelInsight]) -> None:
        self._insights.extend(insights)

    def summarize(self) -> List[ThreatIntelInsight]:
        """Return insights sorted by severity."""
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}
        return sorted(
            self._insights,
            key=lambda insight: severity_order.get(insight.severity.upper(), 5),
        )