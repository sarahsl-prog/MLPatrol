"""Unit tests for threat intelligence aggregation.

This test suite covers:
- ThreatIntelInsight creation
- ThreatIntelAggregator functionality
- CVE summary aggregation
- Dataset findings aggregation
- Severity-based sorting
"""

import pytest

from src.security.threat_intel import ThreatIntelAggregator, ThreatIntelInsight


class TestThreatIntelInsight:
    """Tests for ThreatIntelInsight dataclass."""

    def test_insight_creation(self):
        """Test creating a ThreatIntelInsight."""
        insight = ThreatIntelInsight(
            source="NVD",
            title="Test CVE",
            severity="HIGH",
            details="Test details",
            references=["https://nvd.nist.gov/vuln/detail/CVE-2024-1234"],
        )

        assert insight.source == "NVD"
        assert insight.title == "Test CVE"
        assert insight.severity == "HIGH"
        assert insight.details == "Test details"
        assert len(insight.references) == 1

    def test_insight_empty_references(self):
        """Test creating insight with empty references list."""
        insight = ThreatIntelInsight(
            source="Test",
            title="Title",
            severity="LOW",
            details="Details",
            references=[],
        )

        assert insight.references == []


class TestThreatIntelAggregator:
    """Tests for ThreatIntelAggregator class."""

    def test_aggregator_initialization(self):
        """Test creating a new aggregator."""
        aggregator = ThreatIntelAggregator()

        assert aggregator._insights == []

    def test_add_cve_summary(self):
        """Test adding a CVE summary."""
        aggregator = ThreatIntelAggregator()

        aggregator.add_cve_summary(
            library="tensorflow",
            cve_count=3,
            severity="HIGH",
            references=["https://nvd.nist.gov/vuln/detail/CVE-2024-1"],
        )

        insights = aggregator.summarize()
        assert len(insights) == 1
        assert insights[0].source == "NVD"
        assert "3 CVE(s) affecting tensorflow" in insights[0].title
        assert insights[0].severity == "HIGH"
        assert "tensorflow" in insights[0].details
        assert len(insights[0].references) == 1

    def test_add_cve_summary_without_references(self):
        """Test adding CVE summary without references."""
        aggregator = ThreatIntelAggregator()

        aggregator.add_cve_summary(library="numpy", cve_count=1, severity="MEDIUM")

        insights = aggregator.summarize()
        assert len(insights) == 1
        assert insights[0].references == []

    def test_add_cve_summary_severity_uppercase(self):
        """Test CVE summary converts severity to uppercase."""
        aggregator = ThreatIntelAggregator()

        aggregator.add_cve_summary(library="pandas", cve_count=2, severity="low")

        insights = aggregator.summarize()
        assert insights[0].severity == "LOW"

    def test_add_dataset_findings_poisoning_detected(self):
        """Test adding dataset findings with poisoning detected."""
        aggregator = ThreatIntelAggregator()

        aggregator.add_dataset_findings(quality_score=5.5, suspected_poisoning=True)

        insights = aggregator.summarize()
        assert len(insights) == 1
        assert insights[0].source == "DatasetAnalysis"
        assert insights[0].severity == "HIGH"
        assert "poisoning" in insights[0].details.lower()
        assert "5.5/10" in insights[0].details

    def test_add_dataset_findings_low_quality(self):
        """Test adding dataset findings with low quality (no poisoning)."""
        aggregator = ThreatIntelAggregator()

        aggregator.add_dataset_findings(quality_score=6.0, suspected_poisoning=False)

        insights = aggregator.summarize()
        assert len(insights) == 1
        assert insights[0].severity == "MEDIUM"
        assert "quality degraded" in insights[0].details.lower()

    def test_add_dataset_findings_good_quality(self):
        """Test adding dataset findings with good quality."""
        aggregator = ThreatIntelAggregator()

        aggregator.add_dataset_findings(quality_score=8.5, suspected_poisoning=False)

        insights = aggregator.summarize()
        assert len(insights) == 1
        assert insights[0].severity == "LOW"
        assert "8.5/10" in insights[0].details

    def test_extend_with_insights(self):
        """Test extending aggregator with existing insights."""
        aggregator = ThreatIntelAggregator()

        insights_to_add = [
            ThreatIntelInsight(
                source="Custom",
                title="Test 1",
                severity="HIGH",
                details="Details 1",
                references=[],
            ),
            ThreatIntelInsight(
                source="Custom",
                title="Test 2",
                severity="LOW",
                details="Details 2",
                references=[],
            ),
        ]

        aggregator.extend(insights_to_add)

        insights = aggregator.summarize()
        assert len(insights) == 2

    def test_summarize_sorts_by_severity(self):
        """Test summarize sorts insights by severity (CRITICAL > HIGH > MEDIUM > LOW)."""
        aggregator = ThreatIntelAggregator()

        # Add in random order
        aggregator.add_dataset_findings(
            quality_score=9.0, suspected_poisoning=False
        )  # LOW
        aggregator.add_cve_summary(library="lib1", cve_count=1, severity="CRITICAL")
        aggregator.add_cve_summary(library="lib2", cve_count=1, severity="MEDIUM")
        aggregator.add_cve_summary(library="lib3", cve_count=1, severity="HIGH")

        insights = aggregator.summarize()

        assert len(insights) == 4
        assert insights[0].severity == "CRITICAL"
        assert insights[1].severity == "HIGH"
        assert insights[2].severity == "MEDIUM"
        assert insights[3].severity == "LOW"

    def test_summarize_handles_unknown_severity(self):
        """Test summarize handles unknown severity values."""
        aggregator = ThreatIntelAggregator()

        aggregator._insights.append(
            ThreatIntelInsight(
                source="Test",
                title="Unknown severity",
                severity="UNKNOWN",
                details="Test",
                references=[],
            )
        )
        aggregator.add_cve_summary(library="lib", cve_count=1, severity="HIGH")

        insights = aggregator.summarize()

        assert len(insights) == 2
        # HIGH should come before UNKNOWN
        assert insights[0].severity == "HIGH"
        assert insights[1].severity == "UNKNOWN"

    def test_summarize_stable_sort(self):
        """Test summarize maintains order for same severity."""
        aggregator = ThreatIntelAggregator()

        aggregator.add_cve_summary(library="lib1", cve_count=1, severity="HIGH")
        aggregator.add_cve_summary(library="lib2", cve_count=2, severity="HIGH")
        aggregator.add_cve_summary(library="lib3", cve_count=3, severity="HIGH")

        insights = aggregator.summarize()

        assert len(insights) == 3
        # All HIGH severity, should maintain insertion order
        assert "lib1" in insights[0].title
        assert "lib2" in insights[1].title
        assert "lib3" in insights[2].title

    def test_multiple_operations(self):
        """Test multiple add operations work together."""
        aggregator = ThreatIntelAggregator()

        aggregator.add_cve_summary(library="numpy", cve_count=2, severity="HIGH")
        aggregator.add_cve_summary(library="pandas", cve_count=1, severity="MEDIUM")
        aggregator.add_dataset_findings(quality_score=7.5, suspected_poisoning=False)

        custom_insight = ThreatIntelInsight(
            source="Manual",
            title="Custom finding",
            severity="CRITICAL",
            details="Manual review",
            references=[],
        )
        aggregator.extend([custom_insight])

        insights = aggregator.summarize()

        assert len(insights) == 4
        # CRITICAL first, then HIGH, then MEDIUM, then LOW
        assert insights[0].severity == "CRITICAL"
        assert insights[1].severity == "HIGH"
        assert insights[2].severity == "MEDIUM"
        assert insights[3].severity == "LOW"

    def test_empty_aggregator(self):
        """Test summarize on empty aggregator."""
        aggregator = ThreatIntelAggregator()

        insights = aggregator.summarize()

        assert insights == []
