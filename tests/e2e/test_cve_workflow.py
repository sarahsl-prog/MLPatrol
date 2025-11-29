"""End-to-end tests for CVE monitoring workflow.

This test suite covers the complete workflow:
1. User enters library name
2. System searches NVD (mocked)
3. Results displayed with severity
4. CVE details retrieved
"""

import pytest
import json
from unittest.mock import patch, Mock
from src.security.cve_monitor import CVEMonitor


class TestCVEMonitoringWorkflow:
    """E2E tests for complete CVE monitoring workflow."""

    @patch("src.security.cve_monitor.requests.Session")
    def test_complete_cve_search_workflow(self, mock_session_class, mock_cve_data):
        """Test complete CVE search workflow."""
        # Mock the NVD API response
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2024-0001",
                        "descriptions": [{"value": "Test CVE description"}],
                        "published": "2024-01-01T10:00:00.000",
                        "metrics": {
                            "cvssMetricV31": [
                                {
                                    "cvssData": {
                                        "baseScore": 7.5,
                                        "baseSeverity": "HIGH",
                                    }
                                }
                            ]
                        },
                        "references": [{"url": "https://nvd.nist.gov/vuln/detail/CVE-2024-0001"}],
                    }
                }
            ]
        }
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Execute workflow
        monitor = CVEMonitor(api_key="test-key")
        result = monitor.search_recent("tensorflow", days_back=90)

        # Verify workflow results
        assert result["library"] == "tensorflow"
        assert result["cve_count"] == 1
        assert len(result["cves"]) == 1

        cve = result["cves"][0]
        assert cve["cve_id"] == "CVE-2024-0001"
        assert cve["severity"] == "HIGH"
        assert cve["cvss_score"] == 7.5

    @patch("src.security.cve_monitor.requests.Session")
    def test_workflow_with_no_cves_found(self, mock_session_class):
        """Test workflow when no CVEs are found."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"vulnerabilities": []}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        monitor = CVEMonitor()
        result = monitor.search_recent("safe-library", days_back=30)

        assert result["library"] == "safe-library"
        assert result["cve_count"] == 0
        assert result["cves"] == []

    @patch("src.security.cve_monitor.requests.Session")
    def test_workflow_with_multiple_cves(self, mock_session_class):
        """Test workflow handles multiple CVEs."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": f"CVE-2024-000{i}",
                        "descriptions": [{"value": f"Test CVE {i}"}],
                        "published": "2024-01-01",
                        "metrics": {},
                        "references": [],
                    }
                }
                for i in range(1, 6)  # 5 CVEs
            ]
        }
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        monitor = CVEMonitor()
        result = monitor.search_recent("vulnerable-lib", days_back=90)

        assert result["cve_count"] == 5
        assert len(result["cves"]) == 5

    @patch("src.security.cve_monitor.requests.Session")
    def test_workflow_severity_filtering(self, mock_session_class):
        """Test workflow with different severity levels."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2024-CRITICAL",
                        "descriptions": [{"value": "Critical CVE"}],
                        "published": "2024-01-01",
                        "metrics": {
                            "cvssMetricV31": [
                                {"cvssData": {"baseScore": 9.8, "baseSeverity": "CRITICAL"}}
                            ]
                        },
                        "references": [],
                    }
                },
                {
                    "cve": {
                        "id": "CVE-2024-LOW",
                        "descriptions": [{"value": "Low CVE"}],
                        "published": "2024-01-01",
                        "metrics": {
                            "cvssMetricV31": [
                                {"cvssData": {"baseScore": 3.0, "baseSeverity": "LOW"}}
                            ]
                        },
                        "references": [],
                    }
                },
            ]
        }
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        monitor = CVEMonitor()
        result = monitor.search_recent("test-lib", days_back=90)

        cves = result["cves"]
        assert len(cves) == 2

        # Find the critical CVE
        critical_cve = next(c for c in cves if c["cve_id"] == "CVE-2024-CRITICAL")
        assert critical_cve["severity"] == "CRITICAL"
        assert critical_cve["cvss_score"] == 9.8

        # Find the low CVE
        low_cve = next(c for c in cves if c["cve_id"] == "CVE-2024-LOW")
        assert low_cve["severity"] == "LOW"
        assert low_cve["cvss_score"] == 3.0

    @patch("src.security.cve_monitor.requests.Session")
    def test_workflow_custom_time_range(self, mock_session_class):
        """Test workflow with custom time range."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"vulnerabilities": []}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        monitor = CVEMonitor()
        result = monitor.search_recent("test-lib", days_back=30)

        assert result["days_searched"] == 30

        # Verify correct time range was sent to API
        call_kwargs = mock_session.get.call_args[1]
        params = call_kwargs["params"]
        assert "pubStartDate" in params
        assert "pubEndDate" in params
