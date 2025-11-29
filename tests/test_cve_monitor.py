"""Unit tests for CVE monitoring.

This test suite covers:
- CVERecord creation and serialization
- CVEMonitor initialization
- CVE search functionality
- NVD API integration (mocked)
- Error handling
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from src.security.cve_monitor import CVERecord, CVEMonitor, CVEMonitorError


class TestCVERecord:
    """Tests for CVERecord dataclass."""

    def test_cve_record_creation(self):
        """Test creating a CVERecord."""
        record = CVERecord(
            cve_id="CVE-2024-1234",
            description="Test CVE description",
            cvss_score=7.5,
            severity="HIGH",
            affected_versions=["1.0.0", "1.1.0"],
            published_date="2024-01-15",
            references=["https://nvd.nist.gov/vuln/detail/CVE-2024-1234"],
            library="tensorflow",
        )

        assert record.cve_id == "CVE-2024-1234"
        assert record.description == "Test CVE description"
        assert record.cvss_score == 7.5
        assert record.severity == "HIGH"
        assert len(record.affected_versions) == 2
        assert record.library == "tensorflow"

    def test_cve_record_to_dict(self):
        """Test CVERecord.to_dict() serialization."""
        record = CVERecord(
            cve_id="CVE-2024-5678",
            description="Test description",
            cvss_score=9.8,
            severity="CRITICAL",
            affected_versions=["2.0.0"],
            published_date="2024-02-01",
            references=["https://example.com"],
            library="numpy",
        )

        dict_repr = record.to_dict()

        assert isinstance(dict_repr, dict)
        assert dict_repr["cve_id"] == "CVE-2024-5678"
        assert dict_repr["description"] == "Test description"
        assert dict_repr["cvss_score"] == 9.8
        assert dict_repr["severity"] == "CRITICAL"
        assert dict_repr["affected_versions"] == ["2.0.0"]
        assert dict_repr["published_date"] == "2024-02-01"
        assert dict_repr["references"] == ["https://example.com"]
        assert dict_repr["library"] == "numpy"

    def test_cve_record_empty_references(self):
        """Test CVERecord with empty references."""
        record = CVERecord(
            cve_id="CVE-2024-0001",
            description="No refs",
            cvss_score=5.0,
            severity="MEDIUM",
            affected_versions=[],
            published_date="2024-01-01",
            references=[],
            library="test",
        )

        assert record.references == []
        assert record.to_dict()["references"] == []


class TestCVEMonitor:
    """Tests for CVEMonitor class."""

    def test_monitor_initialization_without_api_key(self):
        """Test creating CVEMonitor without API key."""
        monitor = CVEMonitor()

        assert monitor.api_key is None
        assert monitor.timeout == 10
        assert monitor.session is not None

    def test_monitor_initialization_with_api_key(self):
        """Test creating CVEMonitor with API key."""
        monitor = CVEMonitor(api_key="test-api-key")

        assert monitor.api_key == "test-api-key"

    def test_monitor_initialization_custom_timeout(self):
        """Test creating CVEMonitor with custom timeout."""
        monitor = CVEMonitor(timeout=30)

        assert monitor.timeout == 30

    def test_build_headers_without_api_key(self):
        """Test _build_headers without API key."""
        monitor = CVEMonitor()

        headers = monitor._build_headers()

        assert "User-Agent" in headers
        assert "MLPatrol" in headers["User-Agent"]
        assert "apiKey" not in headers

    def test_build_headers_with_api_key(self):
        """Test _build_headers with API key."""
        monitor = CVEMonitor(api_key="my-api-key")

        headers = monitor._build_headers()

        assert "User-Agent" in headers
        assert "apiKey" in headers
        assert headers["apiKey"] == "my-api-key"

    def test_format_cve_basic(self):
        """Test _format_cve with basic CVE data."""
        monitor = CVEMonitor()

        nvd_entry = {
            "cve": {
                "id": "CVE-2024-1234",
                "descriptions": [{"value": "Test vulnerability description"}],
                "published": "2024-01-15T10:00:00.000",
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
                "references": [
                    {"url": "https://example.com/ref1"},
                    {"url": "https://example.com/ref2"},
                ],
            }
        }

        record = monitor._format_cve(nvd_entry, "tensorflow")

        assert record.cve_id == "CVE-2024-1234"
        assert "Test vulnerability" in record.description
        assert record.cvss_score == 7.5
        assert record.severity == "HIGH"
        assert record.library == "tensorflow"
        assert len(record.references) <= 3

    def test_format_cve_long_description(self):
        """Test _format_cve truncates long descriptions."""
        monitor = CVEMonitor()

        long_desc = "A" * 300  # 300 characters
        nvd_entry = {
            "cve": {
                "id": "CVE-2024-5678",
                "descriptions": [{"value": long_desc}],
                "published": "2024-01-01",
                "metrics": {},
                "references": [],
            }
        }

        record = monitor._format_cve(nvd_entry, "numpy")

        # Should be truncated to 200 chars + "..."
        assert len(record.description) == 203
        assert record.description.endswith("...")

    def test_format_cve_no_metrics(self):
        """Test _format_cve with no CVSS metrics."""
        monitor = CVEMonitor()

        nvd_entry = {
            "cve": {
                "id": "CVE-2024-0001",
                "descriptions": [{"value": "Test"}],
                "published": "2024-01-01",
                "metrics": {},
                "references": [],
            }
        }

        record = monitor._format_cve(nvd_entry, "test-lib")

        assert record.cvss_score == 0.0
        assert record.severity == "UNKNOWN"

    def test_format_cve_no_description(self):
        """Test _format_cve with no description."""
        monitor = CVEMonitor()

        nvd_entry = {
            "cve": {
                "id": "CVE-2024-9999",
                "descriptions": [],
                "published": "2024-01-01",
                "metrics": {},
                "references": [],
            }
        }

        record = monitor._format_cve(nvd_entry, "test-lib")

        assert record.description == "No description"

    def test_format_cve_many_references(self):
        """Test _format_cve limits references to 3."""
        monitor = CVEMonitor()

        nvd_entry = {
            "cve": {
                "id": "CVE-2024-1111",
                "descriptions": [{"value": "Test"}],
                "published": "2024-01-01",
                "metrics": {},
                "references": [
                    {"url": "https://example.com/1"},
                    {"url": "https://example.com/2"},
                    {"url": "https://example.com/3"},
                    {"url": "https://example.com/4"},
                    {"url": "https://example.com/5"},
                ],
            }
        }

        record = monitor._format_cve(nvd_entry, "test-lib")

        assert len(record.references) == 3

    @patch("src.security.cve_monitor.requests.Session")
    def test_search_recent_success(self, mock_session_class):
        """Test search_recent with successful API response."""
        # Mock the session and response
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2024-1234",
                        "descriptions": [{"value": "Test CVE"}],
                        "published": "2024-01-15T10:00:00.000",
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
                        "references": [{"url": "https://example.com"}],
                    }
                }
            ]
        }
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        monitor = CVEMonitor()
        result = monitor.search_recent("tensorflow", days_back=30)

        assert result["library"] == "tensorflow"
        assert result["days_searched"] == 30
        assert result["cve_count"] == 1
        assert len(result["cves"]) == 1
        assert result["cves"][0]["cve_id"] == "CVE-2024-1234"

    @patch("src.security.cve_monitor.requests.Session")
    def test_search_recent_no_results(self, mock_session_class):
        """Test search_recent with no CVEs found."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"vulnerabilities": []}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        monitor = CVEMonitor()
        result = monitor.search_recent("nonexistent-lib", days_back=90)

        assert result["library"] == "nonexistent-lib"
        assert result["cve_count"] == 0
        assert result["cves"] == []

    @patch("src.security.cve_monitor.requests.Session")
    def test_search_recent_multiple_cves(self, mock_session_class):
        """Test search_recent with multiple CVEs."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2024-0001",
                        "descriptions": [{"value": "CVE 1"}],
                        "published": "2024-01-01",
                        "metrics": {},
                        "references": [],
                    }
                },
                {
                    "cve": {
                        "id": "CVE-2024-0002",
                        "descriptions": [{"value": "CVE 2"}],
                        "published": "2024-01-02",
                        "metrics": {},
                        "references": [],
                    }
                },
            ]
        }
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        monitor = CVEMonitor()
        result = monitor.search_recent("pandas", days_back=60)

        assert result["cve_count"] == 2
        assert len(result["cves"]) == 2

    @patch("src.security.cve_monitor.requests.Session")
    def test_search_recent_uses_api_key(self, mock_session_class):
        """Test search_recent includes API key in headers."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"vulnerabilities": []}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        monitor = CVEMonitor(api_key="test-api-key")
        monitor.search_recent("numpy", days_back=30)

        # Check that get was called with headers containing API key
        call_kwargs = mock_session.get.call_args[1]
        assert "headers" in call_kwargs
        assert "apiKey" in call_kwargs["headers"]
        assert call_kwargs["headers"]["apiKey"] == "test-api-key"

    @patch("src.security.cve_monitor.requests.Session")
    def test_search_recent_custom_days_back(self, mock_session_class):
        """Test search_recent with custom days_back parameter."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"vulnerabilities": []}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        monitor = CVEMonitor()
        result = monitor.search_recent("test-lib", days_back=180)

        assert result["days_searched"] == 180

        # Check the params sent to API
        call_kwargs = mock_session.get.call_args[1]
        params = call_kwargs["params"]
        assert "pubStartDate" in params
        assert "pubEndDate" in params

    @patch("src.security.cve_monitor.requests.Session")
    def test_search_recent_http_error(self, mock_session_class):
        """Test search_recent raises exception on HTTP error."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500 Error")
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        monitor = CVEMonitor()

        with pytest.raises(Exception) as exc_info:
            monitor.search_recent("test-lib", days_back=30)

        assert "HTTP 500" in str(exc_info.value)

    @patch("src.security.cve_monitor.requests.Session")
    def test_search_recent_timeout(self, mock_session_class):
        """Test search_recent respects timeout setting."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"vulnerabilities": []}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        monitor = CVEMonitor(timeout=5)
        monitor.search_recent("test-lib", days_back=30)

        # Check that timeout was passed
        call_kwargs = mock_session.get.call_args[1]
        assert call_kwargs["timeout"] == 5
