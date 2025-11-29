"""Utilities for querying the National Vulnerability Database (NVD)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class CVERecord:
    """Structured representation of a CVE entry."""

    cve_id: str
    description: str
    cvss_score: float
    severity: str
    affected_versions: List[str]
    published_date: str
    references: List[str]
    library: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cve_id": self.cve_id,
            "description": self.description,
            "cvss_score": self.cvss_score,
            "severity": self.severity,
            "affected_versions": self.affected_versions,
            "published_date": self.published_date,
            "references": self.references,
            "library": self.library,
        }


class CVEMonitorError(Exception):
    """Base exception for CVE monitor failures."""


class CVEMonitor:
    """Thin wrapper around the NVD REST API."""

    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    def _build_headers(self) -> Dict[str, str]:
        headers = {"User-Agent": "MLPatrol-SecurityAgent/1.0"}
        if self.api_key:
            headers["apiKey"] = self.api_key
        return headers

    def _format_cve(self, entry: Dict[str, Any], library: str) -> CVERecord:
        cve = entry.get("cve", {})
        cve_id = cve.get("id", "UNKNOWN")

        metrics = cve.get("metrics", {})
        cvss_score = 0.0
        severity = "UNKNOWN"
        if "cvssMetricV31" in metrics:
            cvss_data = metrics["cvssMetricV31"][0]["cvssData"]
            cvss_score = cvss_data.get("baseScore", 0.0)
            severity = cvss_data.get("baseSeverity", "UNKNOWN")

        descriptions = cve.get("descriptions", [])
        description = (
            descriptions[0].get("value", "No description")
            if descriptions
            else "No description"
        )
        references = [ref.get("url") for ref in cve.get("references", [])][:3]

        return CVERecord(
            cve_id=cve_id,
            description=(
                description[:200] + "..." if len(description) > 200 else description
            ),
            cvss_score=cvss_score,
            severity=severity,
            affected_versions=["See references for details"],
            published_date=cve.get("published", "Unknown"),
            references=references,
            library=library,
        )

    def search_recent(self, library: str, days_back: int = 90) -> Dict[str, Any]:
        """Fetch CVEs for a library within a time window."""
        logger.info("Querying NVD for %s (last %s days)", library, days_back)

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)

        params = {
            "keywordSearch": library,
            "pubStartDate": start_date.strftime("%Y-%m-%dT%H:%M:%S.000"),
            "pubEndDate": end_date.strftime("%Y-%m-%dT%H:%M:%S.000"),
        }

        response = self.session.get(
            self.base_url,
            params=params,
            headers=self._build_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        vulnerabilities = data.get("vulnerabilities", [])
        records = [
            self._format_cve(vuln, library).to_dict() for vuln in vulnerabilities
        ]

        return {
            "library": library,
            "days_searched": days_back,
            "cve_count": len(records),
            "cves": records,
        }
