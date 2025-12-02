#!/usr/bin/env python3
"""
Script State Management Models

Data models for tracking the lifecycle of security scripts from CVE detection
through generation, review, and execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import uuid


class ScriptStatus(Enum):
    """Status of a script in the automated workflow."""

    DETECTED = "detected"  # CVE found, script not yet generated
    GENERATED = "generated"  # Script created, awaiting review
    UNDER_REVIEW = "under_review"  # LLM is analyzing script
    REVIEWED = "reviewed"  # LLM review complete, awaiting user approval
    APPROVED = "approved"  # User approved, ready to execute
    EXECUTING = "executing"  # Script is running
    COMPLETED = "completed"  # Successfully executed
    FAILED = "failed"  # Execution failed
    REJECTED = "rejected"  # User rejected script
    CANCELLED = "cancelled"  # User cancelled

    def __str__(self) -> str:
        return self.value

    @property
    def is_terminal(self) -> bool:
        """Check if this status is a terminal state."""
        return self in {
            ScriptStatus.COMPLETED,
            ScriptStatus.FAILED,
            ScriptStatus.REJECTED,
            ScriptStatus.CANCELLED
        }

    @property
    def is_pending(self) -> bool:
        """Check if this status requires action."""
        return self in {
            ScriptStatus.DETECTED,
            ScriptStatus.GENERATED,
            ScriptStatus.REVIEWED,
            ScriptStatus.APPROVED
        }


class RiskLevel(Enum):
    """Risk level assigned by script review."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return self.value

    @property
    def color(self) -> str:
        """Get display color for this risk level."""
        colors = {
            RiskLevel.LOW: "#28a745",  # green
            RiskLevel.MEDIUM: "#ffc107",  # yellow
            RiskLevel.HIGH: "#fd7e14",  # orange
            RiskLevel.CRITICAL: "#dc3545"  # red
        }
        return colors[self]


@dataclass
class ReviewResult:
    """Result of LLM review of a generated script."""

    approved: bool
    confidence_score: float  # 0.0-1.0
    risk_level: RiskLevel
    issues_found: List[str]
    recommendations: List[str]
    safe_to_run: bool
    review_summary: str
    detailed_analysis: str
    reviewed_at: datetime = field(default_factory=datetime.now)
    reviewer_model: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "approved": self.approved,
            "confidence_score": self.confidence_score,
            "risk_level": str(self.risk_level),
            "issues_found": self.issues_found,
            "recommendations": self.recommendations,
            "safe_to_run": self.safe_to_run,
            "review_summary": self.review_summary,
            "detailed_analysis": self.detailed_analysis,
            "reviewed_at": self.reviewed_at.isoformat(),
            "reviewer_model": self.reviewer_model
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ReviewResult":
        """Create from dictionary."""
        return cls(
            approved=data["approved"],
            confidence_score=data["confidence_score"],
            risk_level=RiskLevel(data["risk_level"]),
            issues_found=data["issues_found"],
            recommendations=data["recommendations"],
            safe_to_run=data["safe_to_run"],
            review_summary=data["review_summary"],
            detailed_analysis=data["detailed_analysis"],
            reviewed_at=datetime.fromisoformat(data["reviewed_at"]),
            reviewer_model=data.get("reviewer_model")
        )


@dataclass
class ExecutionResult:
    """Result of script execution."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    error_message: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExecutionResult":
        """Create from dictionary."""
        return cls(
            success=data["success"],
            exit_code=data["exit_code"],
            stdout=data["stdout"],
            stderr=data["stderr"],
            duration_seconds=data["duration_seconds"],
            error_message=data.get("error_message"),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class ScriptRecord:
    """Complete record of a script through its lifecycle."""

    # Identifiers
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cve_id: str = ""
    library: str = ""
    severity: str = ""

    # File paths
    script_path: str = ""
    script_hash: Optional[str] = None  # SHA256 of script content

    # Status tracking
    status: ScriptStatus = ScriptStatus.DETECTED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Stage timestamps
    generated_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None

    # Results
    review_result: Optional[ReviewResult] = None
    execution_result: Optional[ExecutionResult] = None

    # Metadata
    metadata: Dict = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)

    def update_status(self, new_status: ScriptStatus, error: Optional[str] = None) -> None:
        """Update status with timestamp and optional error logging."""
        self.status = new_status
        self.updated_at = datetime.now()

        # Update stage-specific timestamps
        if new_status == ScriptStatus.GENERATED:
            self.generated_at = datetime.now()
        elif new_status == ScriptStatus.REVIEWED:
            self.reviewed_at = datetime.now()
        elif new_status == ScriptStatus.APPROVED:
            self.approved_at = datetime.now()
        elif new_status == ScriptStatus.COMPLETED:
            self.executed_at = datetime.now()

        if error:
            self.error_log.append(f"[{datetime.now().isoformat()}] {error}")

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "cve_id": self.cve_id,
            "library": self.library,
            "severity": self.severity,
            "script_path": self.script_path,
            "script_hash": self.script_hash,
            "status": str(self.status),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "review_result": self.review_result.to_dict() if self.review_result else None,
            "execution_result": self.execution_result.to_dict() if self.execution_result else None,
            "metadata": self.metadata,
            "error_log": self.error_log
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ScriptRecord":
        """Create from dictionary."""
        record = cls(
            id=data["id"],
            cve_id=data["cve_id"],
            library=data["library"],
            severity=data["severity"],
            script_path=data["script_path"],
            script_hash=data.get("script_hash"),
            status=ScriptStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            error_log=data.get("error_log", [])
        )

        # Parse optional timestamps
        if data.get("generated_at"):
            record.generated_at = datetime.fromisoformat(data["generated_at"])
        if data.get("reviewed_at"):
            record.reviewed_at = datetime.fromisoformat(data["reviewed_at"])
        if data.get("approved_at"):
            record.approved_at = datetime.fromisoformat(data["approved_at"])
        if data.get("executed_at"):
            record.executed_at = datetime.fromisoformat(data["executed_at"])

        # Parse results
        if data.get("review_result"):
            record.review_result = ReviewResult.from_dict(data["review_result"])
        if data.get("execution_result"):
            record.execution_result = ExecutionResult.from_dict(data["execution_result"])

        return record

    @property
    def age_seconds(self) -> float:
        """Get age of this record in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def is_stale(self) -> bool:
        """Check if record is stale (older than 7 days)."""
        return self.age_seconds > (7 * 24 * 60 * 60)

    @property
    def display_status(self) -> str:
        """Get human-readable status."""
        status_map = {
            ScriptStatus.DETECTED: "🔍 CVE Detected",
            ScriptStatus.GENERATED: "📝 Script Generated",
            ScriptStatus.UNDER_REVIEW: "🤖 Under Review",
            ScriptStatus.REVIEWED: "✅ Reviewed",
            ScriptStatus.APPROVED: "👍 Approved",
            ScriptStatus.EXECUTING: "⚙️ Executing",
            ScriptStatus.COMPLETED: "✔️ Completed",
            ScriptStatus.FAILED: "❌ Failed",
            ScriptStatus.REJECTED: "🚫 Rejected",
            ScriptStatus.CANCELLED: "⛔ Cancelled"
        }
        return status_map.get(self.status, str(self.status))
