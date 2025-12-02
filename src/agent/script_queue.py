#!/usr/bin/env python3
"""
Script Queue Manager

Thread-safe queue manager for tracking security scripts through their lifecycle.
Provides persistent storage using SQLite and event hooks for workflow automation.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from loguru import logger

from src.models.script_state import (
    ExecutionResult,
    ReviewResult,
    ScriptRecord,
    ScriptStatus,
)


class ScriptQueue:
    """
    Thread-safe queue manager for security scripts.

    Manages the lifecycle of scripts from CVE detection through execution,
    with persistent SQLite storage and event-driven hooks.
    """

    def __init__(self, db_path: str = "data/scripts/script_queue.db"):
        """
        Initialize the script queue.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.RLock()
        self._event_hooks: Dict[str, List[Callable]] = {
            "on_script_added": [],
            "on_script_generated": [],
            "on_review_complete": [],
            "on_execution_complete": [],
            "on_status_changed": []
        }

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()
        logger.info(f"ScriptQueue initialized with database: {self.db_path}")

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Scripts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scripts (
                    id TEXT PRIMARY KEY,
                    cve_id TEXT NOT NULL,
                    library TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    script_path TEXT,
                    script_hash TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    generated_at TEXT,
                    reviewed_at TEXT,
                    approved_at TEXT,
                    executed_at TEXT,
                    metadata TEXT,
                    error_log TEXT
                )
            """)

            # Reviews table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    script_id TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    confidence_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    issues_found TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    safe_to_run INTEGER NOT NULL,
                    review_summary TEXT NOT NULL,
                    detailed_analysis TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    reviewer_model TEXT,
                    FOREIGN KEY (script_id) REFERENCES scripts(id)
                )
            """)

            # Executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    script_id TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    exit_code INTEGER NOT NULL,
                    stdout TEXT,
                    stderr TEXT,
                    duration_seconds REAL NOT NULL,
                    error_message TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (script_id) REFERENCES scripts(id)
                )
            """)

            # Create indices for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_status ON scripts(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_cve_id ON scripts(cve_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_library ON scripts(library)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scripts_created_at ON scripts(created_at)")

            conn.commit()
            conn.close()

    def add_script(
        self,
        cve_id: str,
        library: str,
        severity: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a new script to the queue.

        Args:
            cve_id: CVE identifier (e.g., "CVE-2024-12345")
            library: Library name (e.g., "numpy")
            severity: Severity level (e.g., "HIGH", "CRITICAL")
            metadata: Optional metadata dictionary

        Returns:
            Script ID (UUID)
        """
        with self._lock:
            record = ScriptRecord(
                cve_id=cve_id,
                library=library,
                severity=severity,
                status=ScriptStatus.DETECTED,
                metadata=metadata or {}
            )

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO scripts (
                    id, cve_id, library, severity, script_path, script_hash,
                    status, created_at, updated_at, generated_at, reviewed_at,
                    approved_at, executed_at, metadata, error_log
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id,
                record.cve_id,
                record.library,
                record.severity,
                record.script_path,
                record.script_hash,
                str(record.status),
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                None,  # generated_at
                None,  # reviewed_at
                None,  # approved_at
                None,  # executed_at
                json.dumps(record.metadata),
                json.dumps(record.error_log)
            ))

            conn.commit()
            conn.close()

            logger.info(f"Added script {record.id} for {cve_id} ({library})")
            self._trigger_event("on_script_added", record)

            return record.id

    def get_script(self, script_id: str) -> Optional[ScriptRecord]:
        """
        Get a script by ID.

        Args:
            script_id: Script UUID

        Returns:
            ScriptRecord or None if not found
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM scripts WHERE id = ?", (script_id,))
            row = cursor.fetchone()

            if not row:
                conn.close()
                return None

            # Load review result if exists
            cursor.execute("""
                SELECT * FROM reviews
                WHERE script_id = ?
                ORDER BY reviewed_at DESC
                LIMIT 1
            """, (script_id,))
            review_row = cursor.fetchone()

            # Load execution result if exists
            cursor.execute("""
                SELECT * FROM executions
                WHERE script_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (script_id,))
            execution_row = cursor.fetchone()

            conn.close()

            # Build ScriptRecord
            record = self._row_to_record(row, review_row, execution_row)
            return record

    def update_status(
        self,
        script_id: str,
        status: ScriptStatus,
        metadata: Optional[Dict] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Update script status.

        Args:
            script_id: Script UUID
            status: New status
            metadata: Optional metadata to merge
            error: Optional error message to log

        Returns:
            True if updated, False if script not found
        """
        with self._lock:
            record = self.get_script(script_id)
            if not record:
                logger.warning(f"Script {script_id} not found for status update")
                return False

            old_status = record.status
            record.update_status(status, error)

            if metadata:
                record.metadata.update(metadata)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE scripts SET
                    status = ?,
                    updated_at = ?,
                    generated_at = ?,
                    reviewed_at = ?,
                    approved_at = ?,
                    executed_at = ?,
                    metadata = ?,
                    error_log = ?
                WHERE id = ?
            """, (
                str(record.status),
                record.updated_at.isoformat(),
                record.generated_at.isoformat() if record.generated_at else None,
                record.reviewed_at.isoformat() if record.reviewed_at else None,
                record.approved_at.isoformat() if record.approved_at else None,
                record.executed_at.isoformat() if record.executed_at else None,
                json.dumps(record.metadata),
                json.dumps(record.error_log),
                script_id
            ))

            conn.commit()
            conn.close()

            logger.info(f"Updated script {script_id}: {old_status} -> {status}")
            self._trigger_event("on_status_changed", record, old_status)

            # Trigger specific events
            if status == ScriptStatus.GENERATED:
                self._trigger_event("on_script_generated", record)
            elif status == ScriptStatus.REVIEWED:
                self._trigger_event("on_review_complete", record)
            elif status in {ScriptStatus.COMPLETED, ScriptStatus.FAILED}:
                self._trigger_event("on_execution_complete", record)

            return True

    def update_script_path(self, script_id: str, script_path: str, script_hash: str) -> bool:
        """
        Update script file path and hash.

        Args:
            script_id: Script UUID
            script_path: Path to generated script file
            script_hash: SHA256 hash of script content

        Returns:
            True if updated, False if not found
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE scripts SET
                    script_path = ?,
                    script_hash = ?,
                    updated_at = ?
                WHERE id = ?
            """, (script_path, script_hash, datetime.now().isoformat(), script_id))

            updated = cursor.rowcount > 0
            conn.commit()
            conn.close()

            if updated:
                logger.info(f"Updated script path for {script_id}: {script_path}")
            return updated

    def add_review_result(self, script_id: str, review: ReviewResult) -> bool:
        """
        Add review result for a script.

        Args:
            script_id: Script UUID
            review: ReviewResult object

        Returns:
            True if added, False if script not found
        """
        with self._lock:
            record = self.get_script(script_id)
            if not record:
                logger.warning(f"Script {script_id} not found for review")
                return False

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO reviews (
                    script_id, approved, confidence_score, risk_level,
                    issues_found, recommendations, safe_to_run,
                    review_summary, detailed_analysis, reviewed_at,
                    reviewer_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                script_id,
                1 if review.approved else 0,
                review.confidence_score,
                str(review.risk_level),
                json.dumps(review.issues_found),
                json.dumps(review.recommendations),
                1 if review.safe_to_run else 0,
                review.review_summary,
                review.detailed_analysis,
                review.reviewed_at.isoformat(),
                review.reviewer_model
            ))

            conn.commit()
            conn.close()

            logger.info(f"Added review for script {script_id}: {review.risk_level}, confidence={review.confidence_score:.2f}")
            return True

    def add_execution_result(self, script_id: str, execution: ExecutionResult) -> bool:
        """
        Add execution result for a script.

        Args:
            script_id: Script UUID
            execution: ExecutionResult object

        Returns:
            True if added, False if script not found
        """
        with self._lock:
            record = self.get_script(script_id)
            if not record:
                logger.warning(f"Script {script_id} not found for execution result")
                return False

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO executions (
                    script_id, success, exit_code, stdout, stderr,
                    duration_seconds, error_message, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                script_id,
                1 if execution.success else 0,
                execution.exit_code,
                execution.stdout,
                execution.stderr,
                execution.duration_seconds,
                execution.error_message,
                execution.timestamp.isoformat()
            ))

            conn.commit()
            conn.close()

            logger.info(f"Added execution result for script {script_id}: success={execution.success}, exit_code={execution.exit_code}")
            return True

    def get_pending_scripts(self) -> List[ScriptRecord]:
        """
        Get all scripts that require action.

        Returns:
            List of scripts with pending status
        """
        return self.get_scripts_by_status([
            ScriptStatus.DETECTED,
            ScriptStatus.GENERATED,
            ScriptStatus.REVIEWED,
            ScriptStatus.APPROVED
        ])

    def get_scripts_by_status(self, statuses: List[ScriptStatus]) -> List[ScriptRecord]:
        """
        Get scripts by status.

        Args:
            statuses: List of statuses to filter by

        Returns:
            List of matching scripts
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            status_strings = [str(s) for s in statuses]
            placeholders = ",".join("?" * len(status_strings))

            cursor.execute(f"""
                SELECT * FROM scripts
                WHERE status IN ({placeholders})
                ORDER BY created_at DESC
            """, status_strings)

            rows = cursor.fetchall()
            conn.close()

            records = []
            for row in rows:
                # Get review and execution for each script
                record = self.get_script(row["id"])
                if record:
                    records.append(record)

            return records

    def get_all_scripts(self, limit: int = 100) -> List[ScriptRecord]:
        """
        Get all scripts.

        Args:
            limit: Maximum number of scripts to return

        Returns:
            List of scripts
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM scripts
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            records = []
            for row in rows:
                record = self.get_script(row["id"])
                if record:
                    records.append(record)

            return records

    def archive_script(self, script_id: str) -> bool:
        """
        Archive a script (currently just marks as cancelled).

        Args:
            script_id: Script UUID

        Returns:
            True if archived, False if not found
        """
        return self.update_status(script_id, ScriptStatus.CANCELLED)

    def delete_script(self, script_id: str) -> bool:
        """
        Permanently delete a script and all associated data.

        Args:
            script_id: Script UUID

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete from all tables
            cursor.execute("DELETE FROM reviews WHERE script_id = ?", (script_id,))
            cursor.execute("DELETE FROM executions WHERE script_id = ?", (script_id,))
            cursor.execute("DELETE FROM scripts WHERE id = ?", (script_id,))

            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()

            if deleted:
                logger.info(f"Deleted script {script_id}")
            return deleted

    def register_hook(self, event: str, callback: Callable) -> None:
        """
        Register an event hook.

        Args:
            event: Event name (on_script_added, on_script_generated, etc.)
            callback: Callback function to invoke

        Raises:
            ValueError: If event name is invalid
        """
        if event not in self._event_hooks:
            raise ValueError(f"Invalid event: {event}. Valid events: {list(self._event_hooks.keys())}")

        with self._lock:
            self._event_hooks[event].append(callback)
            logger.info(f"Registered hook for event: {event}")

    def _trigger_event(self, event: str, *args, **kwargs) -> None:
        """
        Trigger event hooks.

        Args:
            event: Event name
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks
        """
        if event not in self._event_hooks:
            return

        for callback in self._event_hooks[event]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event hook {event}: {e}")

    def _row_to_record(
        self,
        row: sqlite3.Row,
        review_row: Optional[sqlite3.Row] = None,
        execution_row: Optional[sqlite3.Row] = None
    ) -> ScriptRecord:
        """
        Convert database row to ScriptRecord.

        Args:
            row: Script row
            review_row: Optional review row
            execution_row: Optional execution row

        Returns:
            ScriptRecord object
        """
        record = ScriptRecord(
            id=row["id"],
            cve_id=row["cve_id"],
            library=row["library"],
            severity=row["severity"],
            script_path=row["script_path"] or "",
            script_hash=row["script_hash"],
            status=ScriptStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            error_log=json.loads(row["error_log"]) if row["error_log"] else []
        )

        # Parse timestamps
        if row["generated_at"]:
            record.generated_at = datetime.fromisoformat(row["generated_at"])
        if row["reviewed_at"]:
            record.reviewed_at = datetime.fromisoformat(row["reviewed_at"])
        if row["approved_at"]:
            record.approved_at = datetime.fromisoformat(row["approved_at"])
        if row["executed_at"]:
            record.executed_at = datetime.fromisoformat(row["executed_at"])

        # Parse review result
        if review_row:
            from src.models.script_state import RiskLevel
            record.review_result = ReviewResult(
                approved=bool(review_row["approved"]),
                confidence_score=review_row["confidence_score"],
                risk_level=RiskLevel(review_row["risk_level"]),
                issues_found=json.loads(review_row["issues_found"]),
                recommendations=json.loads(review_row["recommendations"]),
                safe_to_run=bool(review_row["safe_to_run"]),
                review_summary=review_row["review_summary"],
                detailed_analysis=review_row["detailed_analysis"],
                reviewed_at=datetime.fromisoformat(review_row["reviewed_at"]),
                reviewer_model=review_row["reviewer_model"]
            )

        # Parse execution result
        if execution_row:
            record.execution_result = ExecutionResult(
                success=bool(execution_row["success"]),
                exit_code=execution_row["exit_code"],
                stdout=execution_row["stdout"] or "",
                stderr=execution_row["stderr"] or "",
                duration_seconds=execution_row["duration_seconds"],
                error_message=execution_row["error_message"],
                timestamp=datetime.fromisoformat(execution_row["timestamp"])
            )

        return record

    def get_statistics(self) -> Dict:
        """
        Get queue statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Count by status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM scripts
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())

            # Count by severity
            cursor.execute("""
                SELECT severity, COUNT(*) as count
                FROM scripts
                GROUP BY severity
            """)
            severity_counts = dict(cursor.fetchall())

            # Average review confidence
            cursor.execute("SELECT AVG(confidence_score) FROM reviews")
            avg_confidence = cursor.fetchone()[0] or 0.0

            # Success rate
            cursor.execute("""
                SELECT
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
                FROM executions
            """)
            result = cursor.fetchone()
            success_rate = result[0] if result[0] else 0.0

            conn.close()

            return {
                "total_scripts": sum(status_counts.values()),
                "by_status": status_counts,
                "by_severity": severity_counts,
                "avg_review_confidence": avg_confidence,
                "execution_success_rate": success_rate
            }
