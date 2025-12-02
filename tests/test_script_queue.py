#!/usr/bin/env python3
"""
Unit tests for ScriptQueue.
"""

import os
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

from src.agent.script_queue import ScriptQueue
from src.models.script_state import (
    ExecutionResult,
    ReviewResult,
    RiskLevel,
    ScriptStatus,
)


class TestScriptQueue(unittest.TestCase):
    """Test cases for ScriptQueue."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database for testing
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_queue.db")
        self.queue = ScriptQueue(db_path=self.db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary database
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_add_script(self):
        """Test adding a script to the queue."""
        script_id = self.queue.add_script(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH",
            metadata={"test": "data"}
        )

        self.assertIsNotNone(script_id)
        self.assertIsInstance(script_id, str)

        # Verify script was added
        script = self.queue.get_script(script_id)
        self.assertIsNotNone(script)
        self.assertEqual(script.cve_id, "CVE-2024-12345")
        self.assertEqual(script.library, "numpy")
        self.assertEqual(script.severity, "HIGH")
        self.assertEqual(script.status, ScriptStatus.DETECTED)
        self.assertEqual(script.metadata["test"], "data")

    def test_get_nonexistent_script(self):
        """Test getting a script that doesn't exist."""
        script = self.queue.get_script("nonexistent-id")
        self.assertIsNone(script)

    def test_update_status(self):
        """Test updating script status."""
        script_id = self.queue.add_script(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH"
        )

        # Update to GENERATED
        success = self.queue.update_status(script_id, ScriptStatus.GENERATED)
        self.assertTrue(success)

        script = self.queue.get_script(script_id)
        self.assertEqual(script.status, ScriptStatus.GENERATED)
        self.assertIsNotNone(script.generated_at)

        # Update to REVIEWED
        success = self.queue.update_status(script_id, ScriptStatus.REVIEWED)
        self.assertTrue(success)

        script = self.queue.get_script(script_id)
        self.assertEqual(script.status, ScriptStatus.REVIEWED)
        self.assertIsNotNone(script.reviewed_at)

    def test_update_status_with_error(self):
        """Test updating status with error logging."""
        script_id = self.queue.add_script(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH"
        )

        self.queue.update_status(
            script_id,
            ScriptStatus.FAILED,
            error="Test error message"
        )

        script = self.queue.get_script(script_id)
        self.assertEqual(script.status, ScriptStatus.FAILED)
        self.assertEqual(len(script.error_log), 1)
        self.assertIn("Test error message", script.error_log[0])

    def test_update_nonexistent_script(self):
        """Test updating a script that doesn't exist."""
        success = self.queue.update_status("nonexistent-id", ScriptStatus.GENERATED)
        self.assertFalse(success)

    def test_update_script_path(self):
        """Test updating script path and hash."""
        script_id = self.queue.add_script(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH"
        )

        success = self.queue.update_script_path(
            script_id,
            "/path/to/script.py",
            "abc123hash"
        )
        self.assertTrue(success)

        script = self.queue.get_script(script_id)
        self.assertEqual(script.script_path, "/path/to/script.py")
        self.assertEqual(script.script_hash, "abc123hash")

    def test_add_review_result(self):
        """Test adding a review result."""
        script_id = self.queue.add_script(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH"
        )

        review = ReviewResult(
            approved=True,
            confidence_score=0.85,
            risk_level=RiskLevel.LOW,
            issues_found=["Minor style issue"],
            recommendations=["Consider adding docstring"],
            safe_to_run=True,
            review_summary="Script looks good",
            detailed_analysis="The script properly checks versions...",
            reviewer_model="claude-sonnet-4"
        )

        success = self.queue.add_review_result(script_id, review)
        self.assertTrue(success)

        script = self.queue.get_script(script_id)
        self.assertIsNotNone(script.review_result)
        self.assertEqual(script.review_result.confidence_score, 0.85)
        self.assertEqual(script.review_result.risk_level, RiskLevel.LOW)
        self.assertTrue(script.review_result.safe_to_run)
        self.assertEqual(script.review_result.reviewer_model, "claude-sonnet-4")

    def test_add_execution_result(self):
        """Test adding an execution result."""
        script_id = self.queue.add_script(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH"
        )

        execution = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="Environment is safe",
            stderr="",
            duration_seconds=1.5,
            error_message=None
        )

        success = self.queue.add_execution_result(script_id, execution)
        self.assertTrue(success)

        script = self.queue.get_script(script_id)
        self.assertIsNotNone(script.execution_result)
        self.assertTrue(script.execution_result.success)
        self.assertEqual(script.execution_result.exit_code, 0)
        self.assertEqual(script.execution_result.duration_seconds, 1.5)

    def test_get_pending_scripts(self):
        """Test getting pending scripts."""
        # Add scripts with different statuses
        id1 = self.queue.add_script("CVE-2024-001", "numpy", "HIGH")
        id2 = self.queue.add_script("CVE-2024-002", "pandas", "MEDIUM")
        id3 = self.queue.add_script("CVE-2024-003", "scipy", "LOW")

        self.queue.update_status(id2, ScriptStatus.COMPLETED)

        pending = self.queue.get_pending_scripts()
        self.assertEqual(len(pending), 2)

        pending_ids = {s.id for s in pending}
        self.assertIn(id1, pending_ids)
        self.assertNotIn(id2, pending_ids)
        self.assertIn(id3, pending_ids)

    def test_get_scripts_by_status(self):
        """Test getting scripts by status."""
        # Add scripts
        id1 = self.queue.add_script("CVE-2024-001", "numpy", "HIGH")
        id2 = self.queue.add_script("CVE-2024-002", "pandas", "MEDIUM")
        id3 = self.queue.add_script("CVE-2024-003", "scipy", "LOW")

        # Update statuses
        self.queue.update_status(id1, ScriptStatus.GENERATED)
        self.queue.update_status(id2, ScriptStatus.REVIEWED)
        self.queue.update_status(id3, ScriptStatus.COMPLETED)

        # Get by status
        generated = self.queue.get_scripts_by_status([ScriptStatus.GENERATED])
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].id, id1)

        reviewed = self.queue.get_scripts_by_status([ScriptStatus.REVIEWED])
        self.assertEqual(len(reviewed), 1)
        self.assertEqual(reviewed[0].id, id2)

        # Multiple statuses
        active = self.queue.get_scripts_by_status([
            ScriptStatus.GENERATED,
            ScriptStatus.REVIEWED
        ])
        self.assertEqual(len(active), 2)

    def test_get_all_scripts(self):
        """Test getting all scripts."""
        # Add multiple scripts
        for i in range(5):
            self.queue.add_script(f"CVE-2024-00{i}", "numpy", "HIGH")

        all_scripts = self.queue.get_all_scripts()
        self.assertEqual(len(all_scripts), 5)

    def test_archive_script(self):
        """Test archiving a script."""
        script_id = self.queue.add_script("CVE-2024-12345", "numpy", "HIGH")

        success = self.queue.archive_script(script_id)
        self.assertTrue(success)

        script = self.queue.get_script(script_id)
        self.assertEqual(script.status, ScriptStatus.CANCELLED)

    def test_delete_script(self):
        """Test deleting a script."""
        script_id = self.queue.add_script("CVE-2024-12345", "numpy", "HIGH")

        # Add review and execution
        review = ReviewResult(
            approved=True,
            confidence_score=0.9,
            risk_level=RiskLevel.LOW,
            issues_found=[],
            recommendations=[],
            safe_to_run=True,
            review_summary="Good",
            detailed_analysis="Details"
        )
        self.queue.add_review_result(script_id, review)

        execution = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="OK",
            stderr="",
            duration_seconds=1.0,
            error_message=None
        )
        self.queue.add_execution_result(script_id, execution)

        # Delete
        success = self.queue.delete_script(script_id)
        self.assertTrue(success)

        # Verify deleted
        script = self.queue.get_script(script_id)
        self.assertIsNone(script)

    def test_event_hooks(self):
        """Test event hook system."""
        events_triggered = []

        def on_added(record):
            events_triggered.append(("added", record.id))

        def on_generated(record):
            events_triggered.append(("generated", record.id))

        # Register hooks
        self.queue.register_hook("on_script_added", on_added)
        self.queue.register_hook("on_script_generated", on_generated)

        # Add script
        script_id = self.queue.add_script("CVE-2024-12345", "numpy", "HIGH")

        # Update status
        self.queue.update_status(script_id, ScriptStatus.GENERATED)

        # Verify hooks were called
        self.assertEqual(len(events_triggered), 2)
        self.assertEqual(events_triggered[0], ("added", script_id))
        self.assertEqual(events_triggered[1], ("generated", script_id))

    def test_invalid_event_hook(self):
        """Test registering invalid event hook."""
        with self.assertRaises(ValueError):
            self.queue.register_hook("invalid_event", lambda: None)

    def test_get_statistics(self):
        """Test getting queue statistics."""
        # Add scripts with different statuses and severities
        id1 = self.queue.add_script("CVE-2024-001", "numpy", "HIGH")
        id2 = self.queue.add_script("CVE-2024-002", "pandas", "MEDIUM")
        id3 = self.queue.add_script("CVE-2024-003", "scipy", "CRITICAL")

        self.queue.update_status(id1, ScriptStatus.COMPLETED)
        self.queue.update_status(id2, ScriptStatus.FAILED)

        # Add reviews
        review = ReviewResult(
            approved=True,
            confidence_score=0.85,
            risk_level=RiskLevel.LOW,
            issues_found=[],
            recommendations=[],
            safe_to_run=True,
            review_summary="OK",
            detailed_analysis="Details"
        )
        self.queue.add_review_result(id1, review)

        # Add executions
        exec_success = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="OK",
            stderr="",
            duration_seconds=1.0,
            error_message=None
        )
        self.queue.add_execution_result(id1, exec_success)

        exec_fail = ExecutionResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="Error",
            duration_seconds=0.5,
            error_message="Failed"
        )
        self.queue.add_execution_result(id2, exec_fail)

        # Get statistics
        stats = self.queue.get_statistics()

        self.assertEqual(stats["total_scripts"], 3)
        self.assertIn("completed", stats["by_status"])
        self.assertIn("failed", stats["by_status"])
        self.assertIn("HIGH", stats["by_severity"])
        self.assertEqual(stats["avg_review_confidence"], 0.85)
        self.assertEqual(stats["execution_success_rate"], 0.5)

    def test_thread_safety(self):
        """Test thread-safe operations."""
        import threading

        script_ids = []
        errors = []

        def add_scripts():
            try:
                for i in range(10):
                    script_id = self.queue.add_script(
                        f"CVE-2024-{i}",
                        "numpy",
                        "HIGH"
                    )
                    script_ids.append(script_id)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=add_scripts) for _ in range(5)]

        # Start threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify no errors
        self.assertEqual(len(errors), 0)

        # Verify all scripts were added
        self.assertEqual(len(script_ids), 50)

        all_scripts = self.queue.get_all_scripts(limit=100)
        self.assertEqual(len(all_scripts), 50)


if __name__ == "__main__":
    unittest.main()
