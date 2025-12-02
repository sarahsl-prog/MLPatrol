#!/usr/bin/env python3
"""
Unit tests for WorkflowOrchestrator.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

from src.agent.script_queue import ScriptQueue
from src.agent.workflow_orchestrator import WorkflowOrchestrator
from src.models.script_state import ReviewResult, RiskLevel, ScriptStatus


class TestWorkflowOrchestrator(unittest.TestCase):
    """Test cases for WorkflowOrchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_queue.db")
        self.script_dir = os.path.join(self.temp_dir, "scripts")
        self.cache_dir = os.path.join(self.temp_dir, "cache")

        # Create queue with test database
        self.queue = ScriptQueue(db_path=self.db_path)

        # Create mock reviewer
        self.mock_reviewer = MagicMock()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = WorkflowOrchestrator(
            queue=self.queue,
            reviewer=self.mock_reviewer,
            script_dir=self.script_dir,
            auto_review=True
        )

        self.assertIsNotNone(orchestrator.queue)
        self.assertIsNotNone(orchestrator.reviewer)
        self.assertTrue(orchestrator.auto_review)

    def test_handle_cve_detected(self):
        """Test CVE detection handling."""
        orchestrator = WorkflowOrchestrator(
            queue=self.queue,
            reviewer=self.mock_reviewer,
            script_dir=self.script_dir,
            auto_review=False  # Disable auto-review for this test
        )

        script_id = orchestrator.handle_cve_detected(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH",
            description="Test CVE",
            affected_versions=["1.21.0", "1.21.1"]
        )

        # Verify script was created
        self.assertIsNotNone(script_id)

        # Verify script record
        record = self.queue.get_script(script_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.cve_id, "CVE-2024-12345")
        self.assertEqual(record.library, "numpy")
        self.assertEqual(record.severity, "HIGH")

        # Verify script was generated
        self.assertEqual(record.status, ScriptStatus.GENERATED)
        self.assertTrue(os.path.exists(record.script_path))

    def test_handle_script_generated_with_review(self):
        """Test script generation triggers review."""
        # Mock review result
        mock_review = ReviewResult(
            approved=True,
            safe_to_run=True,
            confidence_score=0.9,
            risk_level=RiskLevel.LOW,
            issues_found=[],
            recommendations=["Add type hints"],
            review_summary="Script is safe",
            detailed_analysis="Comprehensive analysis"
        )
        self.mock_reviewer.review_script.return_value = mock_review

        orchestrator = WorkflowOrchestrator(
            queue=self.queue,
            reviewer=self.mock_reviewer,
            script_dir=self.script_dir,
            auto_review=False
        )

        # Create a script
        script_id = orchestrator.handle_cve_detected(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH"
        )

        # Manually trigger review (since auto_review=False)
        orchestrator.handle_script_generated(script_id)

        # Verify review was called
        self.mock_reviewer.review_script.assert_called_once()

        # Verify script status updated
        record = self.queue.get_script(script_id)
        self.assertEqual(record.status, ScriptStatus.REVIEWED)

        # Verify review result stored
        self.assertIsNotNone(record.review_result)
        self.assertEqual(record.review_result.confidence_score, 0.9)

    def test_handle_user_approved(self):
        """Test user approval handling."""
        # Setup mock review
        mock_review = ReviewResult(
            approved=True,
            safe_to_run=True,
            confidence_score=0.95,
            risk_level=RiskLevel.LOW,
            issues_found=[],
            recommendations=[],
            review_summary="Safe",
            detailed_analysis="Details"
        )
        self.mock_reviewer.review_script.return_value = mock_review

        orchestrator = WorkflowOrchestrator(
            queue=self.queue,
            reviewer=self.mock_reviewer,
            script_dir=self.script_dir,
            auto_review=False
        )

        # Create and review script
        script_id = orchestrator.handle_cve_detected(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH"
        )
        orchestrator.handle_script_generated(script_id)

        # Approve script
        success = orchestrator.handle_user_approved(script_id)

        self.assertTrue(success)

        # Verify status
        record = self.queue.get_script(script_id)
        self.assertEqual(record.status, ScriptStatus.APPROVED)

    def test_handle_user_approved_unsafe_script(self):
        """Test approval fails for unsafe script."""
        # Setup mock review marking script as unsafe
        mock_review = ReviewResult(
            approved=False,
            safe_to_run=False,
            confidence_score=0.2,
            risk_level=RiskLevel.CRITICAL,
            issues_found=["eval detected"],
            recommendations=["Remove eval"],
            review_summary="Unsafe",
            detailed_analysis="Details"
        )
        self.mock_reviewer.review_script.return_value = mock_review

        orchestrator = WorkflowOrchestrator(
            queue=self.queue,
            reviewer=self.mock_reviewer,
            script_dir=self.script_dir,
            auto_review=False
        )

        # Create and review script
        script_id = orchestrator.handle_cve_detected(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH"
        )
        orchestrator.handle_script_generated(script_id)

        # Try to approve unsafe script
        success = orchestrator.handle_user_approved(script_id)

        self.assertFalse(success)

        # Verify status hasn't changed to APPROVED
        record = self.queue.get_script(script_id)
        self.assertNotEqual(record.status, ScriptStatus.APPROVED)

    def test_handle_user_rejected(self):
        """Test user rejection handling."""
        orchestrator = WorkflowOrchestrator(
            queue=self.queue,
            reviewer=self.mock_reviewer,
            script_dir=self.script_dir,
            auto_review=False
        )

        # Create script
        script_id = orchestrator.handle_cve_detected(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH"
        )

        # Reject script
        success = orchestrator.handle_user_rejected(
            script_id,
            reason="User prefers manual review"
        )

        self.assertTrue(success)

        # Verify status
        record = self.queue.get_script(script_id)
        self.assertEqual(record.status, ScriptStatus.REJECTED)

    def test_handle_execution_complete(self):
        """Test execution completion handling."""
        orchestrator = WorkflowOrchestrator(
            queue=self.queue,
            reviewer=self.mock_reviewer,
            script_dir=self.script_dir,
            auto_review=False
        )

        # Create script
        script_id = orchestrator.handle_cve_detected(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH"
        )

        # Mark as approved
        self.queue.update_status(script_id, ScriptStatus.APPROVED)

        # Handle execution completion
        success = orchestrator.handle_execution_complete(
            script_id=script_id,
            success=True,
            exit_code=0,
            stdout="Environment is safe",
            stderr="",
            duration=1.5
        )

        self.assertTrue(success)

        # Verify execution result stored
        record = self.queue.get_script(script_id)
        self.assertIsNotNone(record.execution_result)
        self.assertTrue(record.execution_result.success)
        self.assertEqual(record.execution_result.exit_code, 0)
        self.assertEqual(record.execution_result.duration_seconds, 1.5)

        # Verify status updated
        self.assertEqual(record.status, ScriptStatus.COMPLETED)

    def test_get_reviewed_scripts(self):
        """Test querying reviewed scripts."""
        mock_review = ReviewResult(
            approved=True,
            safe_to_run=True,
            confidence_score=0.9,
            risk_level=RiskLevel.LOW,
            issues_found=[],
            recommendations=[],
            review_summary="Safe",
            detailed_analysis="Details"
        )
        self.mock_reviewer.review_script.return_value = mock_review

        orchestrator = WorkflowOrchestrator(
            queue=self.queue,
            reviewer=self.mock_reviewer,
            script_dir=self.script_dir,
            auto_review=False
        )

        # Create and review multiple scripts
        for i in range(3):
            script_id = orchestrator.handle_cve_detected(
                cve_id=f"CVE-2024-{i}",
                library="numpy",
                severity="HIGH"
            )
            orchestrator.handle_script_generated(script_id)

        # Get reviewed scripts
        reviewed = orchestrator.get_reviewed_scripts()

        self.assertEqual(len(reviewed), 3)
        for script in reviewed:
            self.assertEqual(script.status, ScriptStatus.REVIEWED)

    def test_get_workflow_statistics(self):
        """Test workflow statistics."""
        orchestrator = WorkflowOrchestrator(
            queue=self.queue,
            reviewer=self.mock_reviewer,
            script_dir=self.script_dir,
            auto_review=False
        )

        # Create scripts in different states
        id1 = orchestrator.handle_cve_detected("CVE-2024-001", "numpy", "HIGH")
        id2 = orchestrator.handle_cve_detected("CVE-2024-002", "pandas", "MEDIUM")
        id3 = orchestrator.handle_cve_detected("CVE-2024-003", "scipy", "LOW")

        # Get statistics
        stats = orchestrator.get_workflow_statistics()

        self.assertIn("total_scripts", stats)
        self.assertIn("by_status", stats)
        self.assertIn("pending_reviews", stats)
        self.assertEqual(stats["total_scripts"], 3)

    def test_callback_registration(self):
        """Test callback registration and triggering."""
        orchestrator = WorkflowOrchestrator(
            queue=self.queue,
            reviewer=self.mock_reviewer,
            script_dir=self.script_dir,
            auto_review=False
        )

        # Register callback
        callback_called = []

        def on_script_ready(record, review):
            callback_called.append((record.id, review.confidence_score))

        orchestrator.register_callback("on_script_ready", on_script_ready)

        # Mock review
        mock_review = ReviewResult(
            approved=True,
            safe_to_run=True,
            confidence_score=0.88,
            risk_level=RiskLevel.LOW,
            issues_found=[],
            recommendations=[],
            review_summary="Safe",
            detailed_analysis="Details"
        )
        self.mock_reviewer.review_script.return_value = mock_review

        # Create and review script
        script_id = orchestrator.handle_cve_detected(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH"
        )
        orchestrator.handle_script_generated(script_id)

        # Verify callback was called
        self.assertEqual(len(callback_called), 1)
        self.assertEqual(callback_called[0][0], script_id)
        self.assertEqual(callback_called[0][1], 0.88)

    def test_retry_review(self):
        """Test retrying review."""
        self.mock_reviewer.review_script.return_value = ReviewResult(
            approved=True,
            safe_to_run=True,
            confidence_score=0.9,
            risk_level=RiskLevel.LOW,
            issues_found=[],
            recommendations=[],
            review_summary="Safe",
            detailed_analysis="Details"
        )

        orchestrator = WorkflowOrchestrator(
            queue=self.queue,
            reviewer=self.mock_reviewer,
            script_dir=self.script_dir,
            auto_review=False
        )

        # Create script
        script_id = orchestrator.handle_cve_detected(
            cve_id="CVE-2024-12345",
            library="numpy",
            severity="HIGH"
        )

        # Clear mock
        self.mock_reviewer.reset_mock()

        # Retry review
        success = orchestrator.retry_review(script_id)

        self.assertTrue(success)
        self.mock_reviewer.review_script.assert_called_once()


if __name__ == "__main__":
    unittest.main()
