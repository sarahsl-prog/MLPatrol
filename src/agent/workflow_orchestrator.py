#!/usr/bin/env python3
"""
Workflow Orchestrator - Automated agent communication coordination.

Manages the complete lifecycle of security scripts from CVE detection through
execution, coordinating between the CVE monitor, script generator, LLM reviewer,
and execution engine.
"""

import hashlib
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Optional

from loguru import logger

from src.agent.script_queue import ScriptQueue
from src.agent.script_reviewer import ScriptReviewer, create_script_reviewer
from src.models.script_state import ScriptRecord, ScriptStatus
from src.security.code_generator import build_cve_security_script


class WorkflowOrchestrator:
    """
    Orchestrates automated workflow for security script lifecycle.

    Coordinates the complete workflow:
    1. CVE Detection → Script Generation
    2. Script Generation → LLM Review
    3. LLM Review → User Approval (GUI notification)
    4. User Approval → Ready for Execution
    5. Execution → Results Storage

    Thread-safe for concurrent operations from GUI and background threads.
    """

    def __init__(
        self,
        queue: Optional[ScriptQueue] = None,
        reviewer: Optional[ScriptReviewer] = None,
        script_dir: str = "generated_checks",
        auto_review: bool = True,
        review_timeout: int = 120,
        min_confidence: float = 0.7
    ):
        """
        Initialize the workflow orchestrator.

        Args:
            queue: ScriptQueue instance (creates new if None)
            reviewer: ScriptReviewer instance (creates new if None)
            script_dir: Directory for generated scripts
            auto_review: Automatically trigger LLM review after generation
            review_timeout: Timeout for review operations (seconds)
            min_confidence: Minimum confidence score to approve
        """
        self.queue = queue or ScriptQueue()
        self.reviewer = reviewer or create_script_reviewer()
        self.script_dir = Path(script_dir)
        self.script_dir.mkdir(parents=True, exist_ok=True)

        self.auto_review = auto_review
        self.review_timeout = review_timeout
        self.min_confidence = min_confidence

        self._lock = threading.RLock()
        self._callbacks: Dict[str, list] = {
            "on_script_ready": [],  # Script reviewed and ready for user approval
            "on_review_failed": [],  # Review failed
            "on_execution_ready": [],  # Script approved by user
        }

        # Register queue event hooks
        self._register_queue_hooks()

        logger.info(f"WorkflowOrchestrator initialized (auto_review={auto_review})")

    def _register_queue_hooks(self) -> None:
        """Register event hooks with the script queue."""
        if self.auto_review:
            self.queue.register_hook("on_script_generated", self._on_script_generated_hook)

    def _on_script_generated_hook(self, record: ScriptRecord) -> None:
        """Hook called when script is generated."""
        try:
            self.handle_script_generated(record.id)
        except Exception as e:
            logger.error(f"Error in script generated hook: {e}", exc_info=True)

    # ========================================================================
    # Stage 1: CVE Detection → Script Generation
    # ========================================================================

    def handle_cve_detected(
        self,
        cve_id: str,
        library: str,
        severity: str,
        description: str = "",
        affected_versions: Optional[list] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Handle CVE detection and initiate script generation.

        Args:
            cve_id: CVE identifier
            library: Library name
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            description: CVE description
            affected_versions: List of affected version strings
            metadata: Additional metadata

        Returns:
            Script ID (UUID)
        """
        with self._lock:
            logger.info(f"CVE detected: {cve_id} ({library}, {severity})")

            # Create script record
            meta = metadata or {}
            meta.update({
                "description": description,
                "affected_versions": affected_versions or [],
                "detected_at": datetime.now().isoformat()
            })

            script_id = self.queue.add_script(
                cve_id=cve_id,
                library=library,
                severity=severity,
                metadata=meta
            )

            # Generate script immediately
            try:
                self._generate_script(script_id, cve_id, library, description, affected_versions)
                logger.info(f"Script generated for {cve_id}: {script_id}")
            except Exception as e:
                logger.error(f"Script generation failed for {cve_id}: {e}", exc_info=True)
                self.queue.update_status(
                    script_id,
                    ScriptStatus.FAILED,
                    error=f"Generation failed: {str(e)}"
                )

            return script_id

    def _generate_script(
        self,
        script_id: str,
        cve_id: str,
        library: str,
        description: str,
        affected_versions: Optional[list]
    ) -> None:
        """
        Generate security script for CVE.

        Args:
            script_id: Script UUID
            cve_id: CVE ID
            library: Library name
            description: CVE description
            affected_versions: Affected version list
        """
        # Generate script content
        purpose = f"Check for {cve_id}"
        script_content = build_cve_security_script(
            purpose=purpose,
            library=library,
            cve_id=cve_id,
            affected_versions=affected_versions
        )

        # Save to file
        script_filename = f"{cve_id.replace('-', '_')}_{library}_check.py"
        script_path = self.script_dir / script_filename

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Calculate hash
        script_hash = hashlib.sha256(script_content.encode()).hexdigest()

        # Update queue
        self.queue.update_script_path(script_id, str(script_path), script_hash)
        self.queue.update_status(script_id, ScriptStatus.GENERATED)

        logger.info(f"Script saved: {script_path}")

    # ========================================================================
    # Stage 2: Script Generation → LLM Review
    # ========================================================================

    def handle_script_generated(self, script_id: str) -> None:
        """
        Handle script generation completion and initiate review.

        Args:
            script_id: Script UUID
        """
        with self._lock:
            record = self.queue.get_script(script_id)
            if not record:
                logger.warning(f"Script {script_id} not found")
                return

            logger.info(f"Initiating review for {record.cve_id}")

            # Update status to UNDER_REVIEW
            self.queue.update_status(script_id, ScriptStatus.UNDER_REVIEW)

            try:
                # Perform review
                review_result = self.reviewer.review_script(
                    script_path=record.script_path,
                    cve_id=record.cve_id,
                    library=record.library,
                    severity=record.severity,
                    description=record.metadata.get("description", "")
                )

                # Store review result
                self.queue.add_review_result(script_id, review_result)

                # Update status to REVIEWED
                self.queue.update_status(script_id, ScriptStatus.REVIEWED)

                logger.info(
                    f"Review complete for {record.cve_id}: "
                    f"risk={review_result.risk_level}, "
                    f"confidence={review_result.confidence_score:.2f}, "
                    f"safe={review_result.safe_to_run}"
                )

                # Trigger callbacks
                self._trigger_callback("on_script_ready", record, review_result)

            except Exception as e:
                logger.error(f"Review failed for {script_id}: {e}", exc_info=True)
                self.queue.update_status(
                    script_id,
                    ScriptStatus.FAILED,
                    error=f"Review failed: {str(e)}"
                )
                self._trigger_callback("on_review_failed", record, str(e))

    # ========================================================================
    # Stage 3: LLM Review → User Approval
    # ========================================================================

    def handle_review_complete(self, script_id: str) -> None:
        """
        Handle review completion.

        This is called when review is complete. The script is now waiting
        for user approval via GUI.

        Args:
            script_id: Script UUID
        """
        record = self.queue.get_script(script_id)
        if not record:
            logger.warning(f"Script {script_id} not found")
            return

        logger.info(f"Script {record.cve_id} ready for user review")

        # Script stays in REVIEWED status until user approves/rejects
        # GUI will display the review results and provide action buttons

    # ========================================================================
    # Stage 4: User Approval → Execution Ready
    # ========================================================================

    def handle_user_approved(self, script_id: str) -> bool:
        """
        Handle user approval of script.

        Args:
            script_id: Script UUID

        Returns:
            True if approved, False if validation failed
        """
        with self._lock:
            record = self.queue.get_script(script_id)
            if not record:
                logger.warning(f"Script {script_id} not found")
                return False

            # Validate review is recent (< 24 hours)
            if record.review_result and record.reviewed_at:
                age = datetime.now() - record.reviewed_at
                if age > timedelta(hours=24):
                    logger.warning(f"Review is stale ({age.total_seconds()/3600:.1f}h old)")
                    return False

            # Validate review exists and is positive
            if not record.review_result:
                logger.warning(f"No review result for {script_id}")
                return False

            if not record.review_result.safe_to_run:
                logger.warning(f"Review marked script as unsafe: {script_id}")
                return False

            # Update status to APPROVED
            self.queue.update_status(script_id, ScriptStatus.APPROVED)

            logger.info(f"Script approved by user: {record.cve_id}")

            # Trigger callbacks
            self._trigger_callback("on_execution_ready", record)

            return True

    def handle_user_rejected(self, script_id: str, reason: str = "") -> bool:
        """
        Handle user rejection of script.

        Args:
            script_id: Script UUID
            reason: Rejection reason

        Returns:
            True if updated, False if not found
        """
        with self._lock:
            record = self.queue.get_script(script_id)
            if not record:
                logger.warning(f"Script {script_id} not found")
                return False

            self.queue.update_status(
                script_id,
                ScriptStatus.REJECTED,
                metadata={"rejection_reason": reason},
                error=f"Rejected by user: {reason}" if reason else "Rejected by user"
            )

            logger.info(f"Script rejected by user: {record.cve_id}")
            return True

    # ========================================================================
    # Stage 5: Execution → Results
    # ========================================================================

    def handle_execution_complete(
        self,
        script_id: str,
        success: bool,
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
        duration: float = 0.0,
        error: Optional[str] = None
    ) -> bool:
        """
        Handle script execution completion.

        Args:
            script_id: Script UUID
            success: Execution succeeded
            exit_code: Process exit code
            stdout: Standard output
            stderr: Standard error
            duration: Execution duration (seconds)
            error: Error message if failed

        Returns:
            True if updated, False if not found
        """
        with self._lock:
            from src.models.script_state import ExecutionResult

            record = self.queue.get_script(script_id)
            if not record:
                logger.warning(f"Script {script_id} not found")
                return False

            # Create execution result
            execution_result = ExecutionResult(
                success=success,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                error_message=error
            )

            # Store execution result
            self.queue.add_execution_result(script_id, execution_result)

            # Update status
            new_status = ScriptStatus.COMPLETED if success else ScriptStatus.FAILED
            self.queue.update_status(
                script_id,
                new_status,
                error=error if error else None
            )

            logger.info(
                f"Execution complete for {record.cve_id}: "
                f"success={success}, exit_code={exit_code}, duration={duration:.2f}s"
            )

            return True

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_pending_reviews(self) -> list:
        """
        Get scripts waiting for review.

        Returns:
            List of scripts with status GENERATED
        """
        return self.queue.get_scripts_by_status([ScriptStatus.GENERATED])

    def get_reviewed_scripts(self) -> list:
        """
        Get scripts that have been reviewed and are awaiting user action.

        Returns:
            List of scripts with status REVIEWED
        """
        return self.queue.get_scripts_by_status([ScriptStatus.REVIEWED])

    def get_approved_scripts(self) -> list:
        """
        Get scripts approved by user and ready for execution.

        Returns:
            List of scripts with status APPROVED
        """
        return self.queue.get_scripts_by_status([ScriptStatus.APPROVED])

    def get_script_status(self, script_id: str) -> Optional[ScriptRecord]:
        """
        Get current status of a script.

        Args:
            script_id: Script UUID

        Returns:
            ScriptRecord or None if not found
        """
        return self.queue.get_script(script_id)

    # ========================================================================
    # Callback Management
    # ========================================================================

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for workflow events.

        Args:
            event: Event name (on_script_ready, on_review_failed, on_execution_ready)
            callback: Callback function

        Raises:
            ValueError: If event name is invalid
        """
        if event not in self._callbacks:
            raise ValueError(f"Invalid event: {event}. Valid: {list(self._callbacks.keys())}")

        with self._lock:
            self._callbacks[event].append(callback)
            logger.info(f"Registered callback for event: {event}")

    def _trigger_callback(self, event: str, *args, **kwargs) -> None:
        """Trigger callbacks for an event."""
        if event not in self._callbacks:
            return

        for callback in self._callbacks[event]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for {event}: {e}", exc_info=True)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def retry_review(self, script_id: str) -> bool:
        """
        Retry LLM review for a script.

        Args:
            script_id: Script UUID

        Returns:
            True if review triggered, False if script not found
        """
        record = self.queue.get_script(script_id)
        if not record:
            logger.warning(f"Script {script_id} not found")
            return False

        logger.info(f"Retrying review for {record.cve_id}")
        self.handle_script_generated(script_id)
        return True

    def get_workflow_statistics(self) -> Dict:
        """
        Get workflow statistics.

        Returns:
            Dictionary with workflow statistics
        """
        stats = self.queue.get_statistics()

        # Add workflow-specific stats
        pending_reviews = len(self.get_pending_reviews())
        awaiting_approval = len(self.get_reviewed_scripts())
        ready_to_execute = len(self.get_approved_scripts())

        stats.update({
            "pending_reviews": pending_reviews,
            "awaiting_user_approval": awaiting_approval,
            "ready_to_execute": ready_to_execute
        })

        return stats

    def cleanup_stale_scripts(self, max_age_days: int = 7) -> int:
        """
        Clean up stale scripts.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of scripts cleaned up
        """
        count = 0
        all_scripts = self.queue.get_all_scripts(limit=1000)

        for script in all_scripts:
            if script.is_stale and script.status.is_terminal:
                self.queue.delete_script(script.id)
                count += 1

        logger.info(f"Cleaned up {count} stale scripts")
        return count


def create_workflow_orchestrator(**kwargs) -> WorkflowOrchestrator:
    """
    Factory function to create a configured WorkflowOrchestrator.

    Args:
        **kwargs: Arguments passed to WorkflowOrchestrator constructor

    Returns:
        Configured WorkflowOrchestrator instance

    Example:
        >>> orchestrator = create_workflow_orchestrator(auto_review=True)
        >>> script_id = orchestrator.handle_cve_detected(
        ...     cve_id="CVE-2024-12345",
        ...     library="numpy",
        ...     severity="HIGH",
        ...     affected_versions=["1.21.0", "1.21.1"]
        ... )
    """
    return WorkflowOrchestrator(**kwargs)
