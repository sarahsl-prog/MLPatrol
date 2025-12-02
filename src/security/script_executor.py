#!/usr/bin/env python3
"""
Script Executor - Safe sandboxed execution of security scripts.

Executes validated security scripts with timeout protection, output capture,
and comprehensive logging. Designed for running user-approved scripts only.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from src.models.script_state import ExecutionResult


class ScriptExecutor:
    """
    Safe executor for security validation scripts.

    Executes scripts in a sandboxed subprocess with timeout protection,
    output capture, and comprehensive audit logging.
    """

    def __init__(
        self,
        log_dir: str = "data/scripts/execution_logs",
        default_timeout: int = 60,
        max_output_size: int = 1024 * 1024  # 1MB
    ):
        """
        Initialize the script executor.

        Args:
            log_dir: Directory for execution logs
            default_timeout: Default timeout in seconds
            max_output_size: Maximum output size in bytes
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.default_timeout = default_timeout
        self.max_output_size = max_output_size

        # Execution log file (JSONL format)
        self.execution_log_file = self.log_dir / "execution_history.jsonl"

        logger.info(f"ScriptExecutor initialized (timeout={default_timeout}s)")

    def execute_script(
        self,
        script_path: str,
        script_hash: Optional[str] = None,
        timeout: Optional[int] = None,
        verify_hash: bool = True
    ) -> ExecutionResult:
        """
        Execute a security script safely.

        Args:
            script_path: Path to the script file
            script_hash: Expected SHA256 hash (for tamper detection)
            timeout: Execution timeout in seconds (None = default)
            verify_hash: Whether to verify script hash

        Returns:
            ExecutionResult with execution details

        Raises:
            FileNotFoundError: If script doesn't exist
            ValueError: If hash verification fails
            subprocess.TimeoutExpired: If execution exceeds timeout
        """
        start_time = time.time()
        timeout_seconds = timeout or self.default_timeout

        # Validate script exists
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Verify script hash (tamper detection)
        if verify_hash and script_hash:
            actual_hash = self._calculate_hash(script_path)
            if actual_hash != script_hash:
                error_msg = f"Script hash mismatch! Expected {script_hash[:16]}..., got {actual_hash[:16]}..."
                logger.error(error_msg)
                raise ValueError(error_msg)

        logger.info(f"Executing script: {script_path} (timeout={timeout_seconds}s)")

        try:
            # Execute script in subprocess
            # IMPORTANT: shell=False for security (no shell injection)
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False  # CRITICAL: Never use shell=True
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                exit_code = process.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                # Kill the process if timeout exceeded
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -1
                timed_out = True
                logger.warning(f"Script execution timed out after {timeout_seconds}s")

            # Truncate output if too large
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n... (output truncated)"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n... (output truncated)"

            # Calculate duration
            duration = time.time() - start_time

            # Determine success
            success = (exit_code == 0) and not timed_out

            # Create execution result
            result = ExecutionResult(
                success=success,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                error_message="Execution timed out" if timed_out else None
            )

            logger.info(
                f"Script execution complete: exit_code={exit_code}, "
                f"duration={duration:.2f}s, success={success}"
            )

            # Log to audit file
            self._log_execution(script_path, result)

            return result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Return failure result
            result = ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                error_message=error_msg
            )

            # Log to audit file
            self._log_execution(script_path, result, error=str(e))

            return result

    def execute_script_with_args(
        self,
        script_path: str,
        args: list,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute a script with command-line arguments.

        Args:
            script_path: Path to the script file
            args: List of command-line arguments
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        timeout_seconds = timeout or self.default_timeout

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        logger.info(f"Executing script with args: {script_path} {' '.join(args)}")

        try:
            # Execute with arguments
            cmd = [sys.executable, script_path] + args

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                exit_code = process.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -1
                timed_out = True

            duration = time.time() - start_time
            success = (exit_code == 0) and not timed_out

            result = ExecutionResult(
                success=success,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                error_message="Execution timed out" if timed_out else None
            )

            self._log_execution(script_path, result, args=args)

            return result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Execution failed: {str(e)}"

            result = ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                error_message=error_msg
            )

            self._log_execution(script_path, result, error=str(e), args=args)

            return result

    def validate_script_safety(self, script_path: str) -> Tuple[bool, list]:
        """
        Perform basic safety checks on a script before execution.

        Args:
            script_path: Path to the script file

        Returns:
            Tuple of (is_safe, issues_found)
        """
        issues = []

        if not os.path.exists(script_path):
            return False, ["Script file not found"]

        # Check file is readable
        if not os.access(script_path, os.R_OK):
            issues.append("Script is not readable")

        # Check file is a Python file
        if not script_path.endswith('.py'):
            issues.append("Script is not a Python file (.py)")

        # Read script content for basic checks
        try:
            with open(script_path, 'r') as f:
                content = f.read()

            # Basic dangerous pattern checks (similar to static analysis)
            dangerous_patterns = [
                ('eval(', 'Contains eval() - arbitrary code execution'),
                ('exec(', 'Contains exec() - arbitrary code execution'),
                ('__import__', 'Contains dynamic imports'),
                ('os.system(', 'Contains os.system() - command injection risk'),
            ]

            for pattern, message in dangerous_patterns:
                if pattern in content:
                    issues.append(message)

        except Exception as e:
            issues.append(f"Failed to read script: {str(e)}")

        is_safe = len(issues) == 0
        return is_safe, issues

    def _calculate_hash(self, script_path: str) -> str:
        """Calculate SHA256 hash of script file."""
        with open(script_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _log_execution(
        self,
        script_path: str,
        result: ExecutionResult,
        error: Optional[str] = None,
        args: Optional[list] = None
    ) -> None:
        """
        Log execution to audit file.

        Args:
            script_path: Path to executed script
            result: Execution result
            error: Error message if failed
            args: Command-line arguments if any
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "script_path": script_path,
            "success": result.success,
            "exit_code": result.exit_code,
            "duration_seconds": result.duration_seconds,
            "error": error or result.error_message,
            "args": args,
            "stdout_length": len(result.stdout),
            "stderr_length": len(result.stderr)
        }

        try:
            with open(self.execution_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write execution log: {e}")

    def get_execution_history(
        self,
        limit: int = 100,
        script_path: Optional[str] = None
    ) -> list:
        """
        Get execution history from audit log.

        Args:
            limit: Maximum number of entries to return
            script_path: Filter by script path (optional)

        Returns:
            List of execution log entries
        """
        if not self.execution_log_file.exists():
            return []

        entries = []
        try:
            with open(self.execution_log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if script_path is None or entry.get("script_path") == script_path:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue

            # Return most recent first
            entries.reverse()
            return entries[:limit]

        except Exception as e:
            logger.warning(f"Failed to read execution history: {e}")
            return []

    def get_statistics(self) -> dict:
        """
        Get execution statistics.

        Returns:
            Dictionary with statistics
        """
        history = self.get_execution_history(limit=1000)

        if not history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 0.0,
                "avg_duration_seconds": 0.0
            }

        total = len(history)
        successful = sum(1 for e in history if e.get("success", False))
        failed = total - successful
        success_rate = successful / total if total > 0 else 0.0

        durations = [e.get("duration_seconds", 0) for e in history if e.get("duration_seconds")]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": success_rate,
            "avg_duration_seconds": avg_duration
        }


def create_script_executor(**kwargs) -> ScriptExecutor:
    """
    Factory function to create a configured ScriptExecutor.

    Args:
        **kwargs: Arguments passed to ScriptExecutor constructor

    Returns:
        Configured ScriptExecutor instance

    Example:
        >>> executor = create_script_executor(default_timeout=30)
        >>> result = executor.execute_script("generated_checks/check_numpy.py")
        >>> print(f"Exit code: {result.exit_code}")
        >>> print(f"Output: {result.stdout}")
    """
    return ScriptExecutor(**kwargs)
