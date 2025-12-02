#!/usr/bin/env python3
"""
Unit tests for ScriptExecutor.
"""

import os
import tempfile
import unittest
from pathlib import Path

from src.security.script_executor import ScriptExecutor


class TestScriptExecutor(unittest.TestCase):
    """Test cases for ScriptExecutor."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")
        self.executor = ScriptExecutor(log_dir=self.log_dir, default_timeout=5)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_executor_initialization(self):
        """Test executor initialization."""
        self.assertIsNotNone(self.executor)
        self.assertEqual(self.executor.default_timeout, 5)
        self.assertTrue(Path(self.log_dir).exists())

    def test_execute_simple_script(self):
        """Test executing a simple script."""
        # Create a simple test script
        script_path = os.path.join(self.temp_dir, "test_script.py")
        with open(script_path, 'w') as f:
            f.write("""#!/usr/bin/env python3
import sys
print("Hello from test script")
sys.exit(0)
""")

        result = self.executor.execute_script(script_path, verify_hash=False)

        self.assertTrue(result.success)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Hello from test script", result.stdout)
        self.assertEqual(result.stderr, "")
        self.assertIsNone(result.error_message)
        self.assertGreater(result.duration_seconds, 0)

    def test_execute_script_with_error(self):
        """Test executing a script that raises an error."""
        script_path = os.path.join(self.temp_dir, "error_script.py")
        with open(script_path, 'w') as f:
            f.write("""#!/usr/bin/env python3
import sys
print("About to error", file=sys.stderr)
raise ValueError("Test error")
""")

        result = self.executor.execute_script(script_path, verify_hash=False)

        self.assertFalse(result.success)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("ValueError", result.stderr)

    def test_execute_script_with_exit_code(self):
        """Test script with specific exit code."""
        script_path = os.path.join(self.temp_dir, "exit_code_script.py")
        with open(script_path, 'w') as f:
            f.write("""#!/usr/bin/env python3
import sys
print("Script with exit code 42")
sys.exit(42)
""")

        result = self.executor.execute_script(script_path, verify_hash=False)

        self.assertFalse(result.success)  # Non-zero exit code = failure
        self.assertEqual(result.exit_code, 42)

    def test_execute_script_timeout(self):
        """Test script execution timeout."""
        script_path = os.path.join(self.temp_dir, "slow_script.py")
        with open(script_path, 'w') as f:
            f.write("""#!/usr/bin/env python3
import time
print("Starting slow script")
time.sleep(10)  # Sleep longer than timeout
print("This should not print")
""")

        result = self.executor.execute_script(script_path, verify_hash=False, timeout=1)

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, -1)
        self.assertEqual(result.error_message, "Execution timed out")

    def test_execute_script_with_args(self):
        """Test executing script with command-line arguments."""
        script_path = os.path.join(self.temp_dir, "args_script.py")
        with open(script_path, 'w') as f:
            f.write("""#!/usr/bin/env python3
import sys
print(f"Args: {sys.argv[1:]}")
sys.exit(0)
""")

        result = self.executor.execute_script_with_args(
            script_path,
            args=["--arg1", "value1", "--arg2", "value2"]
        )

        self.assertTrue(result.success)
        self.assertIn("--arg1", result.stdout)
        self.assertIn("value1", result.stdout)

    def test_execute_nonexistent_script(self):
        """Test executing a script that doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.executor.execute_script("/nonexistent/script.py", verify_hash=False)

    def test_hash_verification(self):
        """Test script hash verification."""
        script_path = os.path.join(self.temp_dir, "hash_script.py")
        script_content = """#!/usr/bin/env python3
print("Test script")
"""
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Calculate correct hash
        import hashlib
        correct_hash = hashlib.sha256(script_content.encode()).hexdigest()

        # Execute with correct hash
        result = self.executor.execute_script(
            script_path,
            script_hash=correct_hash,
            verify_hash=True
        )
        self.assertTrue(result.success)

        # Execute with incorrect hash
        with self.assertRaises(ValueError):
            self.executor.execute_script(
                script_path,
                script_hash="0" * 64,  # Wrong hash
                verify_hash=True
            )

    def test_validate_script_safety(self):
        """Test script safety validation."""
        # Safe script
        safe_script = os.path.join(self.temp_dir, "safe.py")
        with open(safe_script, 'w') as f:
            f.write("print('safe')")

        is_safe, issues = self.executor.validate_script_safety(safe_script)
        self.assertTrue(is_safe)
        self.assertEqual(len(issues), 0)

        # Unsafe script with eval
        unsafe_script = os.path.join(self.temp_dir, "unsafe.py")
        with open(unsafe_script, 'w') as f:
            f.write("result = eval(input('Enter code: '))")

        is_safe, issues = self.executor.validate_script_safety(unsafe_script)
        self.assertFalse(is_safe)
        self.assertTrue(any("eval" in issue.lower() for issue in issues))

    def test_execution_logging(self):
        """Test execution audit logging."""
        script_path = os.path.join(self.temp_dir, "logged_script.py")
        with open(script_path, 'w') as f:
            f.write("print('logged execution')")

        # Execute script
        result = self.executor.execute_script(script_path, verify_hash=False)

        # Check log file exists
        self.assertTrue(self.executor.execution_log_file.exists())

        # Get execution history
        history = self.executor.get_execution_history()

        self.assertGreater(len(history), 0)
        last_entry = history[0]
        self.assertEqual(last_entry["script_path"], script_path)
        self.assertTrue(last_entry["success"])

    def test_get_execution_history_filtered(self):
        """Test filtered execution history."""
        # Create and execute two different scripts
        script1 = os.path.join(self.temp_dir, "script1.py")
        script2 = os.path.join(self.temp_dir, "script2.py")

        for script in [script1, script2]:
            with open(script, 'w') as f:
                f.write("print('test')")

        self.executor.execute_script(script1, verify_hash=False)
        self.executor.execute_script(script2, verify_hash=False)

        # Get filtered history
        history_all = self.executor.get_execution_history()
        history_script1 = self.executor.get_execution_history(script_path=script1)

        self.assertGreaterEqual(len(history_all), 2)
        self.assertEqual(len(history_script1), 1)
        self.assertEqual(history_script1[0]["script_path"], script1)

    def test_get_statistics(self):
        """Test execution statistics."""
        # Execute some scripts
        success_script = os.path.join(self.temp_dir, "success.py")
        fail_script = os.path.join(self.temp_dir, "fail.py")

        with open(success_script, 'w') as f:
            f.write("import sys; sys.exit(0)")

        with open(fail_script, 'w') as f:
            f.write("import sys; sys.exit(1)")

        self.executor.execute_script(success_script, verify_hash=False)
        self.executor.execute_script(fail_script, verify_hash=False)

        stats = self.executor.get_statistics()

        self.assertGreaterEqual(stats["total_executions"], 2)
        self.assertGreaterEqual(stats["successful_executions"], 1)
        self.assertGreaterEqual(stats["failed_executions"], 1)
        self.assertGreater(stats["success_rate"], 0)
        self.assertGreater(stats["avg_duration_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
