# MLPatrol Project Reorganization Plan

**Date:** November 29, 2025
**Status:** PROPOSED - Awaiting Approval

---

## Executive Summary

This plan reorganizes the MLPatrol project structure to improve maintainability, separate concerns, and implement proper logging with Loguru. The reorganization addresses:

1. Temp files consolidation
2. Runtime data file organization
3. Loguru-based logging for all components
4. Test output organization
5. Base directory cleanup

---

## 1. Temp Files → `/temp` Directory

### Current Issues
- Temp files scattered across project root
- No clear separation between temp and permanent files
- `.gitignore` doesn't have consistent temp handling

### Proposed Changes

#### Create `/temp` Directory Structure
```
temp/
├── downloads/          # Temporary file downloads
├── processing/         # Intermediate processing files
├── cache/             # Temporary cache files
└── .gitkeep          # Keep directory in git
```

#### Update `.gitignore`
```gitignore
# Temporary files
temp/
!temp/.gitkeep
```

#### Files to Move
**None currently identified** - No temp files currently being written

#### Code Changes Required
- **None** - No code currently writes to temp locations
- Future temp file operations should use `Path("temp")` directory

---

## 2. Runtime Data Files → `/data` Directory

### Current Issues
- `.last_cve_check` in project root (should be in data/)
- `alerts.json` correctly in data/ but inconsistent
- `data/` directory exists but not fully utilized

### Current State
```
data/
└── alerts.json        # ✅ Already correct
```

### Proposed Changes

#### Expand `/data` Directory Structure
```
data/
├── alerts.json        # Application alerts (existing)
├── cve_cache/         # CVE search cache
│   └── last_check.txt # Last CVE check timestamp
├── user_data/         # User-specific data
└── .gitkeep          # Keep directory structure
```

#### Files to Move
1. **`.last_cve_check`** → **`data/cve_cache/last_check.txt`**
   - Location: Project root → `data/cve_cache/`
   - Used by: `app.py` line 979
   - Purpose: Track last CVE check timestamp

#### Update `.gitignore`
```gitignore
# Data folders
data/
!data/.gitkeep
!data/*/.gitkeep
```

#### Code Changes Required

**File:** `app.py` (line 979)
```python
# BEFORE
last_file = Path(".last_cve_check")

# AFTER
last_file = Path("data/cve_cache/last_check.txt")
```

**Create directory structure:**
```python
# Add to app startup
Path("data/cve_cache").mkdir(parents=True, exist_ok=True)
```

---

## 3. Loguru Logging Implementation

### Current Issues
- Uses standard `logging` module
- Single log file `mlpatrol.log` in root
- No per-component logging
- No agent-specific log files
- File handler configured in `app.py`

### Current Logging Configuration
**File:** `app.py` lines 63-67
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mlpatrol.log"), logging.StreamHandler(sys.stdout)],
)
```

**File:** `src/utils/logger.py`
- Uses standard logging
- No file handler configuration
- Simple `get_logger()` wrapper

### Proposed Changes

#### Create `/logs` Directory Structure
```
logs/
├── app/
│   ├── main_{date}.log           # Main application log
│   └── coordinator_{date}.log    # Background coordinator
├── agents/
│   ├── reasoning_chain_{date}.log  # ReasoningChain agent
│   ├── tools_{date}.log            # Agent tools
│   └── custom_agent_{date}.log     # Custom agents
├── security/
│   ├── cve_monitor_{date}.log      # CVE monitoring
│   ├── code_generator_{date}.log   # Code generation
│   └── threat_intel_{date}.log     # Threat intelligence
├── dataset/
│   ├── bias_analyzer_{date}.log    # Bias analysis
│   └── poisoning_detector_{date}.log # Poisoning detection
└── .gitkeep
```

#### Replace `src/utils/logger.py` with Loguru

**New Implementation:**
```python
"""Logging utilities using Loguru for MLPatrol."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Ensure logs directory exists
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Remove default handler
logger.remove()

# Add console handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


def get_logger(
    name: str,
    log_dir: str = "app",
    rotation: str = "10 MB",
    retention: str = "1 week",
    level: str = "INFO",
):
    """Get a logger with file and console output.

    Args:
        name: Logger name (used in log messages)
        log_dir: Subdirectory under logs/ (app, agents, security, dataset)
        rotation: When to rotate log files
        retention: How long to keep old log files
        level: Logging level

    Returns:
        Loguru logger instance
    """
    # Create subdirectory
    log_path = LOGS_DIR / log_dir
    log_path.mkdir(parents=True, exist_ok=True)

    # Add file handler with rotation
    log_file = log_path / f"{name}_{{time:YYYY-MM-DD}}.log"
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True,  # Thread-safe
    )

    return logger.bind(name=name)


def get_agent_logger(agent_name: str):
    """Get logger for an agent.

    Args:
        agent_name: Name of the agent (e.g., 'reasoning_chain', 'cve_monitor')

    Returns:
        Loguru logger configured for agents
    """
    return get_logger(agent_name, log_dir="agents")


def get_security_logger(component_name: str):
    """Get logger for security components.

    Args:
        component_name: Name of security component

    Returns:
        Loguru logger configured for security components
    """
    return get_logger(component_name, log_dir="security")


def get_dataset_logger(component_name: str):
    """Get logger for dataset components.

    Args:
        component_name: Name of dataset component

    Returns:
        Loguru logger configured for dataset components
    """
    return get_logger(component_name, log_dir="dataset")
```

#### Update All Components

**1. Main Application (`app.py`)**
```python
# Replace lines 63-67 with:
from src.utils.logger import get_logger

logger = get_logger("main", log_dir="app")
```

**2. Reasoning Chain (`src/agent/reasoning_chain.py`)**
```python
from src.utils.logger import get_agent_logger

logger = get_agent_logger("reasoning_chain")
```

**3. Agent Tools (`src/agent/tools.py`)**
```python
from src.utils.logger import get_agent_logger

logger = get_agent_logger("tools")
```

**4. CVE Monitor (`src/security/cve_monitor.py`)**
```python
from src.utils.logger import get_security_logger

logger = get_security_logger("cve_monitor")
```

**5. Code Generator (`src/security/code_generator.py`)**
```python
from src.utils.logger import get_security_logger

logger = get_security_logger("code_generator")
```

**6. Threat Intel (`src/security/threat_intel.py`)**
```python
from src.utils.logger import get_security_logger

logger = get_security_logger("threat_intel")
```

**7. Bias Analyzer (`src/dataset/bias_analyzer.py`)**
```python
from src.utils.logger import get_dataset_logger

logger = get_dataset_logger("bias_analyzer")
```

**8. Poisoning Detector (`src/dataset/poisoning_detector.py`)**
```python
from src.utils.logger import get_dataset_logger

logger = get_dataset_logger("poisoning_detector")
```

#### Add Loguru Dependency

**File:** `requirements.txt`
```python
# Add after other dependencies
loguru>=0.7.0
```

#### Update `.gitignore`
```gitignore
# Logs
*.log
logs/
!logs/.gitkeep
!logs/*/.gitkeep
```

---

## 4. Test Outputs → `/tests/outputs` Directory

### Current Issues
- Test coverage files in `testse2e/` directory (odd naming)
- `.coverage` file in project root
- No clear organization of test artifacts

### Current Test Output Files
```
.coverage                           # Coverage database (root)
testse2e/coverage.json             # Coverage JSON report
testse2e/coverage_baseline.json    # Coverage baseline
testse2e/baseline_tests.txt        # Test baseline
testse2e/test_results.txt          # Test results
```

### Proposed Changes

#### Create `/tests/outputs` Directory Structure
```
tests/
├── outputs/
│   ├── coverage/
│   │   ├── .coverage              # Coverage database
│   │   ├── coverage.json          # Coverage JSON
│   │   ├── coverage_baseline.json # Coverage baseline
│   │   └── htmlcov/               # HTML coverage reports
│   ├── results/
│   │   ├── test_results.txt       # Test execution results
│   │   └── baseline_tests.txt     # Test baselines
│   ├── pytest_cache/              # Pytest cache
│   └── .gitkeep
│
├── e2e/                           # E2E tests (existing)
├── test_*.py                      # Unit tests (existing)
└── conftest.py                    # Pytest config (if needed)
```

#### Files to Move/Delete

**Move:**
1. `.coverage` (root) → `tests/outputs/coverage/.coverage`
2. `testse2e/coverage.json` → `tests/outputs/coverage/coverage.json`
3. `testse2e/coverage_baseline.json` → `tests/outputs/coverage/coverage_baseline.json`
4. `testse2e/baseline_tests.txt` → `tests/outputs/results/baseline_tests.txt`
5. `testse2e/test_results.txt` → `tests/outputs/results/test_results.txt`

**Delete:**
- `testse2e/` directory (entire directory after moving files)

#### Update pytest Configuration

**File:** `pyproject.toml` - Add/update `[tool.pytest.ini_options]`
```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
cache_dir = "tests/outputs/pytest_cache"
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--showlocals",
    "--tb=short",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]
```

#### Update coverage Configuration

**File:** `pyproject.toml` - Update `[tool.coverage.run]`
```toml
[tool.coverage.run]
branch = true
source = ["src", "app.py"]
data_file = "tests/outputs/coverage/.coverage"
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__init__.py",
    "*/venv/*",
    "*/mlvenv/*",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if TYPE_CHECKING:",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "@(abc\\.)?abstractmethod",
]

[tool.coverage.json]
output = "tests/outputs/coverage/coverage.json"

[tool.coverage.html]
directory = "tests/outputs/coverage/htmlcov"
```

#### Update `.gitignore`
```gitignore
# Testing
.coverage
coverage.json
coverage_baseline.json
htmlcov/
.pytest_cache/
tests/outputs/
!tests/outputs/.gitkeep
```

---

## 5. Base Directory Cleanup

### Current Base Directory Files

#### ✅ **KEEP - Essential Files**
1. **`README.md`** - Project documentation
2. **`LICENSE`** - MIT license
3. **`app.py`** - Main application entry point
4. **`.gitignore`** - Git ignore rules
5. **`.env.example`** - Environment template
6. **`.flake8`** - Flake8 configuration
7. **`.pre-commit-config.yaml`** - Pre-commit hooks
8. **`.secrets.baseline`** - Secrets detection baseline
9. **`pyproject.toml`** - Project configuration
10. **`requirements.txt`** - Production dependencies
11. **`requirements-dev.txt`** - Development dependencies
12. **`CONTRIBUTING.md`** - Contribution guidelines
13. **`get_started.py`** - Quick start script
14. **`MLPatrol.code-workspace`** - VS Code workspace

#### ❌ **DELETE - Obsolete/Temp Files**
1. **`=0.2.0`** - Malformed pip install artifact
2. **`=3.5.0`** - Malformed pip install artifact
3. **`requirements.txt.bak`** - Backup file (use git instead)
4. **`.coverage`** - Move to tests/outputs/coverage/
5. **`mlpatrol.log`** - Old log file (will be replaced by loguru logs/)
6. **`.last_cve_check`** - Move to data/cve_cache/last_check.txt
7. **`COMPREHENSIVE_CODE_ANALYSIS_REPORT.md`** - Duplicate (also in docs/changes/)

#### 📦 **MOVE - Relocate to Proper Location**
1. **`analysis_report.md`** → **`docs/changes/analysis_report.md`**
2. **`mlpatrol_logo.png`** → **`assets/images/mlpatrol_logo.png`**

#### 🔍 **INVESTIGATE - Check if Still Needed**
1. **`generated_checks/`** directory - What creates this? Is it needed?

### Proposed Final Base Directory Structure
```
MLPatrol/
├── .github/              # GitHub workflows
├── .vscode/              # VS Code settings
├── assets/               # Static assets
│   ├── images/          # Images (move logo here)
│   └── videos/          # Videos
├── data/                 # Runtime data
│   ├── cve_cache/
│   └── user_data/
├── docs/                 # Documentation
│   └── changes/         # Change logs, summaries
├── examples/             # Example code/datasets
├── logs/                 # Application logs (Loguru)
│   ├── app/
│   ├── agents/
│   ├── security/
│   └── dataset/
├── src/                  # Source code
│   ├── agent/
│   ├── dataset/
│   ├── mcp/
│   ├── security/
│   └── utils/
├── temp/                 # Temporary files
│   ├── downloads/
│   ├── processing/
│   └── cache/
├── tests/                # Test suite
│   ├── e2e/
│   └── outputs/         # Test outputs
│       ├── coverage/
│       └── results/
├── .env.example          # Environment template
├── .flake8               # Flake8 config
├── .gitignore            # Git ignore
├── .pre-commit-config.yaml  # Pre-commit hooks
├── .secrets.baseline     # Secrets baseline
├── app.py                # Main application
├── CONTRIBUTING.md       # Contribution guide
├── get_started.py        # Quick start
├── LICENSE               # MIT license
├── MLPatrol.code-workspace  # VS Code workspace
├── pyproject.toml        # Project config
├── README.md             # Main documentation
├── requirements.txt      # Production deps
└── requirements-dev.txt  # Development deps
```

---

## 6. Implementation Checklist

### Phase 1: Create Directory Structure
- [ ] Create `temp/` with subdirectories
- [ ] Create `data/cve_cache/`
- [ ] Create `logs/` with subdirectories (app, agents, security, dataset)
- [ ] Create `tests/outputs/` with subdirectories (coverage, results)
- [ ] Create `assets/images/`
- [ ] Add `.gitkeep` files to preserve empty directories

### Phase 2: Install Dependencies
- [ ] Add `loguru>=0.7.0` to `requirements.txt`
- [ ] Run `pip install loguru`

### Phase 3: Update Logging
- [ ] Rewrite `src/utils/logger.py` with Loguru implementation
- [ ] Update `app.py` to use new logger
- [ ] Update `src/agent/reasoning_chain.py` with agent logger
- [ ] Update `src/agent/tools.py` with agent logger
- [ ] Update `src/security/cve_monitor.py` with security logger
- [ ] Update `src/security/code_generator.py` with security logger
- [ ] Update `src/security/threat_intel.py` with security logger
- [ ] Update `src/dataset/bias_analyzer.py` with dataset logger
- [ ] Update `src/dataset/poisoning_detector.py` with dataset logger

### Phase 4: Move Runtime Data Files
- [ ] Update `app.py` line 979: `.last_cve_check` → `data/cve_cache/last_check.txt`
- [ ] Add directory creation code in `app.py` startup
- [ ] Test CVE monitoring still works

### Phase 5: Move Test Outputs
- [ ] Update `pyproject.toml` pytest cache_dir
- [ ] Update `pyproject.toml` coverage data_file, json output, html directory
- [ ] Move `.coverage` to `tests/outputs/coverage/`
- [ ] Move `testse2e/*.json` to `tests/outputs/coverage/`
- [ ] Move `testse2e/*.txt` to `tests/outputs/results/`
- [ ] Delete `testse2e/` directory
- [ ] Run tests to verify new locations work

### Phase 6: Clean Up Base Directory
- [ ] Delete `=0.2.0` and `=3.5.0`
- [ ] Delete `requirements.txt.bak`
- [ ] Delete old `mlpatrol.log`
- [ ] Delete duplicate `COMPREHENSIVE_CODE_ANALYSIS_REPORT.md` from root
- [ ] Move `analysis_report.md` to `docs/changes/`
- [ ] Move `mlpatrol_logo.png` to `assets/images/`
- [ ] Investigate `generated_checks/` - delete if not needed

### Phase 7: Update .gitignore
- [ ] Add `temp/` exclusion (keep .gitkeep)
- [ ] Update `logs/` exclusion (keep .gitkeep)
- [ ] Remove `.last_cve_check` (now in data/)
- [ ] Add `tests/outputs/` exclusion (keep .gitkeep)
- [ ] Remove old `coverage.json`, `.coverage` entries (now in tests/outputs/)

### Phase 8: Testing & Validation
- [ ] Run all unit tests: `pytest tests/`
- [ ] Run all E2E tests: `pytest tests/e2e/`
- [ ] Verify logging works (check logs/ directory)
- [ ] Verify CVE check works (check data/cve_cache/)
- [ ] Verify coverage reports generate correctly
- [ ] Run full CI/CD checks (isort, black, flake8, pre-commit)

### Phase 9: Documentation
- [ ] Update README.md with new structure
- [ ] Update CONTRIBUTING.md with new logging usage
- [ ] Update DEVELOPMENT.md (if exists) with new paths
- [ ] Create migration guide for contributors

---

## 7. Benefits of This Reorganization

### ✅ **Improved Organization**
- Clear separation of concerns (data, logs, temp, tests)
- Easier to find files
- Consistent directory structure

### ✅ **Better Logging**
- Per-component log files with Loguru
- Automatic log rotation and compression
- Thread-safe logging
- Colored console output
- Structured logging with context

### ✅ **Cleaner Git Repo**
- All temp/output files properly ignored
- No clutter in root directory
- Clear what's tracked vs generated

### ✅ **Better Testing**
- All test outputs in one place
- Coverage reports organized
- Easy to clean test artifacts

### ✅ **Developer Experience**
- Easier onboarding (clear structure)
- Better debugging (separate log files)
- Faster troubleshooting (find logs by component)

---

## 8. Estimated Effort

- **Phase 1-2 (Setup):** 30 minutes
- **Phase 3 (Logging):** 2 hours
- **Phase 4-5 (File moves):** 1 hour
- **Phase 6-7 (Cleanup):** 30 minutes
- **Phase 8 (Testing):** 1 hour
- **Phase 9 (Documentation):** 1 hour

**Total:** ~6 hours

---

## 9. Risks & Mitigation

### Risk 1: Breaking Tests
**Mitigation:** Test after each phase, commit frequently

### Risk 2: Lost Log Files
**Mitigation:** Keep old `mlpatrol.log` until new logging verified

### Risk 3: CVE Check Fails
**Mitigation:** Test CVE monitoring separately, keep old file initially

### Risk 4: Coverage Reports Broken
**Mitigation:** Verify coverage commands work before deleting old files

---

## 10. Rollback Plan

If issues arise:
1. Revert commits with `git revert`
2. Restore old logger.py from git history
3. Restore old file locations from git history
4. Old structure is preserved in git

---

## Approval Required

**Please review this plan and approve before implementation begins.**

Once approved, implementation will proceed in phases with testing at each step.

---

**END OF PLAN**
