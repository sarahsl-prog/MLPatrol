# MLPatrol - TODO Fixes Tracking

**Created:** November 28, 2025
**Last Updated:** November 28, 2025

---

## 🔴 P0 - CRITICAL (Must Fix Immediately)

### ✅ COMPLETED

- [x] **[ISSUE-1] Fix Dataset Analysis Logic Error** ✅
  - **File:** `src/agent/tools.py` lines 763-823
  - **Problem:** Variables used before definition (poisoning_report, bias_report)
  - **Impact:** Dataset analysis completely broken, will crash with NameError
  - **Fix:** Reordered code - bias/poisoning analysis now runs BEFORE using results
  - **Completed:** November 28, 2025
  - **Time Spent:** 30 minutes

- [x] **[ISSUE-2] Update Python Version Check** ✅
  - **File:** `app.py` line 24, `README.md` line 181
  - **Problem:** Checked for Python 3.10+ instead of 3.12+
  - **Impact:** Incorrect version enforcement
  - **Fix:** Changed version check from (3, 10) to (3, 12), updated docs
  - **Completed:** November 28, 2025
  - **Time Spent:** 15 minutes

- [x] **[ISSUE-3] Fix Missing Test Dependencies** ✅
  - **Files:** `requirements-dev.txt`, `.github/workflows/python-tests.yml`
  - **Problem:** Tests cannot run - missing dependencies
  - **Impact:** Cannot verify code quality or run tests
  - **Fix:** Comprehensive requirements-dev.txt with 30+ packages, updated CI
  - **Completed:** November 28, 2025
  - **Time Spent:** 30 minutes

- [x] **[ISSUE-15] Fix Pre-commit CI Enforcement** ✅ (Bonus)
  - **File:** `.github/workflows/python-tests.yml`
  - **Problem:** `|| true` masked pre-commit failures
  - **Fix:** Removed `|| true` to properly enforce checks
  - **Completed:** November 28, 2025
  - **Time Spent:** 5 minutes

### ⏳ IN PROGRESS

- [ ] **Installing Dependencies in Local Environment**
  - Running: `pip install -r requirements.txt`
  - Need to also run: `pip install -r requirements-dev.txt`
  - Then: Run test suite to verify fixes

### 📋 TODO

- [ ] **[ISSUE-4] Fix Documentation Mismatches**
  - **Files:** `README.md`, `requirements.txt`, all docs
  - **Problem:**
    - README claims Gradio 6, requirements says >=5.0.0
    - Python version inconsistent across docs
  - **Impact:** User confusion, incorrect setup
  - **Fix:**
    - Update README.md Python version from 3.10+ to 3.12+
    - Clarify Gradio version requirements
    - Update all documentation references
  - **Estimated Time:** 1 hour
  - **Assignee:** TBD
  - **Status:** TODO

---

## 🟡 P1 - HIGH PRIORITY (Before Python 3.12 Migration)

- [ ] **[ISSUE-5] Fix Deprecated datetime.utcnow()**
  - **File:** `src/security/cve_monitor.py` line 92
  - **Problem:** Uses deprecated datetime.utcnow() (deprecated in Python 3.12)
  - **Impact:** Deprecation warning, may break in future Python versions
  - **Fix:** Replace with `datetime.now(timezone.utc)`
  - **Estimated Time:** 15 minutes
  - **Status:** TODO

- [ ] **[ISSUE-6] Review NumPy Version Constraint**
  - **File:** `requirements.txt` line 16
  - **Problem:** numpy>=1.26.0,<2.0.0 excludes numpy 2.x
  - **Impact:** Missing performance improvements from numpy 2.x
  - **Fix:** Test with numpy 2.x, update constraint if compatible
  - **Estimated Time:** 2 hours (includes testing)
  - **Status:** TODO

- [ ] **[ISSUE-7] Add Missing Documentation**
  - **Files:** New files to create
  - **Problem:** Missing CONTRIBUTING.md, CHANGELOG.md
  - **Impact:** Harder for contributors, no version tracking
  - **Fix:** Create comprehensive contributor guide and changelog
  - **Estimated Time:** 3 hours
  - **Status:** TODO

- [ ] **[ISSUE-8] Add Dependency Vulnerability Scanning**
  - **File:** `.github/workflows/python-tests.yml`
  - **Problem:** No automated dependency vulnerability checking
  - **Impact:** May ship with known CVEs
  - **Fix:** Add pip-audit or safety to CI pipeline
  - **Estimated Time:** 1 hour
  - **Status:** TODO

- [ ] **[ISSUE-9] Update Dependency Versions**
  - **File:** `requirements.txt`
  - **Problem:** Some dependencies on older versions
  - **Impact:** Missing bug fixes and Python 3.12 optimizations
  - **Fix:** Update langchain packages, pytest, pandas, scipy
  - **Estimated Time:** 4 hours (includes testing)
  - **Status:** TODO

---

## 🟢 P2 - MEDIUM PRIORITY (Quality Improvements)

- [ ] **[ISSUE-10] Remove Duplicate Threading Import**
  - **File:** `app.py` lines 17 and 42
  - **Problem:** Import threading appears twice
  - **Impact:** Code cleanliness
  - **Fix:** Remove line 42
  - **Estimated Time:** 1 minute
  - **Status:** TODO

- [ ] **[ISSUE-11] Refactor create_interface() Function**
  - **File:** `app.py` lines 1219-1741
  - **Problem:** Function is 522 lines - too long
  - **Impact:** Hard to maintain and test
  - **Fix:** Extract each tab into separate functions
  - **Estimated Time:** 4 hours
  - **Status:** TODO

- [ ] **[ISSUE-12] Replace Magic Numbers with Constants**
  - **Files:** `app.py`, `src/agent/tools.py`
  - **Problem:** Magic numbers like 200, 3.0, 10 scattered throughout
  - **Impact:** Unclear intent, hard to modify
  - **Fix:** Define module-level constants
  - **Estimated Time:** 2 hours
  - **Status:** TODO

- [ ] **[ISSUE-13] Improve Error Handling Specificity**
  - **File:** `src/agent/tools.py` lines 785-823
  - **Problem:** Broad Exception handlers may mask bugs
  - **Impact:** Harder to debug issues
  - **Fix:** Use specific exception types
  - **Estimated Time:** 2 hours
  - **Status:** TODO

- [ ] **[ISSUE-14] Add pyproject.toml**
  - **File:** New file to create
  - **Problem:** No centralized tool configuration
  - **Impact:** Inconsistent linting/formatting
  - **Fix:** Create pyproject.toml with black, isort, pytest, mypy config
  - **Estimated Time:** 1 hour
  - **Status:** TODO

- [ ] **[ISSUE-15] Fix Pre-commit CI Enforcement**
  - **File:** `.github/workflows/python-tests.yml` line 48
  - **Problem:** `|| true` masks pre-commit failures
  - **Impact:** Pre-commit checks not enforced
  - **Fix:** Remove `|| true` to enforce checks
  - **Estimated Time:** 5 minutes
  - **Status:** TODO

---

## 🟢 P3 - LOW PRIORITY (Nice to Have)

- [ ] **[ISSUE-16] Add API Documentation Generation**
  - **Files:** New docs setup
  - **Problem:** No auto-generated API docs
  - **Impact:** Harder for users to understand API
  - **Fix:** Set up Sphinx or pdoc
  - **Estimated Time:** 6 hours
  - **Status:** TODO

- [ ] **[ISSUE-17] Add End-to-End Tests**
  - **Files:** New test files
  - **Problem:** No integration tests for full workflows
  - **Impact:** May not catch integration bugs
  - **Fix:** Add E2E tests for major workflows
  - **Estimated Time:** 8 hours
  - **Status:** TODO

- [ ] **[ISSUE-18] Add Performance Tests**
  - **Files:** New test files
  - **Problem:** No performance benchmarks
  - **Impact:** May not catch performance regressions
  - **Fix:** Add performance test suite
  - **Estimated Time:** 4 hours
  - **Status:** TODO

- [ ] **[ISSUE-19] Add Security Testing**
  - **Files:** New test files
  - **Problem:** No security-focused tests
  - **Impact:** May not catch security vulnerabilities
  - **Fix:** Add tests for XSS, injection, etc.
  - **Estimated Time:** 4 hours
  - **Status:** TODO

- [ ] **[ISSUE-20] Add Rate Limiting to Gradio**
  - **File:** `app.py`
  - **Problem:** No rate limiting on endpoints
  - **Impact:** Vulnerable to abuse
  - **Fix:** Add gradio.RateLimiter decorators
  - **Estimated Time:** 2 hours
  - **Status:** TODO

- [ ] **[ISSUE-21] Add Secrets Scanning Pre-commit Hook**
  - **File:** `.pre-commit-config.yaml`
  - **Problem:** No automated secrets detection
  - **Impact:** May accidentally commit API keys
  - **Fix:** Add detect-secrets or gitleaks
  - **Estimated Time:** 1 hour
  - **Status:** TODO

---

## 📊 Progress Summary

**Total Issues:** 21
- P0 (Critical): 4 issues, ~4.5 hours
- P1 (High): 5 issues, ~10.25 hours
- P2 (Medium): 6 issues, ~9 hours
- P3 (Low): 6 issues, ~25 hours

**Completed:** 0 / 21 (0%)
**In Progress:** 0 / 21 (0%)
**TODO:** 21 / 21 (100%)

**Total Estimated Effort:** ~49 hours
**Critical Path to Production:** ~4.5 hours (P0 only)
**To Python 3.12 Ready:** ~14.75 hours (P0 + P1)

---

## Session Notes

### Session 1 - November 28, 2025
- Initial comprehensive code analysis completed
- 21 issues identified and prioritized
- Ready to begin P0 fixes

---

## Next Actions

1. **Immediate (Today):**
   - [ ] Fix ISSUE-1: Dataset analysis logic error
   - [ ] Fix ISSUE-2: Python version check
   - [ ] Fix ISSUE-3: Test dependencies
   - [ ] Fix ISSUE-4: Documentation updates

2. **This Week:**
   - [ ] Fix ISSUE-5: datetime.utcnow() deprecation
   - [ ] Fix ISSUE-6: NumPy version constraint
   - [ ] Fix ISSUE-8: Dependency scanning

3. **Next Week:**
   - [ ] Remaining P1 and P2 issues
   - [ ] Full Python 3.12 migration testing

---

## Notes

- All line numbers reference the current codebase state (Nov 28, 2025)
- Some issues may affect multiple files - check linked issues
- Test all fixes with both Python 3.12 and 3.13
- Run full test suite after each P0/P1 fix
- Update this file as issues are completed
