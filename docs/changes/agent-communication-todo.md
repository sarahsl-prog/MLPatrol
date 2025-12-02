# MLPatrol Agent Communication Implementation - To-Do List

## Overview
Implement automated agent-to-agent communication workflow:
1. **CVE Monitor** detects vulnerability → Triggers **Script Generator**
2. **Script Generator** creates remediation script → Triggers **LLM Reviewer**
3. **LLM Reviewer** analyzes script for safety/correctness → Presents to user
4. **User** approves via GUI button → **Executor** runs script safely
5. Results displayed in GUI with full audit trail

---

## Phase 1: Data Models & Core Infrastructure

### 1.1 Script Lifecycle State Management
**File**: `src/models/script_state.py` (NEW)

- [ ] Create `ScriptStatus` enum:
  - `DETECTED` - CVE found, script not yet generated
  - `GENERATED` - Script created, awaiting review
  - `UNDER_REVIEW` - LLM is analyzing script
  - `REVIEWED` - LLM review complete, awaiting user approval
  - `APPROVED` - User approved, ready to execute
  - `EXECUTING` - Script is running
  - `COMPLETED` - Successfully executed
  - `FAILED` - Execution failed
  - `REJECTED` - User rejected script
  - `CANCELLED` - User cancelled

- [ ] Create `ScriptRecord` dataclass:
  ```python
  @dataclass
  class ScriptRecord:
      id: str  # UUID
      cve_id: str
      library: str
      severity: str
      script_path: str
      status: ScriptStatus
      created_at: datetime
      generated_at: Optional[datetime]
      reviewed_at: Optional[datetime]
      executed_at: Optional[datetime]
      review_result: Optional[Dict]
      execution_result: Optional[Dict]
      metadata: Dict
  ```

- [ ] Create `ReviewResult` dataclass:
  ```python
  @dataclass
  class ReviewResult:
      approved: bool
      confidence_score: float  # 0.0-1.0
      risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
      issues_found: List[str]
      recommendations: List[str]
      safe_to_run: bool
      review_summary: str
      detailed_analysis: str
  ```

### 1.2 Script Queue Manager
**File**: `src/agent/script_queue.py` (NEW)

- [ ] Create `ScriptQueue` class with thread-safe operations:
  - `add_script(cve_id, library, severity) -> str` - Returns script ID
  - `get_script(script_id) -> ScriptRecord`
  - `update_status(script_id, status, metadata=None)`
  - `get_pending_scripts() -> List[ScriptRecord]`
  - `get_scripts_by_status(status) -> List[ScriptRecord]`
  - `archive_script(script_id)`

- [ ] Implement persistence:
  - SQLite database: `data/scripts/script_queue.db`
  - Tables: `scripts`, `reviews`, `executions`
  - Atomic transactions for status updates

- [ ] Add queue event hooks:
  - `on_script_added` - Trigger script generation
  - `on_script_generated` - Trigger LLM review
  - `on_review_complete` - Notify GUI
  - `on_execution_complete` - Log results

---

## Phase 2: LLM Script Reviewer Agent

### 2.1 Script Review Engine
**File**: `src/agent/script_reviewer.py` (NEW)

- [ ] Create `ScriptReviewer` class:
  - Uses same LLM as main agent (Claude/GPT-4/Llama)
  - Specialized prompt for security code review
  - Temperature: 0.0 (deterministic analysis)

- [ ] Implement `review_script(script_path, cve_context) -> ReviewResult`:
  - **Security Analysis**:
    - [ ] Check for dangerous operations (os.system, eval, exec)
    - [ ] Verify no network calls or file writes
    - [ ] Validate input sanitization
    - [ ] Check for privilege escalation

  - **Code Quality Analysis**:
    - [ ] Verify error handling
    - [ ] Check exit codes
    - [ ] Validate version comparison logic
    - [ ] Ensure idempotency

  - **CVE Relevance Analysis**:
    - [ ] Confirm script addresses the specific CVE
    - [ ] Validate vulnerable version checks
    - [ ] Verify remediation logic

  - **Risk Scoring**:
    - [ ] Calculate confidence score based on findings
    - [ ] Assign risk level (LOW/MEDIUM/HIGH/CRITICAL)
    - [ ] Determine if safe to run automatically

- [ ] Add review caching:
  - Cache reviews by script hash
  - Invalidate on script changes
  - Store in `data/scripts/reviews_cache.json`

### 2.2 Review Prompts
**File**: `src/agent/prompts.py` (UPDATE)

- [ ] Add `SCRIPT_REVIEW_SYSTEM_PROMPT`:
  - Role: Security code reviewer
  - Task: Analyze Python security scripts
  - Focus areas: Safety, correctness, CVE relevance
  - Output format: Structured JSON

- [ ] Add `SCRIPT_REVIEW_USER_PROMPT` template:
  ```
  Review this security script:
  CVE: {cve_id}
  Library: {library}
  Severity: {severity}

  Script:
  {script_content}

  Analyze for: security risks, code quality, CVE relevance
  ```

- [ ] Add review examples (few-shot learning):
  - Example 1: Safe script with proper checks
  - Example 2: Script with security issues
  - Example 3: Script with logic errors

---

## Phase 3: Automated Workflow Coordination

### 3.1 Workflow Orchestrator
**File**: `src/agent/workflow_orchestrator.py` (NEW)

- [ ] Create `WorkflowOrchestrator` class:
  - Manages script lifecycle transitions
  - Coordinates between CVE monitor, generator, reviewer
  - Thread-safe async operations

- [ ] Implement workflow stages:

  **Stage 1: CVE Detection → Script Generation**
  - [ ] `handle_cve_detected(cve_id, library, severity)`:
    - Create script record with status `DETECTED`
    - Queue script generation task
    - Update status to `GENERATED` on completion

  **Stage 2: Script Generation → LLM Review**
  - [ ] `handle_script_generated(script_id)`:
    - Load generated script
    - Queue review task
    - Update status to `UNDER_REVIEW`

  **Stage 3: LLM Review → User Approval**
  - [ ] `handle_review_complete(script_id, review_result)`:
    - Store review result
    - Update status to `REVIEWED`
    - Trigger GUI notification

  **Stage 4: User Approval → Execution**
  - [ ] `handle_user_approved(script_id)`:
    - Validate review is recent (< 24 hours)
    - Update status to `APPROVED`
    - Queue for execution or await manual trigger

  **Stage 5: Execution → Results**
  - [ ] `handle_execution_complete(script_id, result)`:
    - Store execution logs
    - Update status to `COMPLETED` or `FAILED`
    - Generate summary report

- [ ] Add error handling:
  - [ ] Retry logic for transient failures
  - [ ] Timeout handling (max 5 minutes per stage)
  - [ ] Fallback to manual review on LLM failures

### 3.2 Background Worker Integration
**File**: `app.py` (UPDATE)

- [ ] Update `BackgroundCoordinator` class (lines 964-1116):
  - [ ] Add workflow orchestrator integration
  - [ ] Modify `_process_library_alerts()` to use orchestrator:
    ```python
    # Old: Just generate script
    script_path = self.code_generator.generate_cve_security_script(...)

    # New: Trigger workflow
    script_id = self.orchestrator.handle_cve_detected(cve_id, library, severity)
    ```
  - [ ] Add periodic review queue processing
  - [ ] Add status monitoring loop

---

## Phase 4: Script Execution System

### 4.1 Safe Script Executor
**File**: `src/security/script_executor.py` (NEW)

- [ ] Create `ScriptExecutor` class with safety features:

  **Sandboxing**:
  - [ ] Use subprocess with timeout (default: 60 seconds)
  - [ ] Limit CPU/memory usage (optional: cgroups/docker)
  - [ ] No shell=True (prevent injection)
  - [ ] Read-only filesystem access where possible

  **Execution Method**:
  - [ ] `execute_script(script_path, timeout=60) -> ExecutionResult`:
    - Run in isolated subprocess
    - Capture stdout, stderr
    - Monitor exit code
    - Timeout protection
    - Exception handling

  **Logging**:
  - [ ] Log all executions to `data/scripts/execution_log.jsonl`
  - [ ] Include: timestamp, script_id, user, result, duration
  - [ ] Rotate logs (max 100MB)

- [ ] Create `ExecutionResult` dataclass:
  ```python
  @dataclass
  class ExecutionResult:
      success: bool
      exit_code: int
      stdout: str
      stderr: str
      duration_seconds: float
      error_message: Optional[str]
      timestamp: datetime
  ```

- [ ] Add pre-execution validation:
  - [ ] Verify script hasn't been modified since review
  - [ ] Check file permissions
  - [ ] Validate Python syntax
  - [ ] Confirm review is still valid

### 4.2 Execution History
**File**: `src/security/execution_history.py` (NEW)

- [ ] Create execution history manager:
  - [ ] Store in SQLite: `data/scripts/executions.db`
  - [ ] Query methods: by script_id, by date, by status
  - [ ] Export to CSV/JSON for audit
  - [ ] Statistics: success rate, avg duration, common errors

---

## Phase 5: GUI Updates

### 5.1 New Script Management Tab
**File**: `app.py` (UPDATE)

- [ ] Add "Script Manager" tab after "Code Generation" tab:
  - **Layout**:
    ```
    ┌─────────────────────────────────────────┐
    │  Pending Scripts (3)                    │
    │  ┌───────────────────────────────────┐  │
    │  │ Script Cards (filterable/sortable)│  │
    │  └───────────────────────────────────┘  │
    │                                         │
    │  Script Details Panel                   │
    │  ┌───────────────────────────────────┐  │
    │  │ CVE Info | Review | Actions       │  │
    │  └───────────────────────────────────┘  │
    └─────────────────────────────────────────┘
    ```

- [ ] **Script Card Component**:
  - CVE ID (badge)
  - Library name
  - Severity (color-coded)
  - Status (with icon)
  - Timestamp
  - Click to expand details

- [ ] **Script Details Panel**:

  **CVE Information**:
  - [ ] CVE ID, description, CVSS score
  - [ ] Affected versions
  - [ ] Published date, references

  **Generated Script**:
  - [ ] Code viewer with syntax highlighting
  - [ ] Download button
  - [ ] View full script in modal

  **LLM Review Results**:
  - [ ] Confidence score gauge (0-100%)
  - [ ] Risk level badge (LOW/MEDIUM/HIGH/CRITICAL)
  - [ ] Issues found (expandable list)
  - [ ] Recommendations (expandable list)
  - [ ] Detailed analysis (collapsible)

  **Action Buttons**:
  - [ ] ✅ **Run Script** (primary, enabled if safe_to_run=True)
  - [ ] 🔍 **Re-review** (trigger new LLM review)
  - [ ] ✏️ **Edit & Re-review** (open in editor)
  - [ ] ❌ **Reject** (mark as rejected)
  - [ ] 📥 **Download** (save locally)

- [ ] **Execution Panel** (shows after clicking "Run Script"):
  - [ ] Pre-execution checklist:
    - ✓ Script reviewed by LLM
    - ✓ Risk level: LOW
    - ✓ No critical issues found
    - ⚠️ Warning: This will check your environment
  - [ ] Confirmation modal with summary
  - [ ] Real-time execution progress
  - [ ] Live output stream (stdout/stderr)
  - [ ] Exit code and status
  - [ ] Success/failure banner

### 5.2 Handler Functions
**File**: `app.py` (UPDATE)

- [ ] Add `handle_load_scripts()`:
  - Query `ScriptQueue` for all scripts
  - Group by status
  - Format for display

- [ ] Add `handle_script_selected(script_id)`:
  - Load full script details
  - Load review result
  - Load execution history
  - Return formatted data

- [ ] Add `handle_run_script(script_id)`:
  - Validate review status
  - Show confirmation dialog
  - Execute via `ScriptExecutor`
  - Stream output to GUI
  - Update script status
  - Return execution result

- [ ] Add `handle_rereview_script(script_id)`:
  - Trigger new LLM review
  - Update review result
  - Refresh GUI

- [ ] Add `handle_reject_script(script_id, reason)`:
  - Update status to REJECTED
  - Log rejection reason
  - Remove from pending queue

### 5.3 Dashboard Enhancements
**File**: `app.py` (UPDATE - lines 1437-1464)

- [ ] Add script status summary cards:
  ```
  ┌─────────────────┬─────────────────┬─────────────────┐
  │ Pending Review  │ Ready to Run    │ Executed Today  │
  │      5          │       3         │       12        │
  └─────────────────┴─────────────────┴─────────────────┘
  ```

- [ ] Add recent script activity timeline:
  - Last 10 scripts with timestamps
  - Status transitions
  - Quick action buttons

- [ ] Add automated workflow status indicator:
  - ✅ Automation enabled
  - 🔄 Last scan: 2 minutes ago
  - 📊 Scripts this week: 23

### 5.4 Notifications System
**File**: `src/utils/notifications.py` (NEW)

- [ ] Create notification manager:
  - In-app toast notifications
  - Badge counts on tabs
  - Browser notifications (optional)

- [ ] Add notification triggers:
  - New CVE detected
  - Script generated and reviewed
  - Execution complete
  - Critical issues found in review

---

## Phase 6: Enhanced CVE Monitor Integration

### 6.1 Update CVE Monitor
**File**: `src/security/cve_monitor.py` (UPDATE)

- [ ] Add workflow trigger method:
  ```python
  def on_cve_detected(self, cve_record: CVERecord, library: str):
      # Trigger workflow orchestrator
      script_id = self.orchestrator.handle_cve_detected(
          cve_id=cve_record.id,
          library=library,
          severity=cve_record.severity
      )
      logger.info(f"Initiated workflow for {cve_record.id}: {script_id}")
  ```

- [ ] Add filtering options:
  - [ ] Minimum severity threshold (e.g., only MEDIUM+)
  - [ ] Library whitelist/blacklist
  - [ ] Duplicate detection (skip if already processed)

### 6.2 Update Code Generator
**File**: `src/security/code_generator.py` (UPDATE)

- [ ] Modify `generate_cve_security_script()`:
  - [ ] Return script path AND content hash
  - [ ] Save metadata alongside script
  - [ ] Include generation timestamp
  - [ ] Add script versioning

- [ ] Add script templates versioning:
  - [ ] Track template version in generated scripts
  - [ ] Allow re-generation with newer templates

---

## Phase 7: Configuration & Settings

### 7.1 Workflow Settings
**File**: `config/workflow_settings.yaml` (NEW)

```yaml
automation:
  enabled: true
  auto_review: true          # Auto-trigger LLM review
  auto_approve_safe: false   # Auto-approve LOW risk scripts
  require_manual_approval: true

review:
  model: "claude-sonnet-4"   # LLM for reviews
  temperature: 0.0
  timeout_seconds: 120
  cache_reviews: true
  max_cache_age_hours: 24

execution:
  enabled: true
  timeout_seconds: 60
  max_concurrent: 1
  require_confirmation: true
  allowed_in_production: false

thresholds:
  min_severity: "MEDIUM"     # Don't auto-process LOW
  min_confidence: 0.7        # Min review confidence
  max_risk_level: "MEDIUM"   # Don't auto-approve HIGH/CRITICAL

notifications:
  enable_toast: true
  enable_browser: false
  enable_email: false        # Future
```

### 7.2 Settings UI
**File**: `app.py` (UPDATE)

- [ ] Add "Settings" tab with workflow configuration:
  - [ ] Enable/disable automation
  - [ ] Set severity thresholds
  - [ ] Configure LLM for reviews
  - [ ] Set execution timeouts
  - [ ] Notification preferences

---

## Phase 8: Testing

### 8.1 Unit Tests
**Directory**: `tests/unit/`

- [ ] `test_script_queue.py`:
  - Queue operations (add, get, update, delete)
  - Thread safety
  - Persistence

- [ ] `test_script_reviewer.py`:
  - Review logic for safe scripts
  - Detection of dangerous operations
  - Risk scoring accuracy
  - Caching behavior

- [ ] `test_workflow_orchestrator.py`:
  - State transitions
  - Error handling
  - Timeout behavior
  - Event hooks

- [ ] `test_script_executor.py`:
  - Successful execution
  - Timeout handling
  - Error capture
  - Sandboxing

### 8.2 Integration Tests
**Directory**: `tests/integration/`

- [ ] `test_end_to_end_workflow.py`:
  - Simulate CVE detection
  - Verify script generation
  - Confirm LLM review
  - Test GUI interaction
  - Validate execution

- [ ] `test_cve_to_execution.py`:
  - Full pipeline from CVE API to script execution
  - Multiple concurrent CVEs
  - Error recovery

### 8.3 GUI Tests
**Directory**: `tests/gui/`

- [ ] `test_script_manager_tab.py`:
  - Script list rendering
  - Detail panel display
  - Button interactions
  - Execution flow

- [ ] `test_notifications.py`:
  - Toast notifications
  - Badge updates
  - Real-time updates

### 8.4 Security Tests
**Directory**: `tests/security/`

- [ ] `test_script_safety.py`:
  - Verify dangerous scripts are flagged
  - Test injection attempts
  - Validate sandboxing
  - Path traversal protection

---

## Phase 9: Documentation

### 9.1 Architecture Documentation
**File**: `docs/architecture/agent-communication.md` (NEW)

- [ ] Document workflow architecture:
  - State machine diagram
  - Component interaction diagram
  - Data flow diagrams
  - Sequence diagrams for each stage

### 9.2 User Guide
**File**: `docs/user-guide/automated-remediation.md` (NEW)

- [ ] How to use Script Manager tab
- [ ] Understanding review results
- [ ] Running scripts safely
- [ ] Configuring automation settings
- [ ] Troubleshooting common issues

### 9.3 Developer Guide
**File**: `docs/developer-guide/extending-workflow.md` (NEW)

- [ ] Adding new workflow stages
- [ ] Customizing review logic
- [ ] Creating custom executors
- [ ] Integrating with CI/CD

### 9.4 API Documentation
**File**: `docs/api/workflow-api.md` (NEW)

- [ ] ScriptQueue API
- [ ] WorkflowOrchestrator API
- [ ] ScriptReviewer API
- [ ] ScriptExecutor API

---

## Phase 10: Deployment & Monitoring

### 10.1 Database Migrations
**File**: `scripts/migrate_script_queue.py` (NEW)

- [ ] Create initial schema
- [ ] Add indexes for performance
- [ ] Migration for existing installations

### 10.2 Logging Enhancements
**File**: Multiple files (UPDATE)

- [ ] Add structured logging for workflow events:
  - Workflow initiated
  - Stage transitions
  - Review results
  - Execution results
  - Errors and retries

- [ ] Log levels:
  - INFO: Normal workflow progress
  - WARNING: Low-confidence reviews, retries
  - ERROR: Workflow failures, execution errors
  - CRITICAL: Security issues detected

### 10.3 Metrics & Monitoring
**File**: `src/utils/metrics.py` (NEW)

- [ ] Track workflow metrics:
  - CVEs detected per day
  - Scripts generated per day
  - Review confidence distribution
  - Execution success rate
  - Average time per stage
  - Queue depth over time

- [ ] Create metrics dashboard (optional):
  - Add to main Dashboard tab
  - Plotly charts for trends
  - Export to CSV

### 10.4 Alerting
**File**: `src/utils/alerting.py` (NEW)

- [ ] Alert on critical events:
  - High/Critical CVE detected
  - Script review finds critical issues
  - Execution failures
  - Queue backlog (>10 pending)

---

## Implementation Priority

### High Priority (Core Functionality)
1. ✅ Phase 1: Data Models & Infrastructure
2. ✅ Phase 2: LLM Script Reviewer
3. ✅ Phase 3: Workflow Orchestrator
4. ✅ Phase 5.1-5.3: GUI Script Manager Tab

### Medium Priority (Enhanced UX)
5. ✅ Phase 4: Script Executor
6. ✅ Phase 6: CVE Monitor Integration
7. ✅ Phase 5.4: Notifications
8. ✅ Phase 7: Configuration

### Low Priority (Polish)
9. ⏸️ Phase 8: Comprehensive Testing
10. ⏸️ Phase 9: Documentation
11. ⏸️ Phase 10: Monitoring & Metrics

---

## Estimated Effort

| Phase | Estimated Lines of Code | Complexity |
|-------|-------------------------|------------|
| Phase 1 | ~300 | Medium |
| Phase 2 | ~400 | High |
| Phase 3 | ~500 | High |
| Phase 4 | ~250 | Medium |
| Phase 5 | ~800 | Medium |
| Phase 6 | ~150 | Low |
| Phase 7 | ~200 | Low |
| Phase 8 | ~600 | Medium |
| Phase 9 | ~2000 (docs) | Low |
| Phase 10 | ~300 | Medium |
| **Total** | **~5,500** | **High** |

---

## Success Criteria

- [ ] CVE detected → Script generated in <30 seconds
- [ ] Script reviewed by LLM in <60 seconds
- [ ] Review confidence >0.7 for 90%+ of scripts
- [ ] User can run approved script with 1 click
- [ ] Execution completes in <60 seconds
- [ ] Full audit trail from CVE to execution
- [ ] Zero false positives in danger detection
- [ ] GUI responsive throughout workflow
- [ ] All transitions logged and traceable
- [ ] No manual intervention required (optional auto-mode)

---

## Risk Mitigation

### Security Risks
- **Risk**: Auto-execution of malicious scripts
  - **Mitigation**: Require manual approval, LLM review, sandboxing

- **Risk**: Script injection or tampering
  - **Mitigation**: Hash verification, file permissions, atomic updates

### Performance Risks
- **Risk**: LLM review bottleneck
  - **Mitigation**: Async processing, caching, timeout limits

- **Risk**: GUI blocking on execution
  - **Mitigation**: Background workers, streaming output, progress indicators

### Reliability Risks
- **Risk**: Workflow state corruption
  - **Mitigation**: Atomic DB transactions, retry logic, state validation

- **Risk**: LLM API failures
  - **Mitigation**: Fallback to manual review, queue persistence, graceful degradation

---

## Future Enhancements

- [ ] Multi-agent architecture (separate CVE, Review, Execution agents)
- [ ] Machine learning for review confidence calibration
- [ ] Integration with SIEM systems
- [ ] Slack/email notifications
- [ ] Scheduled execution windows
- [ ] Rollback mechanisms
- [ ] A/B testing of remediation strategies
- [ ] Community script sharing (reviewed library)
