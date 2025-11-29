# Changelog

All notable changes to MLPatrol will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CONTRIBUTING.md with comprehensive contribution guidelines
- CHANGELOG.md for version tracking
- Dependency vulnerability scanning in CI pipeline with pip-audit

### Changed
- Updated Python requirement from 3.10+ to 3.12+
- Updated NumPy to version 2.1.0+ for Python 3.13+ compatibility
- Replaced deprecated `datetime.utcnow()` with `datetime.now(timezone.utc)`

### Fixed
- Dataset analysis logic error where variables were used before definition (P0)
- Python version check in app.py now correctly enforces Python 3.12+
- Pre-commit CI enforcement - removed `|| true` that masked failures

## [0.2.0] - 2025-11-28

### Added
- Enhanced dataset analysis with statistical tests and summaries
- Automatic CVE monitoring with background agent initialization
- Timezone-aware datetime handling across the application
- Alert persistence system to maintain security alerts across sessions
- Retry logic with exponential backoff for agent coordinator
- Improved error handling in dataset analysis tools

### Changed
- Updated Gradio Chatbot to use `type='messages'` API
- Consolidated dashboard tabs for better UX
- Made agent initialization non-blocking with background thread
- Moved helper functions before main execution for better code organization

### Fixed
- AgentState initialization and state management
- Duplicate kwargs in `parse_dataset_analysis` function
- Various issues identified through comprehensive code analysis

## [0.1.0] - 2025-11-20

### Added
- Initial release of MLPatrol
- CVE threat intelligence monitoring for ML libraries
- Dataset security analysis (poisoning detection and bias analysis)
- AI-powered security agent using LangGraph
- Multi-LLM support (OpenAI, Anthropic, Ollama)
- Gradio-based web interface with multiple tabs:
  - CVE Monitoring
  - Dataset Analysis
  - Security Chat
  - Alerts Dashboard
- Statistical tests for dataset validation:
  - Duplicate detection
  - Missing value analysis
  - Outlier detection
  - Class imbalance checks
- Web search integration via Tavily API
- Pre-commit hooks for code quality
- Comprehensive test suite with pytest
- CI/CD pipeline with GitHub Actions

### Security
- NVD API integration for CVE data
- Dataset poisoning detection algorithms
- Bias detection and quantification
- Automated security report generation

## Version History

### Breaking Changes

#### v0.2.0
- Minimum Python version increased to 3.12
- NumPy 2.x required (breaking change from NumPy 1.x)
- Gradio API updated to use `type='messages'` (requires Gradio 5.0+)

## Development Milestones

### Phase 1: Core Security Features (Completed)
- ✅ CVE monitoring system
- ✅ Dataset analysis tools
- ✅ Agent framework with LangGraph
- ✅ Web interface with Gradio

### Phase 2: Python 3.12+ Migration (In Progress)
- ✅ Updated Python version requirements
- ✅ Fixed deprecated `datetime.utcnow()` usage
- ✅ Updated NumPy to 2.x
- ✅ Fixed critical logic errors
- ⏳ Complete dependency updates
- ⏳ Enhanced test coverage
- ⏳ Performance optimizations

### Phase 3: Production Readiness (Planned)
- 📋 API documentation generation
- 📋 End-to-end integration tests
- 📋 Performance benchmarks
- 📋 Security hardening (rate limiting, secrets scanning)
- 📋 Docker containerization
- 📋 Deployment documentation

## Support

For questions, issues, or contributions, please visit:
- [GitHub Issues](https://github.com/sarahsl-prog/MLPatrol/issues)
- [Contributing Guide](CONTRIBUTING.md)

## Credits

MLPatrol is built for MCP's 1st Birthday Hackathon - Track 2: Agent Apps (Productivity)

### Technologies
- **LangGraph**: Agent orchestration
- **Gradio**: Web interface
- **Anthropic Claude & OpenAI GPT**: LLM providers
- **Tavily**: Web search API
- **NVD**: CVE database

---

**Note**: Version numbers follow [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backward-compatible functionality additions
- PATCH version for backward-compatible bug fixes
