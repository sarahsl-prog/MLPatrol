# Contributing to MLPatrol

Thank you for your interest in contributing to MLPatrol! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

By participating in this project, you agree to:
- Be respectful and inclusive
- Focus on constructive feedback
- Prioritize the project's goals over personal preferences
- Help create a welcoming environment for all contributors

## Getting Started

### Prerequisites

- **Python 3.12 or higher** (Python 3.13+ recommended)
- **Git** for version control
- **API Keys** (optional): OpenAI, Anthropic, or Tavily for full functionality

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MLPatrol.git
   cd MLPatrol
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/sarahsl-prog/MLPatrol.git
   ```

## Development Setup

### 1. Create a Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install runtime dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys (optional)
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
# TAVILY_API_KEY=your_key_here
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks run automatically before each commit.

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug Fixes**: Fix issues in existing code
- **New Features**: Add new functionality
- **Documentation**: Improve README, docstrings, or guides
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Refactoring**: Improve code structure without changing behavior

### Before You Start

1. **Check existing issues**: Browse [open issues](https://github.com/sarahsl-prog/MLPatrol/issues) to see if someone is already working on it
2. **Create an issue**: If you're planning significant changes, open an issue first to discuss
3. **Claim an issue**: Comment on an issue to let others know you're working on it

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: Maximum 100 characters (enforced by Black)
- **Imports**: Organized using `isort`
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings

### Code Formatting

We use automated formatters:

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Run all pre-commit checks
pre-commit run --all-files
```

### Naming Conventions

- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Example

```python
from typing import Dict, List, Optional

class BiasAnalyzer:
    """Analyzes datasets for potential bias.

    This class provides methods to detect and quantify various types
    of bias in machine learning datasets.

    Attributes:
        threshold: Maximum acceptable bias score (0.0 to 1.0)
        features: List of features to analyze
    """

    DEFAULT_THRESHOLD = 0.3

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        """Initialize the analyzer.

        Args:
            threshold: Bias threshold value between 0.0 and 1.0
        """
        self.threshold = threshold
        self._cache: Dict[str, float] = {}

    def analyze_distribution(
        self,
        data: List[float],
        feature_name: Optional[str] = None
    ) -> Dict[str, float]:
        """Analyze the distribution of values.

        Args:
            data: List of numeric values to analyze
            feature_name: Optional name of the feature

        Returns:
            Dictionary containing bias metrics

        Raises:
            ValueError: If data is empty or contains invalid values
        """
        if not data:
            raise ValueError("Data cannot be empty")

        # Implementation here
        return {"bias_score": 0.15, "distribution": "normal"}
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_bias_analyzer.py

# Run tests matching pattern
pytest -k "test_bias"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names that explain what's being tested
- Include both positive and negative test cases
- Mock external API calls

### Example Test

```python
import pytest
from src.dataset.bias_analyzer import BiasAnalyzer

def test_bias_analyzer_detects_imbalanced_data():
    """Test that BiasAnalyzer detects highly imbalanced datasets."""
    analyzer = BiasAnalyzer(threshold=0.3)

    # Highly imbalanced data: 90% class 0, 10% class 1
    data = [0] * 90 + [1] * 10

    result = analyzer.analyze_distribution(data)

    assert result["bias_score"] > 0.3, "Should detect bias in imbalanced data"
    assert "distribution" in result


def test_bias_analyzer_handles_empty_data():
    """Test that BiasAnalyzer raises error for empty data."""
    analyzer = BiasAnalyzer()

    with pytest.raises(ValueError, match="Data cannot be empty"):
        analyzer.analyze_distribution([])
```

## Pull Request Process

### 1. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a new branch
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, focused commits
- Follow the coding standards
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Your Changes

```bash
# Add files
git add .

# Commit with descriptive message
git commit -m "Add bias detection for categorical features

- Implement chi-square test for independence
- Add support for multi-class imbalance detection
- Update BiasAnalyzer with new methods
- Add tests covering edge cases"
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code restructuring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- **Clear title** describing the change
- **Description** explaining what and why
- **References** to related issues (e.g., "Fixes #123")
- **Testing** notes on how you tested the changes

### 5. PR Review Process

- Maintainers will review your PR
- Address feedback by pushing new commits
- Once approved, a maintainer will merge your PR

## Issue Guidelines

### Creating Issues

When creating an issue, please include:

1. **Clear title**: Summarize the issue in one line
2. **Description**: Detailed explanation of the problem or feature
3. **Steps to reproduce** (for bugs):
   - What you did
   - What you expected
   - What actually happened
4. **Environment**:
   - Python version
   - Operating system
   - Relevant dependencies
5. **Screenshots/Logs**: If applicable

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to docs
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `priority-high`: Important issues
- `priority-low`: Nice to have

## Development Workflow

### Typical Workflow

1. **Pick an issue** or create one
2. **Fork and clone** the repository
3. **Create a branch** for your work
4. **Make changes** following our standards
5. **Write tests** for your changes
6. **Run tests** to ensure everything works
7. **Commit changes** with clear messages
8. **Push to your fork**
9. **Create Pull Request**
10. **Address review feedback**
11. **Celebrate** when merged! 🎉

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge into your main branch
git checkout main
git merge upstream/main

# Push updates to your fork
git push origin main
```

## Questions?

If you have questions about contributing:

1. Check existing [documentation](README.md)
2. Search [closed issues](https://github.com/sarahsl-prog/MLPatrol/issues?q=is%3Aissue+is%3Aclosed)
3. Open a new issue with the `question` label

## License

By contributing to MLPatrol, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing to MLPatrol! Your efforts help make ML systems more secure for everyone. 🛡️
