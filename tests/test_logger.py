"""Unit tests for Loguru logging utilities.

This test suite covers:
- Logger configuration
- Logger instance creation
- Component-specific loggers
"""

import pytest

from src.utils.logger import (
    configure_logging,
    get_agent_logger,
    get_dataset_logger,
    get_logger,
    get_security_logger,
)


class TestLoggerCreation:
    """Tests for logger creation functions."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test_component")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")

    def test_get_agent_logger(self):
        """Test that get_agent_logger returns a logger."""
        logger = get_agent_logger("test_agent")
        assert logger is not None

    def test_get_security_logger(self):
        """Test that get_security_logger returns a logger."""
        logger = get_security_logger("test_security")
        assert logger is not None

    def test_get_dataset_logger(self):
        """Test that get_dataset_logger returns a logger."""
        logger = get_dataset_logger("test_dataset")
        assert logger is not None

    def test_configure_logging_exists(self):
        """Test that configure_logging function exists for backwards compatibility."""
        # Should not raise
        configure_logging()
        configure_logging("INFO")


class TestLogging:
    """Tests for actual logging functionality."""

    def test_logger_can_log(self):
        """Test that logger can write log messages."""
        logger = get_logger("test_logging")
        # These should not raise
        logger.info("Test info message")
        logger.warning("Test warning")
        logger.error("Test error")
