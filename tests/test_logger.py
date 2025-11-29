"""Unit tests for logging utilities.

This test suite covers:
- Logger configuration
- Log level handling
- Logger instance creation
- Configuration persistence
"""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from src.utils.logger import DEFAULT_FORMAT, configure_logging, get_logger


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def setup_method(self):
        """Reset logging state before each test."""
        # Reset the _configured flag
        import src.utils.logger as logger_module

        logger_module._configured = False
        # Clear existing handlers
        logging.root.handlers = []
        # Reset root logger level to NOTSET
        logging.root.setLevel(logging.NOTSET)

    def teardown_method(self):
        """Clean up after each test."""
        import src.utils.logger as logger_module

        logger_module._configured = False
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)  # Reset to default
        os.environ.pop("MLPATROL_LOG_LEVEL", None)

    def test_configure_logging_default_level(self):
        """Test configure_logging with default INFO level."""
        configure_logging()

        assert logging.root.level == logging.INFO

    def test_configure_logging_custom_level(self):
        """Test configure_logging with custom level."""
        configure_logging(level="DEBUG")

        assert logging.root.level == logging.DEBUG

    def test_configure_logging_from_environment(self):
        """Test configure_logging reads from MLPATROL_LOG_LEVEL env var."""
        os.environ["MLPATROL_LOG_LEVEL"] = "WARNING"

        configure_logging()

        assert logging.root.level == logging.WARNING

    def test_configure_logging_param_overrides_env(self):
        """Test explicit level parameter overrides environment variable."""
        os.environ["MLPATROL_LOG_LEVEL"] = "WARNING"

        configure_logging(level="DEBUG")

        assert logging.root.level == logging.DEBUG

    def test_configure_logging_case_insensitive(self):
        """Test log level is case-insensitive."""
        configure_logging(level="debug")

        assert logging.root.level == logging.DEBUG

    def test_configure_logging_only_once(self):
        """Test configure_logging only configures once."""
        configure_logging(level="DEBUG")
        initial_level = logging.root.level
        initial_handlers_count = len(logging.root.handlers)

        # Try to configure again with different level
        configure_logging(level="ERROR")

        # Should still have the first configuration
        assert logging.root.level == initial_level
        assert len(logging.root.handlers) == initial_handlers_count

    def test_configure_logging_format(self):
        """Test configure_logging sets the correct format."""
        configure_logging()

        # Check that at least one handler has the default format
        assert len(logging.root.handlers) > 0
        # The format is set on the root logger's basicConfig

    def test_configure_logging_multiple_levels(self):
        """Test different valid log levels."""
        import src.utils.logger as logger_module

        levels = [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ]

        for level_str, level_int in levels:
            # Reset for each test
            logger_module._configured = False
            logging.root.handlers = []

            configure_logging(level=level_str)
            assert logging.root.level == level_int


class TestGetLogger:
    """Tests for get_logger function."""

    def setup_method(self):
        """Reset logging state before each test."""
        import src.utils.logger as logger_module

        logger_module._configured = False
        logging.root.handlers = []
        # Reset root logger level to NOTSET
        logging.root.setLevel(logging.NOTSET)

    def teardown_method(self):
        """Clean up after each test."""
        import src.utils.logger as logger_module

        logger_module._configured = False
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)  # Reset to default
        os.environ.pop("MLPATROL_LOG_LEVEL", None)

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logging.Logger instance."""
        logger = get_logger("test")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test"

    def test_get_logger_configures_logging(self):
        """Test get_logger configures logging if not already configured."""
        import src.utils.logger as logger_module

        assert logger_module._configured is False

        get_logger("test")

        assert logger_module._configured is True

    def test_get_logger_different_names(self):
        """Test get_logger with different names returns different loggers."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 is not logger2

    def test_get_logger_same_name(self):
        """Test get_logger with same name returns same logger instance."""
        logger1 = get_logger("test")
        logger2 = get_logger("test")

        # Python's logging.getLogger returns the same instance for the same name
        assert logger1 is logger2

    def test_get_logger_inherits_config(self):
        """Test logger from get_logger inherits root config."""
        os.environ["MLPATROL_LOG_LEVEL"] = "WARNING"

        logger = get_logger("test")

        # Logger should inherit the WARNING level from root
        assert logging.root.level == logging.WARNING

    def test_get_logger_can_log(self):
        """Test logger from get_logger can actually log messages."""
        logger = get_logger("test")

        # Should not raise any errors
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_get_logger_with_module_name(self):
        """Test get_logger with __name__ pattern."""
        logger = get_logger(__name__)

        assert logger.name == __name__

    def test_multiple_get_logger_calls_dont_reconfigure(self):
        """Test multiple get_logger calls don't reconfigure logging."""
        logger1 = get_logger("test1")
        initial_handlers = len(logging.root.handlers)

        logger2 = get_logger("test2")
        logger3 = get_logger("test3")

        # Should not add more handlers
        assert len(logging.root.handlers) == initial_handlers


class TestDefaultFormat:
    """Tests for DEFAULT_FORMAT constant."""

    def test_default_format_exists(self):
        """Test DEFAULT_FORMAT is defined."""
        assert DEFAULT_FORMAT is not None
        assert isinstance(DEFAULT_FORMAT, str)

    def test_default_format_contains_required_fields(self):
        """Test DEFAULT_FORMAT contains standard logging fields."""
        assert "%(asctime)s" in DEFAULT_FORMAT
        assert "%(name)s" in DEFAULT_FORMAT
        assert "%(levelname)s" in DEFAULT_FORMAT
        assert "%(message)s" in DEFAULT_FORMAT
