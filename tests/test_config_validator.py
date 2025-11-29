"""Unit tests for configuration validation.

This test suite covers:
- Configuration validation logic
- LLM configuration validation (cloud vs local)
- Web search configuration validation
- API key validation
- Error and warning generation
"""

import os
import pytest
from io import StringIO
from unittest.mock import patch
from src.utils.config_validator import (
    ValidationResult,
    validate_config,
    print_validation_results,
    validate_and_exit_on_error,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(
            valid=True,
            errors=[],
            warnings=["test warning"],
            config_summary={"key": "value"},
        )
        assert result.valid is True
        assert result.errors == []
        assert len(result.warnings) == 1
        assert result.config_summary == {"key": "value"}


class TestValidateConfig:
    """Tests for validate_config function."""

    def setup_method(self):
        """Clear environment before each test."""
        self.original_env = os.environ.copy()
        # Clear all MLPatrol-related env vars
        env_vars = [
            "USE_LOCAL_LLM",
            "LOCAL_LLM_MODEL",
            "LOCAL_LLM_URL",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "ENABLE_WEB_SEARCH",
            "USE_TAVILY_SEARCH",
            "USE_BRAVE_SEARCH",
            "TAVILY_API_KEY",
            "BRAVE_API_KEY",
            "NVD_API_KEY",
            "LOG_LEVEL",
            "MAX_AGENT_ITERATIONS",
        ]
        for var in env_vars:
            os.environ.pop(var, None)

    def teardown_method(self):
        """Restore original environment after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    # ======================================================================
    # Cloud LLM Configuration Tests
    # ======================================================================

    def test_valid_config_with_anthropic(self):
        """Test valid configuration with Anthropic API key."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-1234567890123456789012345678901234567890"

        result = validate_config()

        assert result.valid is True
        assert len(result.errors) == 0
        assert result.config_summary["llm_mode"] == "cloud"
        assert result.config_summary["llm_provider"] == "Anthropic (Claude)"

    def test_valid_config_with_openai(self):
        """Test valid configuration with OpenAI API key."""
        os.environ["OPENAI_API_KEY"] = "sk-1234567890123456789012345678901234567890"

        result = validate_config()

        assert result.valid is True
        assert len(result.errors) == 0
        assert result.config_summary["llm_mode"] == "cloud"
        assert result.config_summary["llm_provider"] == "OpenAI (GPT)"

    def test_error_no_llm_configured(self):
        """Test error when no LLM is configured."""
        # No API keys, no local LLM
        result = validate_config()

        assert result.valid is False
        assert len(result.errors) == 1
        assert "No LLM API key found" in result.errors[0]

    def test_warning_anthropic_key_too_short(self):
        """Test warning for suspiciously short Anthropic API key."""
        os.environ["ANTHROPIC_API_KEY"] = "short"

        result = validate_config()

        assert result.valid is True  # Warning, not error
        assert any("ANTHROPIC_API_KEY seems too short" in w for w in result.warnings)

    def test_warning_openai_key_too_short(self):
        """Test warning for suspiciously short OpenAI API key."""
        os.environ["OPENAI_API_KEY"] = "short"

        result = validate_config()

        assert result.valid is True  # Warning, not error
        assert any("OPENAI_API_KEY seems too short" in w for w in result.warnings)

    # ======================================================================
    # Local LLM Configuration Tests
    # ======================================================================

    def test_valid_local_llm_config(self):
        """Test valid local LLM configuration."""
        os.environ["USE_LOCAL_LLM"] = "true"
        os.environ["LOCAL_LLM_MODEL"] = "ollama/llama3.1:8b"
        os.environ["LOCAL_LLM_URL"] = "http://localhost:11434"

        result = validate_config()

        assert result.valid is True
        assert result.config_summary["llm_mode"] == "local"
        assert result.config_summary["llm_model"] == "ollama/llama3.1:8b"
        assert result.config_summary["llm_url"] == "http://localhost:11434"

    def test_error_local_llm_without_model(self):
        """Test error when USE_LOCAL_LLM=true but no model specified."""
        os.environ["USE_LOCAL_LLM"] = "true"
        # No LOCAL_LLM_MODEL set

        result = validate_config()

        assert result.valid is False
        assert any("LOCAL_LLM_MODEL is not set" in e for e in result.errors)

    def test_error_invalid_local_llm_url(self):
        """Test error when LOCAL_LLM_URL doesn't start with http:// or https://."""
        os.environ["USE_LOCAL_LLM"] = "true"
        os.environ["LOCAL_LLM_MODEL"] = "model"
        os.environ["LOCAL_LLM_URL"] = "localhost:11434"  # Missing http://

        result = validate_config()

        assert result.valid is False
        assert any("must start with http://" in e for e in result.errors)

    def test_local_llm_with_https_url(self):
        """Test local LLM with HTTPS URL is valid."""
        os.environ["USE_LOCAL_LLM"] = "true"
        os.environ["LOCAL_LLM_MODEL"] = "model"
        os.environ["LOCAL_LLM_URL"] = "https://secure-llm.example.com"

        result = validate_config()

        assert result.valid is True
        assert result.config_summary["llm_url"] == "https://secure-llm.example.com"

    # ======================================================================
    # Web Search Configuration Tests
    # ======================================================================

    def test_web_search_disabled(self):
        """Test web search disabled."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["ENABLE_WEB_SEARCH"] = "false"

        result = validate_config()

        assert result.valid is True
        assert result.config_summary["web_search_enabled"] == "False"

    def test_web_search_enabled_with_tavily(self):
        """Test web search enabled with Tavily."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["ENABLE_WEB_SEARCH"] = "true"
        os.environ["USE_TAVILY_SEARCH"] = "true"
        os.environ["TAVILY_API_KEY"] = "tvly-1234567890"

        result = validate_config()

        assert result.valid is True
        assert "Tavily" in result.config_summary["web_search_providers"]

    def test_web_search_enabled_with_brave(self):
        """Test web search enabled with Brave."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["ENABLE_WEB_SEARCH"] = "true"
        os.environ["USE_BRAVE_SEARCH"] = "true"
        os.environ["BRAVE_API_KEY"] = "BSA1234567890"

        result = validate_config()

        assert result.valid is True
        assert "Brave" in result.config_summary["web_search_providers"]

    def test_warning_web_search_enabled_no_providers(self):
        """Test warning when web search enabled but no providers enabled."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["ENABLE_WEB_SEARCH"] = "true"
        os.environ["USE_TAVILY_SEARCH"] = "false"
        os.environ["USE_BRAVE_SEARCH"] = "false"

        result = validate_config()

        assert result.valid is True
        assert any("no search providers are enabled" in w for w in result.warnings)

    def test_warning_tavily_enabled_no_key(self):
        """Test warning when Tavily enabled but no API key."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["ENABLE_WEB_SEARCH"] = "true"
        os.environ["USE_TAVILY_SEARCH"] = "true"
        # No TAVILY_API_KEY

        result = validate_config()

        assert result.valid is True
        assert any("TAVILY_API_KEY is not set" in w for w in result.warnings)

    def test_warning_tavily_placeholder_key(self):
        """Test warning when Tavily has placeholder key."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["ENABLE_WEB_SEARCH"] = "true"
        os.environ["USE_TAVILY_SEARCH"] = "true"
        os.environ["TAVILY_API_KEY"] = "your_tavily_api_key_here"

        result = validate_config()

        assert result.valid is True
        assert any("TAVILY_API_KEY is not set" in w for w in result.warnings)

    def test_warning_brave_enabled_no_key(self):
        """Test warning when Brave enabled but no API key."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["ENABLE_WEB_SEARCH"] = "true"
        os.environ["USE_BRAVE_SEARCH"] = "true"
        # No BRAVE_API_KEY

        result = validate_config()

        assert result.valid is True
        assert any("BRAVE_API_KEY is not set" in w for w in result.warnings)

    def test_web_search_both_providers(self):
        """Test web search with both Tavily and Brave."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["ENABLE_WEB_SEARCH"] = "true"
        os.environ["USE_TAVILY_SEARCH"] = "true"
        os.environ["TAVILY_API_KEY"] = "tvly-key"
        os.environ["USE_BRAVE_SEARCH"] = "true"
        os.environ["BRAVE_API_KEY"] = "brave-key"

        result = validate_config()

        assert result.valid is True
        assert "Tavily" in result.config_summary["web_search_providers"]
        assert "Brave" in result.config_summary["web_search_providers"]

    # ======================================================================
    # NVD API Configuration Tests
    # ======================================================================

    def test_nvd_api_key_configured(self):
        """Test NVD API key configured."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["NVD_API_KEY"] = "nvd-api-key-12345"

        result = validate_config()

        assert result.valid is True
        assert "higher rate limits" in result.config_summary["nvd_api"]
        # Should not have NVD warning
        assert not any("NVD_API_KEY not set" in w for w in result.warnings)

    def test_warning_nvd_api_key_not_configured(self):
        """Test warning when NVD API key not configured."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        # No NVD_API_KEY

        result = validate_config()

        assert result.valid is True
        assert "Not configured" in result.config_summary["nvd_api"]
        assert any("NVD_API_KEY not set" in w for w in result.warnings)

    def test_nvd_api_key_placeholder(self):
        """Test NVD API key with placeholder value is treated as not set."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["NVD_API_KEY"] = "your_nvd_api_key_here"

        result = validate_config()

        assert result.valid is True
        assert "Not configured" in result.config_summary["nvd_api"]

    # ======================================================================
    # Log Level Tests
    # ======================================================================

    def test_valid_log_levels(self):
        """Test all valid log levels."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"

        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            os.environ["LOG_LEVEL"] = level
            result = validate_config()

            assert result.valid is True
            assert result.config_summary["log_level"] == level

    def test_log_level_case_insensitive(self):
        """Test log level is case-insensitive."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["LOG_LEVEL"] = "debug"

        result = validate_config()

        assert result.valid is True
        assert result.config_summary["log_level"] == "DEBUG"

    def test_warning_invalid_log_level(self):
        """Test warning for invalid log level."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["LOG_LEVEL"] = "INVALID"

        result = validate_config()

        assert result.valid is True
        assert any("Invalid LOG_LEVEL" in w for w in result.warnings)
        assert result.config_summary["log_level"] == "INFO"  # Defaults to INFO

    # ======================================================================
    # Max Iterations Tests
    # ======================================================================

    def test_valid_max_iterations(self):
        """Test valid MAX_AGENT_ITERATIONS."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["MAX_AGENT_ITERATIONS"] = "25"

        result = validate_config()

        assert result.valid is True
        assert result.config_summary["max_iterations"] == "25"

    def test_warning_max_iterations_too_low(self):
        """Test warning when MAX_AGENT_ITERATIONS < 1."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["MAX_AGENT_ITERATIONS"] = "0"

        result = validate_config()

        assert result.valid is True
        assert any("must be >= 1" in w for w in result.warnings)
        assert result.config_summary["max_iterations"] == "10"  # Defaults to 10

    def test_warning_max_iterations_very_high(self):
        """Test warning when MAX_AGENT_ITERATIONS > 50."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["MAX_AGENT_ITERATIONS"] = "100"

        result = validate_config()

        assert result.valid is True
        assert any("may cause very long execution times" in w for w in result.warnings)
        assert result.config_summary["max_iterations"] == "100"

    def test_warning_max_iterations_invalid(self):
        """Test warning when MAX_AGENT_ITERATIONS is not a number."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"
        os.environ["MAX_AGENT_ITERATIONS"] = "not-a-number"

        result = validate_config()

        assert result.valid is True
        assert any("must be a number" in w for w in result.warnings)
        assert result.config_summary["max_iterations"] == "10"  # Defaults to 10

    # ======================================================================
    # Complex/Integration Tests
    # ======================================================================

    def test_complete_valid_config(self):
        """Test completely valid configuration with all options."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-1234567890123456789012345678901234567890"
        os.environ["OPENAI_API_KEY"] = "sk-1234567890123456789012345678901234567890"
        os.environ["NVD_API_KEY"] = "nvd-api-key-12345"
        os.environ["ENABLE_WEB_SEARCH"] = "true"
        os.environ["USE_TAVILY_SEARCH"] = "true"
        os.environ["TAVILY_API_KEY"] = "tvly-key"
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["MAX_AGENT_ITERATIONS"] = "20"

        result = validate_config()

        assert result.valid is True
        assert len(result.errors) == 0
        # May have some warnings but should be valid

    def test_multiple_errors(self):
        """Test multiple configuration errors."""
        os.environ["USE_LOCAL_LLM"] = "true"
        # No LOCAL_LLM_MODEL
        os.environ["LOCAL_LLM_URL"] = "invalid-url"

        result = validate_config()

        assert result.valid is False
        assert len(result.errors) >= 2


class TestPrintValidationResults:
    """Tests for print_validation_results function."""

    def test_print_valid_config(self, capsys):
        """Test printing valid configuration."""
        result = ValidationResult(
            valid=True,
            errors=[],
            warnings=[],
            config_summary={"llm_mode": "cloud", "llm_provider": "Anthropic"},
        )

        print_validation_results(result)

        captured = capsys.readouterr()
        assert "Configuration Summary" in captured.out
        assert "Llm Mode: cloud" in captured.out
        assert "Configuration is valid!" in captured.out

    def test_print_config_with_warnings(self, capsys):
        """Test printing configuration with warnings."""
        result = ValidationResult(
            valid=True,
            errors=[],
            warnings=["Warning 1", "Warning 2"],
            config_summary={"key": "value"},
        )

        print_validation_results(result)

        captured = capsys.readouterr()
        assert "2 Warning(s)" in captured.out
        assert "Warning 1" in captured.out
        assert "Warning 2" in captured.out

    def test_print_config_with_errors(self, capsys):
        """Test printing configuration with errors."""
        result = ValidationResult(
            valid=False,
            errors=["Error 1", "Error 2"],
            warnings=[],
            config_summary={},
        )

        print_validation_results(result)

        captured = capsys.readouterr()
        assert "2 Error(s)" in captured.out
        assert "Error 1" in captured.out
        assert "Error 2" in captured.out
        assert "Please fix the above errors" in captured.out


class TestValidateAndExitOnError:
    """Tests for validate_and_exit_on_error function."""

    def test_validate_and_exit_on_error_success(self):
        """Test validate_and_exit_on_error with valid config."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-1234567890123456789012"

        # Should not raise SystemExit
        try:
            validate_and_exit_on_error()
        except SystemExit:
            pytest.fail("Should not exit with valid config")

    def test_validate_and_exit_on_error_failure(self):
        """Test validate_and_exit_on_error with invalid config."""
        # Clear all API keys to cause validation failure
        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "USE_LOCAL_LLM",
            "LOCAL_LLM_MODEL",
        ]:
            os.environ.pop(key, None)

        # Should raise SystemExit
        with pytest.raises(SystemExit) as exc_info:
            validate_and_exit_on_error()

        assert exc_info.value.code == 1
