"""Unit tests for configuration utilities.

This test suite covers:
- Settings loading and caching
- Environment variable handling
- API key validation
- Local vs cloud LLM configuration
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.utils.config import Settings, _load_env_file, get_settings, refresh_settings


class TestSettings:
    """Tests for Settings dataclass."""

    def test_settings_creation(self):
        """Test creating a Settings object."""
        settings = Settings(
            anthropic_api_key="test-key",
            openai_api_key=None,
            nvd_api_key=None,
            use_local_llm=False,
            local_llm_model="test-model",
            local_llm_url="http://localhost:11434",
            log_level="DEBUG",
        )
        assert settings.anthropic_api_key == "test-key"
        assert settings.openai_api_key is None
        assert settings.log_level == "DEBUG"

    def test_has_cloud_llm_with_anthropic(self):
        """Test has_cloud_llm returns True with Anthropic key."""
        settings = Settings(
            anthropic_api_key="test-key",
            openai_api_key=None,
            nvd_api_key=None,
            use_local_llm=False,
            local_llm_model="model",
            local_llm_url="url",
        )
        assert settings.has_cloud_llm is True

    def test_has_cloud_llm_with_openai(self):
        """Test has_cloud_llm returns True with OpenAI key."""
        settings = Settings(
            anthropic_api_key=None,
            openai_api_key="test-key",
            nvd_api_key=None,
            use_local_llm=False,
            local_llm_model="model",
            local_llm_url="url",
        )
        assert settings.has_cloud_llm is True

    def test_has_cloud_llm_with_both_keys(self):
        """Test has_cloud_llm returns True with both keys."""
        settings = Settings(
            anthropic_api_key="anthropic-key",
            openai_api_key="openai-key",
            nvd_api_key=None,
            use_local_llm=False,
            local_llm_model="model",
            local_llm_url="url",
        )
        assert settings.has_cloud_llm is True

    def test_has_cloud_llm_without_keys(self):
        """Test has_cloud_llm returns False without any keys."""
        settings = Settings(
            anthropic_api_key=None,
            openai_api_key=None,
            nvd_api_key=None,
            use_local_llm=False,
            local_llm_model="model",
            local_llm_url="url",
        )
        assert settings.has_cloud_llm is False

    def test_settings_immutable(self):
        """Test that Settings is immutable (frozen dataclass)."""
        settings = Settings(
            anthropic_api_key="test",
            openai_api_key=None,
            nvd_api_key=None,
            use_local_llm=False,
            local_llm_model="model",
            local_llm_url="url",
        )
        with pytest.raises(AttributeError):
            settings.anthropic_api_key = "new-key"  # type: ignore


class TestLoadEnvFile:
    """Tests for _load_env_file function."""

    def test_load_env_file_when_none(self):
        """Test _load_env_file does nothing when env_file is None."""
        # Should not raise any errors
        _load_env_file(None)

    @patch("src.utils.config.load_dotenv")
    def test_load_env_file_when_exists(self, mock_load_dotenv, tmp_path):
        """Test _load_env_file loads existing file."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=test_value\n")

        _load_env_file(str(env_file))

        mock_load_dotenv.assert_called_once()

    def test_load_env_file_when_not_exists(self, tmp_path):
        """Test _load_env_file handles non-existent file."""
        env_file = tmp_path / "nonexistent.env"
        # Should not raise errors
        _load_env_file(str(env_file))


class TestGetSettings:
    """Tests for get_settings function."""

    def setup_method(self):
        """Clear settings cache and environment variables before each test."""
        # Clear environment variables first
        env_vars = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "NVD_API_KEY",
            "USE_LOCAL_LLM",
            "LOCAL_LLM_MODEL",
            "LOCAL_LLM_URL",
            "MLPATROL_LOG_LEVEL",
        ]
        for var in env_vars:
            os.environ.pop(var, None)
        # Then clear cache
        refresh_settings()

    def teardown_method(self):
        """Clean up environment variables after each test."""
        env_vars = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "NVD_API_KEY",
            "USE_LOCAL_LLM",
            "LOCAL_LLM_MODEL",
            "LOCAL_LLM_URL",
            "MLPATROL_LOG_LEVEL",
        ]
        for var in env_vars:
            os.environ.pop(var, None)
        refresh_settings()

    def test_get_settings_defaults(self):
        """Test get_settings returns defaults when no env vars set."""
        settings = get_settings()

        assert settings.anthropic_api_key is None
        assert settings.openai_api_key is None
        assert settings.nvd_api_key is None
        assert settings.use_local_llm is False
        assert settings.local_llm_model == "ollama/llama3.1:8b"
        assert settings.local_llm_url == "http://localhost:11434"
        assert settings.log_level == "INFO"

    def test_get_settings_with_anthropic_key(self):
        """Test get_settings reads ANTHROPIC_API_KEY from environment."""
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
        settings = refresh_settings()

        assert settings.anthropic_api_key == "test-anthropic-key"

    def test_get_settings_with_openai_key(self):
        """Test get_settings reads OPENAI_API_KEY from environment."""
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        settings = refresh_settings()

        assert settings.openai_api_key == "test-openai-key"

    def test_get_settings_with_nvd_key(self):
        """Test get_settings reads NVD_API_KEY from environment."""
        os.environ["NVD_API_KEY"] = "test-nvd-key"
        settings = refresh_settings()

        assert settings.nvd_api_key == "test-nvd-key"

    def test_get_settings_use_local_llm_true(self):
        """Test get_settings handles USE_LOCAL_LLM=true."""
        os.environ["USE_LOCAL_LLM"] = "true"
        settings = refresh_settings()

        assert settings.use_local_llm is True

    def test_get_settings_use_local_llm_false(self):
        """Test get_settings handles USE_LOCAL_LLM=false."""
        os.environ["USE_LOCAL_LLM"] = "false"
        settings = refresh_settings()

        assert settings.use_local_llm is False

    def test_get_settings_use_local_llm_case_insensitive(self):
        """Test get_settings handles USE_LOCAL_LLM case-insensitively."""
        os.environ["USE_LOCAL_LLM"] = "TRUE"
        settings = refresh_settings()

        assert settings.use_local_llm is True

    def test_get_settings_custom_local_model(self):
        """Test get_settings reads custom LOCAL_LLM_MODEL."""
        os.environ["LOCAL_LLM_MODEL"] = "ollama/mistral:latest"
        settings = refresh_settings()

        assert settings.local_llm_model == "ollama/mistral:latest"

    def test_get_settings_custom_local_url(self):
        """Test get_settings reads custom LOCAL_LLM_URL."""
        os.environ["LOCAL_LLM_URL"] = "http://localhost:8080"
        settings = refresh_settings()

        assert settings.local_llm_url == "http://localhost:8080"

    def test_get_settings_custom_log_level(self):
        """Test get_settings reads MLPATROL_LOG_LEVEL."""
        os.environ["MLPATROL_LOG_LEVEL"] = "debug"
        settings = refresh_settings()

        assert settings.log_level == "DEBUG"

    def test_get_settings_cached(self):
        """Test get_settings returns cached value on second call."""
        # Clear any existing cache first
        refresh_settings()

        os.environ["ANTHROPIC_API_KEY"] = "first-key"
        settings1 = get_settings()

        # Change environment but don't refresh
        os.environ["ANTHROPIC_API_KEY"] = "second-key"
        settings2 = get_settings()

        # Should still have first key (cached)
        assert settings2.anthropic_api_key == "first-key"
        assert settings1 is settings2  # Same object

    def test_refresh_settings_clears_cache(self):
        """Test refresh_settings clears cache and reloads."""
        os.environ["ANTHROPIC_API_KEY"] = "first-key"
        settings1 = refresh_settings()

        os.environ["ANTHROPIC_API_KEY"] = "second-key"
        settings2 = refresh_settings()

        assert settings1.anthropic_api_key == "first-key"
        assert settings2.anthropic_api_key == "second-key"
        assert settings1 is not settings2  # Different objects

    def test_get_settings_all_env_vars(self):
        """Test get_settings with all environment variables set."""
        os.environ["ANTHROPIC_API_KEY"] = "anthropic-key"
        os.environ["OPENAI_API_KEY"] = "openai-key"
        os.environ["NVD_API_KEY"] = "nvd-key"
        os.environ["USE_LOCAL_LLM"] = "true"
        os.environ["LOCAL_LLM_MODEL"] = "custom-model"
        os.environ["LOCAL_LLM_URL"] = "http://custom:9999"
        os.environ["MLPATROL_LOG_LEVEL"] = "WARNING"

        settings = refresh_settings()

        assert settings.anthropic_api_key == "anthropic-key"
        assert settings.openai_api_key == "openai-key"
        assert settings.nvd_api_key == "nvd-key"
        assert settings.use_local_llm is True
        assert settings.local_llm_model == "custom-model"
        assert settings.local_llm_url == "http://custom:9999"
        assert settings.log_level == "WARNING"

    def test_get_settings_with_env_file(self, tmp_path):
        """Test get_settings loads from .env file."""
        env_file = tmp_path / ".env.test"
        env_file.write_text("ANTHROPIC_API_KEY=from-file\n")

        # Mock load_dotenv to actually load our test file
        with patch("src.utils.config.load_dotenv") as mock_load:

            def side_effect(path):
                # Simulate loading the env file
                with open(path) as f:
                    for line in f:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            os.environ[key] = value

            mock_load.side_effect = side_effect

            settings = refresh_settings(str(env_file))
            assert settings.anthropic_api_key == "from-file"
