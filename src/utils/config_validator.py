"""Configuration validation utility for MLPatrol.

This module validates environment configuration at startup to provide
clear, actionable error messages before the application starts.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation.

    Attributes:
        valid: Whether the configuration is valid
        errors: List of error messages
        warnings: List of warning messages
        config_summary: Summary of detected configuration
    """
    valid: bool
    errors: List[str]
    warnings: List[str]
    config_summary: Dict[str, str]


def validate_config() -> ValidationResult:
    """Validate MLPatrol configuration at startup.

    Returns:
        ValidationResult with validation status and messages
    """
    errors = []
    warnings = []
    config_summary = {}

    # Check LLM configuration
    use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    config_summary["llm_mode"] = "local" if use_local else "cloud"

    if use_local:
        # Validate local LLM configuration
        model = os.getenv("LOCAL_LLM_MODEL", "")
        url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")

        if not model:
            errors.append("USE_LOCAL_LLM=true but LOCAL_LLM_MODEL is not set")
        else:
            config_summary["llm_model"] = model

        config_summary["llm_url"] = url

        if not url.startswith("http://") and not url.startswith("https://"):
            errors.append(f"LOCAL_LLM_URL must start with http:// or https://, got: {url}")

    else:
        # Validate cloud LLM configuration
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")

        if not anthropic_key and not openai_key:
            errors.append(
                "No LLM API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
                "or set USE_LOCAL_LLM=true to use Ollama"
            )
        elif anthropic_key:
            config_summary["llm_provider"] = "Anthropic (Claude)"
            if len(anthropic_key) < 20:
                warnings.append("ANTHROPIC_API_KEY seems too short, may be invalid")
        else:
            config_summary["llm_provider"] = "OpenAI (GPT)"
            if len(openai_key) < 20:
                warnings.append("OPENAI_API_KEY seems too short, may be invalid")

    # Check web search configuration
    enable_search = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
    config_summary["web_search_enabled"] = str(enable_search)

    if enable_search:
        use_tavily = os.getenv("USE_TAVILY_SEARCH", "true").lower() == "true"
        use_brave = os.getenv("USE_BRAVE_SEARCH", "true").lower() == "true"

        if not use_tavily and not use_brave:
            warnings.append(
                "ENABLE_WEB_SEARCH=true but no search providers are enabled. "
                "Set USE_TAVILY_SEARCH=true or USE_BRAVE_SEARCH=true"
            )

        providers = []
        if use_tavily:
            tavily_key = os.getenv("TAVILY_API_KEY", "")
            if not tavily_key or tavily_key == "your_tavily_api_key_here":
                warnings.append(
                    "USE_TAVILY_SEARCH=true but TAVILY_API_KEY is not set. "
                    "Web search via Tavily will not work. Get key at https://tavily.com"
                )
            else:
                providers.append("Tavily")

        if use_brave:
            brave_key = os.getenv("BRAVE_API_KEY", "")
            if not brave_key or brave_key == "your_brave_api_key_here":
                warnings.append(
                    "USE_BRAVE_SEARCH=true but BRAVE_API_KEY is not set. "
                    "Web search via Brave will not work. Get key at https://brave.com/search/api/"
                )
            else:
                providers.append("Brave")

        if providers:
            config_summary["web_search_providers"] = ", ".join(providers)
        else:
            config_summary["web_search_providers"] = "None (no valid API keys)"

    # Check NVD API key (optional)
    nvd_key = os.getenv("NVD_API_KEY", "")
    if nvd_key and nvd_key != "your_nvd_api_key_here":
        config_summary["nvd_api"] = "Configured (higher rate limits)"
    else:
        config_summary["nvd_api"] = "Not configured (5 requests/30s limit)"
        warnings.append(
            "NVD_API_KEY not set. CVE searches limited to 5 requests per 30 seconds. "
            "Get a free key at https://nvd.nist.gov/developers/request-an-api-key"
        )

    # Check log level
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        warnings.append(
            f"Invalid LOG_LEVEL: {log_level}. Must be one of: {', '.join(valid_levels)}. "
            f"Defaulting to INFO"
        )
        log_level = "INFO"
    config_summary["log_level"] = log_level

    # Check max iterations
    try:
        max_iter = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
        if max_iter < 1:
            warnings.append("MAX_AGENT_ITERATIONS must be >= 1, using default: 10")
            max_iter = 10
        elif max_iter > 50:
            warnings.append("MAX_AGENT_ITERATIONS > 50 may cause very long execution times")
        config_summary["max_iterations"] = str(max_iter)
    except ValueError:
        warnings.append("MAX_AGENT_ITERATIONS must be a number, using default: 10")
        config_summary["max_iterations"] = "10"

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        config_summary=config_summary
    )


def print_validation_results(result: ValidationResult) -> None:
    """Print validation results to console.

    Args:
        result: ValidationResult to print
    """
    print("\n" + "="*70)
    print("MLPatrol Configuration Validation")
    print("="*70)

    # Print configuration summary
    print("\n📋 Configuration Summary:")
    for key, value in result.config_summary.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")

    # Print warnings
    if result.warnings:
        print(f"\n⚠️  {len(result.warnings)} Warning(s):")
        for i, warning in enumerate(result.warnings, 1):
            print(f"  {i}. {warning}")

    # Print errors
    if result.errors:
        print(f"\n❌ {len(result.errors)} Error(s):")
        for i, error in enumerate(result.errors, 1):
            print(f"  {i}. {error}")
        print("\n🔧 Please fix the above errors before starting MLPatrol.")
    else:
        print("\n✅ Configuration is valid!")

    print("="*70 + "\n")


def validate_and_exit_on_error() -> None:
    """Validate configuration and exit if there are errors.

    This function should be called at application startup.
    """
    result = validate_config()
    print_validation_results(result)

    if not result.valid:
        logger.error("Configuration validation failed. Exiting.")
        raise SystemExit(1)

    # Log warnings
    for warning in result.warnings:
        logger.warning(warning)
