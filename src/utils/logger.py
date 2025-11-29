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

# Add console handler with colors
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
        rotation: When to rotate log files (e.g., "10 MB", "1 day")
        retention: How long to keep old log files (e.g., "1 week", "30 days")
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Loguru logger instance bound to the component name
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
        agent_name: Name of the agent (e.g., 'reasoning_chain', 'tools')

    Returns:
        Loguru logger configured for agents
    """
    return get_logger(agent_name, log_dir="agents")


def get_security_logger(component_name: str):
    """Get logger for security components.

    Args:
        component_name: Name of security component (e.g., 'cve_monitor')

    Returns:
        Loguru logger configured for security components
    """
    return get_logger(component_name, log_dir="security")


def get_dataset_logger(component_name: str):
    """Get logger for dataset components.

    Args:
        component_name: Name of dataset component (e.g., 'bias_analyzer')

    Returns:
        Loguru logger configured for dataset components
    """
    return get_logger(component_name, log_dir="dataset")


# For backwards compatibility with code using get_logger(name)
# This ensures existing code continues to work
def configure_logging(level: Optional[str] = None) -> None:
    """Configure logging (legacy compatibility function).

    Args:
        level: Logging level (optional, for compatibility)
    """
    # Loguru is already configured above, this is just for compatibility
    pass
