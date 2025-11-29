"""Minimal registry for MCP connectors used by MLPatrol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MCPConnector:
    """Metadata about an external tool/connector."""

    name: str
    description: str
    config: Dict[str, str]


class ConnectorRegistry:
    """In-memory registry that mimics how MCP connectors are exposed."""

    def __init__(self) -> None:
        self._connectors: Dict[str, MCPConnector] = {}

    def register(self, connector: MCPConnector) -> None:
        self._connectors[connector.name] = connector

    def get(self, name: str) -> Optional[MCPConnector]:
        return self._connectors.get(name)

    def all(self) -> Dict[str, MCPConnector]:
        return dict(self._connectors)


def load_default_connectors() -> ConnectorRegistry:
    """Pre-populate a registry with the connectors MLPatrol relies on."""
    registry = ConnectorRegistry()
    registry.register(
        MCPConnector(
            name="nvd",
            description="National Vulnerability Database REST API",
            config={"base_url": "https://services.nvd.nist.gov"},
        )
    )
    registry.register(
        MCPConnector(
            name="tavily",
            description="Tavily AI search connector",
            config={"env": "TAVILY_API_KEY"},
        )
    )
    registry.register(
        MCPConnector(
            name="brave",
            description="Brave Search API connector",
            config={"env": "BRAVE_API_KEY"},
        )
    )
    return registry
