"""Unit tests for MCP connector registry.

This test suite covers:
- MCPConnector dataclass
- ConnectorRegistry functionality
- Connector registration and retrieval
- Default connectors loading
"""

import pytest
from src.mcp.connectors import MCPConnector, ConnectorRegistry, load_default_connectors


class TestMCPConnector:
    """Tests for MCPConnector dataclass."""

    def test_connector_creation(self):
        """Test creating an MCPConnector."""
        connector = MCPConnector(
            name="test-connector",
            description="Test description",
            config={"key": "value"},
        )

        assert connector.name == "test-connector"
        assert connector.description == "Test description"
        assert connector.config == {"key": "value"}

    def test_connector_empty_config(self):
        """Test creating connector with empty config."""
        connector = MCPConnector(
            name="minimal", description="Minimal connector", config={}
        )

        assert connector.config == {}


class TestConnectorRegistry:
    """Tests for ConnectorRegistry class."""

    def test_registry_initialization(self):
        """Test creating a new registry."""
        registry = ConnectorRegistry()

        assert registry._connectors == {}

    def test_register_connector(self):
        """Test registering a connector."""
        registry = ConnectorRegistry()
        connector = MCPConnector(
            name="test", description="Test", config={"url": "http://test"}
        )

        registry.register(connector)

        assert "test" in registry._connectors
        assert registry._connectors["test"] == connector

    def test_register_multiple_connectors(self):
        """Test registering multiple connectors."""
        registry = ConnectorRegistry()

        connector1 = MCPConnector(name="conn1", description="First", config={})
        connector2 = MCPConnector(name="conn2", description="Second", config={})

        registry.register(connector1)
        registry.register(connector2)

        assert len(registry._connectors) == 2
        assert "conn1" in registry._connectors
        assert "conn2" in registry._connectors

    def test_register_overwrites_existing(self):
        """Test registering with same name overwrites previous."""
        registry = ConnectorRegistry()

        connector1 = MCPConnector(name="test", description="First", config={"v": "1"})
        connector2 = MCPConnector(name="test", description="Second", config={"v": "2"})

        registry.register(connector1)
        registry.register(connector2)

        assert len(registry._connectors) == 1
        assert registry._connectors["test"].description == "Second"

    def test_get_existing_connector(self):
        """Test getting an existing connector."""
        registry = ConnectorRegistry()
        connector = MCPConnector(name="test", description="Test", config={})
        registry.register(connector)

        retrieved = registry.get("test")

        assert retrieved is not None
        assert retrieved.name == "test"
        assert retrieved is connector  # Same object

    def test_get_nonexistent_connector(self):
        """Test getting a non-existent connector returns None."""
        registry = ConnectorRegistry()

        retrieved = registry.get("nonexistent")

        assert retrieved is None

    def test_get_from_empty_registry(self):
        """Test getting from empty registry returns None."""
        registry = ConnectorRegistry()

        assert registry.get("anything") is None

    def test_all_returns_copy(self):
        """Test all() returns a copy of connectors dict."""
        registry = ConnectorRegistry()
        connector1 = MCPConnector(name="conn1", description="First", config={})
        connector2 = MCPConnector(name="conn2", description="Second", config={})

        registry.register(connector1)
        registry.register(connector2)

        all_connectors = registry.all()

        assert len(all_connectors) == 2
        assert "conn1" in all_connectors
        assert "conn2" in all_connectors

        # Modifying returned dict should not affect registry
        all_connectors["conn3"] = MCPConnector(name="conn3", description="Third", config={})
        assert "conn3" not in registry._connectors

    def test_all_empty_registry(self):
        """Test all() on empty registry."""
        registry = ConnectorRegistry()

        all_connectors = registry.all()

        assert all_connectors == {}


class TestLoadDefaultConnectors:
    """Tests for load_default_connectors function."""

    def test_load_default_connectors_returns_registry(self):
        """Test load_default_connectors returns a ConnectorRegistry."""
        registry = load_default_connectors()

        assert isinstance(registry, ConnectorRegistry)

    def test_load_default_connectors_includes_nvd(self):
        """Test NVD connector is loaded."""
        registry = load_default_connectors()

        nvd = registry.get("nvd")

        assert nvd is not None
        assert nvd.name == "nvd"
        assert "National Vulnerability Database" in nvd.description
        assert "base_url" in nvd.config
        assert "nvd.nist.gov" in nvd.config["base_url"]

    def test_load_default_connectors_includes_tavily(self):
        """Test Tavily connector is loaded."""
        registry = load_default_connectors()

        tavily = registry.get("tavily")

        assert tavily is not None
        assert tavily.name == "tavily"
        assert "Tavily" in tavily.description
        assert "env" in tavily.config
        assert tavily.config["env"] == "TAVILY_API_KEY"

    def test_load_default_connectors_includes_brave(self):
        """Test Brave connector is loaded."""
        registry = load_default_connectors()

        brave = registry.get("brave")

        assert brave is not None
        assert brave.name == "brave"
        assert "Brave" in brave.description
        assert "env" in brave.config
        assert brave.config["env"] == "BRAVE_API_KEY"

    def test_load_default_connectors_count(self):
        """Test all default connectors are loaded."""
        registry = load_default_connectors()

        all_connectors = registry.all()

        # Should have nvd, tavily, and brave
        assert len(all_connectors) >= 3

    def test_load_default_connectors_multiple_calls(self):
        """Test calling load_default_connectors multiple times creates new registries."""
        registry1 = load_default_connectors()
        registry2 = load_default_connectors()

        # Should be different instances
        assert registry1 is not registry2

        # But should have same connectors
        assert set(registry1.all().keys()) == set(registry2.all().keys())

    def test_default_connectors_can_be_extended(self):
        """Test default connectors registry can be extended."""
        registry = load_default_connectors()

        custom_connector = MCPConnector(
            name="custom", description="Custom connector", config={"custom": "value"}
        )
        registry.register(custom_connector)

        all_connectors = registry.all()

        assert "custom" in all_connectors
        assert len(all_connectors) >= 4  # 3 defaults + 1 custom
