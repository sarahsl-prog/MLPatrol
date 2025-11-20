"""Comprehensive unit tests for MLPatrolAgent class.

This test suite covers:
- Query classification
- Tool orchestration
- Error handling
- Edge cases
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any, List

from src.agent.reasoning_chain import (
    MLPatrolAgent,
    QueryType,
    AgentResult,
    ReasoningStep,
    AgentError,
    QueryClassificationError,
    ToolExecutionError,
    create_mlpatrol_agent,
)
from langchain_core.messages import AIMessage, HumanMessage


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.bind_tools = Mock(return_value=llm)
    llm.invoke = Mock()
    return llm


@pytest.fixture
def mock_tools():
    """Create real Tool objects for testing."""
    from langchain_core.tools import StructuredTool

    tool1 = StructuredTool.from_function(
        name="cve_search",
        description="Search for CVEs",
        func=lambda library, days_back=90: '{"status": "success"}'
    )

    tool2 = StructuredTool.from_function(
        name="web_search",
        description="Search the web",
        func=lambda query, max_results=5: '{"status": "success"}'
    )

    tool3 = StructuredTool.from_function(
        name="analyze_dataset",
        description="Analyze dataset security",
        func=lambda data_path=None, data_json=None: '{"status": "success"}'
    )

    return [tool1, tool2, tool3]


@pytest.fixture
def agent(mock_llm, mock_tools):
    """Create a test agent instance."""
    with patch('src.agent.reasoning_chain.create_mlpatrol_tools', return_value=mock_tools):
        agent_instance = MLPatrolAgent(llm=mock_llm, verbose=False, max_iterations=10)
        return agent_instance


@pytest.fixture
def mock_agent_result():
    """Create a mock agent executor result."""
    return {
        "output": "This is the agent's response",
        "intermediate_steps": [
            (
                Mock(
                    tool="cve_search",
                    tool_input={"library": "numpy", "days_back": 90},
                    log="I need to search for CVEs in numpy"
                ),
                '{"status": "success", "cve_count": 2}'
            )
        ]
    }


# ============================================================================
# Query Classification Tests
# ============================================================================


class TestQueryClassification:
    """Tests for query classification functionality."""

    def test_classify_cve_monitoring_query(self, agent, mock_llm):
        """Test classification of CVE monitoring queries."""
        mock_llm.invoke.return_value = AIMessage(content="CVE_MONITORING")

        query_type = agent.analyze_query("Check for numpy vulnerabilities")

        assert query_type == QueryType.CVE_MONITORING
        assert mock_llm.invoke.called

    def test_classify_dataset_analysis_query(self, agent, mock_llm):
        """Test classification of dataset analysis queries."""
        mock_llm.invoke.return_value = AIMessage(content="DATASET_ANALYSIS")

        query_type = agent.analyze_query("Analyze this dataset for poisoning")

        assert query_type == QueryType.DATASET_ANALYSIS

    def test_classify_code_generation_query(self, agent, mock_llm):
        """Test classification of code generation queries."""
        mock_llm.invoke.return_value = AIMessage(content="CODE_GENERATION")

        query_type = agent.analyze_query("Generate a security validation script")

        assert query_type == QueryType.CODE_GENERATION

    def test_classify_general_security_query(self, agent, mock_llm):
        """Test classification of general security queries."""
        mock_llm.invoke.return_value = AIMessage(content="GENERAL_SECURITY")

        query_type = agent.analyze_query("What are best practices for ML security?")

        assert query_type == QueryType.GENERAL_SECURITY

    def test_classify_with_context(self, agent, mock_llm):
        """Test classification with additional context."""
        mock_llm.invoke.return_value = AIMessage(content="DATASET_ANALYSIS")

        context = {
            "file_path": "/path/to/data.csv",
            "dataset": True
        }

        query_type = agent.analyze_query("Check this data", context=context)

        assert query_type == QueryType.DATASET_ANALYSIS

    def test_classify_ambiguous_query_defaults_to_general(self, agent, mock_llm):
        """Test that ambiguous queries default to GENERAL_SECURITY."""
        mock_llm.invoke.return_value = AIMessage(content="UNKNOWN_TYPE")

        query_type = agent.analyze_query("What is ML?")

        assert query_type == QueryType.GENERAL_SECURITY

    def test_classify_query_handles_llm_error(self, agent, mock_llm):
        """Test error handling when LLM fails during classification."""
        mock_llm.invoke.side_effect = Exception("LLM error")

        with pytest.raises(QueryClassificationError):
            agent.analyze_query("Check numpy CVEs")


# ============================================================================
# Tool Orchestration Tests
# ============================================================================


class TestToolOrchestration:
    """Tests for tool orchestration and reasoning."""

    def test_get_tool_info(self, agent):
        """Test retrieving tool information."""
        tool_info = agent.get_tool_info()

        assert len(tool_info) == 3
        assert all("name" in info and "description" in info for info in tool_info)
        assert any(info["name"] == "cve_search" for info in tool_info)

    def test_reasoning_step_creation(self):
        """Test ReasoningStep creation and serialization."""
        step = ReasoningStep(
            step_number=1,
            thought="I need to search",
            action="cve_search",
            action_input={"library": "numpy"},
            observation="Found 2 CVEs",
            timestamp=time.time(),
            duration_ms=150.5
        )

        step_dict = step.to_dict()

        assert step_dict["step_number"] == 1
        assert step_dict["action"] == "cve_search"
        assert step_dict["duration_ms"] == 150.5

    def test_long_observation_truncated(self):
        """Test that long observations are truncated in dict output."""
        long_text = "x" * 1000
        step = ReasoningStep(
            step_number=1,
            thought="Search",
            action="cve_search",
            action_input={},
            observation=long_text,
            timestamp=time.time()
        )

        step_dict = step.to_dict()

        assert len(step_dict["observation"]) <= 503  # 500 + "..."
        assert step_dict["observation"].endswith("...")

    def test_confidence_clamped_to_valid_range(self, agent):
        """Test that confidence is always between 0 and 1."""
        # Test with no tools
        result_no_tools = agent._calculate_confidence(
            QueryType.GENERAL_SECURITY,
            [],
            []
        )
        assert 0.0 <= result_no_tools <= 1.0

        # Test with tools
        steps = [
            ReasoningStep(1, "thought", "cve_search", {}, "observation", time.time()),
            ReasoningStep(2, "thought", "web_search", {}, "observation", time.time()),
        ]

        result_with_tools = agent._calculate_confidence(
            QueryType.CVE_MONITORING,
            steps,
            ["cve_search", "web_search"]
        )
        assert 0.0 <= result_with_tools <= 1.0
        assert result_with_tools > result_no_tools


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in various scenarios."""

    def test_validate_query_too_short(self, agent):
        """Test validation rejects queries that are too short."""
        is_valid, error = agent._validate_query("hi")

        assert not is_valid
        assert "too short" in error.lower()

    def test_validate_query_too_long(self, agent):
        """Test validation rejects queries that are too long."""
        long_query = "x" * 10001
        is_valid, error = agent._validate_query(long_query)

        assert not is_valid
        assert "too long" in error.lower()

    def test_validate_query_suspicious_script_tag(self, agent):
        """Test validation detects suspicious script tags."""
        is_valid, error = agent._validate_query("Check numpy <script>alert('xss')</script>")

        assert not is_valid
        assert "unsafe" in error.lower()

    def test_validate_query_suspicious_javascript(self, agent):
        """Test validation detects javascript: protocol."""
        is_valid, error = agent._validate_query("javascript:alert('test')")

        assert not is_valid
        assert "unsafe" in error.lower()

    def test_validate_query_suspicious_eval(self, agent):
        """Test validation detects eval() calls."""
        is_valid, error = agent._validate_query("Check eval(malicious_code)")

        assert not is_valid
        assert "unsafe" in error.lower()

    def test_validate_query_valid(self, agent):
        """Test that valid queries pass validation."""
        is_valid, error = agent._validate_query("Check numpy for CVEs")

        assert is_valid
        assert error is None

    def test_initialization_without_llm_raises_error(self):
        """Test that initializing without LLM raises ValueError."""
        with pytest.raises(ValueError, match="LLM instance is required"):
            MLPatrolAgent(llm=None)


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_clear_history(self, agent):
        """Test clearing chat history."""
        # Add some history
        agent.chat_history.append(HumanMessage(content="Test"))
        agent.chat_history.append(AIMessage(content="Response"))
        agent.reasoning_history.append(Mock())

        assert len(agent.chat_history) > 0

        agent.clear_history()

        assert len(agent.chat_history) == 0
        assert len(agent.reasoning_history) == 0

    def test_agent_repr(self, agent):
        """Test agent string representation."""
        repr_str = repr(agent)

        assert "MLPatrolAgent" in repr_str
        assert "tools=" in repr_str
        assert "max_iterations=" in repr_str

    def test_agent_result_serialization(self):
        """Test AgentResult to_dict serialization."""
        result = AgentResult(
            answer="Test answer",
            reasoning_steps=[],
            tools_used=["cve_search"],
            query_type=QueryType.CVE_MONITORING,
            confidence=0.85,
            total_duration_ms=1250.5
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["answer"] == "Test answer"
        assert result_dict["tools_used"] == ["cve_search"]
        assert result_dict["query_type"] == "cve_monitoring"
        assert result_dict["confidence"] == 0.85
        assert result_dict["total_duration_ms"] == 1250.5


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    @patch('langchain_anthropic.ChatAnthropic')
    def test_create_mlpatrol_agent_with_claude(self, mock_claude, mock_tools):
        """Test creating agent with Claude model."""
        mock_llm_instance = Mock()
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)
        mock_claude.return_value = mock_llm_instance

        with patch('src.agent.reasoning_chain.create_mlpatrol_tools', return_value=mock_tools):
            agent = create_mlpatrol_agent(
                api_key="test-key",
                model="claude-sonnet-4",
                verbose=False
            )

        assert agent is not None
        mock_claude.assert_called_once()
        assert mock_claude.call_args[1]["model"] == "claude-sonnet-4"
        assert mock_claude.call_args[1]["anthropic_api_key"] == "test-key"

    @patch('langchain_openai.ChatOpenAI')
    def test_create_mlpatrol_agent_with_gpt4(self, mock_openai, mock_tools):
        """Test creating agent with GPT-4 model."""
        mock_llm_instance = Mock()
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)
        mock_openai.return_value = mock_llm_instance

        with patch('src.agent.reasoning_chain.create_mlpatrol_tools', return_value=mock_tools):
            agent = create_mlpatrol_agent(
                api_key="test-key",
                model="gpt-4",
                verbose=False
            )

        assert agent is not None
        mock_openai.assert_called_once()
        assert mock_openai.call_args[1]["model"] == "gpt-4"

    def test_create_mlpatrol_agent_unsupported_model(self):
        """Test that unsupported model raises error."""
        with pytest.raises(ValueError, match="Unsupported model"):
            create_mlpatrol_agent(
                api_key="test-key",
                model="unsupported-model",
                verbose=False
            )
