"""Agent Reasoning Chain - Main MLPatrol agent implementation.

This module contains the core MLPatrol agent that orchestrates security analysis
through multi-step reasoning, tool execution, and result synthesis using LangGraph.
"""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

try:
    # Use the modern langchain.agents.create_agent (LangGraph 1.0+)
    # This replaces the deprecated langgraph.prebuilt.create_react_agent
    from langchain.agents import create_agent as create_react_agent  # type: ignore
    from langgraph.graph import END, StateGraph
except Exception:
    try:
        # Fallback to legacy langgraph.prebuilt for older versions
        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import create_react_agent  # type: ignore
    except Exception:
        # If neither import is available, keep names defined as None and allow
        # the setup logic to raise a clear error when attempting to use them.
        create_react_agent = None  # type: ignore
        from langgraph.graph import StateGraph, END

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

# Setup logging with Loguru
from src.utils.logger import get_agent_logger

from .prompts import (
    VALIDATION_PATTERNS,
    get_agent_prompt,
    get_classification_prompt,
    get_error_message,
    get_synthesis_prompt,
)
from .tools import create_mlpatrol_tools, parse_cve_results, parse_dataset_analysis

logger = get_agent_logger("reasoning_chain")


# ============================================================================
# Enums and Data Models
# ============================================================================


class QueryType(Enum):
    """Types of queries the agent can handle.

    Attributes:
        CVE_MONITORING: User wants to check for vulnerabilities
        DATASET_ANALYSIS: User wants to analyze dataset security
        CODE_GENERATION: User wants security validation code
        GENERAL_SECURITY: User wants security advice/information
    """

    CVE_MONITORING = "cve_monitoring"
    DATASET_ANALYSIS = "dataset_analysis"
    CODE_GENERATION = "code_generation"
    GENERAL_SECURITY = "general_security"


@dataclass
class ReasoningStep:
    """A single step in the agent's reasoning process.

    Attributes:
        step_number: Sequential step number (1-indexed)
        thought: The agent's reasoning or planning
        action: The tool or action taken (e.g., "cve_search")
        action_input: Input parameters for the action
        observation: Result/output from the action
        timestamp: When this step occurred
        duration_ms: How long the step took in milliseconds
    """

    step_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    timestamp: float
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": (
                self.observation[:500] + "..."
                if len(self.observation) > 500
                else self.observation
            ),
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class AgentResult:
    """Final result from the agent's execution.

    Attributes:
        answer: The agent's final response to the user
        reasoning_steps: List of reasoning steps taken
        tools_used: List of tool names that were called
        query_type: Classified query type
        confidence: Confidence score (0-1)
        total_duration_ms: Total execution time in milliseconds
        error: Error message if execution failed
    """

    answer: str
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    query_type: Optional[QueryType] = None
    confidence: float = 0.0
    total_duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "reasoning_steps": [step.to_dict() for step in self.reasoning_steps],
            "tools_used": self.tools_used,
            "query_type": self.query_type.value if self.query_type else None,
            "confidence": round(self.confidence, 2),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "error": self.error,
        }


# ============================================================================
# Custom Exception Classes
# ============================================================================


class AgentError(Exception):
    """Base exception for agent errors."""

    pass


class QueryClassificationError(AgentError):
    """Raised when query classification fails."""

    pass


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails."""

    pass


# ============================================================================
# LangGraph State Definition
# ============================================================================


class AgentState(TypedDict):
    """State for the LangGraph agent.

    Attributes:
        messages: Conversation history
        query_type: Classified query type
        reasoning_steps: List of reasoning steps
        tools_used: List of tool names used
        context: Additional context (files, datasets, etc.)
    """

    messages: List[BaseMessage]
    query_type: Optional[str]
    reasoning_steps: List[Dict[str, Any]]
    tools_used: List[str]
    context: Optional[Dict[str, Any]]


# ============================================================================
# Main Agent Class
# ============================================================================


class MLPatrolAgent:
    """Core reasoning agent for MLPatrol security analysis.

    This agent orchestrates ML security analysis using LangGraph by:
    - Understanding user queries about ML security
    - Planning multi-step approaches using available tools
    - Executing tools to gather information
    - Synthesizing findings into actionable insights
    - Providing transparent reasoning throughout

    The agent uses LangGraph with Claude or GPT-4 for structured reasoning
    and tool orchestration.

    Attributes:
        llm: Language model instance (Claude or GPT-4)
        tools: List of available tools for the agent
        verbose: Whether to log detailed reasoning steps
        max_iterations: Maximum number of reasoning steps (default: 15)
        max_execution_time: Maximum execution time in seconds (default: 300)

    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> llm = ChatAnthropic(model="claude-sonnet-4-0")
        >>> agent = MLPatrolAgent(llm=llm, verbose=True)
        >>> result = agent.run("Check for numpy CVEs")
        >>> print(result.answer)
        >>> for step in result.reasoning_steps:
        ...     print(f"Step {step.step_number}: {step.action}")
    """

    def __init__(
        self,
        llm: Any,
        tools: Optional[List[Any]] = None,
        verbose: bool = True,
        max_iterations: int = 15,
        max_execution_time: int = 300,
    ):
        """Initialize the MLPatrol agent.

        Args:
            llm: Language model instance (Claude Sonnet 4 or GPT-4)
            tools: Optional list of tools (if None, uses default MLPatrol tools)
            verbose: Whether to log reasoning steps to console
            max_iterations: Maximum number of reasoning iterations
            max_execution_time: Maximum execution time in seconds

        Raises:
            ValueError: If llm is not provided
        """
        if llm is None:
            raise ValueError("LLM instance is required")

        self.llm = llm
        self.tools = tools if tools is not None else create_mlpatrol_tools()
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time

        # State tracking
        self.reasoning_history: List[ReasoningStep] = []
        self.chat_history: List[BaseMessage] = []
        self.current_step = 0

        # Setup agent components
        self._setup_agent()
        self._setup_logging()

        logger.info(
            f"MLPatrol agent initialized with {len(self.tools)} tools, "
            f"max_iterations={max_iterations}, verbose={verbose}"
        )

    def _setup_logging(self) -> None:
        """Configure logging for the agent."""
        # Logging already configured via Loguru in src.utils.logger
        pass

    def _setup_agent(self) -> None:
        """Configure the LangGraph agent with prompts and tools.

        This creates the ReAct agent that orchestrates tool usage and reasoning.
        """
        try:
            # Get the system prompt
            system_prompt = get_agent_prompt()

            # Extract the system message content from the prompt template
            system_message = None
            for msg in system_prompt.messages:
                if hasattr(msg, "prompt") and hasattr(msg.prompt, "template"):
                    if "MLPatrol" in msg.prompt.template:
                        system_message = msg.prompt.template
                        break

            if not system_message:
                # Fallback to a basic system prompt
                system_message = (
                    "You are MLPatrol, an expert AI security agent for ML systems."
                )

            # Create the ReAct agent using the available agent factory.
            # The modern API (langchain.agents.create_agent) uses 'system_prompt',
            # while the legacy API (langgraph.prebuilt.create_react_agent) uses 'prompt'.
            if create_react_agent is None:
                raise AgentError(
                    "No agent factory available (create_react_agent is not imported)"
                )

            import inspect

            sig = inspect.signature(create_react_agent)
            params = sig.parameters

            # Check which parameter name to use for the system message
            if "system_prompt" in params:
                # Modern API: langchain.agents.create_agent
                self.agent_executor = create_react_agent(
                    model=self.llm,
                    tools=self.tools,
                    system_prompt=system_message,
                )
            elif "prompt" in params:
                # Legacy API: langgraph.prebuilt.create_react_agent
                self.agent_executor = create_react_agent(
                    model=self.llm,
                    tools=self.tools,
                    prompt=system_message,
                )
            else:
                # Fallback: try without system message parameter
                self.agent_executor = create_react_agent(
                    model=self.llm,
                    tools=self.tools,
                )

            logger.info("LangGraph agent configured successfully")

        except Exception as e:
            logger.error(f"Failed to setup agent: {e}", exc_info=True)
            raise AgentError(f"Agent setup failed: {e}")

    def analyze_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> QueryType:
        """Classify the type of security query using LLM.

        Args:
            query: User's security question
            context: Optional additional context (uploaded files, etc.)

        Returns:
            The classified QueryType
        """
        logger.info(f"Classifying query: {query[:100]}...")

        # Prepare context string
        context_str = str(context) if context else "None"

        try:
            prompt_value = get_classification_prompt().invoke(
                {"query": query, "context": context_str}
            )
            response = self.llm.invoke(prompt_value.to_messages())
        except Exception as e:
            logger.error(f"Query classification failed: {e}", exc_info=True)
            raise QueryClassificationError(
                "LLM-based query classification failed"
            ) from e

        # Parse response
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = " ".join(str(chunk) for chunk in content)
        content = str(content).strip().upper()

        # Map to enum
        if "CVE_MONITORING" in content:
            return QueryType.CVE_MONITORING
        elif "DATASET_ANALYSIS" in content:
            return QueryType.DATASET_ANALYSIS
        elif "CODE_GENERATION" in content:
            return QueryType.CODE_GENERATION
        elif "GENERAL_SECURITY" in content:
            return QueryType.GENERAL_SECURITY

        # Fallback to regex if LLM returns something unexpected
        logger.warning(
            f"LLM returned unexpected classification: {content}. Falling back to pattern matching."
        )
        return self._analyze_query_regex(query, context)

    def _analyze_query_regex(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> QueryType:
        """Fallback regex-based classification."""
        query_lower = query.lower()

        # CVE/vulnerability keywords
        cve_keywords = [
            "cve",
            "vulnerability",
            "vulnerabilities",
            "security",
            "exploit",
            "patch",
            "nvd",
            "advisory",
            "threat",
        ]
        library_keywords = [
            "numpy",
            "pytorch",
            "tensorflow",
            "sklearn",
            "scikit-learn",
            "pandas",
            "scipy",
            "keras",
            "xgboost",
            "lightgbm",
            "transformers",
            "huggingface",
        ]

        # Check for CVE monitoring queries
        has_cve_keyword = any(kw in query_lower for kw in cve_keywords)
        has_library = any(lib in query_lower for lib in library_keywords)

        if has_cve_keyword and has_library:
            return QueryType.CVE_MONITORING

        # Dataset analysis keywords
        dataset_keywords = [
            "dataset",
            "data",
            "poisoning",
            "bias",
            "outlier",
            "csv",
            "analyze",
            "statistical",
            "quality",
            "distribution",
        ]

        has_dataset_keyword = any(kw in query_lower for kw in dataset_keywords)
        has_file_context = context and ("file_path" in context or "dataset" in context)

        if has_dataset_keyword or has_file_context:
            return QueryType.DATASET_ANALYSIS

        # Code generation keywords
        code_keywords = [
            "generate",
            "code",
            "script",
            "validation",
            "check",
            "create",
            "write",
            "validation code",
            "security check",
        ]
        security_keywords = ["security", "validate", "cve", "verify", "test"]

        has_code_keyword = any(kw in query_lower for kw in code_keywords)
        has_security_keyword = any(kw in query_lower for kw in security_keywords)

        if has_code_keyword and has_security_keyword:
            return QueryType.CODE_GENERATION

        # Default to GENERAL_SECURITY
        return QueryType.GENERAL_SECURITY

    def _validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate user query for security and format issues.

        Args:
            query: User's input query

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check query length
        if len(query) < 5:
            return False, "Query too short. Please provide more details."

        if len(query) > 10000:
            return False, "Query too long. Please be more concise."

        # Check for potential injection attempts (basic sanitization)
        suspicious_patterns = [
            r"<script",
            r"javascript:",
            r"eval\(",
            r"exec\(",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected in query: {pattern}")
                return False, "Query contains potentially unsafe content."

        return True, None

    def _extract_reasoning_steps(
        self, messages: List[BaseMessage]
    ) -> List[ReasoningStep]:
        """Extract reasoning steps from agent's message history.

        Args:
            messages: List of messages from LangGraph execution

        Returns:
            List of ReasoningStep objects
        """
        reasoning_steps = []
        step_num = 0

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                # This is a tool call message
                for tool_call in msg.tool_calls:
                    step_num += 1
                    step = ReasoningStep(
                        step_number=step_num,
                        thought="",  # LangGraph doesn't expose explicit thoughts
                        action=tool_call.get("name", "unknown"),
                        action_input=tool_call.get("args", {}),
                        observation="",  # Will be filled from next message
                        timestamp=time.time(),
                    )
                    reasoning_steps.append(step)

            elif (
                hasattr(msg, "content")
                and reasoning_steps
                and not reasoning_steps[-1].observation
            ):
                # This might be a tool response
                reasoning_steps[-1].observation = str(msg.content)

        if self.verbose:
            for step in reasoning_steps:
                logger.info(
                    f"Step {step.step_number}: {step.action}({step.action_input})"
                )

        return reasoning_steps

    def _calculate_confidence(
        self,
        query_type: QueryType,
        reasoning_steps: List[ReasoningStep],
        tools_used: List[str],
    ) -> float:
        """Calculate confidence score for the agent's answer.

        Confidence is based on:
        - Number of tools successfully used
        - Whether appropriate tools were used for the query type
        - Number of reasoning steps (more steps can mean thorough analysis)

        Args:
            query_type: Classified query type
            reasoning_steps: Steps taken by the agent
            tools_used: List of tools that were called

        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence

        # Increase confidence if tools were used successfully
        if tools_used:
            confidence += 0.1 * min(len(tools_used), 3)  # Up to +0.3

        # Check if appropriate tools were used for query type
        expected_tools = {
            QueryType.CVE_MONITORING: ["cve_search", "web_search"],
            QueryType.DATASET_ANALYSIS: ["analyze_dataset"],
            QueryType.CODE_GENERATION: ["generate_security_code"],
            QueryType.GENERAL_SECURITY: ["web_search"],
        }

        if query_type in expected_tools:
            relevant_tools = set(expected_tools[query_type]) & set(tools_used)
            if relevant_tools:
                confidence += 0.2

        # Check reasoning depth (more steps = more thorough)
        if len(reasoning_steps) >= 2:
            confidence += 0.1

        # Check for errors in observations
        error_count = sum(
            1
            for step in reasoning_steps
            if "error" in step.observation.lower()
            or "failed" in step.observation.lower()
        )

        if error_count > 0:
            confidence -= 0.1 * error_count

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> AgentResult:
        """Execute the agent on a security query.

        This is the main entry point for running the agent. It orchestrates the entire
        reasoning process:
        1. Validates the query
        2. Classifies the query type
        3. Executes the agent with tools using LangGraph
        4. Extracts reasoning steps
        5. Calculates confidence
        6. Returns structured results

        Args:
            query: User's security question
            context: Optional additional context (uploaded files, dataset, etc.)
            chat_history: Optional conversation history for multi-turn interaction

        Returns:
            AgentResult containing answer, reasoning steps, and metadata

        Raises:
            AgentError: If agent execution fails
            QueryClassificationError: If query classification fails

        Example:
            >>> agent = MLPatrolAgent(llm)
            >>> result = agent.run("Check for recent numpy vulnerabilities")
            >>> print(result.answer)
            >>> print(f"Confidence: {result.confidence}")
            >>> for step in result.reasoning_steps:
            ...     print(f"  {step.action}: {step.observation[:100]}")
        """
        start_time = time.time()

        try:
            # Step 1: Validate query
            logger.info(f"Processing query: {query[:100]}...")
            is_valid, error_msg = self._validate_query(query)
            if not is_valid:
                logger.warning(f"Invalid query: {error_msg}")
                return AgentResult(
                    answer=f"Invalid query: {error_msg}",
                    error=error_msg,
                    confidence=0.0,
                )

            # Step 2: Classify query type
            try:
                query_type = self.analyze_query(query, context)
            except QueryClassificationError as e:
                logger.warning(f"Classification failed, using GENERAL_SECURITY: {e}")
                query_type = QueryType.GENERAL_SECURITY

            # Step 3: Prepare messages for LangGraph
            messages = chat_history or self.chat_history

            # Add context to query if provided
            full_query = query
            if context:
                context_str = "\n\nAdditional Context:\n"
                if "file_path" in context:
                    context_str += f"- File: {context['file_path']}\n"
                if "dataset" in context:
                    context_str += f"- Dataset provided for analysis\n"
                full_query += context_str

            # Add user message
            messages_with_query = messages + [HumanMessage(content=full_query)]

            # Step 4: Execute agent using LangGraph
            logger.info("Executing LangGraph agent...")
            try:
                # Run the agent with configuration
                config = RunnableConfig(
                    recursion_limit=self.max_iterations,
                    max_concurrency=1,
                )

                result = self.agent_executor.invoke(
                    {"messages": messages_with_query}, config=config
                )

            except Exception as e:
                logger.error(f"Agent execution failed: {e}", exc_info=True)
                return AgentResult(
                    answer=f"I encountered an error while processing your query: {str(e)}. Please try rephrasing your question or check if the query is valid.",
                    error=str(e),
                    query_type=query_type,
                    confidence=0.0,
                )

            # Step 5: Extract answer and reasoning steps from LangGraph output
            output_messages = result.get("messages", [])

            # Extract reasoning steps first (needed for fallback answer)
            reasoning_steps = self._extract_reasoning_steps(output_messages)

            # Get the final answer (last AI message with actual text content, not just tool calls)
            answer = None
            for msg in reversed(output_messages):
                if isinstance(msg, AIMessage):
                    # Skip messages that only contain tool calls without text content
                    has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
                    content_str = str(msg.content).strip() if msg.content else ""

                    # Check if content looks like tool call JSON (common patterns)
                    looks_like_tool_call = (
                        '"name"' in content_str and '"parameters"' in content_str
                    ) or content_str.startswith('{"name"')

                    # Prefer messages with actual text content (not tool call JSON)
                    if (
                        content_str
                        and not looks_like_tool_call
                        and len(content_str) > 20
                    ):
                        answer = content_str
                        break
                    # If we only have tool calls, skip this message
                    elif has_tool_calls and (not content_str or looks_like_tool_call):
                        continue

            # If we still don't have a good answer but have tool results, synthesize one
            if answer == "I couldn't generate a response." or len(answer) < 20:
                if reasoning_steps:
                    # Build a summary from tool results
                    tool_summaries = []
                    for step in reasoning_steps:
                        if step.observation and len(step.observation) > 20:
                            tool_summaries.append(f"Completed {step.action} analysis.")
                    if tool_summaries:
                        answer = (
                            "I've completed the analysis using the following tools: "
                            + ", ".join(tool_summaries[:3])
                        )
            # If we didn't find an answer, synthesize one from tool results
            if not answer:
                if reasoning_steps:
                    # Extract key findings from the most recent tool results
                    recent_results = []
                    for step in reversed(reasoning_steps[-3:]):  # Last 3 steps
                        if step.observation and len(step.observation) > 50:
                            # Try to extract the first sentence or meaningful content
                            first_part = step.observation[:300].strip()
                            if "\n" in first_part:
                                first_part = first_part.split("\n")[0]
                            recent_results.append(first_part)

                    if recent_results:
                        answer = "Based on my analysis:\n\n" + "\n\n".join(
                            recent_results
                        )
                        if len(reasoning_steps) > 3:
                            answer += f"\n\n(Analysis completed using {len(reasoning_steps)} steps. See reasoning chain for full details.)"
                    else:
                        # Fall back to listing tools used
                        tools_list = list(set(step.action for step in reasoning_steps))
                        answer = f"I completed the analysis using {len(tools_list)} tool(s): {', '.join(tools_list)}. Please review the reasoning steps below for detailed findings."
                else:
                    # No reasoning steps means something went wrong
                    answer = "I couldn't generate a response. No analysis steps were completed. Please try rephrasing your query or check the logs for errors."

            # Step 6: Determine which tools were used
            tools_used = list(set(step.action for step in reasoning_steps))

            # Step 7: Calculate confidence
            confidence = self._calculate_confidence(
                query_type, reasoning_steps, tools_used
            )

            # Step 8: Calculate total duration
            total_duration_ms = (time.time() - start_time) * 1000

            # Step 9: Update chat history
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=answer))

            # Step 10: Create and return result
            agent_result = AgentResult(
                answer=answer,
                reasoning_steps=reasoning_steps,
                tools_used=tools_used,
                query_type=query_type,
                confidence=confidence,
                total_duration_ms=total_duration_ms,
            )

            logger.info(
                f"Agent completed in {total_duration_ms:.0f}ms with confidence {confidence:.2f}"
            )

            return agent_result

        except Exception as e:
            logger.error(f"Unexpected error in agent.run(): {e}", exc_info=True)
            return AgentResult(
                answer=f"An unexpected error occurred: {str(e)}",
                error=str(e),
                confidence=0.0,
                total_duration_ms=(time.time() - start_time) * 1000,
            )

    def clear_history(self) -> None:
        """Clear the agent's chat history.

        Use this to start a fresh conversation or reset context.
        """
        self.chat_history = []
        self.reasoning_history = []
        logger.info("Agent history cleared")

    def get_tool_info(self) -> List[Dict[str, str]]:
        """Get information about available tools.

        Returns:
            List of dictionaries with tool name and description

        Example:
            >>> agent = MLPatrolAgent(llm)
            >>> tools = agent.get_tool_info()
            >>> for tool in tools:
            ...     print(f"{tool['name']}: {tool['description']}")
        """
        return [
            {"name": tool.name, "description": tool.description} for tool in self.tools
        ]

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"MLPatrolAgent(tools={len(self.tools)}, "
            f"max_iterations={self.max_iterations}, "
            f"verbose={self.verbose})"
        )


# ============================================================================
# Helper Functions
# ============================================================================


def create_mlpatrol_agent(
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-0",
    base_url: Optional[str] = None,
    verbose: bool = True,
    **kwargs,
) -> MLPatrolAgent:
    """Factory function to create a configured MLPatrol agent.

    This is a convenience function that handles LLM initialization and
    agent creation with sensible defaults.

    Args:
        api_key: API key for cloud LLM providers (Anthropic/OpenAI). Not needed for Ollama.
        model: Model to use. Options:
            Cloud:
            - "claude-sonnet-4-0" (default, recommended - alias for latest)
            - "claude-sonnet-4-20250514" (specific version)
            - "claude-opus-4-0"
            - "gpt-4"
            - "gpt-4-turbo"

            Local (Ollama):
            - "ollama/llama3.1:8b" (fast, recommended for local)
            - "ollama/llama3.1:70b" (high accuracy)
            - "ollama/mistral-small:3.1" (balanced)
            - "ollama/qwen2.5:14b" (code analysis)
            - "ollama/deepseek-r1:7b" (reasoning-optimized)

        base_url: Base URL for local LLM servers (Ollama: http://localhost:11434)
        verbose: Whether to enable verbose logging
        **kwargs: Additional arguments passed to MLPatrolAgent

    Returns:
        Configured MLPatrolAgent instance

    Raises:
        ValueError: If model is not supported or API key is missing (for cloud models)

    Example:
        Cloud:
        >>> agent = create_mlpatrol_agent(
        ...     api_key="your-api-key",
        ...     model="claude-sonnet-4-0",
        ...     verbose=True
        ... )

        Local (Ollama):
        >>> agent = create_mlpatrol_agent(
        ...     model="ollama/llama3.1:8b",
        ...     verbose=True
        ... )
        >>> result = agent.run("Check numpy for CVEs")
    """
    try:
        model_lower = model.lower()

        # Cloud LLMs
        if "claude" in model_lower:
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(
                model=model,
                anthropic_api_key=api_key,
                temperature=0.1,  # Low temperature for factual security analysis
                max_tokens=4096,
            )
            logger.info(f"Initialized Claude LLM: {model}")

        elif "gpt" in model_lower:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                temperature=0.1,
                max_tokens=4096,
            )
            logger.info(f"Initialized OpenAI LLM: {model}")

        # Local LLMs (Ollama)
        elif "ollama" in model_lower:
            from langchain_ollama import ChatOllama

            # Extract model name (remove "ollama/" prefix if present)
            ollama_model = model.replace("ollama/", "").replace("Ollama/", "")

            # Use provided base_url or default to localhost
            ollama_url = base_url or "http://localhost:11434"

            llm = ChatOllama(
                model=ollama_model,
                base_url=ollama_url,
                temperature=0.1,
                num_ctx=4096,  # Context window (equivalent to max_tokens)
            )
            logger.info(f"Initialized Ollama LLM: {ollama_model} at {ollama_url}")

        else:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Cloud: claude-sonnet-4-0, claude-opus-4-0, gpt-4, gpt-4-turbo. "
                f"Local: ollama/llama3.1:8b, ollama/mistral-small:3.1, etc."
            )

        # Create and return agent
        agent = MLPatrolAgent(llm=llm, verbose=verbose, **kwargs)
        return agent

    except ImportError as e:
        logger.error(f"Failed to import LLM library: {e}")
        raise ValueError(
            f"Required library not installed. "
            f"For Claude: pip install langchain-anthropic. "
            f"For OpenAI: pip install langchain-openai. "
            f"For Ollama: pip install langchain-ollama"
        )
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        raise


def demo_agent() -> None:
    """Demonstration of the MLPatrol agent.

    This function provides a simple interactive demo. Run it to test the agent.

    Example:
        >>> from src.agent.reasoning_chain import demo_agent
        >>> demo_agent()
    """
    import os

    print("=" * 70)
    print("MLPatrol Security Agent - Demo")
    print("=" * 70)
    print()

    # Check for local LLM first
    use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

    if use_local:
        model = os.getenv("LOCAL_LLM_MODEL", "ollama/llama3.1:8b")
        base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
        api_key = None
        print(f"Using local LLM: {model}")
    else:
        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = None

        if not api_key:
            print(
                "ERROR: Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable"
            )
            print("OR set USE_LOCAL_LLM=true to use Ollama")
            return

        # Determine which model to use
        model = "claude-sonnet-4-0" if os.getenv("ANTHROPIC_API_KEY") else "gpt-4"

    print(f"Initializing agent with model: {model}")
    print()

    try:
        agent = create_mlpatrol_agent(
            api_key=api_key, model=model, base_url=base_url, verbose=True
        )

        # Example queries
        example_queries = [
            "Check for recent vulnerabilities in numpy",
            "What security measures should I take for my PyTorch deployment?",
            "How can I detect data poisoning in my training dataset?",
        ]

        print("Example queries you can try:")
        for i, q in enumerate(example_queries, 1):
            print(f"{i}. {q}")
        print()

        # Interactive loop
        while True:
            query = input("Enter your security query (or 'quit' to exit): ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            print("\nProcessing query...")
            print("-" * 70)

            result = agent.run(query)

            print("\nANSWER:")
            print(result.answer)
            print()
            print(
                f"Query Type: {result.query_type.value if result.query_type else 'unknown'}"
            )
            print(f"Confidence: {result.confidence:.2%}")
            print(f"Tools Used: {', '.join(result.tools_used)}")
            print(f"Duration: {result.total_duration_ms:.0f}ms")

            if result.reasoning_steps:
                print(f"\nReasoning Steps: {len(result.reasoning_steps)}")
                for step in result.reasoning_steps:
                    print(f"  Step {step.step_number}: {step.action}")

            print("-" * 70)
            print()

    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run demo if executed directly
    demo_agent()
