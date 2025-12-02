#!/usr/bin/env python3
"""
Script Reviewer - LLM-based security script analysis.

Analyzes generated security scripts for safety, correctness, and effectiveness
using a dedicated LLM configured separately from the main agent.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from src.agent.prompts import (
    SCRIPT_REVIEW_EXAMPLES,
    SCRIPT_REVIEW_SYSTEM_PROMPT,
    SCRIPT_REVIEW_USER_PROMPT,
)
from src.models.script_state import ReviewResult, RiskLevel


class ScriptReviewer:
    """
    LLM-based security script reviewer.

    Uses a dedicated LLM (configurable via .env) to analyze generated security
    scripts for safety, correctness, and CVE relevance.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: str = "data/scripts/reviews_cache",
        use_cache: bool = True
    ):
        """
        Initialize the script reviewer.

        Args:
            api_key: API key for cloud LLM (if None, reads from env)
            model: Model name (if None, reads from env)
            provider: Provider (anthropic, openai, local - if None, reads from env)
            base_url: Base URL for local LLM
            cache_dir: Directory for review cache
            use_cache: Whether to use cached reviews
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

        # Load configuration from environment
        self._load_config(api_key, model, provider, base_url)

        # Initialize LLM
        self.llm = self._initialize_llm()

        logger.info(f"ScriptReviewer initialized with model: {self.model}")

    def _load_config(
        self,
        api_key: Optional[str],
        model: Optional[str],
        provider: Optional[str],
        base_url: Optional[str]
    ) -> None:
        """Load configuration from environment variables."""
        # Check if using separate reviewer LLM
        use_separate = os.getenv("USE_SEPARATE_REVIEWER_LLM", "false").lower() == "true"

        if use_separate:
            # Use dedicated reviewer LLM configuration
            self.provider = provider or os.getenv("REVIEWER_LLM_PROVIDER", "anthropic")
            self.model = model or os.getenv(
                "REVIEWER_LLM_MODEL",
                "claude-sonnet-4-0"
            )

            if self.provider == "anthropic":
                self.api_key = api_key or os.getenv("REVIEWER_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "openai":
                self.api_key = api_key or os.getenv("REVIEWER_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            elif self.provider == "local":
                self.api_key = None
                self.model = os.getenv("REVIEWER_LOCAL_LLM_MODEL", "ollama/llama3.1:70b")
                self.base_url = base_url or os.getenv("REVIEWER_LOCAL_LLM_URL", "http://localhost:11434")
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            logger.info(f"Using separate reviewer LLM: {self.provider}/{self.model}")
        else:
            # Use same LLM as main agent
            use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

            if use_local:
                self.provider = "local"
                self.model = model or os.getenv("LOCAL_LLM_MODEL", "ollama/llama3.1:8b")
                self.base_url = base_url or os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
                self.api_key = None
            else:
                # Cloud LLM - try Anthropic first, then OpenAI
                anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                openai_key = os.getenv("OPENAI_API_KEY")

                if anthropic_key:
                    self.provider = "anthropic"
                    self.model = model or "claude-sonnet-4-0"
                    self.api_key = api_key or anthropic_key
                elif openai_key:
                    self.provider = "openai"
                    self.model = model or "gpt-4"
                    self.api_key = api_key or openai_key
                else:
                    raise ValueError(
                        "No LLM configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                        "or USE_LOCAL_LLM=true in .env"
                    )

            logger.info(f"Using main agent LLM for reviews: {self.provider}/{self.model}")

    def _initialize_llm(self):
        """Initialize the LLM based on provider."""
        try:
            if self.provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                return ChatAnthropic(
                    model=self.model,
                    anthropic_api_key=self.api_key,
                    temperature=0.0,  # Deterministic for reviews
                    max_tokens=4096
                )

            elif self.provider == "openai":
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model=self.model,
                    openai_api_key=self.api_key,
                    temperature=0.0,
                    max_tokens=4096
                )

            elif self.provider == "local":
                from langchain_ollama import ChatOllama

                ollama_model = self.model.replace("ollama/", "").replace("Ollama/", "")

                return ChatOllama(
                    model=ollama_model,
                    base_url=self.base_url,
                    temperature=0.0,
                    num_ctx=4096
                )

            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except ImportError as e:
            logger.error(f"Failed to import LLM library: {e}")
            raise ValueError(
                f"Required library not installed. "
                f"For Anthropic: pip install langchain-anthropic. "
                f"For OpenAI: pip install langchain-openai. "
                f"For local: pip install langchain-ollama"
            )

    def review_script(
        self,
        script_path: str,
        cve_id: str,
        library: str,
        severity: str,
        description: str = "",
        use_cache: Optional[bool] = None
    ) -> ReviewResult:
        """
        Review a security script for safety and correctness.

        Args:
            script_path: Path to the script file
            cve_id: CVE identifier
            library: Library name
            severity: Severity level
            description: CVE description
            use_cache: Override instance cache setting

        Returns:
            ReviewResult with analysis

        Raises:
            FileNotFoundError: If script doesn't exist
            ValueError: If script is invalid
        """
        # Read script content
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        with open(script_path, 'r') as f:
            script_content = f.read()

        # Check cache
        should_use_cache = use_cache if use_cache is not None else self.use_cache
        if should_use_cache:
            cached_review = self._get_cached_review(script_content, cve_id)
            if cached_review:
                logger.info(f"Using cached review for {cve_id}")
                return cached_review

        # Perform static analysis first (fast, deterministic)
        static_issues = self._static_analysis(script_content)

        # If critical issues found, return immediately without LLM call
        if any("CRITICAL" in issue for issue in static_issues):
            logger.warning(f"Critical issues found in static analysis for {cve_id}")
            return ReviewResult(
                approved=False,
                safe_to_run=False,
                confidence_score=0.0,
                risk_level=RiskLevel.CRITICAL,
                issues_found=static_issues,
                recommendations=[
                    "Fix critical security issues before proceeding",
                    "Remove dangerous operations (eval, exec, os.system)",
                    "Ensure no user input is used in system commands"
                ],
                review_summary="CRITICAL: Script contains dangerous operations that pose security risks. "
                              "Static analysis detected unsafe patterns. Script must be rewritten.",
                detailed_analysis=f"Static analysis detected {len(static_issues)} critical issues. "
                                 f"The script uses dangerous operations that could compromise system security. "
                                 f"Issues: {'; '.join(static_issues)}",
                reviewer_model=f"{self.provider}/{self.model} (static-only)"
            )

        # Perform LLM-based review
        logger.info(f"Performing LLM review for {cve_id} with {self.model}")

        try:
            review_result = self._llm_review(
                script_content,
                cve_id,
                library,
                severity,
                description,
                static_issues
            )

            # Cache the result
            if should_use_cache:
                self._cache_review(script_content, cve_id, review_result)

            return review_result

        except Exception as e:
            logger.error(f"LLM review failed: {e}", exc_info=True)
            # Fallback to conservative review
            return ReviewResult(
                approved=False,
                safe_to_run=False,
                confidence_score=0.3,
                risk_level=RiskLevel.HIGH,
                issues_found=static_issues + [f"LLM review failed: {str(e)}"],
                recommendations=[
                    "Manual review required due to automated review failure",
                    "Verify script safety before execution"
                ],
                review_summary=f"Automated review failed. Static analysis found {len(static_issues)} issues. "
                              f"Manual review required.",
                detailed_analysis=f"The LLM-based review encountered an error: {str(e)}. "
                                 f"Static analysis results: {'; '.join(static_issues) if static_issues else 'No static issues detected'}. "
                                 f"Recommend manual code review before execution.",
                reviewer_model=f"{self.provider}/{self.model} (failed)"
            )

    def _static_analysis(self, script_content: str) -> list:
        """
        Perform fast static analysis for dangerous patterns.

        Args:
            script_content: Script code

        Returns:
            List of issues found
        """
        issues = []

        # Dangerous function patterns
        dangerous_patterns = {
            r'\beval\s*\(': "CRITICAL: eval() detected - arbitrary code execution risk",
            r'\bexec\s*\(': "CRITICAL: exec() detected - arbitrary code execution risk",
            r'\bcompile\s*\(': "CRITICAL: compile() detected - code generation risk",
            r'\bos\.system\s*\(': "CRITICAL: os.system() detected - command injection risk",
            r'\bsubprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True': "CRITICAL: subprocess with shell=True - command injection risk",
            r'\b__import__\s*\(': "HIGH: Dynamic imports detected - review carefully",
            r'\binput\s*\(': "MEDIUM: User input detected - ensure proper validation",
            r'\bopen\s*\([^)]*["\']w["\']': "MEDIUM: File write operation detected",
            r'\brequests\.(get|post|put|delete)': "MEDIUM: Network call detected",
            r'\burllib\.request': "MEDIUM: Network call detected",
            r'\bsocket\.': "MEDIUM: Socket operation detected",
        }

        for pattern, message in dangerous_patterns.items():
            if re.search(pattern, script_content, re.IGNORECASE):
                issues.append(message)

        # Check for proper error handling
        if 'try:' in script_content:
            if 'except:' in script_content and 'except Exception' not in script_content:
                issues.append("LOW: Bare except clause detected - should catch specific exceptions")
        else:
            # No try-except at all
            if any(keyword in script_content for keyword in ['version(', 'import ', 'open(']):
                issues.append("MEDIUM: No error handling detected - add try-except blocks")

        return issues

    def _llm_review(
        self,
        script_content: str,
        cve_id: str,
        library: str,
        severity: str,
        description: str,
        static_issues: list
    ) -> ReviewResult:
        """
        Perform LLM-based review.

        Args:
            script_content: Script code
            cve_id: CVE ID
            library: Library name
            severity: Severity level
            description: CVE description
            static_issues: Issues from static analysis

        Returns:
            ReviewResult
        """
        # Format the prompt
        user_prompt = SCRIPT_REVIEW_USER_PROMPT.format(
            cve_id=cve_id,
            library=library,
            severity=severity,
            description=description or "No description provided",
            script_content=script_content
        )

        # Add static analysis results to prompt
        if static_issues:
            user_prompt += f"\n\n## Static Analysis Results:\n"
            for issue in static_issues:
                user_prompt += f"- {issue}\n"
            user_prompt += "\nPlease consider these static analysis findings in your review."

        # Call LLM
        messages = [
            {"role": "system", "content": SCRIPT_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm.invoke(messages)
        response_text = response.content

        # Parse JSON response
        review_data = self._parse_review_response(response_text)

        # Create ReviewResult
        return ReviewResult(
            approved=review_data.get("approved", False),
            safe_to_run=review_data.get("safe_to_run", False),
            confidence_score=review_data.get("confidence_score", 0.5),
            risk_level=RiskLevel(review_data.get("risk_level", "MEDIUM").upper()),
            issues_found=review_data.get("issues_found", []),
            recommendations=review_data.get("recommendations", []),
            review_summary=review_data.get("review_summary", "Review completed"),
            detailed_analysis=review_data.get("detailed_analysis", "No detailed analysis provided"),
            reviewer_model=f"{self.provider}/{self.model}"
        )

    def _parse_review_response(self, response_text: str) -> Dict:
        """
        Parse LLM response and extract JSON.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed review data dictionary
        """
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from LLM response")

        # Fallback: try to parse the entire response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.error("Could not parse LLM response as JSON")
            # Return conservative default
            return {
                "approved": False,
                "safe_to_run": False,
                "confidence_score": 0.3,
                "risk_level": "MEDIUM",
                "issues_found": ["Failed to parse LLM response"],
                "recommendations": ["Manual review required"],
                "review_summary": "Automated review parsing failed",
                "detailed_analysis": f"Raw response: {response_text[:500]}"
            }

    def _get_cached_review(self, script_content: str, cve_id: str) -> Optional[ReviewResult]:
        """
        Get cached review if available.

        Args:
            script_content: Script code
            cve_id: CVE ID

        Returns:
            Cached ReviewResult or None
        """
        script_hash = self._hash_script(script_content)
        cache_file = self.cache_dir / f"{cve_id}_{script_hash}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return ReviewResult.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load cached review: {e}")
                return None

        return None

    def _cache_review(self, script_content: str, cve_id: str, review: ReviewResult) -> None:
        """
        Cache review result.

        Args:
            script_content: Script code
            cve_id: CVE ID
            review: ReviewResult to cache
        """
        script_hash = self._hash_script(script_content)
        cache_file = self.cache_dir / f"{cve_id}_{script_hash}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump(review.to_dict(), f, indent=2)
            logger.debug(f"Cached review for {cve_id}")
        except Exception as e:
            logger.warning(f"Failed to cache review: {e}")

    def _hash_script(self, script_content: str) -> str:
        """
        Calculate hash of script content.

        Args:
            script_content: Script code

        Returns:
            SHA256 hash (first 16 characters)
        """
        return hashlib.sha256(script_content.encode()).hexdigest()[:16]

    def clear_cache(self) -> int:
        """
        Clear all cached reviews.

        Returns:
            Number of cache files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} cached reviews")
        return count


def create_script_reviewer(**kwargs) -> ScriptReviewer:
    """
    Factory function to create a configured ScriptReviewer.

    Args:
        **kwargs: Arguments passed to ScriptReviewer constructor

    Returns:
        Configured ScriptReviewer instance

    Example:
        >>> reviewer = create_script_reviewer()
        >>> result = reviewer.review_script(
        ...     script_path="generated_checks/cve_check.py",
        ...     cve_id="CVE-2024-12345",
        ...     library="numpy",
        ...     severity="HIGH"
        ... )
    """
    return ScriptReviewer(**kwargs)
