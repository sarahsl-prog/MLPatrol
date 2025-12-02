#!/usr/bin/env python3
"""
Unit tests for ScriptReviewer.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

from src.agent.script_reviewer import ScriptReviewer
from src.models.script_state import RiskLevel


class TestScriptReviewer(unittest.TestCase):
    """Test cases for ScriptReviewer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_static_analysis_detects_eval(self):
        """Test static analysis detects eval()."""
        reviewer = ScriptReviewer(
            model="test-model",
            provider="test",
            cache_dir=self.cache_dir
        )

        script_content = """
import sys
result = eval(input("Enter code: "))
print(result)
"""
        issues = reviewer._static_analysis(script_content)

        # Should detect eval and input
        self.assertTrue(any("eval" in issue.lower() for issue in issues))
        self.assertTrue(any("CRITICAL" in issue for issue in issues))

    def test_static_analysis_detects_os_system(self):
        """Test static analysis detects os.system()."""
        reviewer = ScriptReviewer(
            model="test-model",
            provider="test",
            cache_dir=self.cache_dir
        )

        script_content = """
import os
os.system("ls -la")
"""
        issues = reviewer._static_analysis(script_content)

        self.assertTrue(any("os.system" in issue for issue in issues))
        self.assertTrue(any("CRITICAL" in issue for issue in issues))

    def test_static_analysis_safe_script(self):
        """Test static analysis on a safe script."""
        reviewer = ScriptReviewer(
            model="test-model",
            provider="test",
            cache_dir=self.cache_dir
        )

        script_content = """
#!/usr/bin/env python3
import sys
from importlib.metadata import version

def check_vulnerability():
    try:
        current = version("numpy")
        if current in ["1.21.0", "1.21.1"]:
            print(f"VULNERABLE: {current}")
            return 1
        print(f"SAFE: {current}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(check_vulnerability())
"""
        issues = reviewer._static_analysis(script_content)

        # Should have no CRITICAL issues
        critical_issues = [i for i in issues if "CRITICAL" in i]
        self.assertEqual(len(critical_issues), 0)

    def test_hash_script(self):
        """Test script hashing."""
        reviewer = ScriptReviewer(
            model="test-model",
            provider="test",
            cache_dir=self.cache_dir
        )

        script1 = "print('hello')"
        script2 = "print('world')"

        hash1 = reviewer._hash_script(script1)
        hash2 = reviewer._hash_script(script2)

        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 16)  # First 16 chars of SHA256
        self.assertNotEqual(hash1, hash2)

        # Same content should produce same hash
        hash1_again = reviewer._hash_script(script1)
        self.assertEqual(hash1, hash1_again)

    @patch.dict(os.environ, {"USE_SEPARATE_REVIEWER_LLM": "false", "USE_LOCAL_LLM": "false", "ANTHROPIC_API_KEY": "test-key"})
    def test_reviewer_initialization_anthropic(self):
        """Test reviewer initialization with Anthropic."""
        with patch('src.agent.script_reviewer.ChatAnthropic') as mock_anthropic:
            mock_llm = MagicMock()
            mock_anthropic.return_value = mock_llm

            reviewer = ScriptReviewer(cache_dir=self.cache_dir)

            self.assertEqual(reviewer.provider, "anthropic")
            self.assertIsNotNone(reviewer.llm)
            mock_anthropic.assert_called_once()

    def test_parse_review_response_valid_json(self):
        """Test parsing valid JSON response."""
        reviewer = ScriptReviewer(
            model="test-model",
            provider="test",
            cache_dir=self.cache_dir
        )

        response = """
{
    "approved": true,
    "safe_to_run": true,
    "confidence_score": 0.95,
    "risk_level": "LOW",
    "issues_found": [],
    "recommendations": ["Add type hints"],
    "review_summary": "Script is safe",
    "detailed_analysis": "Comprehensive analysis here"
}
"""
        result = reviewer._parse_review_response(response)

        self.assertTrue(result["approved"])
        self.assertTrue(result["safe_to_run"])
        self.assertEqual(result["confidence_score"], 0.95)
        self.assertEqual(result["risk_level"], "LOW")

    def test_parse_review_response_embedded_json(self):
        """Test parsing JSON embedded in text."""
        reviewer = ScriptReviewer(
            model="test-model",
            provider="test",
            cache_dir=self.cache_dir
        )

        response = """
Here's my review:

{
    "approved": false,
    "safe_to_run": false,
    "confidence_score": 0.3,
    "risk_level": "CRITICAL",
    "issues_found": ["eval detected"],
    "recommendations": ["Remove eval"],
    "review_summary": "Dangerous",
    "detailed_analysis": "Details"
}

Hope this helps!
"""
        result = reviewer._parse_review_response(response)

        self.assertFalse(result["approved"])
        self.assertEqual(result["risk_level"], "CRITICAL")

    def test_parse_review_response_invalid(self):
        """Test parsing invalid response returns safe defaults."""
        reviewer = ScriptReviewer(
            model="test-model",
            provider="test",
            cache_dir=self.cache_dir
        )

        response = "This is not JSON at all!"

        result = reviewer._parse_review_response(response)

        # Should return conservative defaults
        self.assertFalse(result["approved"])
        self.assertFalse(result["safe_to_run"])
        self.assertLessEqual(result["confidence_score"], 0.5)

    def test_review_cache(self):
        """Test review caching functionality."""
        reviewer = ScriptReviewer(
            model="test-model",
            provider="test",
            cache_dir=self.cache_dir,
            use_cache=True
        )

        script_content = "print('test')"
        cve_id = "CVE-2024-12345"

        # Create a mock review result
        from src.models.script_state import ReviewResult

        review = ReviewResult(
            approved=True,
            safe_to_run=True,
            confidence_score=0.9,
            risk_level=RiskLevel.LOW,
            issues_found=[],
            recommendations=[],
            review_summary="Cached review",
            detailed_analysis="Details"
        )

        # Cache the review
        reviewer._cache_review(script_content, cve_id, review)

        # Retrieve from cache
        cached = reviewer._get_cached_review(script_content, cve_id)

        self.assertIsNotNone(cached)
        self.assertEqual(cached.confidence_score, 0.9)
        self.assertEqual(cached.review_summary, "Cached review")

    def test_clear_cache(self):
        """Test clearing review cache."""
        reviewer = ScriptReviewer(
            model="test-model",
            provider="test",
            cache_dir=self.cache_dir,
            use_cache=True
        )

        # Create some cached reviews
        from src.models.script_state import ReviewResult

        for i in range(3):
            review = ReviewResult(
                approved=True,
                safe_to_run=True,
                confidence_score=0.9,
                risk_level=RiskLevel.LOW,
                issues_found=[],
                recommendations=[],
                review_summary=f"Review {i}",
                detailed_analysis="Details"
            )
            reviewer._cache_review(f"script_{i}", f"CVE-2024-{i}", review)

        # Clear cache
        count = reviewer.clear_cache()

        self.assertEqual(count, 3)

        # Verify cache is empty
        cached = reviewer._get_cached_review("script_0", "CVE-2024-0")
        self.assertIsNone(cached)


if __name__ == "__main__":
    unittest.main()
