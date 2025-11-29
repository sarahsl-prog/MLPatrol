"""Unit tests for security code generation tools.

This test suite covers:
- Security code generation
- Template rendering
- Parameter handling
"""

import json

import pytest

from src.agent.tools import generate_security_code_impl


class TestSecurityCodeGeneration:
    """Tests for generate_security_code_impl."""

    def test_generate_basic_validation(self):
        """Test generating basic validation code."""
        result_json = generate_security_code_impl(
            purpose="Validate numpy version", library="numpy"
        )
        result = json.loads(result_json)

        assert result["status"] == "success"
        assert "code" in result
        assert "import numpy" in result["code"]
        assert "check_numpy_security" in result["code"]

    def test_generate_cve_specific_code(self):
        """Test generating code for specific CVE."""
        result_json = generate_security_code_impl(
            purpose="Check for CVE-2023-1234",
            library="tensorflow",
            cve_id="CVE-2023-1234",
        )
        result = json.loads(result_json)

        assert result["status"] == "success"
        assert "CVE-2023-1234" in result["code"]
        assert "check_tensorflow_security" in result["code"]

    def test_generate_with_affected_versions(self):
        """Test generating code with affected versions list."""
        affected_versions = ["1.2.0", "1.2.1", "1.2.2"]
        result_json = generate_security_code_impl(
            purpose="Check vulnerable versions",
            library="pandas",
            affected_versions=affected_versions,
        )
        result = json.loads(result_json)

        assert result["status"] == "success"
        code = result["code"]

        # Verify versions are present in the code
        for version in affected_versions:
            assert version in code

        # Verify the list structure is correct in generated code
        assert "['1.2.0', '1.2.1', '1.2.2']" in code

    def test_generate_error_handling(self):
        """Test error handling in code generation."""
        # Currently the implementation is quite robust and doesn't raise many errors,
        # but we can test that it returns valid JSON even with empty inputs
        result_json = generate_security_code_impl(purpose="", library="unknown_lib")
        result = json.loads(result_json)

        assert result["status"] == "success"
        assert "check_unknown_lib_security" in result["code"]
