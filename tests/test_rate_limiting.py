"""Tests for rate limiting functionality."""

import time

import pytest

from src.utils.rate_limiter import RateLimiter


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_allows_calls_within_limit(self):
        """Test that calls within the limit are allowed."""
        limiter = RateLimiter(max_calls=5, period=10)

        for i in range(5):
            assert (
                limiter.is_allowed("test_key") is True
            ), f"Call {i+1} should be allowed"

    def test_blocks_calls_over_limit(self):
        """Test that calls over the limit are blocked."""
        limiter = RateLimiter(max_calls=3, period=10)

        # First 3 calls should pass
        for _ in range(3):
            assert limiter.is_allowed("test_key") is True

        # 4th call should be blocked
        assert limiter.is_allowed("test_key") is False

    def test_resets_after_period(self):
        """Test that limit resets after the time period."""
        limiter = RateLimiter(max_calls=2, period=1)

        # Use up the limit
        assert limiter.is_allowed("test_key") is True
        assert limiter.is_allowed("test_key") is True
        assert limiter.is_allowed("test_key") is False

        # Wait for period to pass
        time.sleep(1.1)

        # Should be allowed again
        assert limiter.is_allowed("test_key") is True

    def test_different_keys_tracked_separately(self):
        """Test that different keys have separate rate limits."""
        limiter = RateLimiter(max_calls=2, period=10)

        # Use up limit for key1
        assert limiter.is_allowed("key1") is True
        assert limiter.is_allowed("key1") is True
        assert limiter.is_allowed("key1") is False

        # key2 should still be allowed
        assert limiter.is_allowed("key2") is True
        assert limiter.is_allowed("key2") is True

    def test_decorator_allows_calls(self):
        """Test decorator allows calls within limit."""
        limiter = RateLimiter(max_calls=2, period=10)

        @limiter
        def test_func():
            return "success"

        # First 2 calls should work
        assert test_func() == "success"
        assert test_func() == "success"

    def test_decorator_blocks_excess_calls(self):
        """Test decorator blocks calls over limit."""
        limiter = RateLimiter(max_calls=2, period=10)

        @limiter
        def test_func():
            return "success"

        # Use up the limit
        test_func()
        test_func()

        # 3rd call should return error message
        result = test_func()
        assert isinstance(result, str)
        assert "rate limit" in result.lower()
        assert "exceeded" in result.lower()

    def test_decorator_error_message_format(self):
        """Test that error message contains useful information."""
        limiter = RateLimiter(max_calls=5, period=60)

        @limiter
        def test_func():
            return "success"

        # Use up the limit
        for _ in range(5):
            test_func()

        # Get error message
        error_msg = test_func()

        # Should mention the limits
        assert "5" in error_msg  # max_calls
        assert "60" in error_msg  # period
        assert "wait" in error_msg.lower()

    def test_decorator_with_arguments(self):
        """Test decorator works with functions that take arguments."""
        limiter = RateLimiter(max_calls=3, period=10)

        @limiter
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5
        assert add(10, 20) == 30
        assert add(1, 1) == 2

        # 4th call should be blocked
        result = add(5, 5)
        assert isinstance(result, str)
        assert "rate limit" in result.lower()

    def test_sliding_window(self):
        """Test that rate limiting uses sliding window."""
        limiter = RateLimiter(max_calls=2, period=2)

        # First call at t=0
        assert limiter.is_allowed("test") is True
        time.sleep(0.5)

        # Second call at t=0.5
        assert limiter.is_allowed("test") is True

        # Third call at t=0.5 should be blocked (2 calls in window)
        assert limiter.is_allowed("test") is False

        # Wait until first call expires (t>2.0)
        time.sleep(1.6)

        # Should be allowed again (only 1 call in last 2 seconds)
        assert limiter.is_allowed("test") is True

    def test_zero_max_calls_blocks_all(self):
        """Test that max_calls=0 blocks all calls."""
        limiter = RateLimiter(max_calls=0, period=10)

        assert limiter.is_allowed("test") is False

    def test_concurrent_calls_thread_safe(self):
        """Test that rate limiter is thread-safe."""
        import threading

        limiter = RateLimiter(max_calls=10, period=5)
        results = []

        def make_call():
            result = limiter.is_allowed("shared_key")
            results.append(result)

        # Create 20 threads trying to make calls simultaneously
        threads = [threading.Thread(target=make_call) for _ in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 10 should succeed, 10 should fail
        assert sum(results) == 10
        assert len([r for r in results if not r]) == 10
