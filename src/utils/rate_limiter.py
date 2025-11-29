"""Rate limiting utilities for Gradio endpoints."""

import threading
from functools import wraps
from time import time
from typing import Any, Callable, Dict


class RateLimiter:
    """Simple in-memory rate limiter.

    Tracks call counts per key within a sliding time window.
    Thread-safe for concurrent access.
    """

    def __init__(self, max_calls: int, period: int):
        """Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed within the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: Dict[str, list] = {}
        self.lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        """Check if a call is allowed for the given key.

        Args:
            key: Unique identifier for the rate limit (e.g., function name, user ID)

        Returns:
            True if the call is allowed, False if rate limit exceeded
        """
        with self.lock:
            now = time()
            if key not in self.calls:
                self.calls[key] = []

            # Remove old calls outside the time window
            self.calls[key] = [t for t in self.calls[key] if now - t < self.period]

            if len(self.calls[key]) < self.max_calls:
                self.calls[key].append(now)
                return True
            return False

    def __call__(self, func: Callable) -> Callable:
        """Decorator to rate limit a function.

        Args:
            func: Function to rate limit

        Returns:
            Wrapped function that enforces rate limits
        """

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Use function name as key (could be improved with user ID/IP)
            key = func.__name__

            if not self.is_allowed(key):
                # Return error message instead of raising exception
                # (more user-friendly for Gradio UI)
                error_msg = (
                    f"⚠️ Rate limit exceeded. "
                    f"Maximum {self.max_calls} requests per {self.period} seconds. "
                    f"Please wait and try again."
                )
                return error_msg

            return func(*args, **kwargs)

        return wrapper


# Create rate limiters for different endpoints
# Adjust these limits based on your server capacity and requirements

cve_search_limiter = RateLimiter(max_calls=10, period=60)  # 10 requests per minute

dataset_analysis_limiter = RateLimiter(
    max_calls=5, period=60
)  # 5 requests per minute (resource-intensive)

code_gen_limiter = RateLimiter(max_calls=20, period=60)  # 20 requests per minute

chat_limiter = RateLimiter(max_calls=30, period=60)  # 30 requests per minute
