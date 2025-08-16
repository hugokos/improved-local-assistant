"""
Circuit breaker implementation for external dependencies.

This module provides a circuit breaker pattern implementation to handle
failures in external dependencies gracefully and prevent cascading failures.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable
from collections.abc import Callable
from enum import Enum
from typing import Any
from typing import TypeVar

# Type variables for generic functions
T = TypeVar("T")

# Configure logging
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing state, requests are blocked
    HALF_OPEN = "half_open"  # Testing state, limited requests pass through


class CircuitBreaker:
    """
    Circuit breaker implementation for handling external dependency failures.

    The circuit breaker pattern prevents cascading failures by failing fast
    when a dependency is unavailable, and allows for graceful recovery.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        reset_timeout: float = 60.0,
    ):
        """
        Initialize the circuit breaker.

        Args:
            name: Name of the circuit breaker (for logging)
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds before attempting recovery
            half_open_max_calls: Maximum number of calls in half-open state
            reset_timeout: Time in seconds before resetting failure count
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.reset_timeout = reset_timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = time.time()
        self.half_open_calls = 0

        self._lock = asyncio.Lock()

        logger.info(f"Circuit breaker '{name}' initialized")

    async def execute(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function call

        Raises:
            CircuitBreakerOpenError: If the circuit is open
            Exception: Any exception raised by the function
        """
        async with self._lock:
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    # Transition to half-open state
                    logger.info(
                        f"Circuit breaker '{self.name}' transitioning from OPEN to HALF_OPEN"
                    )
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    # Circuit is open, fail fast
                    logger.warning(f"Circuit breaker '{self.name}' is OPEN, failing fast")
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")

            # Check if we've exceeded half-open call limit
            if (
                self.state == CircuitState.HALF_OPEN
                and self.half_open_calls >= self.half_open_max_calls
            ):
                logger.warning(f"Circuit breaker '{self.name}' exceeded half-open call limit")
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' exceeded half-open call limit"
                )

            # Increment half-open call count
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1

        # Execute the function
        try:
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Handle success
            async with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    # Transition to closed state on success
                    logger.info(
                        f"Circuit breaker '{self.name}' transitioning from HALF_OPEN to CLOSED"
                    )
                    self.state = CircuitState.CLOSED

                # Reset failure count on success
                self.failure_count = 0
                self.last_success_time = time.time()

                # Log success
                logger.debug(
                    f"Circuit breaker '{self.name}' call succeeded in {execution_time:.2f}s"
                )

            return result

        except Exception as e:
            # Handle failure
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                # Check if we should open the circuit
                if (
                    self.state == CircuitState.CLOSED
                    and self.failure_count >= self.failure_threshold
                ):
                    logger.warning(
                        f"Circuit breaker '{self.name}' transitioning from CLOSED to OPEN "
                        f"after {self.failure_count} failures"
                    )
                    self.state = CircuitState.OPEN

                # If in half-open state, go back to open on failure
                if self.state == CircuitState.HALF_OPEN:
                    logger.warning(
                        f"Circuit breaker '{self.name}' transitioning from HALF_OPEN to OPEN after failure"
                    )
                    self.state = CircuitState.OPEN

                # Log failure
                logger.error(f"Circuit breaker '{self.name}' call failed: {str(e)}")

            # Re-raise the exception
            raise

    def get_state(self) -> dict[str, Any]:
        """
        Get the current state of the circuit breaker.

        Returns:
            Dict[str, Any]: Current state information
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "half_open_calls": self.half_open_calls,
        }

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED state")


class CircuitBreakerOpenError(Exception):
    """Exception raised when a circuit breaker is open."""


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    This class provides a centralized way to manage and monitor
    multiple circuit breakers in the application.
    """

    def __init__(self):
        """Initialize the circuit breaker registry."""
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
        logger.info("Circuit breaker registry initialized")

    async def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        reset_timeout: float = 60.0,
    ) -> CircuitBreaker:
        """
        Get an existing circuit breaker or create a new one.

        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds before attempting recovery
            half_open_max_calls: Maximum number of calls in half-open state
            reset_timeout: Time in seconds before resetting failure count

        Returns:
            CircuitBreaker: The requested circuit breaker
        """
        async with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    half_open_max_calls=half_open_max_calls,
                    reset_timeout=reset_timeout,
                )
                logger.info(f"Created new circuit breaker '{name}'")

            return self.circuit_breakers[name]

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """
        Get the current state of all circuit breakers.

        Returns:
            Dict[str, Dict[str, Any]]: Current state information for all circuit breakers
        """
        return {name: cb.get_state() for name, cb in self.circuit_breakers.items()}

    async def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        async with self._lock:
            for _name, cb in self.circuit_breakers.items():
                cb.reset()
            logger.info("All circuit breakers reset to CLOSED state")


# Global registry instance
registry = CircuitBreakerRegistry()


async def with_circuit_breaker(
    func: Callable[..., Awaitable[T]],
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    *args,
    **kwargs,
) -> T:
    """
    Execute a function with circuit breaker protection using the global registry.

    Args:
        func: Async function to execute
        name: Name of the circuit breaker
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time in seconds before attempting recovery
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function call

    Raises:
        CircuitBreakerOpenError: If the circuit is open
        Exception: Any exception raised by the function
    """
    cb = await registry.get_or_create(
        name=name, failure_threshold=failure_threshold, recovery_timeout=recovery_timeout
    )
    return await cb.execute(func, *args, **kwargs)
