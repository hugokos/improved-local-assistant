"""
Graceful degradation implementation for component failures.

This module provides mechanisms for graceful degradation when components fail,
allowing the system to continue operating with reduced functionality.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable
from collections.abc import Callable
from enum import Enum
from typing import Any, Union
from typing import TypeVar

# Type variables for generic functions
T = TypeVar("T")

# Configure logging
logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component status enum."""

    OPERATIONAL = "operational"  # Fully operational
    DEGRADED = "degraded"  # Operating with reduced functionality
    FAILED = "failed"  # Completely failed


class DegradationManager:
    """
    Manages graceful degradation of system components.

    This class tracks the status of system components and provides
    fallback mechanisms when components fail.
    """

    def __init__(self):
        """Initialize the degradation manager."""
        self.component_status: dict[str, ComponentStatus] = {}
        self.fallback_handlers: dict[str, list[Callable[..., Any]]] = {}
        self._lock = asyncio.Lock()
        logger.info("Degradation manager initialized")

    async def set_component_status(self, component: str, status: ComponentStatus) -> None:
        """
        Set the status of a component.

        Args:
            component: Component name
            status: New component status
        """
        async with self._lock:
            old_status = self.component_status.get(component, None)
            self.component_status[component] = status

            if old_status != status:
                logger.info(f"Component '{component}' status changed from {old_status} to {status}")

                # If component failed, log a warning
                if status == ComponentStatus.FAILED:
                    logger.warning(f"Component '{component}' has failed")

                # If component recovered, log an info message
                if old_status == ComponentStatus.FAILED and status != ComponentStatus.FAILED:
                    logger.info(f"Component '{component}' has recovered to {status} state")

    def get_component_status(self, component: str) -> ComponentStatus:
        """
        Get the status of a component.

        Args:
            component: Component name

        Returns:
            ComponentStatus: Current component status
        """
        return self.component_status.get(component, ComponentStatus.OPERATIONAL)

    def register_fallback(self, component: str, handler: Callable[..., Any]) -> None:
        """
        Register a fallback handler for a component.

        Args:
            component: Component name
            handler: Fallback handler function
        """
        if component not in self.fallback_handlers:
            self.fallback_handlers[component] = []

        self.fallback_handlers[component].append(handler)
        logger.info(f"Registered fallback handler for component '{component}'")

    async def execute_with_fallback(
        self, component: str, primary_func: Callable[..., Awaitable[T]], *args, **kwargs
    ) -> Union[T, Any]:
        """
        Execute a function with fallback if it fails.

        Args:
            component: Component name
            primary_func: Primary function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the primary function or fallback

        Raises:
            Exception: If all fallbacks fail
        """
        # Check if component is already failed
        if self.get_component_status(component) == ComponentStatus.FAILED:
            logger.warning(f"Component '{component}' is already in FAILED state, using fallback")
            return await self._execute_fallbacks(component, *args, **kwargs)

        # Try primary function
        try:
            result = await primary_func(*args, **kwargs)

            # If component was degraded and succeeded, mark as operational
            if self.get_component_status(component) == ComponentStatus.DEGRADED:
                await self.set_component_status(component, ComponentStatus.OPERATIONAL)

            return result

        except Exception as e:
            logger.error(f"Primary function for component '{component}' failed: {str(e)}")

            # Mark component as degraded
            await self.set_component_status(component, ComponentStatus.DEGRADED)

            # Try fallbacks
            try:
                return await self._execute_fallbacks(component, *args, **kwargs)
            except Exception as fallback_error:
                # All fallbacks failed, mark component as failed
                await self.set_component_status(component, ComponentStatus.FAILED)
                logger.error(
                    f"All fallbacks for component '{component}' failed: {str(fallback_error)}"
                )
                raise

    async def _execute_fallbacks(self, component: str, *args, **kwargs) -> Any:
        """
        Execute fallback handlers for a component.

        Args:
            component: Component name
            *args: Positional arguments for the handlers
            **kwargs: Keyword arguments for the handlers

        Returns:
            The result of the first successful fallback

        Raises:
            Exception: If all fallbacks fail
        """
        if component not in self.fallback_handlers or not self.fallback_handlers[component]:
            raise ValueError(f"No fallback handlers registered for component '{component}'")

        # Try each fallback in order
        last_error = None
        for handler in self.fallback_handlers[component]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await handler(*args, **kwargs)
                else:
                    return handler(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Fallback handler for component '{component}' failed: {str(e)}")
                last_error = e

        # All fallbacks failed
        if last_error:
            raise last_error
        else:
            raise RuntimeError(f"All fallbacks for component '{component}' failed")

    def get_all_statuses(self) -> dict[str, str]:
        """
        Get the status of all components.

        Returns:
            Dict[str, str]: Component statuses
        """
        return {component: status.value for component, status in self.component_status.items()}


# Global degradation manager instance
degradation_manager = DegradationManager()


async def with_degradation(
    component: str, primary_func: Callable[..., Awaitable[T]], *args, **kwargs
) -> Union[T, Any]:
    """
    Execute a function with graceful degradation using the global manager.

    Args:
        component: Component name
        primary_func: Primary function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the primary function or fallback

    Raises:
        Exception: If all fallbacks fail
    """
    return await degradation_manager.execute_with_fallback(component, primary_func, *args, **kwargs)


def register_fallback(component: str, handler: Callable[..., Any]) -> None:
    """
    Register a fallback handler for a component using the global manager.

    Args:
        component: Component name
        handler: Fallback handler function
    """
    degradation_manager.register_fallback(component, handler)


# Common fallback handlers


def empty_response_fallback(*args, **kwargs) -> dict[str, Any]:
    """
    Fallback handler that returns an empty response.

    Returns:
        Dict[str, Any]: Empty response with error information
    """
    return {
        "error": "Service temporarily unavailable",
        "fallback": "empty_response",
        "timestamp": time.time(),
    }


def cached_response_fallback(cache_key: str, cache: dict[str, Any]) -> Callable[..., Any]:
    """
    Create a fallback handler that returns a cached response.

    Args:
        cache_key: Key to look up in the cache
        cache: Cache dictionary

    Returns:
        Callable: Fallback handler function
    """

    def fallback(*args, **kwargs) -> Any:
        if cache_key in cache:
            return cache[cache_key]
        else:
            return empty_response_fallback(*args, **kwargs)

    return fallback


def default_value_fallback(default_value: Any) -> Callable[..., Any]:
    """
    Create a fallback handler that returns a default value.

    Args:
        default_value: Default value to return

    Returns:
        Callable: Fallback handler function
    """

    def fallback(*args, **kwargs) -> Any:
        return default_value

    return fallback
