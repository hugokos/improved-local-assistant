"""
Error handling and user-friendly error message generation.

This module provides comprehensive error handling and user-friendly
error message generation for various types of errors.
"""

import logging
import re
import traceback
from enum import Enum
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error category enum."""

    MODEL = "model"  # Model-related errors
    KNOWLEDGE_GRAPH = "kg"  # Knowledge graph errors
    CONVERSATION = "conversation"  # Conversation errors
    API = "api"  # API errors
    SYSTEM = "system"  # System errors
    NETWORK = "network"  # Network errors
    AUTHENTICATION = "auth"  # Authentication errors
    UNKNOWN = "unknown"  # Unknown errors


class ErrorHandler:
    """
    Comprehensive error handling and user-friendly error message generation.

    This class categorizes errors, generates user-friendly error messages,
    and provides recovery suggestions.
    """

    def __init__(self):
        """Initialize the error handler."""
        # Error patterns for categorization
        self.error_patterns = {
            ErrorCategory.MODEL: [
                r"model.*not.*found",
                r"model.*not.*available",
                r"ollama.*error",
                r"inference.*error",
                r"context.*window.*exceeded",
                r"token.*limit.*exceeded",
            ],
            ErrorCategory.KNOWLEDGE_GRAPH: [
                r"graph.*not.*found",
                r"index.*error",
                r"triplet.*error",
                r"node.*not.*found",
                r"edge.*not.*found",
                r"query.*error",
            ],
            ErrorCategory.CONVERSATION: [
                r"session.*not.*found",
                r"context.*error",
                r"message.*too.*large",
                r"history.*error",
            ],
            ErrorCategory.API: [
                r"api.*error",
                r"endpoint.*not.*found",
                r"invalid.*request",
                r"method.*not.*allowed",
                r"websocket.*error",
            ],
            ErrorCategory.SYSTEM: [
                r"memory.*error",
                r"disk.*error",
                r"cpu.*error",
                r"resource.*limit",
                r"timeout.*error",
            ],
            ErrorCategory.NETWORK: [
                r"connection.*refused",
                r"network.*error",
                r"timeout",
                r"socket.*error",
                r"http.*error",
            ],
            ErrorCategory.AUTHENTICATION: [
                r"auth.*error",
                r"unauthorized",
                r"forbidden",
                r"permission.*denied",
                r"access.*denied",
            ],
        }

        # User-friendly error messages
        self.user_messages = {
            ErrorCategory.MODEL: "The AI model is currently experiencing issues.",
            ErrorCategory.KNOWLEDGE_GRAPH: "There was a problem with the knowledge graph.",
            ErrorCategory.CONVERSATION: "There was an issue with your conversation session.",
            ErrorCategory.API: "The API encountered an error processing your request.",
            ErrorCategory.SYSTEM: "The system is currently experiencing resource constraints.",
            ErrorCategory.NETWORK: "There was a network connectivity issue.",
            ErrorCategory.AUTHENTICATION: "There was an authentication or permission issue.",
            ErrorCategory.UNKNOWN: "An unexpected error occurred.",
        }

        # Recovery suggestions
        self.recovery_suggestions = {
            ErrorCategory.MODEL: [
                "Try again in a few moments.",
                "Try using a different model if available.",
                "Your request might be too complex, try simplifying it.",
            ],
            ErrorCategory.KNOWLEDGE_GRAPH: [
                "Try a different query.",
                "The knowledge graph might not have information on this topic.",
                "Try refreshing the knowledge graph data.",
            ],
            ErrorCategory.CONVERSATION: [
                "Try starting a new conversation session.",
                "Your message might be too long, try breaking it into smaller parts.",
                "Try refreshing the page to reset your session.",
            ],
            ErrorCategory.API: [
                "Check if your request is properly formatted.",
                "Try again in a few moments.",
                "The service might be temporarily unavailable.",
            ],
            ErrorCategory.SYSTEM: [
                "The system is under high load, try again later.",
                "Try simplifying your request to reduce resource usage.",
                "Consider using a more efficient approach.",
            ],
            ErrorCategory.NETWORK: [
                "Check your internet connection.",
                "The server might be temporarily unavailable.",
                "Try again in a few moments.",
            ],
            ErrorCategory.AUTHENTICATION: [
                "Your session might have expired, try logging in again.",
                "You might not have permission to access this resource.",
                "Contact the administrator if you believe this is an error.",
            ],
            ErrorCategory.UNKNOWN: [
                "Try again in a few moments.",
                "Refresh the page and try again.",
                "If the problem persists, contact support.",
            ],
        }

        logger.info("Error handler initialized")

    def categorize_error(self, error: Exception) -> ErrorCategory:
        """
        Categorize an error based on its message and type.

        Args:
            error: The exception to categorize

        Returns:
            ErrorCategory: The error category
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Check each category's patterns
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_str) or re.search(pattern, error_type):
                    return category

        # Special case handling for common exception types
        if isinstance(error, ConnectionError | TimeoutError):
            return ErrorCategory.NETWORK
        elif isinstance(error, MemoryError):
            return ErrorCategory.SYSTEM
        elif isinstance(error, PermissionError):
            return ErrorCategory.AUTHENTICATION
        elif isinstance(error, ValueError) and "session" in error_str:
            return ErrorCategory.CONVERSATION

        # Default to unknown
        return ErrorCategory.UNKNOWN

    def get_user_message(self, error: Exception) -> str:
        """
        Get a user-friendly error message for an exception.

        Args:
            error: The exception

        Returns:
            str: User-friendly error message
        """
        category = self.categorize_error(error)
        return self.user_messages[category]

    def get_recovery_suggestion(self, error: Exception) -> str:
        """
        Get a recovery suggestion for an exception.

        Args:
            error: The exception

        Returns:
            str: Recovery suggestion
        """
        category = self.categorize_error(error)
        suggestions = self.recovery_suggestions[category]

        # Choose the most appropriate suggestion based on the error
        error_str = str(error).lower()

        # Model errors
        if category == ErrorCategory.MODEL:
            if "context window" in error_str or "token limit" in error_str:
                return "Your request is too long. Try breaking it into smaller parts."
            elif "not found" in error_str:
                return "The requested model is not available. Try using a different model."

        # Knowledge graph errors
        elif category == ErrorCategory.KNOWLEDGE_GRAPH:
            if "not found" in error_str:
                return "The requested knowledge graph was not found. Try using a different one."

        # Conversation errors
        elif category == ErrorCategory.CONVERSATION:
            if "session not found" in error_str:
                return "Your session has expired. Please refresh the page to start a new session."

        # Network errors
        elif category == ErrorCategory.NETWORK and "timeout" in error_str:
            return (
                "The request timed out. The server might be under heavy load, try again later."
            )

        # Default to the first suggestion
        return suggestions[0]

    def format_error_response(
        self, error: Exception, include_details: bool = False, error_code: str | None = None
    ) -> dict[str, Any]:
        """
        Format an error response for API endpoints.

        Args:
            error: The exception
            include_details: Whether to include technical details
            error_code: Optional error code

        Returns:
            Dict[str, Any]: Formatted error response
        """
        category = self.categorize_error(error)

        response: dict[str, Any] = {
            "status": "error",
            "message": self.get_user_message(error),
            "suggestion": self.get_recovery_suggestion(error),
            "category": category.value,
        }

        if error_code:
            response["code"] = error_code

        if include_details:
            response["details"] = {"error_type": type(error).__name__, "error_message": str(error)}

        return response

    def log_error(
        self, error: Exception, context: dict[str, Any] | None = None, level: int = logging.ERROR
    ) -> None:
        """
        Log an error with context information.

        Args:
            error: The exception
            context: Additional context information
            level: Logging level
        """
        category = self.categorize_error(error)

        # Format error message
        message = f"Error [{category.value}]: {type(error).__name__}: {str(error)}"

        # Add context if provided
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            message += f" (Context: {context_str})"

        # Log with appropriate level
        logger.log(level, message)

        # Log traceback at debug level
        logger.debug(f"Traceback for {message}:\n{traceback.format_exc()}")

    def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        include_details: bool = False,
        error_code: str | None = None,
        log_level: int = logging.ERROR,
    ) -> dict[str, Any]:
        """
        Comprehensive error handling: log the error and format a response.

        Args:
            error: The exception
            context: Additional context information
            include_details: Whether to include technical details in the response
            error_code: Optional error code
            log_level: Logging level

        Returns:
            Dict[str, Any]: Formatted error response
        """
        # Log the error
        self.log_error(error, context, log_level)

        # Format and return response
        return self.format_error_response(error, include_details, error_code)


# Global error handler instance
error_handler = ErrorHandler()


def handle_error(
    error: Exception,
    context: dict[str, Any] | None = None,
    include_details: bool = False,
    error_code: str | None = None,
    log_level: int = logging.ERROR,
) -> dict[str, Any]:
    """
    Handle an error using the global error handler.

    Args:
        error: The exception
        context: Additional context information
        include_details: Whether to include technical details in the response
        error_code: Optional error code
        log_level: Logging level

    Returns:
        Dict[str, Any]: Formatted error response
    """
    return error_handler.handle_error(
        error=error,
        context=context,
        include_details=include_details,
        error_code=error_code,
        log_level=log_level,
    )


def get_user_message(error: Exception) -> str:
    """
    Get a user-friendly error message using the global error handler.

    Args:
        error: The exception

    Returns:
        str: User-friendly error message
    """
    return error_handler.get_user_message(error)


def get_recovery_suggestion(error: Exception) -> str:
    """
    Get a recovery suggestion using the global error handler.

    Args:
        error: The exception

    Returns:
        str: Recovery suggestion
    """
    return error_handler.get_recovery_suggestion(error)
