"""
Constants and default values for the Improved Local AI Assistant.

This module centralizes magic numbers, timeouts, and thresholds
to make them easier to configure and maintain.
"""

from typing import Any

# Timeout constants (in seconds)
DEFAULTS = {
    # WebSocket timeouts
    "ws_heartbeat": 30,
    "ws_receive_timeout": 1.0,
    "ws_close_timeout": 5.0,
    # Processing timeouts
    "process_timeout": 300.0,  # 5 minutes
    "conversation_timeout": 300.0,  # 5 minutes - increased from 30s
    "knowledge_query_timeout": 60.0,  # 1 minute - increased from 10s
    "health_check_timeout": 10.0,  # increased from 2s
    # Visualization timeouts
    "viz_timeout": 5.0,
    # System thresholds (percentages)
    "cpu_warn": 80.0,
    "mem_warn": 80.0,
    "disk_warn": 90.0,
    # Resource limits
    "max_message_length": 10000,  # 10KB
    "max_history_length": 50,
    "summarize_threshold": 20,
    "context_window_tokens": 8000,
    # Retry settings
    "max_retries": 3,
    "retry_delay": 2.0,
    "service_init_retries": 3,
    # Monitoring intervals
    "monitoring_interval": 5.0,
    "cleanup_interval": 3600.0,  # 1 hour
    # Model settings
    "ollama_timeout": 120,
    "max_parallel": 2,
    "max_loaded_models": 2,
    # Knowledge graph settings
    "max_triplets_per_chunk": 4,
    "kg_query_cache_size": 1000,
    # Log rotation
    "log_backup_count": 7,
    "log_rotation": "midnight",
}

# HTTP status codes for common scenarios
HTTP_STATUS = {
    "SERVICE_UNAVAILABLE": 503,
    "NOT_IMPLEMENTED": 501,
    "TIMEOUT": 408,
    "TOO_LARGE": 413,
}

# Error codes for consistent error handling
ERROR_CODES = {
    "SESSION_NOT_FOUND": "SESSION_NOT_FOUND",
    "MODEL_ERROR": "MODEL_ERROR",
    "KNOWLEDGE_GRAPH_ERROR": "KNOWLEDGE_GRAPH_ERROR",
    "CONVERSATION_ERROR": "CONVERSATION_ERROR",
    "CIRCUIT_BREAKER_OPEN": "CIRCUIT_BREAKER_OPEN",
    "TIMEOUT_ERROR": "TIMEOUT_ERROR",
    "VALIDATION_ERROR": "VALIDATION_ERROR",
}


def get_timeout(key: str, config: dict[str, Any] | None = None) -> float:
    """
    Get timeout value from config or defaults.

    Args:
        key: Timeout key
        config: Configuration dictionary

    Returns:
        float: Timeout value in seconds
    """
    if config and "timeouts" in config and key in config["timeouts"]:
        return float(config["timeouts"][key])
    default_value = DEFAULTS.get(key)
    if default_value is not None:
        return float(default_value)
    return 30.0


def get_threshold(key: str, config: dict[str, Any] | None = None) -> float:
    """
    Get threshold value from config or defaults.

    Args:
        key: Threshold key
        config: Configuration dictionary

    Returns:
        float: Threshold value
    """
    if config and "thresholds" in config and key in config["thresholds"]:
        return float(config["thresholds"][key])
    default_value = DEFAULTS.get(key)
    if default_value is not None:
        return float(default_value)
    return 80.0


def get_limit(key: str, config: dict[str, Any] | None = None) -> int:
    """
    Get limit value from config or defaults.

    Args:
        key: Limit key
        config: Configuration dictionary

    Returns:
        int: Limit value
    """
    if config and "limits" in config and key in config["limits"]:
        return int(config["limits"][key])
    default_value = DEFAULTS.get(key)
    if default_value is not None:
        return int(default_value)
    return 50
