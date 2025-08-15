"""
FastAPI dependency injection for core services.

This module provides dependency injection functions for:
• Dynamic model manager
• Connection pool manager
• System monitor
• Other core services
"""

from typing import Any
from typing import Dict

from services.connection_pool_manager import ConnectionPoolManager
from services.dynamic_model_manager import DynamicModelManager
from services.system_monitor import SystemMonitor

# Global instances (initialized by main app)
_dynamic_model_manager: DynamicModelManager = None
_connection_pool_manager: ConnectionPoolManager = None
_system_monitor: SystemMonitor = None
_config: Dict[str, Any] = None


def initialize_dependencies(
    config: Dict[str, Any], connection_pool: ConnectionPoolManager, system_monitor: SystemMonitor
) -> None:
    """
    Initialize global dependency instances.

    Args:
        config: Application configuration
        connection_pool: Connection pool manager instance
        system_monitor: System monitor instance
    """
    global _dynamic_model_manager, _connection_pool_manager, _system_monitor, _config

    _config = config
    _connection_pool_manager = connection_pool
    _system_monitor = system_monitor
    _dynamic_model_manager = DynamicModelManager(config, connection_pool)


def get_dynamic_model_manager() -> DynamicModelManager:
    """Get the dynamic model manager instance."""
    if _dynamic_model_manager is None:
        raise RuntimeError("Dynamic model manager not initialized")
    return _dynamic_model_manager


def get_connection_pool_manager() -> ConnectionPoolManager:
    """Get the connection pool manager instance."""
    if _connection_pool_manager is None:
        raise RuntimeError("Connection pool manager not initialized")
    return _connection_pool_manager


def get_system_monitor() -> SystemMonitor:
    """Get the system monitor instance."""
    if _system_monitor is None:
        raise RuntimeError("System monitor not initialized")
    return _system_monitor


def get_config() -> Dict[str, Any]:
    """Get the application configuration."""
    if _config is None:
        raise RuntimeError("Configuration not initialized")
    return _config
