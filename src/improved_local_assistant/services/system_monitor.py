"""
System Monitor for tracking resource usage and performance metrics.

This module provides the SystemMonitor class that tracks CPU, memory, and storage usage,
implements adaptive resource management, and provides health check endpoints.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any
from typing import Dict

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import psutil
from asyncio_throttle import Throttler


class SystemMonitor:
    """
    Monitors system resources and performance metrics.

    Tracks CPU, memory, and storage usage, implements adaptive resource management,
    provides health check endpoints, and logs system events and errors.
    """

    def __init__(self, config=None):
        """
        Initialize SystemMonitor with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Extract configuration values
        system_config = self.config.get("system", {})
        self.max_memory_gb = system_config.get("max_memory_gb", 12)
        self.cpu_cores = system_config.get("cpu_cores", psutil.cpu_count())
        self.enable_gpu = system_config.get("enable_gpu", False)
        self.quiet_monitoring = system_config.get("quiet_monitoring", False)

        # Performance configuration
        perf_config = self.config.get("performance", {})
        self.cleanup_interval = perf_config.get("cleanup_interval", 3600)  # 1 hour

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task = None
        self.throttler = Throttler(rate_limit=1, period=5.0)  # Limit to 1 check per 5 seconds

        # Import constants
        from services.constants import get_threshold

        # Resource thresholds (use environment overrides)
        self.cpu_threshold = system_config.get(
            "cpu_threshold_percent", get_threshold("cpu_warn", self.config)
        )
        self.memory_threshold = system_config.get(
            "memory_threshold_percent", get_threshold("mem_warn", self.config)
        )
        self.disk_threshold = get_threshold("disk_warn", self.config)

        # Monitoring intervals
        self.monitor_interval = system_config.get("monitor_interval", 10)
        self.monitor_debounce = system_config.get("monitor_debounce", 60)
        self._last_action_time = 0

        # Startup grace period to ignore initial CPU spikes
        self._startup_time = time.time()
        self._startup_grace_period = 30  # 30 seconds

        # Performance metrics
        self.metrics = {
            "system": {
                "start_time": datetime.now().isoformat(),
                "uptime_seconds": 0,
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "memory_used_gb": 0.0,
                "memory_total_gb": 0.0,
                "disk_percent": 0.0,
                "disk_used_gb": 0.0,
                "disk_total_gb": 0.0,
            },
            "performance": {
                "high_load_events": 0,
                "last_high_load": None,
                "adaptive_actions": 0,
                "last_cleanup": None,
            },
            "history": {"cpu": [], "memory": [], "timestamps": []},
        }

        # Initialize history with empty values
        self._update_history(0.0, 0.0)

    async def start_monitoring(self) -> None:
        """
        Start monitoring system resources.

        This method starts a background task that periodically checks system
        resources and updates metrics.
        """
        if self.is_monitoring:
            self.logger.warning("Monitoring is already running")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("System monitoring started")

    async def stop_monitoring(self) -> None:
        """
        Stop monitoring system resources.

        This method stops the background monitoring task.
        """
        if not self.is_monitoring:
            self.logger.warning("Monitoring is not running")
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None

        self.logger.info("System monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """
        Background task for monitoring system resources.

        This method runs in the background and periodically checks system
        resources, updates metrics, and triggers adaptive actions if needed.
        """
        last_cleanup_time = time.time()

        while self.is_monitoring:
            try:
                # Use throttler to limit check frequency
                async with self.throttler:
                    # Update resource metrics
                    await self._update_resource_metrics()

                    # Check for high load
                    await self._check_high_load()

                    # Perform periodic cleanup if needed
                    current_time = time.time()
                    if current_time - last_cleanup_time > self.cleanup_interval:
                        await self._perform_cleanup()
                        last_cleanup_time = current_time
                        self.metrics["performance"]["last_cleanup"] = datetime.now().isoformat()

                    # Update uptime
                    self.metrics["system"]["uptime_seconds"] = (
                        datetime.now()
                        - datetime.fromisoformat(self.metrics["system"]["start_time"])
                    ).total_seconds()

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")

            # Sleep before next check (use configurable interval)
            await asyncio.sleep(self.monitor_interval)

    async def _update_resource_metrics(self) -> None:
        """
        Update resource usage metrics.

        This method collects current CPU, memory, and disk usage metrics
        and updates the metrics dictionary.
        """
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)

            # Get disk usage for the current directory
            disk = psutil.disk_usage(os.getcwd())
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)

            # Update metrics
            self.metrics["system"]["cpu_percent"] = cpu_percent
            self.metrics["system"]["memory_percent"] = memory_percent
            self.metrics["system"]["memory_used_gb"] = round(memory_used_gb, 2)
            self.metrics["system"]["memory_total_gb"] = round(memory_total_gb, 2)
            self.metrics["system"]["disk_percent"] = disk_percent
            self.metrics["system"]["disk_used_gb"] = round(disk_used_gb, 2)
            self.metrics["system"]["disk_total_gb"] = round(disk_total_gb, 2)

            # Update history
            self._update_history(cpu_percent, memory_percent)

        except Exception as e:
            self.logger.error(f"Error updating resource metrics: {str(e)}")

    def _update_history(self, cpu_percent: float, memory_percent: float) -> None:
        """
        Update resource usage history.

        Args:
            cpu_percent: Current CPU usage percentage
            memory_percent: Current memory usage percentage
        """
        # Keep only the last 60 data points (5 minutes at 5-second intervals)
        max_history = 60

        # Add new data point
        self.metrics["history"]["cpu"].append(cpu_percent)
        self.metrics["history"]["memory"].append(memory_percent)
        self.metrics["history"]["timestamps"].append(datetime.now().isoformat())

        # Trim history if needed
        if len(self.metrics["history"]["cpu"]) > max_history:
            self.metrics["history"]["cpu"] = self.metrics["history"]["cpu"][-max_history:]
            self.metrics["history"]["memory"] = self.metrics["history"]["memory"][-max_history:]
            self.metrics["history"]["timestamps"] = self.metrics["history"]["timestamps"][
                -max_history:
            ]

    async def _check_high_load(self) -> None:
        """
        Check for high system load and take adaptive actions if needed.

        This method checks if CPU or memory usage is above thresholds and
        triggers adaptive resource management actions if needed.
        """
        cpu_percent = self.metrics["system"]["cpu_percent"]
        memory_percent = self.metrics["system"]["memory_percent"]

        # Check for high load
        is_high_load = cpu_percent > self.cpu_threshold or memory_percent > self.memory_threshold

        if is_high_load:
            current_time = time.time()

            # Skip high load detection during startup grace period
            if current_time - self._startup_time < self._startup_grace_period:
                self.logger.debug(
                    f"Ignoring high load during startup grace period: CPU {cpu_percent}%, Memory {memory_percent}%"
                )
                return

            # Debounce actions - don't act too frequently
            if current_time - self._last_action_time < self.monitor_debounce:
                return

            if not self.quiet_monitoring:
                self.logger.warning(
                    f"High system load detected: CPU {cpu_percent}%, Memory {memory_percent}%"
                )
            else:
                self.logger.debug(
                    f"High system load detected: CPU {cpu_percent}%, Memory {memory_percent}%"
                )

            # Update metrics
            self.metrics["performance"]["high_load_events"] += 1
            self.metrics["performance"]["last_high_load"] = datetime.now().isoformat()

            # Take adaptive actions
            await self._adaptive_resource_management()
            self._last_action_time = current_time

    async def _adaptive_resource_management(self) -> None:
        """
        Implement adaptive resource management under high load.

        This method takes actions to reduce resource usage when the system
        is under high load, such as triggering garbage collection, reducing
        cache sizes, or limiting background tasks.
        """
        try:
            self.logger.debug("Implementing adaptive resource management")

            # Trigger garbage collection
            import gc

            gc.collect()

            # Update metrics
            self.metrics["performance"]["adaptive_actions"] += 1

            self.logger.debug("Adaptive resource management completed")

        except Exception as e:
            self.logger.error(f"Error in adaptive resource management: {str(e)}")

    async def _perform_cleanup(self) -> None:
        """
        Perform periodic cleanup of resources.

        This method performs cleanup tasks such as clearing caches,
        removing temporary files, and optimizing memory usage.
        """
        try:
            self.logger.debug("Performing periodic cleanup")

            # Trigger garbage collection
            import gc

            gc.collect()

            # Clean up temporary files
            temp_dir = os.path.join(os.getcwd(), "temp")
            if os.path.exists(temp_dir):
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        self.logger.error(f"Error deleting {file_path}: {str(e)}")

            self.logger.debug("Cleanup completed")

        except Exception as e:
            self.logger.error(f"Error in cleanup: {str(e)}")

    def get_resource_usage(self) -> Dict[str, float]:
        """
        Get current resource usage statistics.

        Returns:
            Dict[str, float]: Resource usage statistics
        """
        return {
            "cpu_percent": self.metrics["system"]["cpu_percent"],
            "memory_percent": self.metrics["system"]["memory_percent"],
            "memory_used_gb": self.metrics["system"]["memory_used_gb"],
            "memory_total_gb": self.metrics["system"]["memory_total_gb"],
            "disk_percent": self.metrics["system"]["disk_percent"],
            "disk_used_gb": self.metrics["system"]["disk_used_gb"],
            "disk_total_gb": self.metrics["system"]["disk_total_gb"],
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.

        Returns:
            Dict[str, Any]: Performance metrics
        """
        return {
            "uptime_seconds": self.metrics["system"]["uptime_seconds"],
            "high_load_events": self.metrics["performance"]["high_load_events"],
            "last_high_load": self.metrics["performance"]["last_high_load"],
            "adaptive_actions": self.metrics["performance"]["adaptive_actions"],
            "last_cleanup": self.metrics["performance"]["last_cleanup"],
        }

    def check_health(self) -> Dict[str, str]:
        """
        Perform a health check on all system components.

        Returns:
            Dict[str, str]: Health check results
        """
        health = {"status": "ok", "timestamp": datetime.now().isoformat(), "components": {}}

        # Check CPU usage
        cpu_percent = self.metrics["system"]["cpu_percent"]
        if cpu_percent > self.cpu_threshold:
            health["components"]["cpu"] = "warning"
            health["status"] = "warning"
        else:
            health["components"]["cpu"] = "ok"

        # Check memory usage
        memory_percent = self.metrics["system"]["memory_percent"]
        if memory_percent > self.memory_threshold:
            health["components"]["memory"] = "warning"
            health["status"] = "warning"
        else:
            health["components"]["memory"] = "ok"

        # Check disk usage
        disk_percent = self.metrics["system"]["disk_percent"]
        if disk_percent > self.disk_threshold:
            health["components"]["disk"] = "warning"
            health["status"] = "warning"
        else:
            health["components"]["disk"] = "ok"

        return health

    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Log a system event.

        Args:
            event_type: Type of event
            details: Event details
        """
        event = {"type": event_type, "timestamp": datetime.now().isoformat(), "details": details}

        self.logger.info(f"System event: {event_type}")
        self.logger.debug(json.dumps(event))

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get detailed system information.

        Returns:
            Dict[str, Any]: System information
        """
        try:
            info = {
                "platform": sys.platform,
                "python_version": sys.version,
                "cpu_count": psutil.cpu_count(),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "disk_total_gb": round(psutil.disk_usage(os.getcwd()).total / (1024**3), 2),
                "hostname": os.uname().nodename if hasattr(os, "uname") else None,
                "process_id": os.getpid(),
                "process_memory_mb": round(
                    psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2
                ),
            }

            return info
        except Exception as e:
            self.logger.error(f"Error getting system info: {str(e)}")
            return {"error": str(e)}

    def get_resource_limits(self) -> Dict[str, Any]:
        """
        Get resource limits from configuration.

        Returns:
            Dict[str, Any]: Resource limits
        """
        return {
            "max_memory_gb": self.max_memory_gb,
            "cpu_cores": self.cpu_cores,
            "enable_gpu": self.enable_gpu,
            "cpu_threshold": self.cpu_threshold,
            "memory_threshold": self.memory_threshold,
            "disk_threshold": self.disk_threshold,
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics and system information.

        Returns:
            Dict[str, Any]: All metrics and system information
        """
        return {
            "system_info": self.get_system_info(),
            "resource_usage": self.get_resource_usage(),
            "performance_metrics": self.get_performance_metrics(),
            "resource_limits": self.get_resource_limits(),
            "health": self.check_health(),
        }
