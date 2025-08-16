"""
Configuration loader for the Improved Local AI Assistant.

This module provides functions to load and validate configuration from YAML files.
"""

import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Load configuration from a YAML file and override with environment variables.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary with environment variable overrides
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Override with environment variables for performance tuning
        startup_config = config.setdefault("startup", {})
        startup_config["preload"] = os.getenv(
            "STARTUP_PRELOAD", startup_config.get("preload", "models")
        )
        startup_config["lazy_load_graphs"] = os.getenv("GRAPH_LAZY_LOAD", "true").lower() == "true"
        startup_config["ollama_healthcheck"] = os.getenv(
            "OLLAMA_HEALTHCHECK", startup_config.get("ollama_healthcheck", "version")
        )

        embedding_config = config.setdefault("embedding", {})
        embedding_config["device"] = os.getenv(
            "EMBED_MODEL_DEVICE", embedding_config.get("device", "cpu")
        )
        embedding_config["int8_quantization"] = (
            os.getenv("EMBED_MODEL_INT8", "true").lower() == "true"
        )

        system_config = config.setdefault("system", {})
        system_config["monitor_interval"] = int(
            os.getenv("RESOURCE_MONITOR_INTERVAL", system_config.get("monitor_interval", 10))
        )
        system_config["monitor_debounce"] = int(
            os.getenv("RESOURCE_MONITOR_DEBOUNCE", system_config.get("monitor_debounce", 60))
        )
        system_config["memory_threshold_percent"] = int(
            float(os.getenv("MEM_PRESSURE_THRESHOLD", "0.95")) * 100
        )

        logger.info("Configuration loaded successfully with environment overrides")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}
