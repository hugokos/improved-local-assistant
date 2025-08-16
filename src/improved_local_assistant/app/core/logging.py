"""
Logging configuration for the Improved Local AI Assistant.

This module provides a centralized logging setup with file rotation
and proper formatting.
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from typing import Any

from services import DEFAULTS


def setup_logging(config: dict[str, Any] = None) -> logging.Logger:
    """
    Configure logging with rotation and formatting.

    Args:
        config: Optional configuration dictionary

    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create logger
    logger = logging.getLogger("api")
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Get rotation settings from config or defaults
    rotation_when = DEFAULTS["log_rotation"]
    backup_count = DEFAULTS["log_backup_count"]

    if config and "logging" in config:
        log_config = config.get("logging", {})
        rotation_when = log_config.get("rotation", rotation_when)
        backup_count = log_config.get("backup_count", backup_count)

    # File handler with rotation (daily rotation, keep 7 days)
    file_handler = TimedRotatingFileHandler(
        "logs/api.log", when=rotation_when, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Set root logger level and add our handlers
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    return logging.getLogger(__name__)
