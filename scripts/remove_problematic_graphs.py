#!/usr/bin/env python3
"""
Script to remove problematic knowledge graphs.

This script removes the survivalist graph that's causing encoding issues.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def remove_problematic_graphs():
    """Remove problematic knowledge graphs."""
    # Define paths
    project_root = Path(__file__).resolve().parent.parent
    survivalist_dir = project_root / "data" / "prebuilt_graphs" / "survivalist"
    survivalist_backup_dir = project_root / "data" / "prebuilt_graphs" / "survivalist_backup"

    # Remove survivalist graph
    if survivalist_dir.exists():
        logger.info(f"Removing survivalist graph directory: {survivalist_dir}")
        try:
            shutil.rmtree(survivalist_dir)
            logger.info("Successfully removed survivalist graph directory")
        except Exception as e:
            logger.error(f"Error removing survivalist graph directory: {str(e)}")
            return False
    else:
        logger.info(f"Survivalist graph directory not found: {survivalist_dir}")

    # Remove survivalist backup graph
    if survivalist_backup_dir.exists():
        logger.info(f"Removing survivalist backup graph directory: {survivalist_backup_dir}")
        try:
            shutil.rmtree(survivalist_backup_dir)
            logger.info("Successfully removed survivalist backup graph directory")
        except Exception as e:
            logger.error(f"Error removing survivalist backup graph directory: {str(e)}")
            return False
    else:
        logger.info(f"Survivalist backup graph directory not found: {survivalist_backup_dir}")

    # Remove any temporary directories
    import tempfile

    temp_dir = tempfile.gettempdir()
    for item in os.listdir(temp_dir):
        if item.startswith("graph_fix_"):
            item_path = os.path.join(temp_dir, item)
            if os.path.isdir(item_path):
                logger.info(f"Removing temporary graph directory: {item_path}")
                try:
                    shutil.rmtree(item_path)
                except Exception as e:
                    logger.warning(f"Error removing temporary directory: {str(e)}")

    return True


def main():
    """Main function."""
    logger.info("Starting removal of problematic graphs")

    if remove_problematic_graphs():
        logger.info("Successfully removed problematic graphs")
        return 0
    else:
        logger.error("Failed to remove problematic graphs")
        return 1


if __name__ == "__main__":
    sys.exit(main())
