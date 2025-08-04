#!/usr/bin/env python3
"""
Script to install Windows-specific dependencies.

This script installs the pywin32 package which is required for Windows-specific
functionality like adjusting process priority.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_pywin32():
    """Install pywin32 package."""
    if sys.platform != "win32":
        logger.info("Not running on Windows, skipping pywin32 installation")
        return True
    
    try:
        logger.info("Checking if pywin32 is already installed...")
        try:
            import win32api
            logger.info("pywin32 is already installed")
            return True
        except ImportError:
            logger.info("pywin32 is not installed, installing...")
        
        # Try to install pywin32
        logger.info("Installing pywin32...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])
        
        # Verify installation
        try:
            import win32api
            logger.info("pywin32 installed successfully")
            return True
        except ImportError:
            logger.error("Failed to import win32api after installation")
            return False
    
    except Exception as e:
        logger.error(f"Error installing pywin32: {str(e)}")
        return False

def main():
    """Main function."""
    logger.info("Installing Windows-specific dependencies...")
    
    if install_pywin32():
        logger.info("All Windows-specific dependencies installed successfully")
        return 0
    else:
        logger.error("Failed to install Windows-specific dependencies")
        return 1

if __name__ == "__main__":
    sys.exit(main())