"""
Installation script for the Improved Local AI Assistant.

This script checks dependencies, sets up the environment, and configures the application.
"""

import os
import sys
import subprocess
import platform
import shutil
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("install.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)

# Define required directories
REQUIRED_DIRS = [
    "logs",
    "data",
    "data/prebuilt_graphs",
    "data/dynamic_graph",
    "data/sessions"
]

# Define required Python packages
REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "ollama",
    "llama-index",
    "networkx",
    "pyvis",
    "psutil",
    "pyyaml",
    "python-dotenv",
    "sentence-transformers"
]

# Define optional Python packages
OPTIONAL_PACKAGES = [
    "pytest",
    "httpx",
    "websockets"
]

def check_python_version() -> bool:
    """
    Check if Python version is compatible.
    
    Returns:
        bool: True if Python version is compatible
    """
    major, minor, _ = sys.version_info
    
    if major < 3 or (major == 3 and minor < 8):
        logger.error(f"Python 3.8 or higher is required, but you have {major}.{minor}")
        return False
    
    logger.info(f"Python version {major}.{minor} is compatible")
    return True

def check_ollama_installation() -> bool:
    """
    Check if Ollama is installed.
    
    Returns:
        bool: True if Ollama is installed
    """
    try:
        # Try to run ollama version
        result = subprocess.run(
            ["ollama", "version"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            logger.info(f"Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            logger.warning(f"Ollama command failed: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        logger.warning("Ollama is not installed or not in PATH")
        return False

def create_virtual_environment() -> bool:
    """
    Create a virtual environment.
    
    Returns:
        bool: True if virtual environment was created successfully
    """
    try:
        # Check if virtual environment already exists
        if os.path.exists(".venv"):
            logger.info("Virtual environment already exists")
            return True
        
        # Create virtual environment
        logger.info("Creating virtual environment...")
        subprocess.run(
            [sys.executable, "-m", "venv", ".venv"],
            check=True
        )
        
        logger.info("Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating virtual environment: {str(e)}")
        return False

def install_dependencies(include_optional: bool = False) -> bool:
    """
    Install dependencies.
    
    Args:
        include_optional: Whether to include optional dependencies
        
    Returns:
        bool: True if dependencies were installed successfully
    """
    try:
        # Get pip path
        if sys.platform == "win32":
            pip_path = os.path.join(".venv", "Scripts", "pip")
        else:
            pip_path = os.path.join(".venv", "bin", "pip")
        
        # Upgrade pip
        logger.info("Upgrading pip...")
        subprocess.run(
            [pip_path, "install", "--upgrade", "pip"],
            check=True
        )
        
        # Install required packages
        logger.info("Installing required packages...")
        packages = REQUIRED_PACKAGES.copy()
        
        # Add optional packages if requested
        if include_optional:
            logger.info("Including optional packages...")
            packages.extend(OPTIONAL_PACKAGES)
        
        subprocess.run(
            [pip_path, "install"] + packages,
            check=True
        )
        
        # Install platform-specific packages
        if sys.platform == "win32":
            logger.info("Installing Windows-specific packages...")
            try:
                subprocess.run(
                    [pip_path, "install", "pywin32>=306"],
                    check=True
                )
                logger.info("Windows-specific packages installed successfully")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error installing Windows-specific packages: {str(e)}")
                logger.warning("Some Windows-specific features may not work properly")
        
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        return False

def create_directories() -> bool:
    """
    Create required directories.
    
    Returns:
        bool: True if directories were created successfully
    """
    try:
        for directory in REQUIRED_DIRS:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        return False

def create_config_file() -> bool:
    """
    Create configuration file.
    
    Returns:
        bool: True if configuration file was created successfully
    """
    try:
        # Check if config file already exists
        if os.path.exists("config.yaml"):
            logger.info("Configuration file already exists")
            return True
        
        # Create config file
        import yaml
        
        # Detect system information
        import psutil
        cpu_cores = psutil.cpu_count()
        memory_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        
        # Determine max memory based on system memory
        max_memory_gb = min(12, memory_gb * 0.8)
        
        config = {
            "ollama": {
                "host": "http://localhost:11434",
                "timeout": 120,
                "max_parallel": 2,
                "max_loaded_models": 2
            },
            "models": {
                "conversation": {
                    "name": "hermes3:3b",
                    "context_window": 8000,
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                "knowledge": {
                    "name": "tinyllama",
                    "context_window": 2048,
                    "temperature": 0.2,
                    "max_tokens": 1024
                }
            },
            "knowledge_graphs": {
                "prebuilt_directory": "./data/prebuilt_graphs",
                "dynamic_storage": "./data/dynamic_graph",
                "max_triplets_per_chunk": 4,
                "enable_visualization": True
            },
            "conversation": {
                "max_history_length": 50,
                "summarize_threshold": 20,
                "context_window_tokens": 8000
            },
            "system": {
                "max_memory_gb": max_memory_gb,
                "cpu_cores": cpu_cores,
                "memory_threshold_percent": 80,
                "cpu_threshold_percent": 80
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_origins": ["*"]
            },
            "environment": "development",
            "environment_production": {
                "api": {
                    "cors_origins": ["http://localhost:8000"]
                },
                "system": {
                    "memory_threshold_percent": 70,
                    "cpu_threshold_percent": 70
                }
            }
        }
        
        # Write config file
        with open("config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("Configuration file created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating configuration file: {str(e)}")
        return False

def create_env_file() -> bool:
    """
    Create .env file.
    
    Returns:
        bool: True if .env file was created successfully
    """
    try:
        # Check if .env file already exists
        if os.path.exists(".env"):
            logger.info(".env file already exists")
            return True
        
        # Create .env file
        with open(".env", "w") as f:
            f.write("# Environment variables for the Improved Local AI Assistant\n")
            f.write("CONVERSATION_MODEL=hermes3:3b\n")
            f.write("KNOWLEDGE_MODEL=tinyllama\n")
            f.write("OLLAMA_HOST=http://localhost:11434\n")
            f.write("OLLAMA_NUM_PARALLEL=2\n")
            f.write("OLLAMA_MAX_LOADED_MODELS=2\n")
            f.write("LOG_LEVEL=INFO\n")
            f.write("API_HOST=0.0.0.0\n")
            f.write("API_PORT=8000\n")
        
        logger.info(".env file created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating .env file: {str(e)}")
        return False

def create_startup_scripts() -> bool:
    """
    Create startup scripts.
    
    Returns:
        bool: True if startup scripts were created successfully
    """
    try:
        # Create Windows batch file
        with open("start_app.bat", "w") as f:
            f.write("@echo off\n")
            f.write("echo Starting Improved Local AI Assistant...\n")
            f.write(".venv\\Scripts\\python app\\main.py\n")
            f.write("pause\n")
        
        # Create Unix shell script
        with open("start_app.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("echo Starting Improved Local AI Assistant...\n")
            f.write(".venv/bin/python app/main.py\n")
        
        # Make shell script executable on Unix-like systems
        if sys.platform != "win32":
            os.chmod("start_app.sh", 0o755)
        
        logger.info("Startup scripts created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating startup scripts: {str(e)}")
        return False

def check_models() -> bool:
    """
    Check if required models are available in Ollama.
    
    Returns:
        bool: True if models are available
    """
    try:
        # Check if Ollama is installed
        if not check_ollama_installation():
            logger.warning("Ollama is not installed, skipping model check")
            return False
        
        # Check if models are available
        logger.info("Checking if required models are available in Ollama...")
        
        # Get list of models
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check if required models are in the list
        models_output = result.stdout.lower()
        hermes_available = "hermes" in models_output
        tinyllama_available = "tinyllama" in models_output
        
        if hermes_available and tinyllama_available:
            logger.info("All required models are available")
            return True
        else:
            missing_models = []
            if not hermes_available:
                missing_models.append("hermes3:3b")
            if not tinyllama_available:
                missing_models.append("tinyllama")
            
            logger.warning(f"Missing models: {', '.join(missing_models)}")
            logger.info("You can pull the missing models with:")
            for model in missing_models:
                logger.info(f"  ollama pull {model}")
            
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking models: {str(e)}")
        return False
    except FileNotFoundError:
        logger.warning("Ollama command not found, skipping model check")
        return False

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Install the Improved Local AI Assistant")
    parser.add_argument("--with-optional", action="store_true", help="Install optional dependencies")
    args = parser.parse_args()
    
    logger.info("Starting installation of Improved Local AI Assistant")
    
    # Check Python version
    if not check_python_version():
        logger.error("Installation failed: Incompatible Python version")
        return 1
    
    # Create directories
    if not create_directories():
        logger.error("Installation failed: Could not create directories")
        return 1
    
    # Create virtual environment
    if not create_virtual_environment():
        logger.error("Installation failed: Could not create virtual environment")
        return 1
    
    # Install dependencies
    if not install_dependencies(include_optional=args.with_optional):
        logger.error("Installation failed: Could not install dependencies")
        return 1
    
    # Create configuration file
    if not create_config_file():
        logger.error("Installation failed: Could not create configuration file")
        return 1
    
    # Create .env file
    if not create_env_file():
        logger.error("Installation failed: Could not create .env file")
        return 1
    
    # Create startup scripts
    if not create_startup_scripts():
        logger.error("Installation failed: Could not create startup scripts")
        return 1
    
    # Check Ollama installation
    check_ollama_installation()
    
    # Check models
    check_models()
    
    logger.info("Installation completed successfully")
    logger.info("You can start the application with:")
    if sys.platform == "win32":
        logger.info("  start_app.bat")
    else:
        logger.info("  ./start_app.sh")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())