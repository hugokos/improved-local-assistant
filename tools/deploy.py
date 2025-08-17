"""
Deployment script for the Improved Local AI Assistant.

This script handles deployment to different platforms, including Windows PC,
Raspberry Pi, and preparing for iPhone integration.
"""

import argparse
import json
import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("deploy.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Define platform-specific configurations
PLATFORM_CONFIGS = {
    "windows": {
        "environment": "production",
        "system": {"max_memory_gb": 8, "memory_threshold_percent": 75, "cpu_threshold_percent": 75},
        "api": {"host": "localhost", "port": 8000},
    },
    "raspberry_pi": {
        "environment": "production",
        "system": {"max_memory_gb": 4, "memory_threshold_percent": 70, "cpu_threshold_percent": 70},
        "api": {"host": "0.0.0.0", "port": 8000},
        "models": {
            "conversation": {"name": "hermes3:3b", "context_window": 4000, "max_tokens": 1024},
            "knowledge": {"name": "tinyllama", "context_window": 1024, "max_tokens": 512},
        },
    },
    "iphone": {
        "environment": "production",
        "system": {"max_memory_gb": 2, "memory_threshold_percent": 60, "cpu_threshold_percent": 60},
        "api": {"host": "localhost", "port": 3000},
        "models": {
            "conversation": {"name": "hermes3:3b", "context_window": 2000, "max_tokens": 512},
            "knowledge": {"name": "tinyllama", "context_window": 512, "max_tokens": 256},
        },
    },
}


def detect_platform() -> str:
    """
    Detect the current platform.

    Returns:
        str: Platform name ("windows", "raspberry_pi", "linux", "macos", or "unknown")
    """
    system = platform.system().lower()

    if system == "windows":
        return "windows"
    elif system == "linux":
        # Check if it's a Raspberry Pi
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
            if (
                "raspberry pi" in cpuinfo.lower()
                or "model name" in cpuinfo.lower()
                and "raspberry" in cpuinfo.lower()
            ):
                return "raspberry_pi"
            else:
                return "linux"
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Could not read /proc/cpuinfo: {str(e)}")
            return "linux"
    elif system == "darwin":
        return "macos"
    else:
        return "unknown"


def create_platform_config(platform_name: str) -> dict[str, Any]:
    """
    Create platform-specific configuration.

    Args:
        platform_name: Platform name

    Returns:
        Dict[str, Any]: Platform-specific configuration
    """
    if platform_name not in PLATFORM_CONFIGS:
        logger.warning(f"No specific configuration for platform {platform_name}, using defaults")
        return {}

    return PLATFORM_CONFIGS[platform_name]


def update_config_for_platform(platform_name: str) -> bool:
    """
    Update configuration for the specified platform.

    Args:
        platform_name: Platform name

    Returns:
        bool: True if configuration was updated successfully
    """
    try:
        # Import config validator
        project_root = Path(__file__).resolve().parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from services.config_validator import ConfigManager

        # Create config manager
        config_manager = ConfigManager()

        # Load existing configuration
        config = config_manager.load_config("config.yaml")

        # Get platform-specific configuration
        platform_config = create_platform_config(platform_name)

        # Update configuration
        for key, value in platform_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                # Merge dictionaries
                config[key].update(value)
            else:
                # Set value
                config[key] = value

        # Set environment to production
        config["environment"] = "production"

        # Process configuration
        processed_config = config_manager.process_config(config)

        # Save configuration
        with open("config.yaml", "w") as f:
            import yaml

            yaml.dump(processed_config, f, default_flow_style=False)

        logger.info(f"Configuration updated for platform {platform_name}")
        return True
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return False


def create_systemd_service() -> bool:
    """
    Create systemd service file for Linux/Raspberry Pi.

    Returns:
        bool: True if service file was created successfully
    """
    try:
        # Get absolute path to the application
        app_path = os.path.abspath(os.path.dirname(__file__))

        # Create systemd service file
        service_content = f"""[Unit]
Description=Improved Local AI Assistant
After=network.target

[Service]
Type=simple
User={os.getlogin()}
WorkingDirectory={app_path}
ExecStart={app_path}/.venv/bin/python {app_path}/app/main.py
Restart=on-failure
RestartSec=5
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=local-ai-assistant

[Install]
WantedBy=multi-user.target
"""

        # Write service file
        service_path = os.path.join(app_path, "local-ai-assistant.service")
        with open(service_path, "w") as f:
            f.write(service_content)

        logger.info(f"Systemd service file created at {service_path}")
        logger.info("To install the service, run:")
        logger.info(f"  sudo cp {service_path} /etc/systemd/system/")
        logger.info("  sudo systemctl daemon-reload")
        logger.info("  sudo systemctl enable local-ai-assistant")
        logger.info("  sudo systemctl start local-ai-assistant")

        return True
    except Exception as e:
        logger.error(f"Error creating systemd service file: {str(e)}")
        return False


def create_windows_service() -> bool:
    """
    Create Windows service.

    Returns:
        bool: True if service was created successfully
    """
    try:
        # Get absolute path to the application
        app_path = os.path.abspath(os.path.dirname(__file__))

        # Create batch file for the service
        batch_content = f"""@echo off
cd /d {app_path}
.\\venv\\Scripts\\python app\\main.py
"""

        # Write batch file
        batch_path = os.path.join(app_path, "service.bat")
        with open(batch_path, "w") as f:
            f.write(batch_content)

        # Create NSSM command to install the service
        nssm_cmd = f"""
To install as a Windows service, download NSSM from https://nssm.cc/ and run:

nssm install LocalAIAssistant "{batch_path}"
nssm set LocalAIAssistant DisplayName "Local AI Assistant"
nssm set LocalAIAssistant Description "Improved Local AI Assistant"
nssm set LocalAIAssistant Start SERVICE_AUTO_START
nssm start LocalAIAssistant
"""

        # Write NSSM instructions
        nssm_path = os.path.join(app_path, "install_service.txt")
        with open(nssm_path, "w") as f:
            f.write(nssm_cmd)

        logger.info(f"Windows service batch file created at {batch_path}")
        logger.info(f"Service installation instructions written to {nssm_path}")

        return True
    except Exception as e:
        logger.error(f"Error creating Windows service: {str(e)}")
        return False


def prepare_for_iphone() -> bool:
    """
    Prepare API for iPhone integration.

    Returns:
        bool: True if preparation was successful
    """
    try:
        # Create iPhone API documentation
        api_docs = {
            "api_version": "1.0.0",
            "base_url": "http://localhost:3000",
            "endpoints": [
                {
                    "path": "/api/chat",
                    "method": "POST",
                    "description": "Send a chat message",
                    "request_body": {"message": "string", "session_id": "string (optional)"},
                    "response": {"response": "string", "session_id": "string"},
                },
                {
                    "path": "/api/session/{session_id}",
                    "method": "GET",
                    "description": "Get session information",
                    "response": {
                        "session_id": "string",
                        "created_at": "string (ISO datetime)",
                        "updated_at": "string (ISO datetime)",
                        "message_count": "integer",
                        "has_summary": "boolean",
                    },
                },
                {
                    "path": "/ws/{session_id}",
                    "method": "WebSocket",
                    "description": "Real-time chat via WebSocket",
                    "messages": {
                        "incoming": "string (chat message)",
                        "outgoing": {
                            "text": "string (response token)",
                            "json": {
                                "type": "string (system, typing, knowledge_graph, error, etc.)",
                                "data": "object (depends on type)",
                            },
                        },
                    },
                },
            ],
            "authentication": "None (local network only)",
            "notes": "This API is designed for local network use only. For security reasons, it should not be exposed to the internet.",
        }

        # Write API documentation
        with open("iphone_api_docs.json", "w") as f:
            json.dump(api_docs, f, indent=2)

        # Create Swift example code
        swift_example = """import Foundation
import Combine

class LocalAIAssistant {
    private let baseURL = "http://localhost:3000"
    private var sessionID: String?
    private var cancellables = Set<AnyCancellable>()

    func sendMessage(message: String, completion: @escaping (Result<String, Error>) -> Void) {
        guard let url = URL(string: "\\(baseURL)/api/chat") else {
            completion(.failure(NSError(domain: "Invalid URL", code: 0)))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "message": message,
            "session_id": sessionID as Any
        ]

        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        URLSession.shared.dataTaskPublisher(for: request)
            .map { $0.data }
            .decode(type: ChatResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { completion in
                    if case let .failure(error) = completion {
                        completion(.failure(error))
                    }
                },
                receiveValue: { response in
                    self.sessionID = response.session_id
                    completion(.success(response.response))
                }
            )
            .store(in: &cancellables)
    }
}

struct ChatResponse: Codable {
    let response: String
    let session_id: String
}
"""

        # Write Swift example
        with open("iphone_example.swift", "w") as f:
            f.write(swift_example)

        logger.info("iPhone API documentation created at iphone_api_docs.json")
        logger.info("Swift example code created at iphone_example.swift")

        return True
    except Exception as e:
        logger.error(f"Error preparing for iPhone: {str(e)}")
        return False


def optimize_for_platform(platform_name: str) -> bool:
    """
    Optimize the application for the specified platform.

    Args:
        platform_name: Platform name

    Returns:
        bool: True if optimization was successful
    """
    try:
        logger.info(f"Optimizing for platform: {platform_name}")

        if platform_name == "windows":
            # Windows-specific optimizations
            logger.info("Applying Windows-specific optimizations")

            # Create Windows service
            create_windows_service()

        elif platform_name == "raspberry_pi":
            # Raspberry Pi-specific optimizations
            logger.info("Applying Raspberry Pi-specific optimizations")

            # Create systemd service
            create_systemd_service()

            # Create thermal management script
            with open("thermal_management.sh", "w") as f:
                f.write(
                    """#!/bin/bash
# Thermal management script for Raspberry Pi
while true; do
    temp=$(vcgencmd measure_temp | cut -d= -f2 | cut -d\\' -f1)
    if (( $(echo "$temp > 75" | bc -l) )); then
        echo "Temperature too high: ${temp}°C, throttling CPU"
        sudo cpufreq-set -g powersave
    elif (( $(echo "$temp < 65" | bc -l) )); then
        echo "Temperature normal: ${temp}°C, restoring CPU"
        sudo cpufreq-set -g ondemand
    fi
    sleep 30
done
"""
                )
            os.chmod("thermal_management.sh", 0o755)

            logger.info("Created thermal management script: thermal_management.sh")
            logger.info("To use it, install cpufrequtils and run: sudo ./thermal_management.sh &")

        elif platform_name == "iphone":
            # iPhone-specific optimizations
            logger.info("Applying iPhone-specific optimizations")

            # Prepare API for iPhone integration
            prepare_for_iphone()

        else:
            logger.info(f"No specific optimizations for platform {platform_name}")

        return True
    except Exception as e:
        logger.error(f"Error optimizing for platform: {str(e)}")
        return False


def create_deployment_package(platform_name: str, output_dir: str) -> bool:
    """
    Create deployment package for the specified platform.

    Args:
        platform_name: Platform name
        output_dir: Output directory

    Returns:
        bool: True if package was created successfully
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create package directory
        package_dir = os.path.join(output_dir, f"local-ai-assistant-{platform_name}")
        os.makedirs(package_dir, exist_ok=True)

        # Copy required files
        required_dirs = ["app", "services", "data", "cli"]

        required_files = ["config.yaml", "requirements.txt", ".env.example"]

        # Copy directories
        for directory in required_dirs:
            if os.path.exists(directory):
                shutil.copytree(directory, os.path.join(package_dir, directory), dirs_exist_ok=True)

        # Copy files
        for file in required_files:
            if os.path.exists(file):
                shutil.copy2(file, os.path.join(package_dir, file))

        # Create platform-specific files
        if platform_name == "windows":
            # Create Windows batch file
            with open(os.path.join(package_dir, "start.bat"), "w") as f:
                f.write(
                    """@echo off
echo Installing dependencies...
python -m venv .venv
.\\venv\\Scripts\\pip install -r requirements.txt
echo Starting Local AI Assistant...
.\\venv\\Scripts\\python app\\main.py
pause
"""
                )

        elif platform_name == "raspberry_pi" or platform_name == "linux":
            # Create shell script
            with open(os.path.join(package_dir, "start.sh"), "w") as f:
                f.write(
                    """#!/bin/bash
echo "Installing dependencies..."
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
echo "Starting Local AI Assistant..."
./.venv/bin/python app/main.py
"""
                )
            os.chmod(os.path.join(package_dir, "start.sh"), 0o755)

            # Create systemd service file
            with open(os.path.join(package_dir, "install_service.sh"), "w") as f:
                f.write(
                    """#!/bin/bash
echo "Installing Local AI Assistant as a service..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SERVICE_FILE="$SCRIPT_DIR/local-ai-assistant.service"

# Create service file
cat > "$SERVICE_FILE" << EOL
[Unit]
Description=Improved Local AI Assistant
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/.venv/bin/python $SCRIPT_DIR/app/main.py
Restart=on-failure
RestartSec=5
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=local-ai-assistant

[Install]
WantedBy=multi-user.target
EOL

echo "Service file created at $SERVICE_FILE"
echo "To install the service, run:"
echo "  sudo cp $SERVICE_FILE /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable local-ai-assistant"
echo "  sudo systemctl start local-ai-assistant"
"""
                )
            os.chmod(os.path.join(package_dir, "install_service.sh"), 0o755)

        elif platform_name == "iphone":
            # Create iPhone-specific files
            prepare_for_iphone()
            if os.path.exists("iphone_api_docs.json"):
                shutil.copy2(
                    "iphone_api_docs.json", os.path.join(package_dir, "iphone_api_docs.json")
                )
            if os.path.exists("iphone_example.swift"):
                shutil.copy2(
                    "iphone_example.swift", os.path.join(package_dir, "iphone_example.swift")
                )

        # Create README file
        with open(os.path.join(package_dir, "README.md"), "w") as f:
            f.write(
                f"""# Improved Local AI Assistant - {platform_name.capitalize()} Deployment

This package contains the Improved Local AI Assistant optimized for {platform_name}.

## Requirements

- Python 3.8 or higher
- Ollama with hermes3:3b and tinyllama models

## Installation

1. Install Ollama from https://ollama.ai/
2. Pull the required models:
   ```
   ollama pull hermes3:3b
   ollama pull tinyllama
   ```

## Starting the Application

"""
            )

            if platform_name == "windows":
                f.write(
                    """Run `start.bat` to install dependencies and start the application.

Alternatively, you can:
1. Create a virtual environment: `python -m venv .venv`
2. Activate it: `.\\venv\\Scripts\\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Start the application: `python app\\main.py`
"""
                )
            elif platform_name == "raspberry_pi" or platform_name == "linux":
                f.write(
                    """Run `./start.sh` to install dependencies and start the application.

Alternatively, you can:
1. Create a virtual environment: `python3 -m venv .venv`
2. Activate it: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Start the application: `python app/main.py`

To install as a service, run `./install_service.sh` and follow the instructions.
"""
                )
            elif platform_name == "iphone":
                f.write(
                    """This package provides the server component for iPhone integration.

1. Create a virtual environment: `python3 -m venv .venv`
2. Activate it: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Start the application: `python app/main.py`

See `iphone_api_docs.json` for API documentation and `iphone_example.swift` for example Swift code.
"""
                )

        logger.info(f"Deployment package created at {package_dir}")
        return True
    except Exception as e:
        logger.error(f"Error creating deployment package: {str(e)}")
        return False


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy the Improved Local AI Assistant")
    parser.add_argument(
        "--platform",
        choices=["windows", "raspberry_pi", "linux", "macos", "iphone", "auto"],
        default="auto",
        help="Target platform",
    )
    parser.add_argument("--optimize", action="store_true", help="Optimize for the target platform")
    parser.add_argument("--package", action="store_true", help="Create deployment package")
    parser.add_argument(
        "--output-dir", default="dist", help="Output directory for deployment package"
    )
    args = parser.parse_args()

    logger.info("Starting deployment of Improved Local AI Assistant")

    # Detect platform if auto
    platform_name = args.platform
    if platform_name == "auto":
        platform_name = detect_platform()
        logger.info(f"Detected platform: {platform_name}")

    # Update configuration for platform
    if not update_config_for_platform(platform_name):
        logger.error("Deployment failed: Could not update configuration")
        return 1

    # Optimize for platform
    if args.optimize and not optimize_for_platform(platform_name):
        logger.error("Deployment failed: Could not optimize for platform")
        return 1

    # Create deployment package
    if args.package and not create_deployment_package(platform_name, args.output_dir):
        logger.error("Deployment failed: Could not create deployment package")
        return 1

    logger.info("Deployment completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
