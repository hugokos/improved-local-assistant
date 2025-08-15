"""
Configuration validation and management.

This module provides mechanisms for validating and managing configuration,
including environment-specific configurations and default values.
"""

import json
import logging
import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List

import yaml

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ConfigValidationError:
    """Configuration validation error."""

    path: str
    message: str
    severity: str = "error"  # "error", "warning", or "info"


@dataclass
class ConfigValidationResult:
    """Configuration validation result."""

    valid: bool
    errors: List[ConfigValidationError] = field(default_factory=list)
    warnings: List[ConfigValidationError] = field(default_factory=list)


class ConfigValidator:
    """
    Validates configuration against a schema.

    This class provides methods for validating configuration against a schema,
    including type checking, required fields, and value constraints.
    """

    def __init__(self):
        """Initialize the configuration validator."""
        # Define the configuration schema
        self.schema = {
            "ollama": {
                "type": "dict",
                "required": True,
                "schema": {
                    "host": {"type": "string", "required": True},
                    "timeout": {"type": "number", "required": False, "default": 120},
                    "max_parallel": {"type": "integer", "required": False, "default": 2},
                    "max_loaded_models": {"type": "integer", "required": False, "default": 2},
                },
            },
            "models": {
                "type": "dict",
                "required": True,
                "schema": {
                    "conversation": {
                        "type": "dict",
                        "required": True,
                        "schema": {
                            "name": {"type": "string", "required": True},
                            "context_window": {
                                "type": "integer",
                                "required": False,
                                "default": 8000,
                            },
                            "temperature": {"type": "number", "required": False, "default": 0.7},
                            "max_tokens": {"type": "integer", "required": False, "default": 2048},
                        },
                    },
                    "knowledge": {
                        "type": "dict",
                        "required": True,
                        "schema": {
                            "name": {"type": "string", "required": True},
                            "context_window": {
                                "type": "integer",
                                "required": False,
                                "default": 2048,
                            },
                            "temperature": {"type": "number", "required": False, "default": 0.2},
                            "max_tokens": {"type": "integer", "required": False, "default": 1024},
                        },
                    },
                },
            },
            "knowledge_graphs": {
                "type": "dict",
                "required": True,
                "schema": {
                    "prebuilt_directory": {
                        "type": "string",
                        "required": False,
                        "default": "./data/prebuilt_graphs",
                    },
                    "dynamic_storage": {
                        "type": "string",
                        "required": False,
                        "default": "./data/dynamic_graph",
                    },
                    "max_triplets_per_chunk": {"type": "integer", "required": False, "default": 4},
                    "enable_visualization": {"type": "boolean", "required": False, "default": True},
                },
            },
            "conversation": {
                "type": "dict",
                "required": True,
                "schema": {
                    "max_history_length": {"type": "integer", "required": False, "default": 50},
                    "summarize_threshold": {"type": "integer", "required": False, "default": 20},
                    "context_window_tokens": {
                        "type": "integer",
                        "required": False,
                        "default": 8000,
                    },
                },
            },
            "system": {
                "type": "dict",
                "required": True,
                "schema": {
                    "max_memory_gb": {"type": "number", "required": False, "default": 12},
                    "cpu_cores": {"type": "integer", "required": False, "default": None},
                    "memory_threshold_percent": {
                        "type": "number",
                        "required": False,
                        "default": 80,
                    },
                    "cpu_threshold_percent": {"type": "number", "required": False, "default": 80},
                },
            },
            "api": {
                "type": "dict",
                "required": False,
                "schema": {
                    "host": {"type": "string", "required": False, "default": "0.0.0.0"},
                    "port": {"type": "integer", "required": False, "default": 8000},
                    "cors_origins": {"type": "list", "required": False, "default": ["*"]},
                },
            },
            "environment": {
                "type": "string",
                "required": False,
                "default": "development",
                "allowed": ["development", "testing", "production"],
            },
        }

        logger.info("Configuration validator initialized")

    def validate(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """
        Validate configuration against the schema.

        Args:
            config: Configuration dictionary

        Returns:
            ConfigValidationResult: Validation result
        """
        errors = []
        warnings = []

        # Validate configuration against schema
        self._validate_dict(config, self.schema, "", errors, warnings)

        # Check for unknown fields
        self._check_unknown_fields(config, self.schema, "", warnings)

        # Check for environment-specific configurations
        if "environment" in config:
            environment = config["environment"]
            env_key = f"environment_{environment}"
            if env_key in config:
                # Environment-specific configuration exists
                env_config = config[env_key]
                if not isinstance(env_config, dict):
                    errors.append(
                        ConfigValidationError(
                            path=env_key,
                            message="Environment-specific configuration must be a dictionary",
                        )
                    )

        # Return validation result
        valid = len(errors) == 0
        return ConfigValidationResult(valid=valid, errors=errors, warnings=warnings)

    def _validate_dict(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any],
        path: str,
        errors: List[ConfigValidationError],
        warnings: List[ConfigValidationError],
    ) -> None:
        """
        Validate a dictionary against a schema.

        Args:
            config: Configuration dictionary
            schema: Schema dictionary
            path: Current path in the configuration
            errors: List to collect errors
            warnings: List to collect warnings
        """
        # Check required fields
        for key, field_schema in schema.items():
            field_path = f"{path}.{key}" if path else key

            if field_schema.get("required", False) and key not in config:
                # Check if there's a default value
                if "default" in field_schema:
                    warnings.append(
                        ConfigValidationError(
                            path=field_path,
                            message=f"Missing required field, using default value: {field_schema['default']}",
                            severity="warning",
                        )
                    )
                else:
                    errors.append(
                        ConfigValidationError(path=field_path, message="Missing required field")
                    )

            # If field exists, validate it
            if key in config:
                field_value = config[key]
                field_type = field_schema.get("type")

                # Check type
                if field_type == "dict":
                    if not isinstance(field_value, dict):
                        errors.append(
                            ConfigValidationError(
                                path=field_path,
                                message=f"Expected dictionary, got {type(field_value).__name__}",
                            )
                        )
                    elif "schema" in field_schema:
                        # Recursively validate nested dictionary
                        self._validate_dict(
                            field_value, field_schema["schema"], field_path, errors, warnings
                        )

                elif field_type == "list":
                    if not isinstance(field_value, list):
                        errors.append(
                            ConfigValidationError(
                                path=field_path,
                                message=f"Expected list, got {type(field_value).__name__}",
                            )
                        )
                    elif "schema" in field_schema:
                        # Validate each item in the list
                        for i, item in enumerate(field_value):
                            item_path = f"{field_path}[{i}]"
                            self._validate_value(
                                item, field_schema["schema"], item_path, errors, warnings
                            )

                elif field_type == "string":
                    if not isinstance(field_value, str):
                        errors.append(
                            ConfigValidationError(
                                path=field_path,
                                message=f"Expected string, got {type(field_value).__name__}",
                            )
                        )
                    elif "allowed" in field_schema and field_value not in field_schema["allowed"]:
                        errors.append(
                            ConfigValidationError(
                                path=field_path,
                                message=f"Value must be one of: {', '.join(field_schema['allowed'])}",
                            )
                        )

                elif field_type == "integer":
                    if not isinstance(field_value, int):
                        errors.append(
                            ConfigValidationError(
                                path=field_path,
                                message=f"Expected integer, got {type(field_value).__name__}",
                            )
                        )
                    elif "min" in field_schema and field_value < field_schema["min"]:
                        errors.append(
                            ConfigValidationError(
                                path=field_path,
                                message=f"Value must be at least {field_schema['min']}",
                            )
                        )
                    elif "max" in field_schema and field_value > field_schema["max"]:
                        errors.append(
                            ConfigValidationError(
                                path=field_path,
                                message=f"Value must be at most {field_schema['max']}",
                            )
                        )

                elif field_type == "number":
                    if not isinstance(field_value, (int, float)):
                        errors.append(
                            ConfigValidationError(
                                path=field_path,
                                message=f"Expected number, got {type(field_value).__name__}",
                            )
                        )
                    elif "min" in field_schema and field_value < field_schema["min"]:
                        errors.append(
                            ConfigValidationError(
                                path=field_path,
                                message=f"Value must be at least {field_schema['min']}",
                            )
                        )
                    elif "max" in field_schema and field_value > field_schema["max"]:
                        errors.append(
                            ConfigValidationError(
                                path=field_path,
                                message=f"Value must be at most {field_schema['max']}",
                            )
                        )

                elif field_type == "boolean":
                    if not isinstance(field_value, bool):
                        errors.append(
                            ConfigValidationError(
                                path=field_path,
                                message=f"Expected boolean, got {type(field_value).__name__}",
                            )
                        )

    def _validate_value(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[ConfigValidationError],
        warnings: List[ConfigValidationError],
    ) -> None:
        """
        Validate a value against a schema.

        Args:
            value: Value to validate
            schema: Schema dictionary
            path: Current path in the configuration
            errors: List to collect errors
            warnings: List to collect warnings
        """
        field_type = schema.get("type")

        if field_type == "dict":
            if not isinstance(value, dict):
                errors.append(
                    ConfigValidationError(
                        path=path, message=f"Expected dictionary, got {type(value).__name__}"
                    )
                )
            elif "schema" in schema:
                # Recursively validate nested dictionary
                self._validate_dict(value, schema["schema"], path, errors, warnings)

        elif field_type == "list":
            if not isinstance(value, list):
                errors.append(
                    ConfigValidationError(
                        path=path, message=f"Expected list, got {type(value).__name__}"
                    )
                )
            elif "schema" in schema:
                # Validate each item in the list
                for i, item in enumerate(value):
                    item_path = f"{path}[{i}]"
                    self._validate_value(item, schema["schema"], item_path, errors, warnings)

        elif field_type == "string":
            if not isinstance(value, str):
                errors.append(
                    ConfigValidationError(
                        path=path, message=f"Expected string, got {type(value).__name__}"
                    )
                )
            elif "allowed" in schema and value not in schema["allowed"]:
                errors.append(
                    ConfigValidationError(
                        path=path, message=f"Value must be one of: {', '.join(schema['allowed'])}"
                    )
                )

        elif field_type == "integer":
            if not isinstance(value, int):
                errors.append(
                    ConfigValidationError(
                        path=path, message=f"Expected integer, got {type(value).__name__}"
                    )
                )
            elif "min" in schema and value < schema["min"]:
                errors.append(
                    ConfigValidationError(
                        path=path, message=f"Value must be at least {schema['min']}"
                    )
                )
            elif "max" in schema and value > schema["max"]:
                errors.append(
                    ConfigValidationError(
                        path=path, message=f"Value must be at most {schema['max']}"
                    )
                )

        elif field_type == "number":
            if not isinstance(value, (int, float)):
                errors.append(
                    ConfigValidationError(
                        path=path, message=f"Expected number, got {type(value).__name__}"
                    )
                )
            elif "min" in schema and value < schema["min"]:
                errors.append(
                    ConfigValidationError(
                        path=path, message=f"Value must be at least {schema['min']}"
                    )
                )
            elif "max" in schema and value > schema["max"]:
                errors.append(
                    ConfigValidationError(
                        path=path, message=f"Value must be at most {schema['max']}"
                    )
                )

        elif field_type == "boolean":
            if not isinstance(value, bool):
                errors.append(
                    ConfigValidationError(
                        path=path, message=f"Expected boolean, got {type(value).__name__}"
                    )
                )

    def _check_unknown_fields(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any],
        path: str,
        warnings: List[ConfigValidationError],
    ) -> None:
        """
        Check for unknown fields in the configuration.

        Args:
            config: Configuration dictionary
            schema: Schema dictionary
            path: Current path in the configuration
            warnings: List to collect warnings
        """
        for key in config:
            field_path = f"{path}.{key}" if path else key

            # Skip environment-specific configurations
            if key.startswith("environment_"):
                continue

            if key not in schema:
                warnings.append(
                    ConfigValidationError(
                        path=field_path, message="Unknown field", severity="warning"
                    )
                )
            elif isinstance(config[key], dict) and "schema" in schema.get(key, {}):
                # Recursively check nested dictionaries
                self._check_unknown_fields(config[key], schema[key]["schema"], field_path, warnings)

    def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values to missing fields.

        Args:
            config: Configuration dictionary

        Returns:
            Dict[str, Any]: Configuration with defaults applied
        """
        return self._apply_defaults_to_dict(config, self.schema)

    def _apply_defaults_to_dict(
        self, config: Dict[str, Any], schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply default values to missing fields in a dictionary.

        Args:
            config: Configuration dictionary
            schema: Schema dictionary

        Returns:
            Dict[str, Any]: Configuration with defaults applied
        """
        result = config.copy()

        # Apply defaults to missing fields
        for key, field_schema in schema.items():
            if key not in result and "default" in field_schema:
                result[key] = field_schema["default"]

            # Recursively apply defaults to nested dictionaries
            if (
                key in result
                and isinstance(result[key], dict)
                and field_schema.get("type") == "dict"
                and "schema" in field_schema
            ):
                result[key] = self._apply_defaults_to_dict(result[key], field_schema["schema"])

        return result

    def merge_environment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge environment-specific configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Dict[str, Any]: Merged configuration
        """
        result = config.copy()

        # Check for environment-specific configuration
        if "environment" in result:
            environment = result["environment"]
            env_key = f"environment_{environment}"

            if env_key in result and isinstance(result[env_key], dict):
                # Merge environment-specific configuration
                env_config = result[env_key]

                # Deep merge
                result = self._deep_merge(result, env_config)

                # Remove environment-specific configuration
                del result[env_key]

        return result

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Dict[str, Any]: Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            # Skip environment-specific configurations
            if key.startswith("environment_"):
                continue

            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = value

        return result


class ConfigManager:
    """
    Manages configuration loading, validation, and application.

    This class provides methods for loading configuration from files,
    validating it against a schema, and applying environment-specific
    configurations and default values.
    """

    def __init__(self):
        """Initialize the configuration manager."""
        self.validator = ConfigValidator()
        self.config = {}
        logger.info("Configuration manager initialized")

    def load_config(self, config_file: str = "config.yaml") -> Dict[str, Any]:
        """
        Load configuration from a file.

        Args:
            config_file: Path to configuration file

        Returns:
            Dict[str, Any]: Loaded configuration
        """
        try:
            # Check if file exists
            if not os.path.exists(config_file):
                logger.warning(f"Configuration file {config_file} not found")
                return {}

            # Load configuration from file
            with open(config_file) as f:
                if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                    config = yaml.safe_load(f)
                elif config_file.endswith(".json"):
                    config = json.load(f)
                else:
                    logger.error(f"Unsupported configuration file format: {config_file}")
                    return {}

            logger.info(f"Loaded configuration from {config_file}")
            return config

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}

    def validate_config(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """
        Validate configuration.

        Args:
            config: Configuration dictionary

        Returns:
            ConfigValidationResult: Validation result
        """
        return self.validator.validate(config)

    def process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process configuration by validating, applying defaults, and merging environment-specific configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Dict[str, Any]: Processed configuration
        """
        # Validate configuration
        validation_result = self.validate_config(config)

        # Log validation results
        if not validation_result.valid:
            for error in validation_result.errors:
                logger.error(f"Configuration error: {error.path}: {error.message}")

        for warning in validation_result.warnings:
            logger.warning(f"Configuration warning: {warning.path}: {warning.message}")

        # Apply defaults
        config_with_defaults = self.validator.apply_defaults(config)

        # Merge environment-specific configuration
        merged_config = self.validator.merge_environment_config(config_with_defaults)

        # Store processed configuration
        self.config = merged_config

        return merged_config

    def load_and_process_config(self, config_file: str = "config.yaml") -> Dict[str, Any]:
        """
        Load and process configuration from a file.

        Args:
            config_file: Path to configuration file

        Returns:
            Dict[str, Any]: Processed configuration
        """
        # Load configuration
        config = self.load_config(config_file)

        # Process configuration
        processed_config = self.process_config(config)

        return processed_config

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            Dict[str, Any]: Current configuration
        """
        return self.config

    def get_value(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by path.

        Args:
            path: Configuration path (e.g., "models.conversation.name")
            default: Default value if path not found

        Returns:
            Any: Configuration value
        """
        parts = path.split(".")
        value = self.config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set_value(self, path: str, value: Any) -> None:
        """
        Set a configuration value by path.

        Args:
            path: Configuration path (e.g., "models.conversation.name")
            value: Value to set
        """
        parts = path.split(".")
        config = self.config

        # Navigate to the parent of the target field
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
            config = config[part]

        # Set the value
        config[parts[-1]] = value

    def save_config(self, config_file: str = "config.yaml") -> bool:
        """
        Save configuration to a file.

        Args:
            config_file: Path to configuration file

        Returns:
            bool: True if save was successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)

            # Save configuration to file
            with open(config_file, "w") as f:
                if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                    yaml.dump(self.config, f, default_flow_style=False)
                elif config_file.endswith(".json"):
                    json.dump(self.config, f, indent=2)
                else:
                    logger.error(f"Unsupported configuration file format: {config_file}")
                    return False

            logger.info(f"Saved configuration to {config_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False


# Global configuration manager instance
config_manager = ConfigManager()


def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load and process configuration using the global configuration manager.

    Args:
        config_file: Path to configuration file

    Returns:
        Dict[str, Any]: Processed configuration
    """
    return config_manager.load_and_process_config(config_file)


def get_config() -> Dict[str, Any]:
    """
    Get the current configuration using the global configuration manager.

    Returns:
        Dict[str, Any]: Current configuration
    """
    return config_manager.get_config()


def get_value(path: str, default: Any = None) -> Any:
    """
    Get a configuration value by path using the global configuration manager.

    Args:
        path: Configuration path (e.g., "models.conversation.name")
        default: Default value if path not found

    Returns:
        Any: Configuration value
    """
    return config_manager.get_value(path, default)
