"""
Settings management for Improved Local Assistant
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


class Settings(BaseModel):
    """Application settings with environment variable support."""

    # Server settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # Ollama settings
    ollama_host: str = Field(default="http://127.0.0.1:11434", description="Ollama server URL")
    model_chat: str = Field(default="hermes3:3b", description="Chat/inference model")
    model_embed: str = Field(default="nomic-embed-text", description="Embedding model")

    # Storage settings
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    prebuilt_dir: Path = Field(
        default=Path("./data/prebuilt_graphs"), description="Prebuilt graphs directory"
    )

    # Router settings
    use_graph: bool = Field(default=True, description="Enable graph retrieval")
    use_vector: bool = Field(default=True, description="Enable vector retrieval")
    use_bm25: bool = Field(default=True, description="Enable BM25 retrieval")
    router_graph_weight: float = Field(default=0.5, description="Graph retrieval weight")
    router_vector_weight: float = Field(default=0.4, description="Vector retrieval weight")
    router_bm25_weight: float = Field(default=0.1, description="BM25 retrieval weight")

    # Performance settings
    edge_optimization: bool = Field(default=True, description="Enable edge optimizations")
    max_concurrent_sessions: int = Field(default=10, description="Max concurrent sessions")

    # Voice settings
    voice_enabled: bool = Field(default=True, description="Enable voice interface")
    stt_model: str = Field(default="vosk-model-en-us-0.22", description="STT model")
    tts_voice: str = Field(default="en_US-lessac-medium", description="TTS voice")

    class Config:
        env_prefix = "ILA_"
        case_sensitive = False

    @classmethod
    def from_env(
        cls, env_key: str = "ILA_CONFIG", default_path: str = "configs/base.yaml"
    ) -> "Settings":
        """Load settings from environment and config file."""
        config_path = Path(os.getenv(env_key, default_path))

        # Start with defaults
        data = {}

        # Load from config file if it exists
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    file_data = yaml.safe_load(f) or {}
                data.update(file_data)
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        else:
            logger.info(f"Config file {config_path} not found, using defaults")

        # Environment variables override config file
        env_overrides = {}
        for key, value in os.environ.items():
            if key.startswith("ILA_"):
                config_key = key[4:].lower()  # Remove ILA_ prefix and lowercase
                env_overrides[config_key] = value

        if env_overrides:
            data.update(env_overrides)
            logger.info(f"Applied environment overrides: {list(env_overrides.keys())}")

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary."""
        return self.dict()


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment and config."""
    global _settings
    _settings = Settings.from_env()
    return _settings
