"""
Service initialization for the Improved Local AI Assistant.

This module provides functions to initialize all required services
and register startup/shutdown events.
"""

import asyncio
import gc
import json
import logging
import os
import time
from typing import Any
from typing import Dict

from fastapi import FastAPI
from services import ConversationManager
from services import KnowledgeGraphManager
from services import ModelConfig
from services import ModelManager
from services import SystemMonitor

logger = logging.getLogger(__name__)


async def load_graphs_background(kg_manager: KnowledgeGraphManager, config: Dict[str, Any]) -> None:
    """Load knowledge graphs in the background to avoid blocking startup."""
    try:
        if not kg_manager:
            return

        logger.info("Starting background graph loading...")

        # Load pre-built graphs
        kg_dir = config.get("knowledge_graphs", {}).get(
            "prebuilt_directory", "./data/prebuilt_graphs"
        )

        # Run in thread to avoid blocking event loop
        loaded_graphs = await asyncio.to_thread(kg_manager.load_prebuilt_graphs, kg_dir)

        logger.info(f"Background loading completed: {len(loaded_graphs)} graphs loaded")

    except Exception as e:
        logger.error(f"Error in background graph loading: {str(e)}")


def create_singleton_embedding_model(config: Dict[str, Any]):
    """Create a singleton embedding model to avoid reloading."""
    embedding_config = config.get("embedding", {})

    if not embedding_config.get("singleton", True):
        return None

    try:
        import torch
        from sentence_transformers import SentenceTransformer

        model_path = embedding_config.get("model_path", "BAAI/bge-small-en-v1.5")
        device = embedding_config.get("device", "cpu")

        # Create model with optimizations and offline support
        try:
            # Try local first, then online
            model = SentenceTransformer(model_path, device=device, local_files_only=False)
        except Exception as e:
            logger.warning(f"Failed to load embedding model {model_path}: {e}")
            logger.info("Falling back to basic embedding model")
            # Fallback to a simpler model or disable embeddings
            return None

        # Apply int8 quantization if requested
        if embedding_config.get("int8_quantization", False):
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Applied int8 quantization to embedding model")
            except Exception as e:
                logger.warning(f"Failed to apply quantization: {e}")

        logger.info(f"Created singleton embedding model: {model_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to create singleton embedding model: {e}")
        return None


# Global singleton embedding model
_EMBEDDING_MODEL = None


async def initialize_services(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize all services required by the application.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary containing initialized service instances
    """
    global _EMBEDDING_MODEL
    services = {}
    startup_mode = config.get("startup", {}).get("preload", "models")

    try:
        # Create singleton embedding model in background to avoid CPU spike
        if _EMBEDDING_MODEL is None:
            # Load embedding model in background thread to avoid blocking startup
            embedding_task = asyncio.create_task(
                asyncio.to_thread(create_singleton_embedding_model, config)
            )
            _EMBEDDING_MODEL = await embedding_task

            # Set the embedding model in LlamaIndex Settings to prevent OpenAI fallback
            if _EMBEDDING_MODEL is not None:
                try:
                    from llama_index.core import Settings
                    from services.embedding_singleton import get_embedding_model

                    # Use the singleton embedding model to configure LlamaIndex globally
                    Settings.embed_model = get_embedding_model("BAAI/bge-small-en-v1.5")
                    logger.info("Set singleton embedding model in LlamaIndex Settings")
                except Exception as e:
                    logger.warning(f"Could not set embedding model in Settings: {e}")

        # Resource manager with optimized settings
        logger.info("Initializing ResourceManager...")
        from services.resource_manager import resource_manager
        from services.resource_manager import start_resource_monitoring

        await start_resource_monitoring(config)
        services["resource_manager"] = resource_manager

        # Model manager with optional orchestration
        logger.info("Initializing ModelManager...")
        ollama_config = config.get("ollama", {})
        host = ollama_config.get("host", "http://localhost:11434")

        # Check if edge optimization is enabled
        use_orchestration = config.get("edge_optimization", {}).get("enabled", False)

        if use_orchestration:
            logger.info("Using orchestrated model manager for edge optimization")
            from services.orchestrated_model_manager import create_model_manager

            try:
                model_manager = create_model_manager(config, use_orchestration=True)
            except Exception as e:
                import sys
                import traceback

                logger.error(f"Failed to create orchestrated model manager: {e}")
                logger.error("Full traceback:")
                traceback.print_exc(file=sys.stderr)
                logger.info("Falling back to standard model manager")
                healthcheck_mode = config.get("startup", {}).get("ollama_healthcheck", "version")
                model_manager = ModelManager(host=host, healthcheck_mode=healthcheck_mode)
        else:
            logger.info("Using standard model manager")
            # Use light health check instead of heavy chat test
            healthcheck_mode = config.get("startup", {}).get("ollama_healthcheck", "version")
            model_manager = ModelManager(host=host, healthcheck_mode=healthcheck_mode)

        services["model_manager"] = model_manager

        # Only initialize models if preload includes models
        if startup_mode in ["models", "all"]:
            model_cfg = config.get("models", {}).get("conversation", {})
            model_config = ModelConfig(
                name=model_cfg.get("name", "hermes3:3b"),
                type="conversation",
                context_window=model_cfg.get("context_window", 8000),
                temperature=model_cfg.get("temperature", 0.7),
                max_tokens=model_cfg.get("max_tokens", 2048),
                timeout=ollama_config.get("timeout", 120),
                max_parallel=ollama_config.get("max_parallel", 2),
                max_loaded=ollama_config.get("max_loaded_models", 2),
            )

            await model_manager.initialize_models(model_config)
        else:
            logger.info("Skipping model initialization (lazy loading enabled)")

        # Knowledge graph manager (no immediate graph loading)
        logger.info("Initializing KnowledgeGraphManager...")
        kg_manager = KnowledgeGraphManager(model_manager=model_manager, config=config)
        services["kg_manager"] = kg_manager

        # Only load graphs immediately if not lazy loading
        if not config.get("startup", {}).get("lazy_load_graphs", True) and startup_mode in [
            "graphs",
            "all",
        ]:
            kg_dir = config.get("knowledge_graphs", {}).get(
                "prebuilt_directory", "./data/prebuilt_graphs"
            )
            kg_manager.load_prebuilt_graphs(kg_dir)
        else:
            logger.info("Graph loading deferred to background task")

        # Lazy initialize optimizer and summarizer
        if startup_mode == "all":
            logger.info("Initializing KnowledgeGraphOptimizer...")
            from services.kg_optimizer import initialize_optimizer

            initialize_optimizer(kg_manager)

            logger.info("Initializing ConversationSummarizer...")
            from services.conversation_summarizer import initialize_summarizer

            initialize_summarizer(model_manager)
        else:
            logger.info("Optimizer and summarizer initialization deferred")

        # Conversation manager
        logger.info("Initializing ConversationManager...")
        conversation_manager = ConversationManager(
            model_manager=model_manager, kg_manager=kg_manager, config=config
        )
        services["conversation_manager"] = conversation_manager

        # System monitor with optimized settings
        logger.info("Initializing SystemMonitor...")
        system_monitor = SystemMonitor(config)
        await system_monitor.start_monitoring()
        services["system_monitor"] = system_monitor

        # Dynamic model manager for runtime model switching
        logger.info("Initializing DynamicModelManager...")
        from services.connection_pool_manager import ConnectionPoolManager
        from services.dynamic_model_manager import DynamicModelManager

        # Create connection pool manager if not already available
        connection_pool = getattr(model_manager, "connection_pool", None)
        if not connection_pool:
            connection_pool = ConnectionPoolManager(config)

        dynamic_model_manager = DynamicModelManager(config, connection_pool)
        services["dynamic_model_manager"] = dynamic_model_manager

        # Voice manager for speech processing
        logger.info("Initializing VoiceManager...")
        try:
            from services.voice_manager import VoiceManager

            voice_manager = VoiceManager(config)
            services["voice_manager"] = voice_manager
            logger.info("Voice manager initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize voice manager: {str(e)}")
            logger.info("Voice features will be disabled")
            services["voice_manager"] = None

        # Initialize dependencies for API
        from app.core.dependencies import initialize_dependencies

        initialize_dependencies(config, connection_pool, system_monitor)

        # Memory pressure callbacks (with debouncing)
        last_cleanup_time = 0
        debounce_seconds = config.get("system", {}).get("monitor_debounce", 60)

        async def conversation_memory_pressure(memory_percent):
            nonlocal last_cleanup_time
            current_time = time.time()

            if current_time - last_cleanup_time < debounce_seconds:
                return  # Skip if we just cleaned up

            logger.warning(
                f"Memory pressure detected ({memory_percent}%), optimizing conversation memory"
            )
            from services.resource_manager import optimize_memory_usage

            await optimize_memory_usage(conversation_manager, kg_manager)
            last_cleanup_time = current_time

        resource_manager.register_memory_pressure_callback(conversation_memory_pressure)

        logger.info(f"Services initialized successfully (mode: {startup_mode})")
        return services
    except Exception as e:
        import sys
        import traceback

        logger.error(f"Error initializing services: {str(e)}")
        logger.error("Full traceback:")
        traceback.print_exc(file=sys.stderr)
        return services


def init_app(app: FastAPI, config: Dict[str, Any]) -> None:
    """
    Register startup and shutdown events for the FastAPI application.

    Args:
        app: FastAPI application instance
        config: Configuration dictionary
    """

    @app.on_event("startup")
    async def startup_event():
        try:
            # Create all required directories
            os.makedirs("logs", exist_ok=True)
            os.makedirs("data", exist_ok=True)
            os.makedirs("data/prebuilt_graphs", exist_ok=True)

            # Create dynamic graph storage directory from config
            dynamic_storage = config.get("knowledge_graphs", {}).get(
                "dynamic_storage", "./data/dynamic_graph"
            )
            os.makedirs(dynamic_storage, exist_ok=True)

            os.makedirs("data/sessions", exist_ok=True)

            # Initialize services (fail-fast, no retries)
            try:
                logger.info("Initializing services...")
                services = await initialize_services(config)

                # Store services in app state
                app.state.model_manager = services.get("model_manager")
                app.state.kg_manager = services.get("kg_manager")
                app.state.conversation_manager = services.get("conversation_manager")
                app.state.system_monitor = services.get("system_monitor")
                app.state.dynamic_model_manager = services.get("dynamic_model_manager")
                app.state.voice_manager = services.get("voice_manager")
                app.state.connection_manager = getattr(app.state, "connection_manager", None)

                # Start background graph loading if enabled
                startup_mode = config.get("startup", {}).get("preload", "models")
                if startup_mode in ["graphs", "all"]:
                    asyncio.create_task(load_graphs_background(app.state.kg_manager, config))

                logger.info("Services initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize services: {str(e)}")
                logger.warning("Running in degraded mode - some features may not work")
        except Exception as e:
            logger.error(f"Error during startup: {str(e)}")
            logger.warning("Running in degraded mode")

    @app.on_event("shutdown")
    async def shutdown_event():
        try:
            logger.info("Application shutting down, cleaning up resources...")

            # Close WebSocket connections
            connection_manager = getattr(app.state, "connection_manager", None)
            if connection_manager:
                for session_id, connection in list(connection_manager.active_connections.items()):
                    try:
                        # Check if connection is still open before trying to close
                        if (
                            hasattr(connection, "client_state")
                            and connection.client_state.name != "DISCONNECTED"
                        ):
                            await connection.close()
                            logger.debug(f"Closed WebSocket connection for session {session_id}")
                        else:
                            logger.debug(f"WebSocket for session {session_id} already closed")
                    except Exception as e:
                        # Ignore common WebSocket close errors during shutdown
                        error_msg = str(e).lower()
                        if any(
                            phrase in error_msg
                            for phrase in [
                                'cannot call "send" once a close message has been sent',
                                "websocket.close",
                                "connection is closed",
                                "already closed",
                            ]
                        ):
                            logger.debug(f"WebSocket for session {session_id} already closed: {e}")
                        else:
                            logger.error(
                                f"Error closing WebSocket connection for session {session_id}: {str(e)}"
                            )

            # Stop system monitoring
            system_monitor = getattr(app.state, "system_monitor", None)
            if system_monitor:
                try:
                    await asyncio.wait_for(system_monitor.stop_monitoring(), timeout=5.0)
                    logger.info("System monitoring stopped")
                except asyncio.TimeoutError:
                    logger.warning("System monitoring shutdown timed out")
                except Exception as e:
                    logger.error(f"Error stopping system monitoring: {str(e)}")

            # Shutdown orchestrated model manager if used (before saving sessions)
            model_manager = getattr(app.state, "model_manager", None)
            if model_manager and hasattr(model_manager, "shutdown"):
                try:
                    await asyncio.wait_for(model_manager.shutdown(), timeout=10.0)
                    logger.info("Orchestrated model manager shutdown completed")
                except asyncio.TimeoutError:
                    logger.warning("Model manager shutdown timed out")
                except Exception as e:
                    logger.error(f"Error shutting down orchestrated model manager: {str(e)}")

            # Save conversation sessions
            conversation_manager = getattr(app.state, "conversation_manager", None)
            if conversation_manager and hasattr(conversation_manager, "sessions"):
                try:
                    session_count = len(conversation_manager.sessions)
                    if session_count > 0:
                        logger.info(f"Saving {session_count} conversation sessions...")
                        os.makedirs("data/sessions", exist_ok=True)
                        session_data = {}

                        # Safely extract session data
                        for sid, s in conversation_manager.sessions.items():
                            try:
                                session_data[sid] = {
                                    "created_at": s["created_at"].isoformat()
                                    if hasattr(s["created_at"], "isoformat")
                                    else str(s["created_at"]),
                                    "updated_at": s["updated_at"].isoformat()
                                    if hasattr(s["updated_at"], "isoformat")
                                    else str(s["updated_at"]),
                                    "message_count": len(s.get("messages", [])),
                                    "has_summary": bool(s.get("summary")),
                                }
                            except Exception as e:
                                logger.warning(f"Error processing session {sid}: {e}")
                                continue

                        if session_data:
                            with open("data/sessions/sessions.json", "w") as f:
                                json.dump(session_data, f, indent=2)
                            logger.info(f"Saved {len(session_data)} conversation sessions")
                        else:
                            logger.info("No valid sessions to save")
                    else:
                        logger.info("No conversation sessions to save")
                except Exception as e:
                    logger.error(f"Error saving conversation sessions: {str(e)}")

            # Save knowledge graph query cache
            try:
                from services.kg_optimizer import optimizer

                if optimizer:
                    logger.info("Saving knowledge graph query cache...")
                    optimizer.save_cache_to_disk()
            except Exception as e:
                logger.error(f"Error saving knowledge graph cache: {str(e)}")

            # Force garbage collection
            gc.collect()

            logger.info("Shutdown completed successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            logger.warning("Forcing clean shutdown despite errors")
