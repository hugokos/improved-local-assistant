"""
Orchestrated Model Manager that integrates with the LLM Orchestration system.

This module provides an enhanced ModelManager that uses the LLM Orchestrator
for turn-by-turn coordination while maintaining backward compatibility with
the existing interface.
"""

import asyncio
import logging
import os
import sys
from typing import Any, AsyncGenerator, Dict, List, Optional

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from services.llm_orchestrator import LLMOrchestrator
from services.working_set_cache import WorkingSetCache
from services.hybrid_retriever import HybridEnsembleRetriever
from services.extraction_pipeline import ExtractionPipeline
from services.system_monitor import SystemMonitor
from services.model_mgr import ModelConfig  # Import original config class


class OrchestratedModelManager:
    """
    Enhanced ModelManager that uses LLM Orchestration for optimal edge performance.
    
    Provides backward compatibility with the original ModelManager interface while
    leveraging the orchestration system for turn-by-turn coordination, resource
    management, and hybrid retrieval.
    """
    
    def __init__(self, config: Dict[str, Any], system_monitor: Optional[SystemMonitor] = None):
        """Initialize Orchestrated Model Manager with configuration."""
        try:
            self.config = config
            self.logger = logging.getLogger(__name__)
            
            # Create or use provided system monitor
            if system_monitor is None:
                try:
                    self.logger.info("Creating SystemMonitor...")
                    self.system_monitor = SystemMonitor(config)
                    self.logger.info("SystemMonitor created successfully")
                except Exception as e:
                    import traceback, sys
                    self.logger.error(f"Failed to create SystemMonitor: {e}")
                    traceback.print_exc(file=sys.stderr)
                    raise
            else:
                self.system_monitor = system_monitor
            
            # Initialize orchestration components
            self.logger.info("Creating LLMOrchestrator...")
            self.orchestrator = LLMOrchestrator(config, self.system_monitor)
            self.logger.info("Creating WorkingSetCache...")
            self.working_set_cache = WorkingSetCache(config)
            self.logger.info("Creating HybridEnsembleRetriever...")
            self.hybrid_retriever = HybridEnsembleRetriever(
                graph_index=None,  # Will be set later when graphs are loaded
                working_set_cache=self.working_set_cache,
                config=config
            )
            self.logger.info("Creating ExtractionPipeline...")
            self.extraction_pipeline = ExtractionPipeline(
                config, 
                self.orchestrator.connection_pool, 
                self.system_monitor
            )
            self.logger.info("All orchestration components created successfully")
        except Exception as e:
            import traceback, sys
            self.logger.error(f"Error in OrchestratedModelManager.__init__: {e}")
            traceback.print_exc(file=sys.stderr)
            raise
        
        # Backward compatibility attributes
        self.host = config.get("ollama", {}).get("host", "http://localhost:11434")
        self.conversation_model = config.get("models", {}).get("conversation", {}).get("name", "hermes3:3b")
        self.knowledge_model = config.get("models", {}).get("knowledge", {}).get("name", "tinyllama")
        
        # Metrics for backward compatibility
        self.metrics = {
            "conversation": {"requests": 0, "tokens_generated": 0, "avg_response_time": 0, "last_response_time": 0},
            "knowledge": {"requests": 0, "tokens_generated": 0, "avg_response_time": 0, "last_response_time": 0},
        }
        
        # Initialization state
        self._initialized = False
    
    async def initialize_models(self, config: ModelConfig) -> bool:
        """
        Initialize all orchestration components.
        
        Args:
            config: ModelConfig object (for backward compatibility)
            
        Returns:
            bool: True if initialization was successful
        """
        try:
            self.logger.info("Initializing Orchestrated Model Manager")
            
            # Start system monitor if not already started
            if hasattr(self.system_monitor, 'start_monitoring'):
                if not hasattr(self.system_monitor, 'monitoring_task') or self.system_monitor.monitoring_task is None:
                    await self.system_monitor.start_monitoring()
            else:
                self.logger.warning("SystemMonitor does not have start_monitoring method")
            
            # Initialize orchestration components
            await self.orchestrator.initialize()
            await self.working_set_cache.initialize()
            
            # Initialize hybrid retriever (placeholder - would need actual indices)
            # Note: HybridEnsembleRetriever is already initialized in constructor
            # No additional initialization needed here
            self.logger.info("HybridEnsembleRetriever already initialized")
            
            self._initialized = True
            self.logger.info("Orchestrated Model Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Orchestrated Model Manager: {e}")
            return False
    
    async def query_conversation_model(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7, 
        max_tokens: int = 2048,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from conversation model using orchestration.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
            session_id: Optional session ID for context
            
        Yields:
            str: Response tokens as they are generated
        """
        if not self._initialized:
            self.logger.error("Model manager not initialized")
            yield "Error: Model manager not initialized"
            return
        
        try:
            # Extract user message from messages
            user_message = ""
            if messages:
                last_message = messages[-1]
                if last_message.get("role") == "user":
                    user_message = last_message.get("content", "")
            
            # Use orchestrator for turn processing
            session_id = session_id or "default_session"
            async for token in self.orchestrator.process_turn(
                session_id=session_id,
                user_message=user_message,
                conversation_history=messages
            ):
                yield token
            
            # Update backward compatibility metrics
            self.metrics["conversation"]["requests"] += 1
            
        except Exception as e:
            self.logger.error(f"Error in orchestrated conversation query: {e}")
            yield f"Error: {str(e)}"
    
    async def query_conversation_model_with_context(
        self,
        messages: List[Dict[str, str]],
        session_id: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:
        """
        Stream response with hybrid retrieval context.
        
        Args:
            messages: List of message dictionaries
            session_id: Session ID for working set context
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
            
        Yields:
            str: Response tokens with context
        """
        if not self._initialized:
            self.logger.error("Model manager not initialized")
            yield "Error: Model manager not initialized"
            return
        
        try:
            # Extract user message
            user_message = ""
            if messages:
                last_message = messages[-1]
                if last_message.get("role") == "user":
                    user_message = last_message.get("content", "")
            
            # Get context from hybrid retriever
            context_chunks = await self.hybrid_retriever.retrieve(
                query=user_message,
                session_id=session_id
            )
            
            # Add context to messages if available
            enhanced_messages = messages.copy()
            if context_chunks:
                context_text = "\n\n".join([chunk.content for chunk in context_chunks])
                context_message = {
                    "role": "system",
                    "content": f"Relevant context:\n{context_text}\n\nUse this context to help answer the user's question."
                }
                enhanced_messages.insert(-1, context_message)  # Insert before last user message
            
            # Use orchestrator with enhanced messages
            async for token in self.orchestrator.process_turn(
                session_id=session_id,
                user_message=user_message,
                conversation_history=enhanced_messages
            ):
                yield token
            
            # Update metrics
            self.metrics["conversation"]["requests"] += 1
            
        except Exception as e:
            self.logger.error(f"Error in context-aware conversation query: {e}")
            yield f"Error: {str(e)}"
    
    async def query_knowledge_model(
        self, 
        text: str, 
        temperature: float = 0.2, 
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Query knowledge model for entity extraction using extraction pipeline.
        
        Args:
            text: Text to extract entities from
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict: Response from the model
        """
        if not self._initialized:
            return {"error": "Model manager not initialized"}
        
        try:
            # Use extraction pipeline for bounded extraction
            triples = await self.extraction_pipeline.extract_bounded(text)
            
            if triples is None:
                return {
                    "content": "Extraction skipped due to resource pressure",
                    "model": self.knowledge_model,
                    "skipped": True
                }
            
            # Format triples as content
            if triples:
                content = "\n".join([
                    f"({triple.subject}, {triple.predicate}, {triple.object}, {triple.confidence})"
                    for triple in triples
                ])
            else:
                content = "No entities extracted."
            
            # Update metrics
            self.metrics["knowledge"]["requests"] += 1
            
            return {
                "content": content,
                "model": self.knowledge_model,
                "triples_count": len(triples) if triples else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in knowledge extraction: {e}")
            return {
                "content": f"Error extracting entities: {str(e)}",
                "model": self.knowledge_model,
                "error": str(e)
            }
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get current status of all models and orchestration components."""
        if not self._initialized:
            return {"error": "Model manager not initialized"}
        
        try:
            # Get orchestrator status
            orchestrator_status = await self.orchestrator.get_status()
            
            # Get component statuses
            cache_stats = self.working_set_cache.get_global_stats()
            retriever_status = self.hybrid_retriever.get_status()
            extraction_status = self.extraction_pipeline.get_status()
            
            return {
                "orchestrator": orchestrator_status,
                "working_set_cache": cache_stats,
                "hybrid_retriever": retriever_status,
                "extraction_pipeline": extraction_status,
                "backward_compatibility_metrics": self.metrics,
                "models": {
                    "conversation_model": self.conversation_model,
                    "knowledge_model": self.knowledge_model
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model status: {e}")
            return {"error": str(e)}
    
    # Backward compatibility methods
    
    def swap_model(self, client_type: str, new_model: str) -> bool:
        """Hot-swap model (backward compatibility)."""
        try:
            if client_type == "chat":
                self.conversation_model = new_model
            elif client_type == "background":
                self.knowledge_model = new_model
            else:
                return False
            
            self.logger.info(f"Swapped {client_type} model to {new_model}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error swapping model: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown all orchestration components."""
        try:
            self.logger.info("Shutting down Orchestrated Model Manager")
            
            if self._initialized:
                await self.orchestrator.shutdown()
                await self.working_set_cache.shutdown()
                if hasattr(self.system_monitor, 'stop_monitoring'):
                    await self.system_monitor.stop_monitoring()
            
            self._initialized = False
            self.logger.info("Orchestrated Model Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Factory function for backward compatibility
def create_model_manager(config: Dict[str, Any], use_orchestration: bool = None) -> Any:
    """
    Factory function to create appropriate model manager.
    
    Args:
        config: Configuration dictionary
        use_orchestration: Whether to use orchestration (auto-detect if None)
        
    Returns:
        ModelManager instance (orchestrated or original)
    """
    # Auto-detect orchestration preference
    if use_orchestration is None:
        use_orchestration = config.get("edge_optimization", {}).get("enabled", False)
    
    if use_orchestration:
        return OrchestratedModelManager(config)
    else:
        # Fall back to original model manager
        from services.model_mgr import ModelManager
        host = config.get("ollama", {}).get("host", "http://localhost:11434")
        return ModelManager(host=host)