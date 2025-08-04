"""
Dynamic Model Manager for runtime model switching.

This module provides the DynamicModelManager class that handles:
• Runtime model switching for conversation and knowledge extraction
• Model validation and availability checking
• Configuration updates and persistence
• WebSocket notifications for UI updates
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from services.connection_pool_manager import ConnectionPoolManager


class DynamicModelManager:
    """
    Manages dynamic model switching for conversation and knowledge extraction.
    
    Provides runtime model switching capabilities with validation, configuration
    updates, and UI synchronization via WebSocket notifications.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        connection_pool: ConnectionPoolManager
    ):
        """Initialize Dynamic Model Manager with configuration."""
        self.config = config
        self.connection_pool = connection_pool
        self.logger = logging.getLogger(__name__)
        
        # Current active models
        self.current_conversation_model = config.get("models", {}).get("conversation", {}).get("name", "hermes3:3b")
        self.current_knowledge_model = config.get("models", {}).get("knowledge", {}).get("name", "tinyllama")
        
        # Available model options
        self.conversation_options = config.get("models", {}).get("conversation", {}).get("options", [])
        self.knowledge_options = config.get("models", {}).get("knowledge", {}).get("options", [])
        
        # WebSocket connections for notifications
        self.websocket_connections = set()
        
        # Metrics
        self.metrics = {
            "model_switches": 0,
            "conversation_model_switches": 0,
            "knowledge_model_switches": 0,
            "switch_failures": 0,
            "avg_switch_time": 0.0,
        }
        
        self.logger.info(f"Initialized DynamicModelManager")
        self.logger.info(f"Current conversation model: {self.current_conversation_model}")
        self.logger.info(f"Current knowledge model: {self.current_knowledge_model}")
        self.logger.info(f"Available conversation models: {[opt['name'] for opt in self.conversation_options]}")
        self.logger.info(f"Available knowledge models: {[opt['name'] for opt in self.knowledge_options]}")
    
    def register_websocket(self, websocket) -> None:
        """Register a WebSocket connection for notifications."""
        self.websocket_connections.add(websocket)
    
    def unregister_websocket(self, websocket) -> None:
        """Unregister a WebSocket connection."""
        self.websocket_connections.discard(websocket)
    
    async def switch_conversation_model(self, model_name: str) -> Dict[str, Any]:
        """
        Switch the conversation model to the specified model.
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            Dict: Switch result with success status and details
        """
        start_time = time.time()
        
        try:
            # Validate model is available
            model_config = self._get_model_config("conversation", model_name)
            if not model_config:
                return {
                    "success": False,
                    "error": f"Model '{model_name}' not available for conversation",
                    "available_models": [opt["name"] for opt in self.conversation_options]
                }
            
            # Check if model is already active
            if model_name == self.current_conversation_model:
                return {
                    "success": True,
                    "message": f"Model '{model_name}' is already active",
                    "model": model_name,
                    "switch_time": 0.0
                }
            
            # Test model availability
            is_available = await self._test_model_availability(model_name)
            if not is_available:
                return {
                    "success": False,
                    "error": f"Model '{model_name}' is not available or not responding",
                    "model": model_name
                }
            
            # Unload current model if different
            if self.current_conversation_model != model_name:
                await self._unload_model(self.current_conversation_model)
            
            # Update current model
            old_model = self.current_conversation_model
            self.current_conversation_model = model_name
            
            # Update configuration
            self.config["models"]["conversation"]["name"] = model_name
            self.config["models"]["conversation"]["context_window"] = model_config["context_window"]
            self.config["models"]["conversation"]["max_tokens"] = model_config["max_tokens"]
            
            # Update metrics
            switch_time = time.time() - start_time
            self.metrics["model_switches"] += 1
            self.metrics["conversation_model_switches"] += 1
            self._update_avg_metric("avg_switch_time", switch_time)
            
            # Notify WebSocket connections
            await self._notify_model_switch("conversation", old_model, model_name, switch_time)
            
            self.logger.info(f"Switched conversation model from '{old_model}' to '{model_name}' in {switch_time:.3f}s")
            
            return {
                "success": True,
                "message": f"Successfully switched conversation model to '{model_name}'",
                "old_model": old_model,
                "new_model": model_name,
                "switch_time": switch_time,
                "model_config": model_config
            }
            
        except Exception as e:
            self.metrics["switch_failures"] += 1
            self.logger.error(f"Error switching conversation model to '{model_name}': {e}")
            return {
                "success": False,
                "error": f"Failed to switch model: {str(e)}",
                "model": model_name
            }
    
    async def switch_knowledge_model(self, model_name: str) -> Dict[str, Any]:
        """
        Switch the knowledge extraction model to the specified model.
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            Dict: Switch result with success status and details
        """
        start_time = time.time()
        
        try:
            # Validate model is available
            model_config = self._get_model_config("knowledge", model_name)
            if not model_config:
                return {
                    "success": False,
                    "error": f"Model '{model_name}' not available for knowledge extraction",
                    "available_models": [opt["name"] for opt in self.knowledge_options]
                }
            
            # Check if model is already active
            if model_name == self.current_knowledge_model:
                return {
                    "success": True,
                    "message": f"Model '{model_name}' is already active",
                    "model": model_name,
                    "switch_time": 0.0
                }
            
            # Test model availability
            is_available = await self._test_model_availability(model_name)
            if not is_available:
                return {
                    "success": False,
                    "error": f"Model '{model_name}' is not available or not responding",
                    "model": model_name
                }
            
            # Unload current model if different
            if self.current_knowledge_model != model_name:
                await self._unload_model(self.current_knowledge_model)
            
            # Update current model
            old_model = self.current_knowledge_model
            self.current_knowledge_model = model_name
            
            # Update configuration
            self.config["models"]["knowledge"]["name"] = model_name
            self.config["models"]["knowledge"]["context_window"] = model_config["context_window"]
            self.config["models"]["knowledge"]["max_tokens"] = model_config["max_tokens"]
            
            # Update metrics
            switch_time = time.time() - start_time
            self.metrics["model_switches"] += 1
            self.metrics["knowledge_model_switches"] += 1
            self._update_avg_metric("avg_switch_time", switch_time)
            
            # Notify WebSocket connections
            await self._notify_model_switch("knowledge", old_model, model_name, switch_time)
            
            self.logger.info(f"Switched knowledge model from '{old_model}' to '{model_name}' in {switch_time:.3f}s")
            
            return {
                "success": True,
                "message": f"Successfully switched knowledge model to '{model_name}'",
                "old_model": old_model,
                "new_model": model_name,
                "switch_time": switch_time,
                "model_config": model_config
            }
            
        except Exception as e:
            self.metrics["switch_failures"] += 1
            self.logger.error(f"Error switching knowledge model to '{model_name}': {e}")
            return {
                "success": False,
                "error": f"Failed to switch model: {str(e)}",
                "model": model_name
            }
    
    def _get_model_config(self, model_type: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific model.
        
        Args:
            model_type: Type of model ("conversation" or "knowledge")
            model_name: Name of the model
            
        Returns:
            Optional[Dict]: Model configuration or None if not found
        """
        options = self.conversation_options if model_type == "conversation" else self.knowledge_options
        
        for option in options:
            if option["name"] == model_name:
                return option
        
        return None
    
    async def _test_model_availability(self, model_name: str) -> bool:
        """
        Test if a model is available and responding.
        
        Args:
            model_name: Name of the model to test
            
        Returns:
            bool: True if model is available
        """
        try:
            # Send a minimal test request
            result = await self.connection_pool.chat_request(
                model=model_name,
                messages=[{"role": "user", "content": "test"}],
                keep_alive=0,  # Don't keep loaded after test
                num_predict=1,  # Minimal response
                timeout=10.0  # Quick timeout
            )
            
            return result is not None and "message" in result
            
        except Exception as e:
            self.logger.debug(f"Model '{model_name}' availability test failed: {e}")
            return False
    
    async def _unload_model(self, model_name: str) -> None:
        """
        Unload a model to free resources.
        
        Args:
            model_name: Name of the model to unload
        """
        try:
            self.logger.debug(f"Unloading model '{model_name}'")
            
            await self.connection_pool.chat_request(
                model=model_name,
                messages=[{"role": "user", "content": "unload"}],
                keep_alive=0,
                num_predict=1
            )
            
        except Exception as e:
            self.logger.debug(f"Error unloading model '{model_name}' (may already be unloaded): {e}")
    
    async def _notify_model_switch(
        self, 
        model_type: str, 
        old_model: str, 
        new_model: str, 
        switch_time: float
    ) -> None:
        """
        Notify WebSocket connections about model switch.
        
        Args:
            model_type: Type of model switched
            old_model: Previous model name
            new_model: New model name
            switch_time: Time taken for switch
        """
        if not self.websocket_connections:
            return
        
        notification = {
            "type": "model_switch",
            "model_type": model_type,
            "old_model": old_model,
            "new_model": new_model,
            "switch_time": switch_time,
            "timestamp": time.time()
        }
        
        # Send to all connected WebSockets
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(notification)
            except Exception as e:
                self.logger.debug(f"Failed to send model switch notification: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected WebSockets
        for websocket in disconnected:
            self.websocket_connections.discard(websocket)
    
    def _update_avg_metric(self, metric_name: str, new_value: float) -> None:
        """Update rolling average metric."""
        current_avg = self.metrics.get(metric_name, 0.0)
        count = self.metrics.get("model_switches", 1)
        
        if count > 1:
            self.metrics[metric_name] = (current_avg * (count - 1) + new_value) / count
        else:
            self.metrics[metric_name] = new_value
    
    def get_current_models(self) -> Dict[str, str]:
        """Get currently active models."""
        return {
            "conversation": self.current_conversation_model,
            "knowledge": self.current_knowledge_model
        }
    
    def get_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get available model options."""
        return {
            "conversation": self.conversation_options,
            "knowledge": self.knowledge_options
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model manager metrics."""
        return self.metrics.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get model manager status."""
        return {
            "current_models": self.get_current_models(),
            "available_models": self.get_available_models(),
            "websocket_connections": len(self.websocket_connections),
            "metrics": self.get_metrics()
        }