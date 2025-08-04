"""
Connection Pool Manager for optimized HTTP client management with Ollama.

This module provides the ConnectionPoolManager class that manages HTTP connections
to Ollama with proper pooling, timeouts, and keep-alive functionality for optimal
performance on edge devices.
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx


class ConnectionPoolManager:
    """
    Manages HTTP connections to Ollama with connection pooling and keep-alive.
    
    Provides optimized connection reuse, proper timeout handling, and model
    residency management for edge device performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Connection Pool Manager with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract connection pool configuration
        pool_config = config.get("connection_pool", {})
        self.max_connections = pool_config.get("max_connections", 10)
        self.max_keepalive_connections = pool_config.get("max_keepalive_connections", 5)
        self.keepalive_expiry = pool_config.get("keepalive_expiry", 30.0)
        
        # Timeout configuration
        timeout_config = pool_config.get("timeout", {})
        self.connect_timeout = timeout_config.get("connect", 5.0)
        self.read_timeout = timeout_config.get("read", 30.0)
        self.write_timeout = timeout_config.get("write", 30.0)
        self.pool_timeout = timeout_config.get("pool", 5.0)
        
        # Keep-alive models
        self.keep_alive_models = pool_config.get("keep_alive_models", ["hermes3:3b"])
        
        # Ollama configuration
        ollama_config = config.get("ollama", {})
        self.ollama_host = ollama_config.get("host", "http://localhost:11434")
        
        # HTTP client (will be initialized in initialize())
        self._client: Optional[httpx.AsyncClient] = None
        
        # Metrics
        self.metrics = {
            "requests_made": 0,
            "connections_created": 0,
            "connections_reused": 0,
            "timeouts": 0,
            "errors": 0,
        }
        
        # Residency check task
        self._residency_check_task: Optional[asyncio.Task] = None
        self._residency_check_interval = 60.0  # Check every minute
    
    async def initialize(self) -> None:
        """Initialize the HTTP client with connection pooling."""
        try:
            self.logger.info("Initializing Connection Pool Manager")
            
            # Create connection limits
            limits = httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
                keepalive_expiry=self.keepalive_expiry
            )
            
            # Create timeout configuration
            timeout = httpx.Timeout(
                connect=self.connect_timeout,
                read=self.read_timeout,
                write=self.write_timeout,
                pool=self.pool_timeout
            )
            
            # Initialize HTTP client
            try:
                # Try to enable HTTP/2 if available
                self._client = httpx.AsyncClient(
                    base_url=self.ollama_host,
                    limits=limits,
                    timeout=timeout,
                    http2=True  # Enable HTTP/2 for better performance
                )
            except Exception as e:
                # Fall back to HTTP/1.1 if HTTP/2 is not available
                self.logger.warning(f"HTTP/2 not available, falling back to HTTP/1.1: {e}")
                self._client = httpx.AsyncClient(
                    base_url=self.ollama_host,
                    limits=limits,
                    timeout=timeout,
                    http2=False
                )
            
            # Start periodic residency checking
            self._residency_check_task = asyncio.create_task(self._periodic_residency_check())
            
            self.logger.info("Connection Pool Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Connection Pool Manager: {e}")
            raise
    
    async def chat_request(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        keep_alive: Optional[Any] = None,
        **options
    ) -> Dict[str, Any]:
        """
        Make a chat request to Ollama with connection pooling.
        
        Args:
            model: Model name
            messages: Chat messages
            keep_alive: Keep-alive parameter for model residency
            **options: Additional Ollama options
            
        Returns:
            Dict: Response from Ollama
        """
        if not self._client:
            raise RuntimeError("Connection pool not initialized")
        
        request_start_time = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                **options
            }
            
            # Add keep_alive if specified
            if keep_alive is not None:
                payload["keep_alive"] = keep_alive
            elif model in self.keep_alive_models:
                # Default keep-alive for specified models
                payload["keep_alive"] = "30m"
            
            # Make request
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            # Update metrics
            self.metrics["requests_made"] += 1
            
            # Parse response
            result = response.json()
            
            # Log timing metrics if available
            if "load_duration" in result:
                self.logger.debug(f"Model load duration: {result['load_duration']}ns")
            if "prompt_eval_duration" in result:
                self.logger.debug(f"Prompt eval duration: {result['prompt_eval_duration']}ns")
            
            return result
            
        except httpx.TimeoutException as e:
            self.metrics["timeouts"] += 1
            self.logger.error(f"Request timeout for model {model}: {e}")
            raise
            
        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Request error for model {model}: {e}")
            raise
    
    async def stream_chat(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        keep_alive: Optional[Any] = None,
        **options
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat request to Ollama with connection pooling.
        
        Args:
            model: Model name
            messages: Chat messages
            keep_alive: Keep-alive parameter for model residency
            **options: Additional Ollama options
            
        Yields:
            str: Response tokens
        """
        if not self._client:
            raise RuntimeError("Connection pool not initialized")
        
        try:
            # Prepare request payload
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                **options
            }
            
            # Add keep_alive if specified
            if keep_alive is not None:
                payload["keep_alive"] = keep_alive
            elif model in self.keep_alive_models:
                # Default keep-alive for specified models
                payload["keep_alive"] = "30m"
            
            # Make streaming request
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            import json
                            chunk = json.loads(line)
                            
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                if content:
                                    yield content
                                    
                            # Check for completion
                            if chunk.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
            # Update metrics
            self.metrics["requests_made"] += 1
            
        except httpx.TimeoutException as e:
            self.metrics["timeouts"] += 1
            self.logger.error(f"Streaming timeout for model {model}: {e}")
            raise
            
        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Streaming error for model {model}: {e}")
            raise
    
    async def verify_model_residency(self, model: str) -> bool:
        """
        Verify if a model is currently resident in memory.
        
        Args:
            model: Model name to check
            
        Returns:
            bool: True if model is resident
        """
        if not self._client:
            return False
        
        try:
            response = await self._client.get("/api/ps")
            response.raise_for_status()
            
            ps_data = response.json()
            models = ps_data.get("models", [])
            
            for model_info in models:
                if model_info.get("name") == model or model_info.get("model") == model:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking model residency: {e}")
            return False
    
    async def get_loaded_models(self) -> List[Dict[str, Any]]:
        """
        Get list of currently loaded models.
        
        Returns:
            List[Dict]: Information about loaded models
        """
        if not self._client:
            return []
        
        try:
            response = await self._client.get("/api/ps")
            response.raise_for_status()
            
            ps_data = response.json()
            return ps_data.get("models", [])
            
        except Exception as e:
            self.logger.error(f"Error getting loaded models: {e}")
            return []
    
    async def health_check(self) -> bool:
        """
        Perform health check on Ollama connection.
        
        Returns:
            bool: True if healthy
        """
        if not self._client:
            return False
        
        try:
            response = await self._client.get("/api/version")
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def _periodic_residency_check(self) -> None:
        """Periodically check and log model residency."""
        while True:
            try:
                await asyncio.sleep(self._residency_check_interval)
                
                loaded_models = await self.get_loaded_models()
                model_names = [m.get("name", m.get("model", "unknown")) for m in loaded_models]
                
                self.logger.debug(f"Currently loaded models: {model_names}")
                
                # Check if expected models are loaded
                for expected_model in self.keep_alive_models:
                    is_loaded = any(expected_model in name for name in model_names)
                    if not is_loaded:
                        self.logger.warning(f"Expected model {expected_model} is not loaded")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic residency check: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics."""
        metrics = self.metrics.copy()
        
        if self._client:
            # Add connection pool info if available
            try:
                # Note: httpx doesn't expose detailed pool metrics directly
                # This is a placeholder for potential future metrics
                metrics["pool_info"] = {
                    "max_connections": self.max_connections,
                    "max_keepalive_connections": self.max_keepalive_connections,
                    "keepalive_expiry": self.keepalive_expiry
                }
            except Exception:
                pass
        
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown connection pool and cleanup resources."""
        try:
            self.logger.info("Shutting down Connection Pool Manager")
            
            # Cancel residency check task
            if self._residency_check_task:
                self._residency_check_task.cancel()
                try:
                    await self._residency_check_task
                except asyncio.CancelledError:
                    pass
            
            # Close HTTP client
            if self._client:
                await self._client.aclose()
                self._client = None
            
            self.logger.info("Connection Pool Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during connection pool shutdown: {e}")