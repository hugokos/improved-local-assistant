"""
Model Manager for handling Ollama model operations.

This module provides the ModelManager class that handles initialization,
configuration, and communication with Ollama models using best practices
from the reference guide.
"""

import asyncio
import os
import logging
import time
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import ollama
from ollama import AsyncClient


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    type: str  # "conversation" or "knowledge_extraction"
    context_window: int = 8000
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120
    max_parallel: int = 2
    max_loaded: int = 2


class ModelManager:
    """
    Manages Ollama models with dual-model architecture.
    
    Uses persistent AsyncClient instances for production to reuse HTTP/2 connections.
    Implements resource management via environment variables and supports
    concurrent dual-model operations without blocking.
    """
    
    def __init__(self, host: str = "http://localhost:11434"):
        """Initialize ModelManager with Ollama host."""
        self.host = host
        self.chat_client = AsyncClient(host=host)  # Hermes 3:3B - user-facing
        self.bg_client = AsyncClient(host=host)    # TinyLlama - background jobs
        
        self.conversation_model = "hermes3:3b"
        self.knowledge_model = "tinyllama"
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Performance metrics
        self.metrics = {
            "conversation": {
                "requests": 0,
                "tokens_generated": 0,
                "avg_response_time": 0,
                "last_response_time": 0
            },
            "knowledge": {
                "requests": 0,
                "tokens_generated": 0,
                "avg_response_time": 0,
                "last_response_time": 0
            }
        }
    
    async def initialize_models(self, config: ModelConfig) -> bool:
        """
        Initialize Ollama models with proper configuration.
        
        Args:
            config: ModelConfig object with resource settings
            
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Set environment variables for resource management
            os.environ["OLLAMA_NUM_PARALLEL"] = str(config.max_parallel)
            os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(config.max_loaded)
            
            self.logger.info(f"Initializing models with max_parallel={config.max_parallel}, max_loaded={config.max_loaded}")
            
            # Test conversation model availability
            self.logger.info(f"Testing conversation model: {self.conversation_model}")
            await self.chat_client.chat(
                model=self.conversation_model, 
                messages=[{"role": "user", "content": "test"}],
                options={
                    "temperature": config.temperature,
                    "num_predict": 1  # Minimal response for testing
                }
            )
            
            # Test knowledge model availability
            self.logger.info(f"Testing knowledge model: {self.knowledge_model}")
            await self.bg_client.chat(
                model=self.knowledge_model, 
                messages=[{"role": "user", "content": "test"}],
                options={
                    "temperature": config.temperature,
                    "num_predict": 1  # Minimal response for testing
                }
            )
            
            self.logger.info("Both models initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            return False
    
    async def query_conversation_model(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from conversation model (Hermes 3:3B).
        
        Args:
            messages: List of message dictionaries with role and content
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
            
        Yields:
            str: Response tokens as they are generated
        """
        start_time = time.time()
        token_count = 0
        
        try:
            stream = await self.chat_client.chat(
                model=self.conversation_model,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            async for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    token_count += 1
                    yield content
            
            # Update metrics
            elapsed = time.time() - start_time
            self._update_metrics("conversation", token_count, elapsed)
            
        except Exception as e:
            self.logger.error(f"Error querying conversation model: {str(e)}")
            yield f"Error: {str(e)}"
    
    async def query_knowledge_model(
        self, 
        text: str, 
        temperature: float = 0.2,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Query knowledge model (TinyLlama) for entity extraction.
        
        Args:
            text: Text to extract entities from
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict: Response from the model
        """
        start_time = time.time()
        
        try:
            prompt = f"""Extract entities and relationships from the following text. 
            Format the output as a list of triples (subject, relation, object):
            
            {text}
            
            Triples:"""
            
            response = await self.bg_client.chat(
                model=self.knowledge_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            # Update metrics
            elapsed = time.time() - start_time
            content = response["message"]["content"]
            token_count = len(content.split())
            self._update_metrics("knowledge", token_count, elapsed)
            
            return {
                "content": content,
                "model": self.knowledge_model,
                "elapsed_time": elapsed
            }
            
        except Exception as e:
            self.logger.error(f"Error querying knowledge model: {str(e)}")
            return {
                "content": f"Error: {str(e)}",
                "model": self.knowledge_model,
                "error": str(e)
            }
    
    def swap_model(self, client_type: str, new_model: str) -> bool:
        """
        Hot-swap model for a client.
        
        Args:
            client_type: "chat" or "background"
            new_model: Name of the new model to use
            
        Returns:
            bool: True if swap was successful
        """
        try:
            if client_type == "chat":
                self.chat_client.headers["x-ollama-model"] = new_model
                self.conversation_model = new_model
                self.logger.info(f"Swapped conversation model to {new_model}")
            elif client_type == "background":
                self.bg_client.headers["x-ollama-model"] = new_model
                self.knowledge_model = new_model
                self.logger.info(f"Swapped knowledge model to {new_model}")
            else:
                self.logger.error(f"Unknown client type: {client_type}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error swapping model: {str(e)}")
            return False
    
    async def get_model_status(self) -> Dict[str, Any]:
        """
        Get current status of all models.
        
        Returns:
            Dict: Status information for all models
        """
        try:
            # Use Ollama API to get model list and status
            response = await self.chat_client.list()
            
            models = response.get("models", [])
            model_info = {}
            
            for model in models:
                model_name = model.get("name", "unknown")
                model_info[model_name] = {
                    "size": model.get("size", 0),
                    "modified_at": model.get("modified_at", ""),
                    "is_conversation_model": model_name == self.conversation_model,
                    "is_knowledge_model": model_name == self.knowledge_model
                }
            
            return {
                "models": model_info,
                "conversation_model": self.conversation_model,
                "knowledge_model": self.knowledge_model,
                "metrics": self.metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model status: {str(e)}")
            return {
                "error": str(e),
                "conversation_model": self.conversation_model,
                "knowledge_model": self.knowledge_model,
                "metrics": self.metrics
            }
    
    async def run_concurrent_queries(
        self, 
        user_message: str
    ) -> Tuple[AsyncGenerator[str, None], asyncio.Task]:
        """
        Run conversation and knowledge queries concurrently.
        
        This implements the dual-model architecture where the conversation
        model handles the user-facing responses while the knowledge model
        processes the message for entity extraction in the background.
        
        Args:
            user_message: User message to process
            
        Returns:
            Tuple: (conversation_stream, background_task)
        """
        # Create conversation stream
        messages = [{"role": "user", "content": user_message}]
        conversation_stream = self.query_conversation_model(messages)
        
        # Create background task for knowledge extraction
        bg_task = asyncio.create_task(
            self.query_knowledge_model(user_message)
        )
        
        return conversation_stream, bg_task
    
    def _update_metrics(self, model_type: str, tokens: int, elapsed: float) -> None:
        """
        Update performance metrics for a model.
        
        Args:
            model_type: "conversation" or "knowledge"
            tokens: Number of tokens generated
            elapsed: Time elapsed in seconds
        """
        metrics = self.metrics.get(model_type, {})
        metrics["requests"] += 1
        metrics["tokens_generated"] += tokens
        metrics["last_response_time"] = elapsed
        
        # Update rolling average
        if metrics["requests"] > 1:
            metrics["avg_response_time"] = (
                (metrics["avg_response_time"] * (metrics["requests"] - 1) + elapsed) / 
                metrics["requests"]
            )
        else:
            metrics["avg_response_time"] = elapsed