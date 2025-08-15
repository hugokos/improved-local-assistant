"""
Mock Ollama client for testing without requiring actual Ollama server.

This module provides MockAsyncClient that mimics the ollama.AsyncClient interface
for use in tests and development environments.
"""

import asyncio
from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import List


class MockAsyncClient:
    """Mock implementation of ollama.AsyncClient for testing."""

    def __init__(self, host: str = "http://localhost:11434"):
        """Initialize mock client with default models."""
        self.host = host
        self._models = ["hermes3:3b", "tinyllama"]
        self._responses = {
            "hermes3:3b": "This is a mock response from Hermes 3B model.",
            "tinyllama": "Mock response from TinyLlama model.",
        }

    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Any:
        """Mock generate method that returns deterministic responses."""
        if model not in self._models:
            raise Exception(f"Model {model} not found")

        base_response = self._responses.get(model, "Mock response")

        if stream:
            # Return async generator for streaming
            async def mock_stream() -> AsyncGenerator[Dict[str, Any], None]:
                tokens = base_response.split()
                for i, token in enumerate(tokens):
                    yield {
                        "response": token + (" " if i < len(tokens) - 1 else ""),
                        "done": i == len(tokens) - 1,
                        "model": model,
                        "created_at": "2024-01-01T00:00:00Z",
                        "context": [1, 2, 3],
                    }
                    await asyncio.sleep(0.01)  # Simulate streaming delay

            return mock_stream()
        else:
            # Return single response
            return {
                "response": base_response,
                "done": True,
                "model": model,
                "created_at": "2024-01-01T00:00:00Z",
                "context": [1, 2, 3],
                "total_duration": 1000000,
                "load_duration": 100000,
                "prompt_eval_count": 10,
                "prompt_eval_duration": 200000,
                "eval_count": 20,
                "eval_duration": 700000,
            }

    async def chat(
        self, model: str, messages: List[Dict[str, str]], stream: bool = False, **kwargs
    ) -> Any:
        """Mock chat method for conversation-style interactions."""
        if model not in self._models:
            raise Exception(f"Model {model} not found")

        # Extract last user message for context
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        response_content = (
            f"Mock response to: {user_message[:50]}..." if user_message else "Mock chat response"
        )

        if stream:

            async def mock_chat_stream() -> AsyncGenerator[Dict[str, Any], None]:
                tokens = response_content.split()
                for i, token in enumerate(tokens):
                    yield {
                        "message": {
                            "role": "assistant",
                            "content": token + (" " if i < len(tokens) - 1 else ""),
                        },
                        "done": i == len(tokens) - 1,
                        "model": model,
                        "created_at": "2024-01-01T00:00:00Z",
                    }
                    await asyncio.sleep(0.01)

            return mock_chat_stream()
        else:
            return {
                "message": {"role": "assistant", "content": response_content},
                "done": True,
                "model": model,
                "created_at": "2024-01-01T00:00:00Z",
                "total_duration": 1000000,
                "load_duration": 100000,
                "prompt_eval_count": len(user_message.split()) if user_message else 0,
                "prompt_eval_duration": 200000,
                "eval_count": len(response_content.split()),
                "eval_duration": 700000,
            }

    async def list(self) -> Dict[str, List[Dict[str, str]]]:
        """Mock list method that returns available models."""
        return {
            "models": [
                {
                    "name": model,
                    "model": model,
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 1000000000,
                    "digest": "mock_digest_" + model.replace(":", "_"),
                }
                for model in self._models
            ]
        }

    async def show(self, model: str) -> Dict[str, Any]:
        """Mock show method that returns model details."""
        if model not in self._models:
            raise Exception(f"Model {model} not found")

        return {
            "model": model,
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 1000000000,
            "digest": f"mock_digest_{model.replace(':', '_')}",
            "details": {
                "family": "llama",
                "format": "gguf",
                "parameter_size": "3B" if "3b" in model else "1B",
                "quantization_level": "Q4_0",
            },
            "modelfile": f"FROM {model}\nPARAMETER temperature 0.7",
            "parameters": {"temperature": 0.7, "top_p": 0.9, "top_k": 40},
        }

    async def pull(self, model: str, stream: bool = False) -> Any:
        """Mock pull method for downloading models."""
        if stream:

            async def mock_pull_stream() -> AsyncGenerator[Dict[str, Any], None]:
                statuses = [
                    "pulling manifest",
                    "downloading",
                    "verifying sha256",
                    "writing manifest",
                    "success",
                ]
                for i, status in enumerate(statuses):
                    yield {
                        "status": status,
                        "digest": f"mock_digest_{i}",
                        "total": 1000000000,
                        "completed": (i + 1) * 200000000,
                    }
                    await asyncio.sleep(0.1)

            return mock_pull_stream()
        else:
            return {"status": "success"}

    async def push(self, model: str, stream: bool = False) -> Any:
        """Mock push method for uploading models."""
        if stream:

            async def mock_push_stream() -> AsyncGenerator[Dict[str, Any], None]:
                yield {"status": "pushing manifest"}
                yield {"status": "success"}

            return mock_push_stream()
        else:
            return {"status": "success"}

    async def delete(self, model: str) -> Dict[str, str]:
        """Mock delete method for removing models."""
        if model in self._models:
            self._models.remove(model)
        return {"status": "success"}


# Convenience function for tests
def create_mock_client(host: str = "http://localhost:11434") -> MockAsyncClient:
    """Create a mock Ollama client for testing."""
    return MockAsyncClient(host=host)


# Mock responses for specific test scenarios
MOCK_RESPONSES = {
    "knowledge_extraction": "Entity: Python, Relationship: is_a, Object: programming_language",
    "conversation": "Hello! I'm a mock AI assistant. How can I help you today?",
    "error_scenario": "Mock error response for testing error handling",
    "empty_response": "",
    "long_response": " ".join(["Mock"] * 100) + " response for testing long content.",
}


class MockStreamResponse:
    """Mock streaming response for testing stream handling."""

    def __init__(self, content: str, model: str = "hermes3:3b"):
        self.content = content
        self.model = model
        self.tokens = content.split()

    async def __aiter__(self):
        """Async iterator for streaming tokens."""
        for i, token in enumerate(self.tokens):
            yield {
                "response": token + (" " if i < len(self.tokens) - 1 else ""),
                "done": i == len(self.tokens) - 1,
                "model": self.model,
            }
            await asyncio.sleep(0.01)
