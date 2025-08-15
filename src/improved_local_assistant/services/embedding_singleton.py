"""
Embedding model singleton to avoid duplicate SentenceTransformer instantiation.
"""

import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Global embedding model instance
_embedding_model: Optional = None
_embedding_model_name: Optional[str] = None


def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Get or create the global embedding model singleton.

    Args:
        model_name: Name of the embedding model to use

    Returns:
        HuggingFaceEmbedding: The singleton embedding model instance
    """
    global _embedding_model, _embedding_model_name

    # Return existing model if it matches the requested model
    if _embedding_model is not None and _embedding_model_name == model_name:
        logger.debug(f"Reusing existing embedding model: {model_name}")
        return _embedding_model

    try:
        # Use the new LlamaIndex 0.14+ import path
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # Set up cache folder
        cache_folder = "C:/hf-cache" if sys.platform == "win32" else "/tmp/hf-cache"
        os.makedirs(cache_folder, exist_ok=True)

        logger.info(f"Creating singleton embedding model: {model_name}")

        # Create the embedding model
        _embedding_model = HuggingFaceEmbedding(
            model_name=model_name,
            trust_remote_code=False,
            device="cpu",
            cache_folder=cache_folder,
            normalize=True,
            embed_batch_size=10,
        )

        _embedding_model_name = model_name

        logger.info(f"Successfully created singleton embedding model: {model_name}")
        return _embedding_model

    except ImportError as e:
        logger.error(f"Could not import HuggingFaceEmbedding: {e}")
        # Fallback to sentence transformers directly if HuggingFaceEmbedding fails
        try:
            from llama_index.core.embeddings import BaseEmbedding
            from sentence_transformers import SentenceTransformer

            class LocalHuggingFaceEmbedding(BaseEmbedding):
                def __init__(self, model_name: str, **kwargs):
                    super().__init__(**kwargs)
                    self.model = SentenceTransformer(model_name)
                    self.model_name = model_name

                def _get_query_embedding(self, query: str):
                    return self.model.encode([query])[0].tolist()

                def _get_text_embedding(self, text: str):
                    return self.model.encode([text])[0].tolist()

                async def _aget_query_embedding(self, query: str):
                    return self._get_query_embedding(query)

                async def _aget_text_embedding(self, text: str):
                    return self._get_text_embedding(text)

            _embedding_model = LocalHuggingFaceEmbedding(model_name=model_name)
            _embedding_model_name = model_name

            logger.info(f"Successfully created fallback embedding model: {model_name}")
            return _embedding_model

        except Exception as fallback_error:
            logger.error(f"Fallback embedding creation failed: {fallback_error}")
            raise
    except Exception as e:
        logger.error(f"Could not create embedding model: {e}")
        raise


def configure_global_embedding(model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Configure the global LlamaIndex embedding model using the singleton.

    Args:
        model_name: Name of the embedding model to use
    """
    try:
        from llama_index.core import Settings

        # Set the embedding model directly to avoid triggering default initialization
        embed_model = get_embedding_model(model_name)
        Settings.embed_model = embed_model
        logger.info(f"Configured global LlamaIndex embedding model: {model_name}")
        logger.info(f"Embedding singleton set to {type(Settings.embed_model)}")

    except Exception as e:
        logger.warning(f"Could not set embedding model in Settings: {e}")
        # Continue without setting global embedding - individual components can still use the singleton


def clear_embedding_singleton():
    """Clear the embedding singleton (for testing purposes)."""
    global _embedding_model, _embedding_model_name
    _embedding_model = None
    _embedding_model_name = None
    logger.debug("Cleared embedding singleton")
