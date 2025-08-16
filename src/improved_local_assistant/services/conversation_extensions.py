"""
Extensions to ConversationManager for CLI compatibility.

This module provides additional methods for the ConversationManager
to support the GraphRAG REPL requirements.
"""

import json
import logging
import os
from collections.abc import AsyncGenerator


class ConversationManagerExtensions:
    """
    Extensions for ConversationManager to support CLI requirements.
    """

    async def get_response(
        self, session_id: str, user_message: str, use_kg: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Get response with optional knowledge graph integration.

        This method provides the interface required by the GraphRAG REPL:
        pass the raw text to ConversationManager.get_response() with use_kg=True
        so the graph context is injected.

        Args:
            session_id: Session ID
            user_message: User message
            use_kg: Whether to use knowledge graph retrieval and extraction

        Yields:
            str: Response tokens as they are generated
        """
        try:
            if use_kg and self.kg_manager:
                # ✅ Always prefer converse_with_context for KG integration
                async for token in self.converse_with_context(session_id, user_message):
                    yield token
            else:
                # Use basic conversation without KG
                async for token in self.process_message(session_id, user_message):
                    yield token

        except Exception as e:
            self.logger.error(f"Error in get_response: {str(e)}")
            yield f"Error: {str(e)}"

    # get_citations() method removed - using the original implementation from ConversationManager
    # which already stores session["last_citations"] and returns real citations
    pass  # This class now only serves as a placeholder for potential future extensions

    def ensure_embedding_compatibility(self):
        """
        Ensure embedding model is configured to match prebuilt graphs.

        This method reads metadata from prebuilt graphs and configures
        the global embedding model to match.
        """
        try:
            from llama_index.core import Settings
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            # Skip if embedding model is already configured
            if Settings.embed_model is not None:
                self.logger.info("Embedding model already configured")
                return True

            # Read embedding model from graph metadata
            embed_model_name = self._get_embedding_model_from_metadata()

            # Configure the embedding model
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=embed_model_name,
                trust_remote_code=False,
                device="cpu",
                normalize=True,
                embed_batch_size=10,
            )

            self.logger.info(f"Configured embedding model for compatibility: {embed_model_name}")
            return True

        except ImportError as e:
            self.logger.error(f"Could not import HuggingFaceEmbedding: {e}")
            self.logger.warning("Install sentence-transformers: pip install sentence-transformers")
            return False
        except Exception as e:
            self.logger.error(f"Error configuring embedding compatibility: {e}")
            return False

    def _get_embedding_model_from_metadata(self) -> str:
        """Get embedding model name from prebuilt graph metadata."""
        default_model = "BAAI/bge-small-en-v1.5"

        try:
            # Check survivalist graph metadata first
            prebuilt_dir = "./data/prebuilt_graphs"
            meta_path = os.path.join(prebuilt_dir, "survivalist", "meta.json")

            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                    embed_model = metadata.get("embed_model", default_model)
                    self.logger.info(f"Found embedding model in metadata: {embed_model}")
                    return embed_model

            # Check other graph directories
            if os.path.exists(prebuilt_dir):
                for graph_dir in os.listdir(prebuilt_dir):
                    meta_path = os.path.join(prebuilt_dir, graph_dir, "meta.json")
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, encoding="utf-8") as f:
                                metadata = json.load(f)
                                embed_model = metadata.get("embed_model", default_model)
                                return embed_model
                        except Exception:
                            continue

            return default_model

        except Exception as e:
            self.logger.warning(f"Error reading graph metadata: {e}")
            return default_model


# Monkey patch the ConversationManager to add the get_response method
def extend_conversation_manager():
    """
    Extend ConversationManager with additional methods for CLI compatibility.
    """
    from services.conversation_manager import ConversationManager

    # Add the get_response method
    ConversationManager.get_response = ConversationManagerExtensions.get_response

    logging.getLogger(__name__).info("Extended ConversationManager with get_response method")


# Auto-extend when this module is imported
extend_conversation_manager()


# Apply extensions to ConversationManager
def apply_extensions():
    """Apply extensions to ConversationManager class."""
    from services.conversation_manager import ConversationManager

    # Add methods from ConversationManagerExtensions to ConversationManager
    for attr_name in dir(ConversationManagerExtensions):
        if not attr_name.startswith("_") and attr_name != "apply_extensions":
            attr = getattr(ConversationManagerExtensions, attr_name)
            if callable(attr):
                setattr(ConversationManager, attr_name, attr)


# Apply extensions when module is imported
try:
    apply_extensions()
except Exception as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.error(f"Failed to apply conversation extensions: {e}")
