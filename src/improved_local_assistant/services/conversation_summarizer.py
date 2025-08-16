"""
Conversation summarization for memory optimization.

This module provides mechanisms for summarizing conversations to reduce
memory usage while preserving context for long conversations.
"""

import logging
from datetime import datetime
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """
    Summarizes conversations to optimize memory usage.

    This class provides methods for summarizing conversations to reduce
    memory usage while preserving context for long conversations.
    """

    def __init__(self, model_manager=None):
        """
        Initialize the conversation summarizer.

        Args:
            model_manager: ModelManager instance for generating summaries
        """
        self.model_manager = model_manager
        logger.info("Conversation summarizer initialized")

    async def summarize_conversation(
        self, messages: list[dict[str, Any]], max_messages: int = 20
    ) -> str:
        """
        Generate a summary of a conversation.

        Args:
            messages: List of conversation messages
            max_messages: Maximum number of messages to include in the summary

        Returns:
            str: Conversation summary
        """
        if not messages:
            return ""

        if not self.model_manager:
            logger.error("Model manager not available for summarization")
            return self._generate_basic_summary(messages)

        # Limit the number of messages to summarize
        messages_to_summarize = messages[:max_messages]

        # Create a prompt for summarization
        summary_prompt = """
        Summarize this conversation concisely, capturing key points, questions, and information.
        Focus on preserving important context that might be needed for future reference.

        Conversation:
        """

        for msg in messages_to_summarize:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            summary_prompt += f"\n{role.capitalize()}: {content}\n"

        summary_prompt += "\nSummary:"

        try:
            # Query the model for a summary
            summary = ""
            async for token in self.model_manager.query_conversation_model(
                [{"role": "user", "content": summary_prompt}]
            ):
                summary += token

            logger.info(f"Generated conversation summary ({len(summary)} chars)")
            return summary

        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            return self._generate_basic_summary(messages)

    def _generate_basic_summary(self, messages: list[dict[str, Any]]) -> str:
        """
        Generate a basic summary without using the model.

        This is a fallback method when the model is not available.

        Args:
            messages: List of conversation messages

        Returns:
            str: Basic conversation summary
        """
        logger.info("Generating basic conversation summary")

        if not messages:
            return ""

        # Count messages by role
        user_messages = sum(1 for msg in messages if msg.get("role") == "user")
        assistant_messages = sum(1 for msg in messages if msg.get("role") == "assistant")

        # Extract topics (simple approach)
        topics = set()
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                # Extract potential topics from the first few words of user messages
                words = content.split()[:10]
                for word in words:
                    if len(word) > 4 and word not in [
                        "about",
                        "what",
                        "when",
                        "where",
                        "which",
                        "would",
                        "could",
                        "should",
                    ]:
                        topics.add(word)

        # Limit to top 5 topics
        topics = list(topics)[:5]

        # Generate summary
        summary = f"Conversation with {user_messages} user messages and {assistant_messages} assistant responses."

        if topics:
            summary += f" Topics discussed include: {', '.join(topics)}."

        # Add first and last messages
        if messages:
            first_user_msg = next(
                (msg.get("content", "") for msg in messages if msg.get("role") == "user"), ""
            )
            if first_user_msg:
                summary += f' Started with: "{first_user_msg[:50]}..."'

            last_user_msg = next(
                (msg.get("content", "") for msg in reversed(messages) if msg.get("role") == "user"),
                "",
            )
            if last_user_msg and last_user_msg != first_user_msg:
                summary += f' Most recent query: "{last_user_msg[:50]}..."'

        return summary

    async def compress_conversation(
        self, messages: list[dict[str, Any]], max_messages: int = 20, keep_recent: int = 10
    ) -> list[dict[str, Any]]:
        """
        Compress a conversation by summarizing older messages.

        Args:
            messages: List of conversation messages
            max_messages: Maximum number of messages before compression
            keep_recent: Number of recent messages to keep unchanged

        Returns:
            List[Dict[str, Any]]: Compressed conversation
        """
        if len(messages) <= max_messages:
            return messages

        logger.info(f"Compressing conversation from {len(messages)} messages")

        # Generate summary of older messages
        older_messages = messages[:-keep_recent]
        summary = await self.summarize_conversation(older_messages)

        # Create compressed conversation
        compressed = [
            {
                "role": "system",
                "content": f"Previous conversation summary: {summary}",
                "timestamp": datetime.now(),
            }
        ]

        # Add recent messages
        compressed.extend(messages[-keep_recent:])

        logger.info(f"Compressed conversation to {len(compressed)} messages")
        return compressed

    def estimate_token_count(self, messages: list[dict[str, Any]]) -> int:
        """
        Estimate the token count for a conversation.

        This is a rough approximation using a simple heuristic.

        Args:
            messages: List of conversation messages

        Returns:
            int: Estimated token count
        """
        # Simple heuristic: 1 token ≈ 4 characters
        total_chars = sum(len(msg.get("content", "")) for msg in messages)

        # Add overhead for message metadata
        overhead = len(messages) * 10

        # Estimate tokens
        estimated_tokens = (total_chars + overhead) // 4

        return estimated_tokens


# Global conversation summarizer instance
summarizer = None


def initialize_summarizer(model_manager=None):
    """
    Initialize the global conversation summarizer.

    Args:
        model_manager: ModelManager instance for generating summaries
    """
    global summarizer
    summarizer = ConversationSummarizer(model_manager)
    return summarizer


async def summarize_conversation(messages: list[dict[str, Any]], max_messages: int = 20) -> str:
    """
    Summarize a conversation using the global summarizer.

    Args:
        messages: List of conversation messages
        max_messages: Maximum number of messages to include in the summary

    Returns:
        str: Conversation summary
    """
    global summarizer
    if not summarizer:
        logger.error("Conversation summarizer not initialized")
        return ""

    return await summarizer.summarize_conversation(messages, max_messages)


async def compress_conversation(
    messages: list[dict[str, Any]], max_messages: int = 20, keep_recent: int = 10
) -> list[dict[str, Any]]:
    """
    Compress a conversation using the global summarizer.

    Args:
        messages: List of conversation messages
        max_messages: Maximum number of messages before compression
        keep_recent: Number of recent messages to keep unchanged

    Returns:
        List[Dict[str, Any]]: Compressed conversation
    """
    global summarizer
    if not summarizer:
        logger.error("Conversation summarizer not initialized")
        return messages

    return await summarizer.compress_conversation(messages, max_messages, keep_recent)
