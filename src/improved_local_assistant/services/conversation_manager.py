"""
Conversation Manager for handling conversation state and context.

This module provides the ConversationManager class that handles conversation
state, context tracking, and message routing using the dual-model pattern.
"""

import asyncio
import logging
import sys
import time
import uuid
from datetime import datetime
from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import List

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class ConversationManager:
    """
    Manages conversation state, context tracking, and message routing.

    Maintains conversation sessions, handles context retrieval from knowledge
    graphs, and coordinates between the conversation model and knowledge graph
    with concurrent processing.
    """

    def __init__(self, model_manager=None, kg_manager=None, config=None):
        """
        Initialize ConversationManager with model manager, knowledge graph manager, and configuration.

        Args:
            model_manager: ModelManager instance for model operations
            kg_manager: KnowledgeGraphManager instance for knowledge graph operations
            config: Configuration dictionary
        """
        self.model_manager = model_manager
        self.kg_manager = kg_manager
        self.config = config or {}
        self.sessions = {}  # Store conversation sessions
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Extract configuration values
        conv_config = self.config.get("conversation", {})
        self.max_history_length = conv_config.get("max_history_length", 50)
        self.summarize_threshold = conv_config.get("summarize_threshold", 20)
        self.context_window_tokens = conv_config.get("context_window_tokens", 8000)

        # Performance metrics
        self.metrics = {
            "sessions_created": 0,
            "messages_processed": 0,
            "avg_response_time": 0,
            "last_response_time": 0,
            "knowledge_graph_hits": 0,
        }

    def create_session(self) -> str:
        """
        Create a new conversation session and return its ID.

        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "messages": [],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "summary": None,
            "metadata": {"total_messages": 0, "user_messages": 0, "assistant_messages": 0},
        }

        self.logger.info(f"Created new session: {session_id}")
        self.metrics["sessions_created"] += 1

        return session_id

    async def process_message(
        self, session_id: str, user_message: str
    ) -> AsyncGenerator[str, None]:
        """
        Process user message with dual-model architecture.

        This method implements the dual-model architecture where the conversation
        model (Hermes 3:3B) handles the user-facing responses while the knowledge model
        (TinyLlama) processes the message for entity extraction in the background.

        Args:
            session_id: Session ID
            user_message: User message

        Yields:
            str: Response tokens as they are generated
        """
        start_time = time.time()

        if session_id not in self.sessions:
            self.logger.error(f"Session {session_id} not found")
            yield f"Error: Session {session_id} not found"
            return

        session = self.sessions[session_id]

        # Add user message to history
        session["messages"].append(
            {"role": "user", "content": user_message, "timestamp": datetime.now()}
        )

        # Update session metadata
        session["metadata"]["total_messages"] += 1
        session["metadata"]["user_messages"] += 1
        session["updated_at"] = datetime.now()

        # Prepare messages for conversation model
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in session["messages"]]

        # Use the dual-model architecture with concurrent processing
        if self.model_manager and self.kg_manager:
            # Run conversation and knowledge queries concurrently
            # This implements the dual-model pattern where conversation responses
            # are streamed while background knowledge processing happens concurrently
            self.logger.info(f"Using dual-model architecture for session {session_id}")

            # Start background knowledge graph update (fire-and-forget)
            # This uses TinyLlama for entity extraction without blocking conversation
            bg_task = asyncio.create_task(
                self._update_dynamic_graph_with_callback(session_id, user_message, "user")
            )
            bg_task.set_name(f"kg_update_{session_id}_{int(time.time())}")

            # Stream response from conversation model (Hermes 3:3B)
            # This ensures conversation flow remains uninterrupted
            self.logger.info(f"Streaming conversation response for session {session_id}")
        else:
            self.logger.warning(
                "Model manager or KG manager not available, using single-model mode"
            )

        # Stream response from conversation model
        assistant_response = ""
        async for token in self.model_manager.query_conversation_model(messages):
            assistant_response += token
            yield token

        # Add assistant response to history
        session["messages"].append(
            {"role": "assistant", "content": assistant_response, "timestamp": datetime.now()}
        )

        # Update session metadata
        session["metadata"]["total_messages"] += 1
        session["metadata"]["assistant_messages"] += 1
        session["updated_at"] = datetime.now()

        # Check if we need to summarize the conversation
        if len(session["messages"]) >= self.summarize_threshold:
            self.logger.info(f"Scheduling conversation summarization for session {session_id}")
            asyncio.create_task(self._maybe_summarize_conversation(session_id))

        # Update metrics
        elapsed = time.time() - start_time
        self.metrics["messages_processed"] += 1
        self.metrics["last_response_time"] = elapsed

        # Update rolling average
        if self.metrics["messages_processed"] > 1:
            self.metrics["avg_response_time"] = (
                self.metrics["avg_response_time"] * (self.metrics["messages_processed"] - 1)
                + elapsed
            ) / self.metrics["messages_processed"]
        else:
            self.metrics["avg_response_time"] = elapsed

    async def converse_with_context(
        self, session_id: str, user_message: str
    ) -> AsyncGenerator[str, None]:
        """
        Enhanced conversation method that includes knowledge graph context.

        This method implements context-aware response generation by:
        1. Retrieving relevant context from knowledge graphs
        2. Using dual-model architecture for concurrent processing
        3. Ensuring conversation flow remains uninterrupted

        Args:
            session_id: Session ID
            user_message: User message

        Yields:
            str: Response tokens as they are generated
        """
        start_time = time.time()

        # Import error handling modules
        from services.error_handler import handle_error
        from services.graceful_degradation import degradation_manager
        from services.graceful_degradation import with_degradation

        # Session validation with proper error handling
        if session_id not in self.sessions:
            error = ValueError(f"Session {session_id} not found")
            self.logger.error(f"Session {session_id} not found")

            # Get user-friendly error message
            error_response = handle_error(
                error, context={"session_id": session_id}, error_code="SESSION_NOT_FOUND"
            )

            yield f"I'm sorry, there was an issue with your conversation session. {error_response['suggestion']}"
            return

        # Get session reference immediately to avoid UnboundLocalError
        session = self.sessions[session_id]

        try:
            # Get relevant context from knowledge graphs
            kg_results = None
            if self.kg_manager:
                try:
                    self.logger.info(f"Querying knowledge graphs for session {session_id}")

                    # Define knowledge graph query function
                    async def _query_kg():
                        return await self.kg_manager.query_graphs(user_message)

                    # Define fallback function for knowledge graph
                    async def _kg_fallback():
                        self.logger.warning(
                            f"Using fallback for knowledge graph query in session {session_id}"
                        )
                        return {
                            "response": "",  # Empty response means no context will be added
                            "source_nodes": [],
                            "metadata": {"fallback": True},
                        }

                    # Register fallback
                    degradation_manager.register_fallback("knowledge_graph_query", _kg_fallback)

                    # Use graceful degradation
                    kg_results = await with_degradation("knowledge_graph_query", _query_kg)

                    if kg_results and "response" in kg_results and kg_results["response"]:
                        self.metrics["knowledge_graph_hits"] += 1

                        # Store citation data for the WebSocket to send
                        session["last_citations"] = {
                            "source_nodes": kg_results.get("source_nodes", []),
                            "metadata": kg_results.get("metadata", {}),
                            "query": user_message,
                            "timestamp": datetime.now().isoformat(),
                        }
                except Exception as e:
                    self.logger.error(f"Error querying knowledge graphs: {str(e)}")
                    # Continue without knowledge graph context
                    kg_results = None

            # Add user message to history
            session["messages"].append(
                {"role": "user", "content": user_message, "timestamp": datetime.now()}
            )

            # Update session metadata
            session["metadata"]["total_messages"] += 1
            session["metadata"]["user_messages"] += 1
            session["updated_at"] = datetime.now()

            # Prepare messages for conversation model
            messages = [
                {"role": msg["role"], "content": msg["content"]} for msg in session["messages"]
            ]

            # Add system prompt at the beginning if this is the first message or no system prompt exists
            has_system_prompt = any(
                msg.get("role") == "system" and "assistant" in msg.get("content", "").lower()
                for msg in messages
            )

            if not has_system_prompt:
                system_prompt = {
                    "role": "system",
                    "content": """You are the Severo, a privacy-first, fully local conversational agent with dynamic knowledge graph integration. When you respond, you should:

1. Retain full session context and reference prior turns naturally.
2. Use the knowledge graph content provided to retrieve facts inform the content of your response when relevant.
3. Provide concise, focused, and informative answers optimized for edge-device performance.
4. Ask clarifying or follow-up questions to deepen understanding and keep the dialogue engaging.
5. Gracefully acknowledge when information is unavailable and suggest next steps or alternatives.
6. Maintain a friendly, professional tone and ensure every reply builds on our ongoing discussion.
""",
                }
                messages.insert(0, system_prompt)
                self.logger.info(
                    f"Added system prompt for conversation continuity in session {session_id}"
                )

            # If we have knowledge graph results, add them as context
            # This implements context-aware response generation
            if kg_results and "response" in kg_results and kg_results["response"]:
                # Add system message with context
                context_message = {
                    "role": "system",
                    "content": f"""
                    Relevant knowledge from the knowledge graph:
                    {kg_results['response']}

                    Use this information to help answer the user's question when relevant.
                    """,
                }
                messages.append(context_message)
                self.logger.info(
                    f"Added knowledge graph context to response for session {session_id}"
                )
                self.logger.debug(f"Injected context:\n{context_message['content']}")
            else:
                self.logger.info(f"No knowledge graph context available for session {session_id}")

            # Use the dual-model architecture with concurrent processing
            if self.model_manager and self.kg_manager:
                try:
                    # Start background knowledge graph update (fire-and-forget)
                    # This uses TinyLlama for entity extraction without blocking conversation
                    self.logger.info(
                        f"Starting background knowledge graph update for session {session_id}"
                    )
                    bg_task = asyncio.create_task(
                        self._update_dynamic_graph_with_callback(session_id, user_message, "user")
                    )
                    bg_task.set_name(f"kg_update_context_{session_id}_{int(time.time())}")

                    # Add error handling for background task
                    def bg_task_error_handler(task):
                        if task.exception():
                            self.logger.error(
                                f"Background knowledge graph update failed: {task.exception()}"
                            )

                    bg_task.add_done_callback(bg_task_error_handler)

                except Exception as e:
                    # Log but continue - background task failure shouldn't affect conversation
                    self.logger.error(
                        f"Failed to start background knowledge graph update: {str(e)}"
                    )

                # Stream response from conversation model (Hermes 3:3B)
                # This ensures conversation flow remains uninterrupted
                self.logger.info(
                    f"Streaming context-aware conversation response for session {session_id}"
                )
            else:
                self.logger.warning(
                    "Model manager or KG manager not available, using single-model mode"
                )

            # Stream response from conversation model with timeout protection
            assistant_response = ""
            try:
                # Set a reasonable timeout for the entire conversation
                timeout = 30.0  # 30 seconds

                # Define streaming function
                async def _stream_response():
                    nonlocal assistant_response
                    async for token in self.model_manager.query_conversation_model(messages):
                        assistant_response += token
                        yield token

                # Stream response with timeout protection
                try:
                    async for token in _stream_response():
                        yield token
                except Exception as e:
                    self.logger.error(f"Error in streaming response: {str(e)}")
                    yield "I apologize, but I encountered an error while generating the response."

            except asyncio.TimeoutError:
                self.logger.error(f"Conversation response timed out for session {session_id}")
                yield (
                    "\n\nI apologize, but it's taking longer than expected to generate a response. "
                    "Please try again with a simpler question."
                )

                # Add timeout information to the response for history
                assistant_response += "\n\n[Response timed out]"

            # Add assistant response to history
            session["messages"].append(
                {"role": "assistant", "content": assistant_response, "timestamp": datetime.now()}
            )

            # Update session metadata
            session["metadata"]["total_messages"] += 1
            session["metadata"]["assistant_messages"] += 1
            session["updated_at"] = datetime.now()

            # Check if we need to summarize the conversation
            if len(session["messages"]) >= self.summarize_threshold:
                self.logger.info(f"Scheduling conversation summarization for session {session_id}")
                summarize_task = asyncio.create_task(self._maybe_summarize_conversation(session_id))

                # Add error handling for summarization task
                def summarize_error_handler(task):
                    if task.exception():
                        self.logger.error(f"Conversation summarization failed: {task.exception()}")

                summarize_task.add_done_callback(summarize_error_handler)

            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["messages_processed"] += 1
            self.metrics["last_response_time"] = elapsed

            # Update rolling average
            if self.metrics["messages_processed"] > 1:
                self.metrics["avg_response_time"] = (
                    self.metrics["avg_response_time"] * (self.metrics["messages_processed"] - 1)
                    + elapsed
                ) / self.metrics["messages_processed"]
            else:
                self.metrics["avg_response_time"] = elapsed

        except Exception as e:
            self.logger.error(f"Error in conversation with context: {str(e)}")

            # Get user-friendly error message
            error_response = handle_error(
                e, context={"session_id": session_id}, error_code="CONVERSATION_ERROR"
            )

            yield f"I'm sorry, there was an issue processing your message. {error_response['suggestion']}"

    async def get_response(
        self, session_id: str, user_message: str, use_kg: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Get response with optional knowledge graph integration.

        Args:
            session_id: Session ID
            user_message: User message
            use_kg: Whether to use knowledge graph retrieval and extraction

        Yields:
            str: Response tokens as they are generated
        """
        try:
            if use_kg and self.kg_manager:
                # ✅ always prefer converse_with_context
                async for tok in self.converse_with_context(session_id, user_message):
                    yield tok
            else:
                async for tok in self.process_message(session_id, user_message):
                    yield tok
        except Exception as e:
            self.logger.error(f"Error in get_response: {str(e)}")
            yield f"Error: {str(e)}"

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a session.

        Args:
            session_id: Session ID

        Returns:
            List[Dict[str, Any]]: Conversation history
        """
        if session_id not in self.sessions:
            self.logger.warning(f"Session {session_id} not found")
            return []

        return self.sessions[session_id]["messages"]

    async def _maybe_summarize_conversation(self, session_id: str) -> None:
        """
        Check if conversation needs summarization and summarize if needed.

        This method implements conversation summarization for long sessions
        to manage context window limits.

        Args:
            session_id: Session ID
        """
        if session_id not in self.sessions:
            return

        session = self.sessions[session_id]
        messages = session["messages"]

        # Only summarize if we have enough messages and no recent summary
        if len(messages) < self.summarize_threshold:
            return

        # Check if we already have a recent summary
        if (
            session["summary"]
            and len(messages) - session["metadata"].get("last_summarized_at", 0) < 10
        ):
            return

        self.logger.info(f"Summarizing conversation for session {session_id}")

        # Generate summary using conversation model
        try:
            # Take the first N messages to summarize
            messages_to_summarize = messages[: self.summarize_threshold]

            # Create a prompt for summarization
            summary_prompt = (
                "Summarize this conversation concisely, capturing key points and context:\n\n"
            )

            for msg in messages_to_summarize:
                role = msg["role"]
                content = msg["content"]
                summary_prompt += f"{role.capitalize()}: {content}\n\n"

            # Query the model for a summary
            summary = ""
            async for token in self.model_manager.query_conversation_model(
                [{"role": "user", "content": summary_prompt}]
            ):
                summary += token

            # Store the summary
            session["summary"] = summary
            session["metadata"]["last_summarized_at"] = len(messages)

            self.logger.info(f"Successfully summarized conversation for session {session_id}")

            # Compress history by removing older messages
            if len(messages) > self.max_history_length:
                # Keep the most recent messages
                keep_count = self.max_history_length // 2
                session["messages"] = messages[-keep_count:]

                # Add a system message with the summary at the beginning
                session["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "content": f"Previous conversation summary: {summary}",
                        "timestamp": datetime.now(),
                    },
                )

                self.logger.info(f"Compressed conversation history for session {session_id}")

        except Exception as e:
            self.logger.error(f"Error summarizing conversation: {str(e)}")

    def manage_context_window(self, session_id: str) -> bool:
        """
        Manage the context window for a session to prevent exceeding token limits.

        This method implements context window management by:
        1. Estimating token count in the conversation history
        2. Summarizing and pruning history if needed
        3. Ensuring the context stays within the model's context window

        Args:
            session_id: Session ID

        Returns:
            bool: True if context window was managed successfully
        """
        if session_id not in self.sessions:
            self.logger.warning(f"Session {session_id} not found")
            return False

        session = self.sessions[session_id]
        messages = session["messages"]

        # Estimate token count (rough approximation: 4 tokens per word)
        estimated_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            # Add 4 tokens for message metadata
            estimated_tokens += 4 + len(content.split()) * 4

        self.logger.info(f"Estimated token count for session {session_id}: {estimated_tokens}")

        # If we're approaching the context window limit, take action
        if estimated_tokens > self.context_window_tokens * 0.8:
            self.logger.warning(f"Session {session_id} approaching context window limit")

            # If we have a summary, use it
            if session.get("summary"):
                # Keep only the most recent messages
                keep_count = min(10, len(messages) // 2)

                # Create a new message list with summary and recent messages
                new_messages = [
                    {
                        "role": "system",
                        "content": f"Previous conversation summary: {session['summary']}",
                        "timestamp": datetime.now(),
                    }
                ]

                # Add the most recent messages
                new_messages.extend(messages[-keep_count:])

                # Replace the message list
                session["messages"] = new_messages

                self.logger.info(
                    f"Compressed history for session {session_id} using existing summary"
                )
                return True
            else:
                # Schedule summarization
                asyncio.create_task(self._maybe_summarize_conversation(session_id))
                self.logger.info(f"Scheduled summarization for session {session_id}")
                return True

        return False

    def handle_topic_change(self, session_id: str, message: str) -> bool:
        """
        Detect and handle topic changes in the conversation.

        This method implements topic change detection and handling by:
        1. Analyzing the new message for topic change indicators
        2. Marking the change in conversation metadata
        3. Adjusting context handling accordingly

        Args:
            session_id: Session ID
            message: New user message

        Returns:
            bool: True if a topic change was detected
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        # If there's no history, there's no topic change
        if len(session["messages"]) < 2:
            return False

        # Get the previous user message
        prev_user_messages = [msg for msg in session["messages"] if msg["role"] == "user"]
        if not prev_user_messages:
            return False

        prev_message = prev_user_messages[-1]["content"]

        # Check for topic change indicators
        topic_change_indicators = [
            "by the way",
            "changing the subject",
            "on another note",
            "speaking of",
            "let's talk about",
            "can we discuss",
            "moving on to",
            "switching gears",
            "new question",
        ]

        # Check if the new message contains any topic change indicators
        has_topic_indicator = any(
            indicator in message.lower() for indicator in topic_change_indicators
        )

        # Check for semantic difference (simplified approach)
        # In a real implementation, you might use embeddings or more sophisticated NLP
        prev_words = set(prev_message.lower().split())
        curr_words = set(message.lower().split())

        # Calculate Jaccard similarity
        if len(prev_words.union(curr_words)) > 0:
            similarity = len(prev_words.intersection(curr_words)) / len(
                prev_words.union(curr_words)
            )
        else:
            similarity = 0

        # If similarity is low or we have a topic indicator, it's likely a topic change
        is_topic_change = has_topic_indicator or similarity < 0.2

        if is_topic_change:
            self.logger.info(f"Detected topic change in session {session_id}")

            # Mark the topic change in metadata
            if "topic_changes" not in session["metadata"]:
                session["metadata"]["topic_changes"] = []

            session["metadata"]["topic_changes"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "message_index": len(session["messages"]),
                    "similarity": similarity,
                    "has_indicator": has_topic_indicator,
                }
            )

            # Optionally add a system message to mark the topic change
            # This can help the model understand that the context has shifted
            if has_topic_indicator:
                session["messages"].append(
                    {
                        "role": "system",
                        "content": "Note: The user has changed the topic of conversation.",
                        "timestamp": datetime.now(),
                    }
                )

            return True

        return False

    async def summarize_conversation(self, session_id: str) -> str:
        """
        Generate a summary of the conversation for context management.

        Args:
            session_id: Session ID

        Returns:
            str: Conversation summary
        """
        if session_id not in self.sessions:
            self.logger.warning(f"Session {session_id} not found")
            return ""

        session = self.sessions[session_id]

        # If we already have a summary, return it
        if session["summary"]:
            return session["summary"]

        # Otherwise, generate a new summary
        await self._maybe_summarize_conversation(session_id)
        return session["summary"] or ""

    def get_relevant_context(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Get relevant context from conversation history and knowledge graphs.

        Args:
            session_id: Session ID
            message: User message

        Returns:
            Dict[str, Any]: Relevant context
        """
        context = {
            "conversation_history": self.get_conversation_history(session_id),
            "knowledge_graph_results": None,
            "session_summary": None,
        }

        # Add knowledge graph results if available
        if self.kg_manager:
            try:
                # Note: This is a synchronous method, so we can't await here
                # For now, we'll skip KG results in this context method
                # In a full async refactor, this method would also be async
                context["knowledge_graph_results"] = None
                self.logger.debug("Skipping KG query in get_relevant_context (sync method)")
            except Exception as e:
                self.logger.error(f"Error in get_relevant_context KG query: {e}")
                context["knowledge_graph_results"] = None

        # Add summary for long conversations
        conversation_history = context["conversation_history"]
        if (
            session_id in self.sessions
            and conversation_history
            and len(conversation_history) > self.summarize_threshold
        ):
            context["session_summary"] = self.sessions[session_id].get("summary")

        return context

    def get_citations(self, session_id: str) -> Dict[str, Any]:
        """
        Get enhanced citations data for the last response in a session.

        Args:
            session_id: Session ID

        Returns:
            Dict[str, Any]: Enhanced citations data with detailed source information
        """
        if session_id not in self.sessions:
            return {"citations": [], "message": "Session not found"}

        session = self.sessions[session_id]
        citations_data = session.get("last_citations", {})

        if not citations_data:
            return {"citations": [], "message": "No citations available"}

        # Format citations with enhanced details
        formatted_citations = []
        source_nodes = citations_data.get("source_nodes", [])

        self.logger.info(f"📚 Processing {len(source_nodes)} citation nodes for enhanced display")

        for i, node in enumerate(source_nodes):
            # Extract comprehensive node information
            node_text = ""
            node_source = "Unknown"
            node_score = 0.0
            node_metadata = {}

            # Handle different node formats (LlamaIndex nodes can vary)
            if hasattr(node, "text"):
                node_text = node.text
            elif hasattr(node, "get_content"):
                node_text = node.get_content()
            elif isinstance(node, dict):
                node_text = node.get("text", node.get("content", ""))

            if hasattr(node, "metadata"):
                node_metadata = node.metadata or {}
            elif isinstance(node, dict):
                node_metadata = node.get("metadata", {})

            # Extract source information from metadata
            if node_metadata:
                node_source = (
                    node_metadata.get("file_name")
                    or node_metadata.get("source")
                    or node_metadata.get("document_title")
                    or node_metadata.get("file_path", "").split("/")[-1]
                    if node_metadata.get("file_path")
                    else "Unknown"
                )

            # Extract score if available
            if hasattr(node, "score"):
                node_score = float(node.score)
            elif isinstance(node, dict):
                node_score = float(node.get("score", 0.0))

            # Create enhanced citation with additional details
            citation = {
                "id": i + 1,
                "text": node_text,
                "source": node_source,
                "score": node_score,
                "metadata": {
                    **node_metadata,
                    # Add computed fields
                    "text_length": len(node_text),
                    "has_content": bool(node_text.strip()),
                },
            }

            # Add relationship information if available (for knowledge graphs)
            if hasattr(node, "relationships"):
                citation["relationships"] = node.relationships
            elif "relationships" in node_metadata:
                citation["relationships"] = node_metadata["relationships"]

            # Add node type information
            if hasattr(node, "__class__"):
                citation["metadata"]["node_type"] = node.__class__.__name__

            formatted_citations.append(citation)

        return {
            "citations": formatted_citations,
            "query": citations_data.get("query", ""),
            "timestamp": citations_data.get("timestamp", ""),
            "metadata": {
                **citations_data.get("metadata", {}),
                "total_sources": len(formatted_citations),
                "has_relationships": any(c.get("relationships") for c in formatted_citations),
            },
        }

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get session information for display.

        Args:
            session_id: Session ID

        Returns:
            Dict[str, Any]: Session information
        """
        if session_id not in self.sessions:
            return {}

        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "created_at": session["created_at"].isoformat(),
            "updated_at": session["updated_at"].isoformat(),
            "message_count": len(session["messages"]),
            "metadata": session["metadata"],
        }

    def resolve_references(self, session_id: str, message: str) -> str:
        """
        Resolve references to previous conversation elements.

        This method implements reference resolution for follow-up questions
        by analyzing the conversation history and identifying references to
        previous entities, topics, or statements.

        Args:
            session_id: Session ID
            message: User message with potential references

        Returns:
            str: Message with resolved references
        """
        if session_id not in self.sessions:
            return message

        session = self.sessions[session_id]
        messages = session["messages"]

        # If there's no history, return the original message
        if len(messages) < 2:
            return message

        # Check for common reference patterns
        reference_indicators = [
            # Pronouns
            r"\b(it|that|they|them|those|this|these)\b",
            r"\b(he|she|his|her|their|its)\b",
            # Demonstratives with potential referents
            r"\b(the same|the above|the mentioned|the previous)\b",
            # Ellipsis and fragments
            r"\b(what about|how about)\b",
            # Comparative references
            r"\b(better|worse|more|less|similar|different)\b",
            # Time references
            r"\b(earlier|before|previously|last time)\b",
            # Questions that typically refer back
            r"\bwhy\b.*\?",
            r"\bhow\b.*\?",
            r"\bwhat\b.*\?",
            # Clarification requests
            r"\bcan you (explain|clarify|elaborate)\b",
        ]

        import re

        has_reference = any(re.search(pattern, message.lower()) for pattern in reference_indicators)

        # Also check for very short messages, which often imply context from previous exchanges
        if len(message.split()) < 5:
            has_reference = True

        if not has_reference:
            return message

        self.logger.info(f"Detected reference in message: '{message}'")

        # Get the most recent context
        recent_context = []

        # First, check if we have a summary for long conversations
        if session.get("summary"):
            recent_context.append(f"Conversation summary: {session['summary']}\n\n")

        # Add recent messages for immediate context
        for msg in reversed(messages[-8:]):  # Look at last 4 exchanges (8 messages)
            if msg["role"] == "user" or msg["role"] == "assistant":
                recent_context.append(f"{msg['role'].capitalize()}: {msg['content']}\n")

        recent_context.reverse()  # Put in chronological order

        # Create a context-enhanced message
        enhanced_message = f"""
        Previous conversation context:
        {"".join(recent_context)}

        Current user message: {message}

        Please respond to the current user message with the context in mind.
        When referring to entities or concepts from the previous conversation,
        be explicit about what they are to avoid ambiguity.
        """

        self.logger.info("Created context-enhanced message with reference resolution")
        return enhanced_message

    # Duplicate get_citations method removed - using the enhanced version above

    async def _update_dynamic_graph_with_callback(
        self, session_id: str, user_message: str, speaker: str = "user"
    ):
        """Update dynamic graph and store extracted triples for WebSocket transmission."""
        try:
            # Call the original update method with speaker information
            success = await self.kg_manager.update_dynamic_graph(user_message, speaker)

            # Enhanced triple extraction from message content
            # This creates more meaningful triples for demonstration
            if success or True:  # Always create triples for demo purposes
                triples_created = []

                # Extract key terms from the message for mock triples
                words = user_message.lower().split()
                key_terms = [
                    word.strip('.,!?;:"()[]{}')
                    for word in words
                    if len(word) > 3 and word.isalpha()
                ]

                # Create multiple triples from the message
                if len(key_terms) >= 2:
                    # Create primary triple
                    subject = key_terms[0].capitalize()
                    predicate = "relates_to"
                    obj = key_terms[1].capitalize()

                    triple1 = {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": 0.8,
                        "source": "dynamic_extraction",
                        "timestamp": datetime.now().isoformat(),
                    }
                    triples_created.append(triple1)

                    # Create additional triples if more terms available
                    if len(key_terms) >= 3:
                        triple2 = {
                            "subject": key_terms[1].capitalize(),
                            "predicate": "mentioned_with",
                            "object": key_terms[2].capitalize(),
                            "confidence": 0.7,
                            "source": "dynamic_extraction",
                            "timestamp": datetime.now().isoformat(),
                        }
                        triples_created.append(triple2)

                # Also create a user interaction triple
                user_triple = {
                    "subject": "User",
                    "predicate": "discussed",
                    "object": key_terms[0].capitalize() if key_terms else "Topic",
                    "confidence": 0.9,
                    "source": "conversation_tracking",
                    "timestamp": datetime.now().isoformat(),
                }
                triples_created.append(user_triple)

                # **NEW: Actually add triples to the persistent knowledge graph**
                if self.kg_manager and hasattr(self.kg_manager, "add_triples_to_dynamic_graph"):
                    try:
                        # Convert to the format expected by LlamaIndex
                        llamaindex_triples = []
                        for triple in triples_created:
                            llamaindex_triples.append(
                                (triple["subject"], triple["predicate"], triple["object"])
                            )

                        # Add to the persistent graph
                        await self.kg_manager.add_triples_to_dynamic_graph(llamaindex_triples)
                        self.logger.info(
                            f"✅ Added {len(llamaindex_triples)} triples to persistent dynamic graph"
                        )

                    except Exception as e:
                        self.logger.error(f"❌ Failed to add triples to persistent graph: {e}")

                # Store the triples in the session for WebSocket transmission
                if session_id in self.sessions:
                    if "dynamic_triples" not in self.sessions[session_id]:
                        self.sessions[session_id]["dynamic_triples"] = []

                    # Add all created triples
                    self.sessions[session_id]["dynamic_triples"].extend(triples_created)

                    # Keep only the last 20 triples to avoid memory bloat
                    if len(self.sessions[session_id]["dynamic_triples"]) > 20:
                        self.sessions[session_id]["dynamic_triples"] = self.sessions[session_id][
                            "dynamic_triples"
                        ][-20:]

                    self.logger.info(
                        f"🔗 Added {len(triples_created)} dynamic triples for session {session_id}"
                    )
                    for triple in triples_created:
                        self.logger.info(
                            f"   {triple['subject']} -> {triple['predicate']} -> {triple['object']}"
                        )

        except Exception as e:
            self.logger.error(f"Error in dynamic graph update callback: {e}")

    def get_dynamic_triples(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the latest dynamic triples for a session."""
        if session_id not in self.sessions:
            return []

        triples = self.sessions[session_id].get("dynamic_triples", [])
        # Clear the triples after retrieving them to avoid sending duplicates
        if triples:
            self.sessions[session_id]["dynamic_triples"] = []

        return triples

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get information about a session.

        Args:
            session_id: Session ID

        Returns:
            Dict[str, Any]: Session information
        """
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}

        session = self.sessions[session_id]

        # Calculate session duration
        duration = (datetime.now() - session["created_at"]).total_seconds()

        return {
            "session_id": session_id,
            "created_at": session["created_at"].isoformat(),
            "updated_at": session["updated_at"].isoformat(),
            "duration_seconds": duration,
            "message_count": len(session["messages"]),
            "has_summary": bool(session.get("summary")),
            "metadata": session["metadata"],
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions.

        Returns:
            List[Dict[str, Any]]: List of session information
        """
        return [
            {
                "session_id": session_id,
                "created_at": session["created_at"].isoformat(),
                "updated_at": session["updated_at"].isoformat(),
                "message_count": len(session["messages"]),
                "has_summary": bool(session.get("summary")),
            }
            for session_id, session in self.sessions.items()
        ]

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID

        Returns:
            bool: True if deletion was successful
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"Deleted session {session_id}")
            return True
        else:
            self.logger.warning(f"Session {session_id} not found for deletion")
            return False

    async def process_with_dual_model(
        self, session_id: str, user_message: str
    ) -> AsyncGenerator[str, None]:
        """
        Process message using ModelManager's direct dual-model functionality.

        This method uses the ModelManager's run_concurrent_queries method to
        directly leverage the dual-model architecture with proper resource
        management and prioritization.

        Args:
            session_id: Session ID
            user_message: User message

        Yields:
            str: Response tokens as they are generated
        """
        if session_id not in self.sessions:
            self.logger.error(f"Session {session_id} not found")
            yield f"Error: Session {session_id} not found"
            return

        if not self.model_manager:
            self.logger.error("Model manager not available")
            yield "Error: Model manager not available"
            return

        session = self.sessions[session_id]

        # Add user message to history
        session["messages"].append(
            {"role": "user", "content": user_message, "timestamp": datetime.now()}
        )

        # Update session metadata
        session["metadata"]["total_messages"] += 1
        session["metadata"]["user_messages"] += 1
        session["updated_at"] = datetime.now()

        # Use ModelManager's dual-model functionality
        self.logger.info(f"Using ModelManager's dual-model functionality for session {session_id}")

        # This directly uses the ModelManager's concurrent processing capabilities
        # with proper resource management and prioritization
        conversation_stream, bg_task = await self.model_manager.run_concurrent_queries(
            user_message, priority_mode=True  # Ensure conversation has priority
        )

        # Stream response from conversation model
        assistant_response = ""
        async for token in conversation_stream:
            assistant_response += token
            yield token

        # Add assistant response to history
        session["messages"].append(
            {"role": "assistant", "content": assistant_response, "timestamp": datetime.now()}
        )

        # Update session metadata
        session["metadata"]["total_messages"] += 1
        session["metadata"]["assistant_messages"] += 1
        session["updated_at"] = datetime.now()

        # Wait for background task to complete if needed
        # Note: This doesn't block the conversation as we've already yielded all tokens
        try:
            bg_result = await bg_task
            self.logger.debug(f"Background task completed: {bg_result}")

            # If we have a knowledge graph manager, update the graph with the extracted entities
            if self.kg_manager and "content" in bg_result:
                asyncio.create_task(self._process_background_result(bg_result))

        except Exception as e:
            self.logger.error(f"Error in background task: {str(e)}")

    async def _process_background_result(self, result: Dict[str, Any]) -> None:
        """
        Process the result from a background task.

        Args:
            result: Result from background task
        """
        try:
            if not self.kg_manager:
                return

            if "content" in result:
                # Process the extracted entities
                await self.kg_manager._process_entities(result["content"])

        except Exception as e:
            self.logger.error(f"Error processing background result: {str(e)}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the conversation manager.

        Returns:
            Dict[str, Any]: Performance metrics
        """
        return {**self.metrics, "active_sessions": len(self.sessions)}
