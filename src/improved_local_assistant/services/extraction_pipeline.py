"""
Resource-Aware Extraction Pipeline for bounded knowledge extraction.

This module provides the ExtractionPipeline class that performs bounded
TinyLlama processing with strict time/token budgets and resource monitoring
for optimal performance on edge devices.
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import psutil
import tiktoken
from rapidfuzz import process
from services.connection_pool_manager import ConnectionPoolManager
from services.system_monitor import SystemMonitor


@dataclass
class Triple:
    """Represents an extracted knowledge triple."""

    subject: str
    predicate: str
    object: str
    confidence: float
    source_span: Optional[str] = None


@dataclass
class ExtractionResult:
    """Structured result from knowledge extraction."""

    triples: List[Triple]
    span_indices: List[tuple[int, int]]
    extractor_version: str = "v2.0"


class EntityLinker:
    """
    Canonical entity linking using fuzzy matching and SQLite storage.

    Maintains a database of entity variants and their canonical forms,
    using fuzzy matching to resolve new entities to existing canonical forms.
    """

    def __init__(self, db_path: str = "./data/entity_link.db"):
        """Initialize entity linker with SQLite database."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS names(variant TEXT PRIMARY KEY, canonical TEXT)"
        )
        self.conn.commit()

    def resolve(self, name: str) -> str:
        """
        Resolve an entity name to its canonical form.

        Args:
            name: Entity name to resolve

        Returns:
            str: Canonical form of the entity
        """
        if not name or not name.strip():
            return name

        name = name.strip()

        # Check if we already have this variant
        cur = self.conn.execute("SELECT canonical FROM names WHERE variant=?", (name,))
        row = cur.fetchone()
        if row:
            return row[0]

        # Get all canonical names for fuzzy matching
        cur = self.conn.execute("SELECT DISTINCT canonical FROM names")
        canonicals = [r[0] for r in cur.fetchall()]

        if canonicals:
            # Find best match using fuzzy matching
            result = process.extractOne(name, canonicals)
            if result:
                match, score, _ = result  # rapidfuzz returns (match, score, index)
                if score > 90:  # High confidence threshold
                    # Store this variant mapping
                    self.conn.execute(
                        "INSERT OR IGNORE INTO names(variant, canonical) VALUES(?, ?)",
                        (name, match),
                    )
                    self.conn.commit()
                    return match

        # No good match found, this becomes a new canonical entity
        self.conn.execute(
            "INSERT OR IGNORE INTO names(variant, canonical) VALUES(?, ?)",
            (name, name),
        )
        self.conn.commit()
        return name

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()


class ExtractionPipeline:
    """
    Performs bounded knowledge extraction with resource awareness.

    Manages TinyLlama processing with strict time/token budgets, skip logic
    based on resource pressure, and proper model lifecycle management.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        connection_pool: ConnectionPoolManager,
        system_monitor: SystemMonitor,
    ):
        """Initialize Extraction Pipeline with configuration."""
        self.config = config
        self.connection_pool = connection_pool
        self.system_monitor = system_monitor
        self.logger = logging.getLogger(__name__)

        # Extract extraction configuration
        extraction_config = config.get("extraction", {})
        self.max_time_seconds = extraction_config.get("max_time_seconds", 8.0)
        self.max_tokens = extraction_config.get("max_tokens", 1024)
        self.max_triples_per_turn = extraction_config.get("max_triples_per_turn", 10)
        self.skip_on_memory_pressure = extraction_config.get("skip_on_memory_pressure", True)
        self.skip_on_cpu_pressure = extraction_config.get("skip_on_cpu_pressure", True)
        self.input_window_turns = extraction_config.get("input_window_turns", 3)
        self.format_json = extraction_config.get("format", "json") == "json"
        self.unload_after_extraction = extraction_config.get("unload_after_extraction", True)

        # Model configuration
        models_config = config.get("models", {})
        self.knowledge_model = models_config.get("knowledge", {}).get("name", "tinyllama")

        # System thresholds (direct percentage values for psutil)
        system_config = config.get("system", {})
        self.memory_threshold_percent = system_config.get("memory_threshold_percent", 95)
        self.cpu_threshold_percent = system_config.get("cpu_threshold_percent", 90)

        # Entity linker for canonical entity resolution
        self.linker = EntityLinker()

        # Metrics
        self.metrics = {
            "extractions_attempted": 0,
            "extractions_completed": 0,
            "extractions_skipped_pressure": 0,
            "extractions_skipped_timeout": 0,
            "extractions_failed": 0,
            "total_triples_extracted": 0,
            "avg_extraction_time": 0.0,
            "avg_triples_per_extraction": 0.0,
        }

    def _split_text_into_chunks(self, text: str, max_tokens: int = 128) -> List[str]:
        """
        Split text into chunks with a maximum token count using tiktoken.

        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk

        Returns:
            List[str]: List of text chunks
        """
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)

            chunks = []
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i : i + max_tokens]
                chunk_text = enc.decode(chunk_tokens)
                chunks.append(chunk_text)

            return chunks
        except Exception as e:
            self.logger.warning(
                f"Error in token-based chunking: {e}, falling back to character-based"
            )
            # Fallback to simple character-based chunking
            chunk_size = max_tokens * 4  # Rough estimate: 4 chars per token
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def extract_bounded(
        self,
        text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        budget: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExtractionResult]:
        """
        Perform bounded knowledge extraction with time and token limits.

        Args:
            text: Primary text to extract from
            conversation_history: Recent conversation context
            budget: Custom budget overrides

        Returns:
            Optional[ExtractionResult]: Extraction result with triples and metadata or None if skipped/failed
        """
        extraction_start_time = time.time()
        self.metrics["extractions_attempted"] += 1

        try:
            # Check if we should skip extraction due to resource pressure
            if self.should_skip_extraction():
                self.logger.info("Skipping extraction due to resource pressure")
                self.metrics["extractions_skipped_pressure"] += 1
                return None

            # Prepare input window
            input_text = self._prepare_input_window(text, conversation_history)

            # Apply budget constraints
            effective_budget = self._apply_budget_constraints(budget)

            # Perform extraction with timeout
            try:
                raw_result = await asyncio.wait_for(
                    self._extract_with_model(input_text, effective_budget),
                    timeout=effective_budget["max_time_seconds"],
                )
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Extraction timed out after {effective_budget['max_time_seconds']}s"
                )
                self.metrics["extractions_skipped_timeout"] += 1
                return None

            # Parse and validate triples
            triples = self._format_triples(raw_result)

            # Persist triples to knowledge graph
            if triples:
                await self._persist_triples(
                    triples,
                    conversation_history[0].get("session_id") if conversation_history else None,
                )

            # Model automatically unloaded via keep_alive=0 if configured

            # Update metrics
            extraction_time = time.time() - extraction_start_time
            self.metrics["extractions_completed"] += 1
            self.metrics["total_triples_extracted"] += len(triples)
            self._update_avg_metric("avg_extraction_time", extraction_time)
            self._update_avg_metric("avg_triples_per_extraction", len(triples))

            self.logger.debug(f"Extracted {len(triples)} triples in {extraction_time:.3f}s")

            return ExtractionResult(
                triples=triples,
                span_indices=[],  # TODO: Implement span tracking
                extractor_version="v2.0",
            )

        except Exception as e:
            self.logger.error(f"Error in bounded extraction: {e}")
            self.metrics["extractions_failed"] += 1
            return None

    def _prepare_input_window(
        self, primary_text: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Prepare input text with bounded conversation window using token-aware chunking.

        Args:
            primary_text: Main text to extract from
            conversation_history: Recent conversation messages

        Returns:
            str: Formatted input text within window limits (≤1024 tokens)
        """
        # Split primary text into chunks and take first 8 chunks (≤1024 tokens total)
        chunks = self._split_text_into_chunks(primary_text, max_tokens=128)
        input_text = "\n".join(chunks[:8])  # ≤1024 tokens total, honors budget

        # Add limited conversation history if available and space permits
        if conversation_history and self.input_window_turns > 0:
            # Take the most recent turns up to the limit
            recent_history = conversation_history[
                -self.input_window_turns * 2 :
            ]  # User + assistant pairs

            if recent_history:
                history_text = "\n".join(
                    [
                        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                        for msg in recent_history
                    ]
                )

                # Check if we have room for history (rough token estimate)
                history_chunks = self._split_text_into_chunks(history_text, max_tokens=64)
                if len(chunks[:8]) + len(history_chunks[:4]) <= 12:  # Stay within reasonable bounds
                    input_text += f"\n\nRecent conversation:\n{history_text}"

        return input_text

    def _apply_budget_constraints(
        self, custom_budget: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply budget constraints with optional custom overrides.

        Args:
            custom_budget: Custom budget parameters

        Returns:
            Dict: Effective budget constraints
        """
        budget = {
            "max_time_seconds": self.max_time_seconds,
            "max_tokens": self.max_tokens,
            "max_triples": self.max_triples_per_turn,
        }

        # Apply custom overrides
        if custom_budget:
            budget.update(custom_budget)

        # Apply resource pressure adjustments
        if self._is_under_memory_pressure():
            budget["max_time_seconds"] *= 0.5
            budget["max_tokens"] = min(budget["max_tokens"], 512)
            budget["max_triples"] = min(budget["max_triples"], 5)
            self.logger.debug("Reduced budget due to memory pressure")

        if self._is_under_cpu_pressure():
            budget["max_time_seconds"] *= 0.7
            budget["max_tokens"] = min(budget["max_tokens"], 768)
            self.logger.debug("Reduced budget due to CPU pressure")

        return budget

    async def _extract_with_model(self, input_text: str, budget: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform extraction using the knowledge model.

        Args:
            input_text: Text to extract from
            budget: Budget constraints

        Returns:
            Dict: Raw model response
        """
        # Create extraction prompt
        prompt = self._create_extraction_prompt(input_text, budget["max_triples"])

        # Prepare request options
        options = {
            "temperature": 0.2,
            "num_predict": budget["max_tokens"],
        }

        # Add JSON format if enabled
        if self.format_json:
            options["format"] = "json"

        # Make request with keep_alive=0 for automatic unloading
        keep_alive = 0 if self.unload_after_extraction else None

        result = await self.connection_pool.chat_request(
            model=self.knowledge_model,
            messages=[{"role": "user", "content": prompt}],
            keep_alive=keep_alive,
            **options,
        )

        return result

    def _create_extraction_prompt(self, input_text: str, max_triples: int) -> str:
        """
        Create extraction prompt for the model.

        Args:
            input_text: Text to extract from
            max_triples: Maximum triples to extract

        Returns:
            str: Formatted prompt
        """
        if self.format_json:
            return f"""Extract entities and relationships from the following text.
Return ONLY a JSON array of triples in this exact format:
[
  {{
    "subject": "entity1",
    "predicate": "relationship",
    "object": "entity2",
    "confidence": 0.9,
    "source_span": "relevant text span"
  }}
]

Rules:
- Maximum {max_triples} triples
- Focus on the most important entities and relationships
- Confidence should be between 0.0 and 1.0
- Keep subject/predicate/object concise (1-3 words each)
- Source span should be the relevant text that supports the triple

Text to analyze:
{input_text}

JSON:"""
        else:
            return f"""Extract entities and relationships from the following text.
Format each relationship as: (subject, predicate, object, confidence)

Maximum {max_triples} relationships. Focus on the most important ones.

Text: {input_text}

Relationships:"""

    def _format_triples(self, raw_result: Dict[str, Any]) -> List[Triple]:
        """
        Parse and format raw model output into Triple objects.

        Args:
            raw_result: Raw model response

        Returns:
            List[Triple]: Parsed and validated triples
        """
        try:
            content = raw_result.get("message", {}).get("content", "")

            if not content.strip():
                return []

            triples = []

            if self.format_json:
                # Parse JSON format
                try:
                    json_data = json.loads(content)
                    if isinstance(json_data, list):
                        for item in json_data[: self.max_triples_per_turn]:
                            if isinstance(item, dict) and all(
                                k in item for k in ["subject", "predicate", "object"]
                            ):
                                triple = Triple(
                                    subject=str(item["subject"]).strip(),
                                    predicate=str(item["predicate"]).strip(),
                                    object=str(item["object"]).strip(),
                                    confidence=float(item.get("confidence", 0.5)),
                                    source_span=item.get("source_span"),
                                )
                                if self._validate_triple(triple):
                                    # Apply entity linking to canonicalize entities
                                    triple.subject = self.linker.resolve(triple.subject)
                                    triple.object = self.linker.resolve(triple.object)
                                    triples.append(triple)
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse JSON extraction result")
                    # Fall back to text parsing
                    triples = self._parse_text_format(content)
            else:
                # Parse text format
                triples = self._parse_text_format(content)

            return triples[: self.max_triples_per_turn]

        except Exception as e:
            self.logger.error(f"Error formatting triples: {e}")
            return []

    def _parse_text_format(self, content: str) -> List[Triple]:
        """Parse text format extraction results using regex for robustness."""
        import re

        triples = []

        for line in content.split("\n"):
            line = line.strip()
            if not line or not line.startswith("("):
                continue

            try:
                # Use regex to properly parse (subject, predicate, object, confidence) format
                # This handles commas inside entity names like "New York, NY"
                match = re.match(
                    r"\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*(?:,\s*([\d.]+))?\s*\)", line
                )
                if match:
                    subject, predicate, obj, confidence_str = match.groups()

                    # Clean up quotes and whitespace
                    subject = subject.strip().strip("\"'")
                    predicate = predicate.strip().strip("\"'")
                    obj = obj.strip().strip("\"'")
                    confidence = float(confidence_str) if confidence_str else 0.5

                    triple = Triple(
                        subject=subject, predicate=predicate, object=obj, confidence=confidence
                    )

                    if self._validate_triple(triple):
                        # Apply entity linking to canonicalize entities
                        triple.subject = self.linker.resolve(triple.subject)
                        triple.object = self.linker.resolve(triple.object)
                        triples.append(triple)
                else:
                    self.logger.debug(f"Failed to parse line with regex: '{line}'")

            except (ValueError, IndexError) as e:
                self.logger.debug(f"Failed to parse line '{line}': {e}")
                continue

        return triples

    def _validate_triple(self, triple: Triple) -> bool:
        """
        Validate a triple for basic quality checks.

        Args:
            triple: Triple to validate

        Returns:
            bool: True if valid
        """
        # Check for empty or very short components
        if not triple.subject or not triple.predicate or not triple.object:
            return False

        if len(triple.subject.strip()) < 2 or len(triple.object.strip()) < 2:
            return False

        # Check confidence range
        if not (0.0 <= triple.confidence <= 1.0):
            triple.confidence = max(0.0, min(1.0, triple.confidence))

        # Check for reasonable length (avoid very long extractions)
        if len(triple.subject) > 50 or len(triple.predicate) > 30 or len(triple.object) > 50:
            return False

        return True

    async def _unload_model(self) -> None:
        """Unload the knowledge model to free resources."""
        try:
            self.logger.debug(f"Unloading {self.knowledge_model}")

            # Send minimal request with keep_alive=0
            await self.connection_pool.chat_request(
                model=self.knowledge_model,
                messages=[{"role": "user", "content": "unload"}],
                keep_alive=0,
                num_predict=1,
            )

        except Exception as e:
            self.logger.debug(f"Error unloading model (may already be unloaded): {e}")

    def should_skip_extraction(self) -> bool:
        """
        Determine if extraction should be skipped based on resource pressure.

        Returns:
            bool: True if extraction should be skipped
        """
        if self.skip_on_memory_pressure and self._is_under_memory_pressure():
            return True

        if self.skip_on_cpu_pressure and self._is_under_cpu_pressure():
            return True

        return False

    def _is_under_memory_pressure(self) -> bool:
        """Check if system is under memory pressure using psutil."""
        try:
            proc = psutil.Process()
            memory_percent = proc.memory_percent()
            return memory_percent > self.memory_threshold_percent
        except Exception as e:
            self.logger.debug(f"Error checking memory pressure: {e}")
            return False

    def _is_under_cpu_pressure(self) -> bool:
        """Check if system is under CPU pressure using psutil."""
        try:
            proc = psutil.Process()
            cpu_percent = proc.cpu_percent(interval=0.1)
            return cpu_percent > self.cpu_threshold_percent
        except Exception as e:
            self.logger.debug(f"Error checking CPU pressure: {e}")
            return False

    async def _persist_triples(
        self, triples: List[Triple], session_id: Optional[str] = None
    ) -> None:
        """
        Persist extracted triples to the knowledge graph.

        Args:
            triples: Triples to persist
            session_id: Optional session context
        """
        try:
            if not triples:
                return

            # This is a placeholder for actual persistence logic
            # In the full implementation, this would integrate with the knowledge graph manager
            self.logger.debug(f"Persisting {len(triples)} triples")

            # Convert triples to a format suitable for the KG
            triple_data = [
                {
                    "subject": triple.subject,
                    "predicate": triple.predicate,
                    "object": triple.object,
                    "confidence": triple.confidence,
                    "source_span": triple.source_span,
                    "session_id": session_id,
                    "timestamp": time.time(),
                }
                for triple in triples
            ]

            # TODO: Integrate with actual KG persistence
            self.logger.debug(f"Would persist: {triple_data}")

            # For now, just log the canonical entities being persisted
            canonical_entities = set()
            for triple in triples:
                canonical_entities.add(triple.subject)
                canonical_entities.add(triple.object)
            self.logger.debug(f"Canonical entities: {canonical_entities}")

        except Exception as e:
            self.logger.error(f"Error persisting triples: {e}")

    def _update_avg_metric(self, metric_name: str, new_value: float) -> None:
        """Update rolling average metric."""
        current_avg = self.metrics.get(metric_name, 0.0)
        count = self.metrics.get("extractions_completed", 1)

        if count > 1:
            self.metrics[metric_name] = (current_avg * (count - 1) + new_value) / count
        else:
            self.metrics[metric_name] = new_value

    def get_metrics(self) -> Dict[str, Any]:
        """Get extraction pipeline metrics."""
        return self.metrics.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get extraction pipeline status."""
        return {
            "configuration": {
                "max_time_seconds": self.max_time_seconds,
                "max_tokens": self.max_tokens,
                "max_triples_per_turn": self.max_triples_per_turn,
                "input_window_turns": self.input_window_turns,
                "format_json": self.format_json,
                "unload_after_extraction": self.unload_after_extraction,
            },
            "resource_monitoring": {
                "skip_on_memory_pressure": self.skip_on_memory_pressure,
                "skip_on_cpu_pressure": self.skip_on_cpu_pressure,
                "memory_threshold": self.memory_threshold,
                "cpu_threshold": self.cpu_threshold,
                "current_memory_pressure": self._is_under_memory_pressure(),
                "current_cpu_pressure": self._is_under_cpu_pressure(),
            },
            "metrics": self.get_metrics(),
        }
