# Storage and Persistence Testing Guide

This guide explains how data is stored in the improved-local-assistant and how to test storage and persistence over long conversations.

## Data Storage Structure

### Overview

The improved-local-assistant uses a **LlamaIndex-based storage system** with the following structure:

```
data/
├── dynamic_graph/           # Dynamic knowledge graph (grows during conversations)
│   └── main/               # Main dynamic graph index
│       ├── docstore.json           # Document storage
│       ├── graph_store.json        # Entity-relationship storage
│       ├── property_graph_store.json  # Property graph storage
│       ├── default__vector_store.json  # Vector embeddings
│       ├── image__vector_store.json    # Image embeddings (if any)
│       └── index_store.json        # Index metadata
├── prebuilt_graphs/        # Static knowledge graphs loaded at startup
│   └── survivalist/        # Example: survival knowledge graph
│       ├── docstore.json
│       ├── graph_store.json
│       ├── property_graph_store.json
│       ├── default__vector_store.json
│       ├── index_store.json
│       ├── graph_meta.json         # Graph metadata
│       └── meta.json              # Additional metadata
├── sessions/               # Conversation session data
│   └── sessions.json       # Session storage
├── graph_registry.json     # Registry of available graphs
└── kg_cache.json          # Knowledge graph cache
```

### Storage File Types

#### 1. **docstore.json** - Document Storage
```json
{
  "docstore/data": {
    "doc_id_1": {
      "__type__": "Document",
      "__data__": "{\"text\": \"document content\", \"metadata\": {...}}"
    }
  }
}
```

#### 2. **graph_store.json** - Entity-Relationship Storage
```json
{
  "graph_store/data": {
    "rel_map": {
      "entity1": [
        ["relationship", "entity2"],
        ["another_rel", "entity3"]
      ]
    }
  }
}
```

#### 3. **property_graph_store.json** - Property Graph Storage
```json
{
  "nodes": [
    {
      "id": "node_id",
      "type": "entity_type",
      "properties": {"name": "value"}
    }
  ],
  "edges": [
    {
      "source": "node1",
      "target": "node2",
      "type": "relationship_type",
      "properties": {}
    }
  ]
}
```

#### 4. **default__vector_store.json** - Vector Embeddings
```json
{
  "vector_store/data": {
    "embedding_dict": {
      "node_id": [0.1, 0.2, 0.3, ...],  // 384-dimensional vectors
    },
    "text_id_to_ref_doc_id": {
      "node_id": "doc_id"
    }
  }
}
```

#### 5. **index_store.json** - Index Metadata
```json
{
  "index_store/data": {
    "index_id": {
      "__type__": "VectorStoreIndex",
      "__data__": "{\"index_id\": \"...\", \"summary\": \"...\"}"
    }
  }
}
```

## Persistence Mechanism

### Dynamic Graph Updates

1. **Real-time Extraction**: During conversations, entities and relationships are extracted using TinyLlama
2. **Background Processing**: Extraction happens asynchronously without blocking conversation flow
3. **Incremental Updates**: New triples are added to the dynamic graph using `upsert_triplet_and_node()`
4. **Periodic Persistence**: Graph is persisted every 10 updates or 5 minutes (configurable)

### Persistence Triggers

```python
# Automatic persistence conditions
self._persist_update_threshold = 10      # persist after 10 updates
self._persist_interval = 300             # persist every 5 minutes

# Manual persistence
await kg_manager._persist_dynamic_graph()
```

## Testing Storage and Persistence

### 1. Quick Storage Inspection

```bash
# Inspect current storage structure
cd improved-local-assistant
python scripts/inspect_storage_structure.py
```

This shows:
- Directory structure and file sizes
- JSON file contents and statistics
- Storage usage breakdown
- Graph node/edge counts

### 2. Long Conversation Persistence Test

```bash
# Run automated long conversation test
python scripts/test_long_conversation_persistence.py --messages 50 --inspect-interval 10

# Interactive testing mode
python scripts/test_long_conversation_persistence.py --interactive

# Analyze existing storage without running new conversations
python scripts/test_long_conversation_persistence.py --analyze-storage
```

#### Test Features:

- **Automated Conversation**: Generates 50+ diverse test messages about survival topics
- **Storage Snapshots**: Takes periodic snapshots of storage state
- **Growth Analysis**: Shows how storage grows over time
- **Query Performance**: Tests retrieval performance with accumulated knowledge
- **Interactive Mode**: Real-time inspection and testing

### 3. Manual Testing with CLI Tools

```bash
# Test conversation with knowledge graph integration
python cli/test_conversation.py --interactive

# Test knowledge graph operations
python cli/test_knowledge_graph.py

# Inspect graph indices
python scripts/inspect_graph_indices.py
```

### 4. Monitoring Storage Growth

#### Example Test Session:

```bash
# Start the test
python scripts/test_long_conversation_persistence.py --messages 100 --inspect-interval 20

# Expected output:
# Processing message 1/100: Hello, my name is Alex...
# Processing message 20/100: How do I start a fire...
# Taking storage snapshot: after_20_messages
# Dynamic Graph Growth:
#   Total size: 1,234 → 5,678 bytes
#   Node count: 15 → 45
#   Edge count: 8 → 23
```

#### Interactive Inspection:

```bash
python scripts/test_long_conversation_persistence.py --interactive

# Available commands:
Inspection> /snapshot          # Take storage snapshot
Inspection> /analyze           # Analyze growth patterns
Inspection> /inspect           # Inspect storage files
Inspection> /query fire        # Test query performance
Inspection> /stats             # Show current statistics
Inspection> /persist           # Force persistence
```

## Key Metrics to Monitor

### Storage Growth Metrics

1. **File Sizes**: Monitor growth of JSON storage files
2. **Node Count**: Number of entities in the graph
3. **Edge Count**: Number of relationships
4. **Document Count**: Number of stored documents
5. **Vector Count**: Number of embeddings

### Performance Metrics

1. **Query Response Time**: Time to retrieve relevant information
2. **Extraction Time**: Time to extract entities from conversations
3. **Persistence Time**: Time to save graph to disk
4. **Memory Usage**: RAM consumption during operations

### Quality Metrics

1. **Entity Extraction Rate**: Entities extracted per message
2. **Relationship Discovery**: New relationships found
3. **Query Relevance**: Quality of retrieved information
4. **Storage Efficiency**: Data compression and deduplication

## Configuration Options

### Storage Configuration (config.yaml)

```yaml
knowledge_graphs:
  dynamic_storage: ./data/dynamic_graph     # Dynamic graph location
  prebuilt_directory: ./data/prebuilt_graphs  # Static graphs location
  max_triplets_per_chunk: 3                # Entities per extraction
  enable_caching: true                      # Enable query caching

extraction:
  max_time_seconds: 8.0                     # Max extraction time
  max_tokens: 1024                          # Max tokens per extraction
  max_triples_per_turn: 10                  # Max triples per message
  unload_after_extraction: true             # Unload model after use

hybrid_retriever:
  budget:
    max_chunks: 12                          # Max retrieved chunks
    graph_depth: 2                          # Graph traversal depth
```

## Troubleshooting

### Common Issues

1. **Storage Not Growing**: Check if extraction is enabled and models are loaded
2. **Large File Sizes**: Monitor vector storage growth, consider compression
3. **Slow Queries**: Check graph size and indexing performance
4. **Memory Issues**: Monitor RAM usage during long conversations

### Debug Commands

```bash
# Check if dynamic graph is being updated
python -c "
import json
with open('data/dynamic_graph/main/graph_store.json') as f:
    data = json.load(f)
    rel_map = data.get('graph_store/data', {}).get('rel_map', {})
    print(f'Entities: {len(rel_map)}')
    print(f'Relations: {sum(len(rels) for rels in rel_map.values())}')
"

# Monitor file changes during conversation
# On Linux/Mac:
watch -n 5 'ls -la data/dynamic_graph/main/'

# On Windows:
# Use PowerShell: while($true) { ls data/dynamic_graph/main/; sleep 5; clear }
```

## Best Practices

### For Testing

1. **Start Small**: Begin with 10-20 messages to understand the pattern
2. **Monitor Resources**: Watch CPU and memory usage during tests
3. **Take Snapshots**: Regular snapshots help track growth patterns
4. **Test Queries**: Verify that accumulated knowledge improves responses
5. **Check Persistence**: Ensure data survives application restarts

### For Production

1. **Regular Backups**: Backup the data directory regularly
2. **Monitor Growth**: Set up alerts for excessive storage growth
3. **Optimize Queries**: Use appropriate retrieval budgets
4. **Clean Old Data**: Implement data retention policies if needed
5. **Performance Testing**: Regular performance testing with realistic loads

## Example Test Workflow

```bash
# 1. Inspect initial state
python scripts/inspect_storage_structure.py

# 2. Run conversation test
python scripts/test_long_conversation_persistence.py --messages 30

# 3. Analyze results
python scripts/test_long_conversation_persistence.py --analyze-storage

# 4. Test query performance
python scripts/test_long_conversation_persistence.py --query-test

# 5. Interactive exploration
python scripts/test_long_conversation_persistence.py --interactive
```

This comprehensive testing approach ensures that the dynamic graph generation and persistence work correctly over extended conversations, providing insights into storage patterns, performance characteristics, and data quality.
