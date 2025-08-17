# Property Graph Migration Guide

This guide explains how to migrate from the flat triples system to the new Property Graph architecture using LlamaIndex's `PropertyGraphIndex` and `SimplePropertyGraphStore`.

## Overview

The migration converts your GraphRAG stack from:
- **Flat triples**: `(subject, predicate, object)` stored in `triples.json`
- **SimpleGraphStore**: Basic graph storage with limited metadata

To:
- **Property graphs**: Rich nodes and edges with metadata stored in `kg.json`
- **SimplePropertyGraphStore**: In-memory property graph with full persistence

## Benefits of Property Graphs

1. **Rich Metadata**: Nodes and edges can carry arbitrary properties
2. **Better Context**: Source attribution and chunk tracking
3. **Entity Disambiguation**: Canonical vs display names
4. **Relationship Properties**: Edge metadata for confidence, source, etc.
5. **Backward Compatibility**: Automatic `triples.json` generation

## Migration Steps

### 1. Update Configuration

Add property graph configuration to `config.yaml`:

```yaml
graph:
  type: property  # "simple" or "property"
  store: simple   # Use SimplePropertyGraphStore (in-memory)
```

### 2. Run Migration Script

Convert existing graphs to property format:

```bash
python migrate_to_property_graphs.py
```

This script:
- Converts `triples.json` to `kg.json` with property graph structure
- Creates `graph_meta.json` with graph type metadata
- Preserves original `triples.json` for compatibility

### 3. Test the Migration

Run the test script to verify everything works:

```bash
python test_property_graph_conversion.py
```

### 4. Rebuild Graph Registry

Update the graph registry with property graph embeddings:

```bash
python scripts/build_graph_registry.py
```

## Property Graph Structure

### kg.json Format

```json
{
  "nodes": {
    "uuid-1": {
      "label": "ENTITY",
      "properties": {
        "original_name": "Fire",
        "canonical_name": "fire",
        "context_count": 3,
        "migrated_from_triples": true
      }
    }
  },
  "edges": [
    {
      "source_id": "uuid-1",
      "target_id": "uuid-2",
      "label": "requires",
      "properties": {
        "src_display": "Fire",
        "tgt_display": "oxygen",
        "chunk_id": 0,
        "migrated_from_triples": true
      }
    }
  ]
}
```

### Compatibility Layer

The system maintains backward compatibility by:
- Keeping original `triples.json` files
- Auto-generating `triples.json` from property graphs
- Supporting both graph types in the same codebase

## Code Changes

### Graph Builder

```python
# Old approach
from llama_index.core.graph_stores import SimpleGraphStore
graph_store = SimpleGraphStore()

# New approach
from llama_index.core.graph_stores.simple import SimplePropertyGraphStore
graph_store = SimplePropertyGraphStore()
```

### Index Creation

```python
# Old approach
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(graph_store=graph_store)
)

# New approach
index = PropertyGraphIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(property_graph_store=graph_store),
    kg_extractors=[ImplicitPathExtractor()]
)
```

### Dynamic Updates

```python
# Old approach
graph.upsert_triplet((subject, predicate, object))

# New approach
graph_store.upsert_nodes([{
    "id": node_id,
    "label": "ENTITY",
    "properties": {"name": entity_name}
}])

graph_store.upsert_relations([{
    "source_id": src_id,
    "target_id": tgt_id,
    "label": predicate,
    "properties": {"chunk_id": chunk_id}
}])
```

### Graph Querying

```python
# Old approach
for subj, pred, obj in graph.triples:
    process_triple(subj, pred, obj)

# New approach
for edge in property_graph.edges:
    src = property_graph.nodes[edge["source_id"]]
    tgt = property_graph.nodes[edge["target_id"]]
    src_name = src["properties"]["original_name"]
    tgt_name = tgt["properties"]["original_name"]
    predicate = edge["label"]
    process_relationship(src_name, predicate, tgt_name)
```

## Persistence Changes

### File Structure

```
graph_directory/
├── kg.json                    # Main property graph data
├── triples.json              # Compatibility flat triples
├── graph_meta.json           # Graph type and metadata
├── docstore.json            # Document store
├── vector_store.json        # Entity embeddings
└── index_store.json         # Index metadata
```

### Loading Graphs

The system automatically detects graph type from `graph_meta.json`:

```python
# Automatic detection
meta = json.load(open("graph_meta.json"))
if meta.get("graph_type") == "property":
    # Load as PropertyGraphIndex
else:
    # Load as KnowledgeGraphIndex
```

## Retrieval Updates

### Hybrid Retriever

The hybrid retriever automatically adapts to property graphs:

```python
# PropertyGraphIndex uses as_retriever() with include_text=True
retriever = property_graph_index.as_retriever(
    include_text=True,
    similarity_top_k=top_k
)

# KnowledgeGraphIndex uses KnowledgeGraphRAGRetriever
retriever = KnowledgeGraphRAGRetriever(
    index=knowledge_graph_index,
    depth=depth,
    similarity_top_k=top_k
)
```

### Graph Router

The graph router supports both graph types with automatic detection:

```python
# Loads both PropertyGraphIndex and KnowledgeGraphIndex
vector_idx, property_idx = self._load_indices(graph_id)

# Creates appropriate retrievers for each type
if isinstance(idx, PropertyGraphIndex):
    retriever = idx.as_retriever(include_text=True)
elif isinstance(idx, KnowledgeGraphIndex):
    retriever = KnowledgeGraphRAGRetriever(index=idx)
```

## Testing

### Unit Tests

Run the property graph conversion test:

```bash
python test_property_graph_conversion.py
```

### Integration Tests

Test the full GraphRAG pipeline:

```bash
python cli/graphrag_repl.py
```

### Performance Tests

Compare retrieval performance:

```bash
python scripts/test_improved_graphrag.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you have `llama-index==0.13.0` or later
2. **Missing kg.json**: Run the migration script on your graph directories
3. **Graph Type Detection**: Check that `graph_meta.json` exists and has `graph_type` field
4. **Embedding Issues**: Verify embedding model is properly configured

### Debug Commands

```bash
# Check graph structure
python -c "
import json
kg = json.load(open('data/prebuilt_graphs/survivalist/kg.json'))
print(f'Nodes: {len(kg[\"nodes\"])}')
print(f'Edges: {len(kg[\"edges\"])}')
"

# Verify graph type
python -c "
import json
meta = json.load(open('data/prebuilt_graphs/survivalist/graph_meta.json'))
print(f'Graph type: {meta.get(\"graph_type\", \"unknown\")}')
"
```

## Rollback Plan

If you need to rollback to simple graphs:

1. Change `config.yaml`: `graph.type: simple`
2. The system will automatically use `triples.json` files
3. All existing functionality continues to work

## Performance Considerations

- **Memory Usage**: Property graphs use slightly more memory due to rich metadata
- **Persistence**: `kg.json` files are larger than `triples.json` but more informative
- **Retrieval**: Property graphs may have better retrieval quality due to richer context
- **Build Time**: Similar build times, with better entity disambiguation

## Next Steps

After migration:

1. **Test Thoroughly**: Run all test suites to ensure compatibility
2. **Monitor Performance**: Check memory usage and response times
3. **Update Documentation**: Update any custom documentation
4. **Train Team**: Ensure team understands new property graph concepts
5. **Gradual Rollout**: Consider migrating graphs incrementally

## Support

For issues or questions:
- Check the test scripts for examples
- Review the LlamaIndex PropertyGraph documentation
- Examine the migration logs for specific error messages
