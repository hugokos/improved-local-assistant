# Knowledge Graphs Directory

This directory contains prebuilt knowledge graphs for the Improved Local Assistant.

## 📦 Download Graphs

Use the download script to get prebuilt graphs:

```bash
# Interactive selection
python scripts/download_graphs.py

# Download specific graphs
python scripts/download_graphs.py survivalist medical

# Download all graphs
python scripts/download_graphs.py all

# List available graphs
python scripts/download_graphs.py --list
```

## 📊 Available Graphs

### Survivalist Knowledge Base
- **Size**: 45MB compressed, ~180MB extracted
- **Entities**: 2,847 unique entities
- **Relationships**: 8,234 relationships
- **Content**: Outdoor survival, bushcraft, emergency preparedness, wilderness skills
- **Use Cases**: Survival questions, outdoor activities, emergency planning

### Medical Knowledge Base
- **Size**: 78MB compressed, ~320MB extracted  
- **Entities**: 4,521 unique entities
- **Relationships**: 12,847 relationships
- **Content**: Health information, medical procedures, anatomy, pharmacology
- **Use Cases**: Health questions, medical research, symptom analysis

### Technical Documentation
- **Size**: 92MB compressed, ~380MB extracted
- **Entities**: 5,234 unique entities
- **Relationships**: 15,672 relationships
- **Content**: Programming languages, frameworks, APIs, development tools
- **Use Cases**: Programming help, API documentation, technical troubleshooting

## 🔧 Custom Graphs

To create your own knowledge graphs:

```bash
# Use the kg_builder tool
cd kg_builder
python src/graph_builder.py --input your_documents/ --output ../improved-local-assistant/data/graphs/custom/
```

For more information, see the [Graph Builder Documentation](../../kg_builder/README.md).

## 📁 Directory Structure

After downloading graphs, your directory will look like:

```
data/graphs/
├── README.md                 # This file
├── survivalist/             # Survivalist knowledge graph
│   ├── kg.json              # Property graph data
│   ├── graph_meta.json      # Graph metadata
│   ├── triples.json         # Compatibility triples
│   └── vector_store.json    # Vector embeddings
├── medical/                 # Medical knowledge graph
│   └── ...
└── technical/               # Technical knowledge graph
    └── ...
```

## 🚀 Usage

Once graphs are downloaded, they're automatically available in the application:

### Web Interface
1. Start the app: `python run_app.py`
2. Open http://localhost:8000
3. Ask questions related to your downloaded graphs
4. See citations and knowledge graph updates in real-time

### Command Line
```bash
# Interactive GraphRAG REPL
python cli/graphrag_repl.py

# Test with specific questions
> How do I start a fire without matches?
> What are the symptoms of dehydration?
> How do I use Python's asyncio library?
```

### API Integration
```python
import requests

response = requests.post("http://localhost:8000/api/chat", json={
    "message": "Tell me about wilderness first aid",
    "session_id": "demo-session",
    "use_kg": True
})

print(response.json())
```

## 🔍 Troubleshooting

### Graph Not Loading
```bash
# Check if graph exists
ls -la data/graphs/survivalist/

# Validate graph structure
python scripts/validate_graphs.py --graph survivalist

# Rebuild graph index
python scripts/rebuild_graph_index.py --graph survivalist
```

### Download Issues
```bash
# Check network connectivity
python scripts/download_graphs.py --test-connection

# Manual download with verbose output
python scripts/download_graphs.py survivalist --verbose

# Clear download cache
rm -rf /tmp/graph_downloads/
```

### Performance Issues
```bash
# Check graph statistics
python scripts/graph_stats.py --graph survivalist

# Optimize graph for better performance
python scripts/optimize_graph.py --input survivalist --output survivalist_optimized
```

## 📚 Additional Resources

- [Prebuilt Graphs Distribution Guide](../docs/PREBUILT_GRAPHS_GUIDE.md)
- [Graph Builder Documentation](../../kg_builder/README.md)
- [API Documentation](../docs/API.md)
- [GraphRAG Architecture](../docs/ARCHITECTURE.md)

---

**Note**: Prebuilt graphs are distributed separately from the main repository to keep the codebase lightweight. The download script handles all the complexity of fetching, verifying, and extracting graphs automatically.