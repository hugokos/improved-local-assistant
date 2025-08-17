# Quick Start Guide

Get up and running with the Improved Local AI Assistant in minutes.

## First Launch

After [installation](installation.md), start the application:

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Launch the application
python run_app.py
```

The application will start on [http://localhost:8000](http://localhost:8000).

## Basic Usage

### 1. Web Interface

Open your browser and navigate to [http://localhost:8000](http://localhost:8000).

**First Conversation:**
1. Type a message in the input field
2. Press Enter or click Send
3. Watch the AI respond in real-time

**Example Questions:**
- "What is artificial intelligence?"
- "Explain machine learning in simple terms"
- "How do neural networks work?"

### 2. Voice Interface

Enable voice mode for hands-free interaction:

1. Click the microphone button or press `Shift+M`
2. Allow microphone permissions when prompted
3. Start speaking when the orb turns green
4. The AI will respond with voice

**Voice Commands:**
- `"new chat"` - Start a fresh conversation
- `"stop"` - Stop the AI from speaking
- `"faster"` / `"slower"` - Adjust speech speed
- `"cite sources"` - Show knowledge sources

### 3. Knowledge Graphs

Build dynamic knowledge from your conversations:

**Automatic Knowledge Extraction:**
- The system automatically extracts entities and relationships
- Knowledge accumulates across conversations
- View extracted knowledge in the side panels

**Manual Knowledge Import:**
```bash
# Download prebuilt knowledge graphs
python scripts/download_graphs.py survivalist

# Or build from your documents
python cli/graphrag_repl.py
```

## Key Features Demo

### GraphRAG Technology

Ask questions that require connecting information:

```
You: "What's the relationship between Python and machine learning?"
AI: [Provides detailed answer with source citations]
```

The system will:
1. Extract relevant knowledge from the conversation
2. Build connections between concepts
3. Provide sourced, accurate responses

### Conversational Memory

The assistant remembers context across the conversation:

```
You: "Tell me about Einstein"
AI: [Explains Einstein's contributions]

You: "What was his most famous equation?"
AI: [Knows "his" refers to Einstein from context]
```

### Real-time Learning

Watch knowledge grow in the side panels:
- **Recent Entities**: Shows extracted concepts
- **Knowledge Graph**: Displays relationships
- **Sources**: Lists information sources

## Advanced Usage

### Command Line Interface

For power users, use the CLI:

```bash
# Interactive GraphRAG shell
python cli/graphrag_repl.py

# System monitoring
python cli/test_system.py

# Model management
python cli/test_models.py
```

### API Integration

Use the REST API for custom applications:

```python
import requests

# Send a chat message
response = requests.post("http://localhost:8000/api/chat", json={
    "message": "What is quantum computing?",
    "session_id": "my-session",
    "use_kg": True
})

print(response.json()["response"])
```

### WebSocket Streaming

For real-time applications:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');
ws.send(JSON.stringify({
    message: "Explain blockchain technology",
    session_id: "websocket-demo"
}));
```

## Configuration

### Performance Tuning

Optimize for your hardware:

```bash
# Enable edge optimization
python cli/toggle_edge_optimization.py --enable

# Check system resources
python scripts/system_health_check.py
```

### Model Selection

Switch between different AI models:

```yaml
# In config.yaml
models:
  conversation_model: "hermes3:3b"  # Fast, good quality
  # conversation_model: "llama2:7b"  # Alternative option
```

### Voice Settings

Customize voice interaction:

```yaml
# In config.yaml
voice:
  enabled: true
  stt_model: "vosk-model-en-us-0.22"
  tts_voice: "en_US-lessac-medium"
  speech_speed: 1.0
```

## Tips and Tricks

### Maximize Performance
- Use SSD storage for faster model loading
- Close unnecessary applications to free RAM
- Enable edge optimization for resource-constrained systems

### Better Conversations
- Ask follow-up questions to build knowledge
- Use specific terms to improve entity extraction
- Reference previous topics to test memory

### Voice Optimization
- Use a good microphone for better recognition
- Speak clearly and at normal pace
- Use voice commands for efficient control

## Troubleshooting

### Common Issues

**Slow Responses:**
```bash
# Check system resources
python scripts/system_health_check.py

# Optimize memory usage
python scripts/memory_optimizer.py
```

**Voice Not Working:**
```bash
# Test voice components
python scripts/test_voice_fixes_comprehensive.py

# Download voice models
python scripts/download_voice_models.py
```

**Knowledge Graph Issues:**
```bash
# Rebuild knowledge graphs
python scripts/rebuild_survivalist_graph.py

# Test GraphRAG pipeline
python scripts/test_improved_graphrag.py
```

## Next Steps

Now that you're up and running:

1. **Explore Documentation**: Read the [User Guide](user-guide/web-interface.md)
2. **Join Community**: Visit [GitHub Discussions](https://github.com/hugokos/improved-local-assistant/discussions)
3. **Contribute**: See [Contributing Guide](developer-guide/contributing.md)
4. **Customize**: Build your own knowledge graphs and integrations

## Getting Help

- **Documentation**: [https://hugokos.github.io/improved-local-assistant](https://hugokos.github.io/improved-local-assistant)
- **Issues**: [GitHub Issues](https://github.com/hugokos/improved-local-assistant/issues)
- **Community**: [GitHub Discussions](https://github.com/hugokos/improved-local-assistant/discussions)
