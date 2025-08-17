# API Documentation

The Improved Local AI Assistant provides a comprehensive REST API and WebSocket interface for all functionality.

## Base URL

```
http://localhost:8000
```

## Interactive Documentation

Visit `/docs` for interactive Swagger/OpenAPI documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Authentication

This is a local-only application with no authentication required.

## Core Endpoints

### Health Check
```http
GET /api/health
```

Returns system health status and component availability.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-26T12:00:00Z",
  "components": {
    "ollama": "connected",
    "models": "loaded",
    "graphs": "available"
  }
}
```

### Chat API
```http
POST /api/chat
```

Send a message and receive AI response with optional knowledge graph integration.

**Request:**
```json
{
  "message": "What is machine learning?",
  "session_id": "my-session",
  "use_kg": true,
  "max_tokens": 1000
}
```

**Response:**
```json
{
  "response": "Machine learning is...",
  "citations": [
    {
      "id": "1",
      "title": "Introduction to ML",
      "source": "knowledge_graph"
    }
  ],
  "session_id": "my-session",
  "timestamp": "2025-01-26T12:00:00Z"
}
```

### Knowledge Graph Management

#### Get Graph Statistics
```http
GET /api/graph/stats
```

**Response:**
```json
{
  "graphs": {
    "dynamic_main": {
      "nodes": 150,
      "edges": 300,
      "last_updated": "2025-01-26T12:00:00Z"
    },
    "survivalist": {
      "nodes": 500,
      "edges": 1200,
      "last_updated": "2025-01-25T10:00:00Z"
    }
  }
}
```

#### Export Graph
```http
GET /api/graph/{graph_id}/export_native
```

**Parameters:**
- `limit` (optional): Limit number of nodes
- `hops` (optional): Maximum hops from seed nodes
- `max_nodes` (optional): Maximum nodes in export

**Response:** ZIP file containing graph data

#### Import Graph
```http
POST /api/graph/import_native
```

**Form Data:**
- `file`: ZIP file containing graph data
- `graph_id`: Target graph identifier
- `graph_type`: `dynamic` or `modular`
- `replace`: `true` or `false`
- `merge_strategy`: `union`, `prefer_base`, or `prefer_incoming`

#### Get Subgraph
```http
GET /api/graph/{graph_id}/subgraph
```

**Parameters:**
- `query`: Search query
- `max_hops`: Maximum hops from query nodes
- `max_nodes`: Maximum nodes to return

**Response:**
```json
{
  "nodes": [...],
  "edges": [...],
  "metadata": {
    "query": "machine learning",
    "total_nodes": 25,
    "total_edges": 45
  }
}
```

### System Information

#### Get System Info
```http
GET /api/system/info
```

**Response:**
```json
{
  "version": "2.0.0",
  "python_version": "3.11.0",
  "models": {
    "conversation": "hermes3:3b",
    "knowledge": "tinyllama:latest",
    "embedding": "BAAI/bge-small-en-v1.5"
  },
  "system": {
    "cpu_usage": 15.2,
    "memory_usage": 45.8,
    "disk_usage": 23.1
  }
}
```

#### Get Performance Metrics
```http
GET /api/metrics
```

**Response:**
```json
{
  "requests_total": 1250,
  "requests_per_minute": 12.5,
  "average_response_time": 1.8,
  "active_sessions": 3,
  "graph_operations": {
    "queries": 450,
    "updates": 125,
    "exports": 5
  }
}
```

## WebSocket API

### Chat WebSocket
```
ws://localhost:8000/ws/chat/{session_id}
```

Real-time chat interface with streaming responses and live citations.

**Message Format:**
```json
{
  "type": "message",
  "content": "Your message here",
  "use_kg": true
}
```

**Response Format:**
```json
{
  "type": "token",
  "content": "streaming",
  "session_id": "session-123"
}
```

**Citation Format:**
```json
{
  "type": "citation",
  "citations": [
    {
      "id": "1",
      "title": "Source Title",
      "source": "knowledge_graph"
    }
  ]
}
```

### System Monitor WebSocket
```
ws://localhost:8000/ws/monitor
```

Real-time system metrics and performance monitoring.

**Message Format:**
```json
{
  "type": "metrics",
  "timestamp": "2025-01-26T12:00:00Z",
  "cpu_usage": 15.2,
  "memory_usage": 45.8,
  "active_sessions": 3,
  "graph_stats": {
    "total_nodes": 650,
    "total_edges": 1500
  }
}
```

## Error Handling

All API endpoints return consistent error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "session_id",
      "issue": "Required field missing"
    }
  },
  "timestamp": "2025-01-26T12:00:00Z"
}
```

### Common Error Codes
- `VALIDATION_ERROR`: Invalid request parameters
- `MODEL_ERROR`: AI model unavailable or failed
- `GRAPH_ERROR`: Knowledge graph operation failed
- `SESSION_ERROR`: Invalid or expired session
- `SYSTEM_ERROR`: Internal server error

## Rate Limiting

- **Chat API**: 60 requests per minute per session
- **Graph Operations**: 10 requests per minute per client
- **System APIs**: 100 requests per minute per client

## SDK Examples

### Python
```python
import requests
import websocket

# REST API
response = requests.post("http://localhost:8000/api/chat", json={
    "message": "Hello, AI!",
    "session_id": "my-session",
    "use_kg": True
})

# WebSocket
def on_message(ws, message):
    print(f"Received: {message}")

ws = websocket.WebSocketApp("ws://localhost:8000/ws/chat/my-session")
ws.on_message = on_message
ws.run_forever()
```

### JavaScript
```javascript
// REST API
const response = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'Hello, AI!',
    session_id: 'my-session',
    use_kg: true
  })
});

// WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/chat/my-session');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### cURL
```bash
# Chat API
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, AI!", "session_id": "my-session", "use_kg": true}'

# Export graph
curl -X GET "http://localhost:8000/api/graph/my_graph/export_native" \
  -H "accept: application/zip" \
  --output my_graph_export.zip

# System health
curl -X GET "http://localhost:8000/api/health"
```

## Development

### Running in Development Mode
```bash
python app.py --reload --log-level DEBUG
```

### API Testing
```bash
# Run API tests
python -m pytest tests/test_api.py -v

# Test WebSocket functionality
python tests/test_websocket_chat.py
```

### Custom Endpoints

To add custom endpoints, create a new router in `app/api/` and register it in `app/main.py`:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/custom", tags=["custom"])

@router.get("/my-endpoint")
async def my_endpoint():
    return {"message": "Hello from custom endpoint"}
```

For more detailed information, visit the interactive documentation at `/docs` when the server is running.
