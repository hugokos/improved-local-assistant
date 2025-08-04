# Model Testing CLI

This command-line interface provides tools for testing the ModelManager implementation, including model initialization, streaming responses, background model querying, resource monitoring, and error handling.

## Prerequisites

Before using this CLI, ensure you have:

1. Installed Ollama (https://ollama.ai/)
2. Pulled the required models:
   ```
   ollama pull hermes3:3b
   ollama pull tinyllama
   ```
3. Installed Python dependencies:
   ```
   pip install -r requirements.txt
   ```

### Embedding Model Update

The system now uses local embeddings via HuggingFace's sentence-transformers instead of OpenAI embeddings. This change:

1. Eliminates the need for an OpenAI API key
2. Provides offline embedding capabilities
3. Improves performance and reduces costs

If you have existing knowledge graphs built with a different embedding model, you should rebuild them using:

```
python rebuild_knowledge_graphs.py
```

This ensures all knowledge graphs use the same embedding model dimensions.

## Available CLI Tools

### 1. General Model Testing

```
python cli/test_models.py [options]
```

#### Available Options

- `--verify-installation`: Verify Ollama installation and model availability
- `--test-streaming`: Test streaming responses from the conversation model
- `--test-background`: Test background model for knowledge extraction
- `--test-concurrent`: Test concurrent operation of both models
- `--test-error-handling`: Test error handling and recovery mechanisms
- `--test-resource-monitoring`: Test resource monitoring during model operations
- `--interactive`: Run an interactive chat session with model switching and monitoring
- `--all`: Run all tests except interactive mode

### 2. Dual Model Architecture Testing

```
python cli/test_dual_model.py [options]
```

#### Available Options

- `--concurrent`: Test concurrent processing
- `--fire-forget`: Test fire-and-forget pattern
- `--isolation`: Test resource isolation
- `--non-blocking`: Test non-blocking operations
- `--priority`: Test priority management
- `--all`: Run all tests

### 3. Milestone 1 Validation CLI

```
python cli/validate_milestone_1.py [options]
```

#### Available Options

- `--interactive`: Run interactive validation menu (default)

This comprehensive interactive testing interface provides:
- Menu-driven testing for all model operations
- Resource monitoring and performance validation
- Clear pass/fail indicators for each test
- Performance benchmarking
- Detailed test results summary

### 4. Knowledge Graph Testing

```
python cli/test_knowledge_graph.py [options]
```

#### Available Options

- `--create-graph`: Test creating a knowledge graph from documents
- `--load-prebuilt`: Test loading pre-built knowledge graphs
- `--test-updates`: Test dynamic knowledge graph updates
- `--test-queries`: Test graph-based retrieval and queries
- `--visualize`: Test knowledge graph visualization
- `--all`: Run all tests
- `--docs-path PATH`: Specify custom path to documents for graph creation
- `--prebuilt-path PATH`: Specify custom path to pre-built knowledge graphs

### 5. Milestone 2 Validation CLI

```
python cli/validate_milestone_2.py [options]
```

#### Available Options

- `--interactive`: Run interactive validation menu (default)

This comprehensive interactive testing interface provides:
- Menu-driven testing for all knowledge graph operations
- Graph creation, loading, and visualization
- Dynamic graph updates and querying
- Performance benchmarking and memory usage monitoring
- Graph statistics and visualization

### 6. Conversation Testing

```
python cli/test_conversation.py [options]
```

#### Available Options

- `--create-session`: Test creating a conversation session
- `--test-streaming`: Test streaming conversation responses with dual-model architecture
- `--test-context`: Test conversation context handling and reference resolution
- `--interactive`: Run an interactive conversation session with debugging features

### 7. Milestone 3 Validation CLI

```
python cli/validate_milestone_3.py [options]
```

#### Available Options

- `--interactive`: Run interactive validation menu (default)

This comprehensive interactive testing interface provides:
- Multi-turn conversation testing with context validation
- Knowledge graph integration verification
- Conversation analysis and performance metrics
- Reference resolution and topic change detection
- Context window management testing
- Dual-model architecture validation

## Interactive Mode

The interactive modes provide chat interfaces with additional commands:

### test_models.py Interactive Commands

- `exit`, `quit`: End the session
- `status`: Show model status
- `resources`: Show resource usage
- `switch model <model_name>`: Switch conversation model
- `config temperature <value>`: Set temperature (0.0-1.0)
- `config max_tokens <value>`: Set max tokens for response
- `list models`: Show available models
- `help`: Show the help menu

### validate_milestone_1.py Interactive Menu

The milestone 1 validation CLI provides a comprehensive menu-driven interface for testing:
1. Model Initialization
2. Streaming Response
3. Background Model
4. Concurrent Operation
5. Fire-and-Forget Pattern
6. Resource Isolation
7. Non-Blocking Operations
8. Performance Benchmarking
9. Error Handling
10. Interactive Chat Session
11. Run All Tests
12. Show Test Results Summary

### validate_milestone_2.py Interactive Menu

The milestone 2 validation CLI provides a comprehensive menu-driven interface for testing:
1. Create Knowledge Graph from Documents
2. Load Pre-built Knowledge Graphs
3. Test Dynamic Knowledge Graph Updates
4. Test Graph-based Retrieval and Queries
5. Test Knowledge Graph Visualization
6. Run Performance Tests
7. View Graph Statistics

### test_conversation.py Interactive Commands

- `exit`, `quit`, `bye`: End the session
- Any other input will be treated as a message to the assistant

### validate_milestone_3.py Interactive Commands

- `/help`: Show available commands
- `/exit`: Exit the validation CLI
- `/test`: Run automated validation tests
- `/status`: Show test status and results
- `/debug <on|off>`: Toggle debug mode
- `/memory <on|off>`: Toggle memory tracking
- `/new`: Create a new session
- `/sessions`: List all active sessions
- `/switch <id>`: Switch to another session
- `/history`: Show conversation history
- `/info`: Show session information
- `/metrics`: Show performance metrics
- `/analyze`: Analyze conversation context and references
- `/summarize`: Force conversation summarization
- `/kg`: Show knowledge graph information
- `/dual`: Test dual-model architecture
- `/topic <message>`: Test topic change detection
- `/context`: Test context window management
- `/reference <message>`: Test reference resolution
- `/report`: Generate validation report
- Any other input will be treated as a message to the assistant

## Examples

### Verify Installation

```
python cli/test_models.py --verify-installation
```

### Test Streaming Responses

```
python cli/test_models.py --test-streaming
```

### Run Interactive Chat

```
python cli/test_models.py --interactive
```

### Run All Tests

```
python cli/test_models.py --all
```

### Run Milestone 1 Validation

```
python cli/validate_milestone_1.py
```

### Run Knowledge Graph Tests

```
python cli/test_knowledge_graph.py --all
```

### Create and Visualize a Knowledge Graph

```
python cli/test_knowledge_graph.py --create-graph --visualize
```

### Run Milestone 2 Validation

```
python cli/validate_milestone_2.py --interactive
```

### Test Conversation Features

```
python cli/test_conversation.py --test-streaming --test-context
```

### Run Interactive Conversation Session

```
python cli/test_conversation.py --interactive
```

### Run Milestone 3 Validation

```
python cli/validate_milestone_3.py
```

## Comprehensive Testing

### Milestone 1: Model Management

The Milestone 1 tests focus on the dual-model architecture, ensuring proper resource management, concurrent operations, and model performance.

### Milestone 2: Knowledge Graph Management

The Milestone 2 tests validate the knowledge graph functionality:

- **Graph Creation**: Test creating knowledge graphs from documents
- **Pre-built Graph Loading**: Test loading and managing pre-built knowledge graphs
- **Dynamic Updates**: Test real-time graph updates with entity extraction
- **Query System**: Test graph-based retrieval and complex queries
- **Visualization**: Test knowledge graph visualization capabilities
- **Performance**: Test memory usage, query performance, and scalability

### Milestone 3: Conversation Management

The Milestone 3 tests validate the conversation management functionality:

- **Multi-turn Conversations**: Test conversation history and context retention
- **Reference Resolution**: Test resolving references to previous conversation elements
- **Knowledge Graph Integration**: Test seamless integration with knowledge graphs
- **Context Window Management**: Test handling of long conversations and context limits
- **Topic Change Detection**: Test detection and handling of conversation topic changes
- **Dual-model Architecture**: Test concurrent processing with conversation and knowledge models
- **Conversation Summarization**: Test summarization of long conversations
- **Response Time**: Test response time requirements for conversation processing
- **Memory Usage**: Test memory stability during long conversations

## Resource Monitoring

The CLI includes resource monitoring capabilities that track:

- CPU usage
- Memory usage (RSS and VMS)
- Peak resource usage
- Resource usage deltas during operations

This helps identify potential performance issues and resource constraints during model operations.

## Error Handling

The CLI implements error handling and recovery mechanisms for:

- Model initialization failures
- Connection issues
- Invalid model names
- Query timeouts
- Graph creation and loading failures
- Entity extraction errors
- Visualization generation issues

When errors occur, the system attempts to recover automatically when possible and provides clear error messages.

## Running Comprehensive Tests

To run all tests for all milestones:

```bash
# Run all Milestone 1 tests
python cli/validate_milestone_1.py --interactive

# Run all Milestone 2 tests
python cli/validate_milestone_2.py --interactive

# Run all Milestone 3 tests
python cli/validate_milestone_3.py

# Run specific knowledge graph tests
python cli/test_knowledge_graph.py --create-graph --test-queries --visualize

# Run specific conversation tests
python cli/test_conversation.py --test-streaming --test-context --interactive
```

For the best experience with knowledge graph visualization, run the tests in an environment where a web browser can be opened automatically to display the generated graph visualizations.