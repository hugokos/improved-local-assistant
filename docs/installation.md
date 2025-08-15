# Installation Guide

This guide will help you install and set up the Improved Local AI Assistant on your system.

## Prerequisites

### System Requirements
- **Python 3.8+** (Python 3.11+ recommended for optimal performance)
- **8GB RAM minimum** (16GB recommended for development)
- **10GB free disk space** (for models and knowledge graphs)
- **Git** for version control

### Required Software
- **Ollama** - Local AI model runtime
- **Virtual environment** support (venv, conda, etc.)

## Quick Installation

### 1. Install Ollama

Visit [https://ollama.ai](https://ollama.ai) and follow the installation instructions for your platform.

After installation, pull the required models:

```bash
ollama pull hermes3:3b
ollama pull phi3:mini
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/hugokos/improved-local-assistant.git
cd improved-local-assistant

# Run automated setup
python setup_dev.py
```

The setup script will:
- Check system requirements
- Create a virtual environment
- Install all dependencies
- Set up pre-commit hooks
- Run initial quality checks

### 3. Launch the Application

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Start the application
python run_app.py
```

Visit [http://localhost:8000](http://localhost:8000) to access the web interface.

## Manual Installation

If you prefer manual installation or the automated setup fails:

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
# Install in development mode
pip install -e ".[dev]"

# Or for production use only
pip install -e .
```

### 3. Set up Development Tools (Optional)

```bash
# Install pre-commit hooks
pre-commit install

# Run code quality checks
make lint
make format
```

## Configuration

### Environment Variables

Copy the example environment file and customize:

```bash
cp .env.example .env
```

Edit `.env` to configure:
- Model settings
- API endpoints
- Feature flags

### Configuration File

The main configuration is in `config.yaml`. Key settings include:

```yaml
# Model configuration
models:
  conversation_model: "hermes3:3b"
  knowledge_extraction_model: "phi3:mini"

# Performance settings
edge_optimization:
  enabled: true
  mode: "production"

# Voice interface (optional)
voice:
  enabled: true
  stt_model: "vosk-model-en-us-0.22"
  tts_voice: "en_US-lessac-medium"
```

## Verification

### Test the Installation

```bash
# Run system health check
python scripts/system_health_check.py

# Run basic tests
make test

# Test the web interface
python cli/test_web_interface.py
```

### Verify Models

```bash
# Check Ollama models
ollama list

# Test model connectivity
python cli/test_models.py
```

## Troubleshooting

### Common Issues

**Ollama Connection Failed**
```bash
# Check if Ollama is running
ollama list

# Start Ollama service (if needed)
ollama serve
```

**Import Errors**
```bash
# Reinstall in development mode
pip install -e ".[dev]"

# Check Python path
python -c "import sys; print(sys.path)"
```

**Memory Issues**
```bash
# Enable edge optimization
python cli/toggle_edge_optimization.py --enable

# Check memory usage
python scripts/memory_optimizer.py --analyze
```

### Getting Help

- **Documentation**: [Full documentation](https://hugokos.github.io/improved-local-assistant)
- **Issues**: [GitHub Issues](https://github.com/hugokos/improved-local-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hugokos/improved-local-assistant/discussions)

## Next Steps

After installation:

1. **Explore the Interface**: Visit the web interface and try basic conversations
2. **Enable Voice**: Set up voice models for hands-free interaction
3. **Load Knowledge**: Import documents to build knowledge graphs
4. **Customize**: Adjust settings for your specific use case

See the [Quick Start Guide](quickstart.md) for detailed usage instructions.