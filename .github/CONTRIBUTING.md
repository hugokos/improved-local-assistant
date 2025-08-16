# Contributing to Improved Local Assistant

Thank you for your interest in contributing to the Improved Local Assistant! We welcome contributions from developers of all skill levels.

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+** (3.11+ recommended for optimal performance)
- **Git** for version control
- **Ollama** with required models (hermes3:3b, phi3:mini)
- **8GB+ RAM** for development (16GB recommended)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/improved-local-assistant.git
   cd improved-local-assistant
   ```

2. **Set Up Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows
   
   # Install in development mode with all dependencies
   pip install -e ".[dev]"
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Verify Setup**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Check code quality
   ruff check .
   black --check .
   mypy src/
   
   # Verify system functionality
   python cli/validate_milestone_6.py
   ```

## 🔧 Development Workflow

### Code Quality Standards

We maintain high code quality standards using automated tools:

**Formatting and Linting:**
```bash
# Format code (automatically fixes issues)
ruff format .
black .

# Lint code (check for issues)
ruff check .

# Type checking
mypy src/

# Run all quality checks
pre-commit run --all-files
```

**Code Style Guidelines:**
- **Line Length**: 100 characters maximum
- **Docstrings**: Google style for all public functions and classes
- **Type Hints**: Required for all function signatures
- **Import Organization**: Use isort with black profile

# Type checking
python -m mypy services/ --ignore-missing-imports

# Run tests
python -m pytest tests/ -v --cov=services
```

### Commit Messages
Use conventional commit format:
```
feat: add new GraphRAG feature
fix: resolve WebSocket connection issue
docs: update API documentation
test: add integration tests for hybrid retriever
refactor: improve model manager architecture
```

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test improvements

## 🧪 Testing

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_milestone_6_fixed.py -v

# With coverage
python -m pytest tests/ -v --cov=services --cov-report=html
```

### Test Requirements
- **Unit Tests**: 90%+ code coverage required
- **Integration Tests**: End-to-end workflow validation
- **Mock Framework**: Use for external dependencies
- **Performance Tests**: Benchmark critical paths

### Writing Tests
```python
import pytest
from unittest.mock import Mock, patch
from services.model_manager import ModelManager

class TestModelManager:
    @pytest.fixture
    def mock_ollama(self):
        with patch('services.model_manager.requests') as mock:
            yield mock
    
    def test_model_loading(self, mock_ollama):
        # Test implementation
        pass
```

## 📝 Documentation

### Code Documentation
- **Docstrings**: All public functions and classes
- **Type Hints**: Comprehensive type annotations
- **Comments**: Complex logic and business rules
- **API Docs**: OpenAPI/Swagger documentation

### Documentation Updates
- Update README.md for user-facing changes
- Update CHANGELOG.md for all releases
- Add technical docs to docs/ directory
- Update API documentation for endpoint changes

## 🐛 Bug Reports

### Before Reporting
1. Check existing issues
2. Verify with latest version
3. Test with minimal reproduction case
4. Check logs for error details

### Bug Report Template
```markdown
**Bug Description**
Clear description of the issue

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [Windows/Linux/macOS]
- Python version: [3.x.x]
- Ollama version: [x.x.x]
- Models: [hermes3:3b, phi3:mini]

**Logs**
```
Relevant log output
```
```

## 💡 Feature Requests

### Feature Request Template
```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this be implemented?

**Alternatives Considered**
Other approaches considered

**Additional Context**
Screenshots, mockups, references
```

## 🔍 Code Review Process

### Pull Request Requirements
- [ ] Tests pass (`python -m pytest tests/ -v`)
- [ ] Code formatted (`black` and `isort`)
- [ ] Type checking passes (`mypy`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or properly documented)

### Review Checklist
- **Functionality**: Does it work as intended?
- **Performance**: No significant performance regression
- **Security**: No security vulnerabilities introduced
- **Maintainability**: Clean, readable code
- **Testing**: Adequate test coverage
- **Documentation**: Proper documentation

## 🏗️ Architecture Guidelines

### Service Design
- **Single Responsibility**: Each service has one clear purpose
- **Dependency Injection**: Use constructor injection
- **Error Handling**: Comprehensive error handling and logging
- **Async/Await**: Use async patterns for I/O operations

### Code Organization
```
services/
├── model_manager.py      # AI model orchestration
├── graph_manager/        # Knowledge graph management
├── hybrid_retriever.py   # Multi-modal retrieval
└── conversation_manager.py # Session management
```

### Performance Considerations
- **Memory Usage**: Monitor and optimize memory consumption
- **Response Time**: Target <1s for most operations
- **Concurrency**: Support 10+ concurrent users
- **Resource Management**: Proper cleanup and resource pooling

## 🚀 Release Process

### Version Numbering
- **Major**: Breaking changes (2.0.0)
- **Minor**: New features (2.1.0)
- **Patch**: Bug fixes (2.1.1)

### Release Checklist
- [ ] All tests pass
- [ ] Performance benchmarks validated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in setup.py
- [ ] Git tag created
- [ ] Release notes published

## 🤝 Community

### Communication
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Code Reviews**: Constructive feedback on pull requests

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional standards

## 📚 Resources

### Technical Documentation
- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Development History](DEVELOPMENT_HISTORY.md)

### External Resources
- [Ollama Documentation](https://ollama.ai/docs)
- [LlamaIndex Documentation](https://docs.llamaindex.ai)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

---

Thank you for contributing to the Improved Local Assistant! Your contributions help make this project better for everyone.