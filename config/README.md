# Configuration Files

This directory contains development and build configuration files that were moved from the root to reduce clutter.

## Files

- **`mkdocs.yml`** - Documentation build configuration for MkDocs
- **`Makefile`** - Development automation commands

## Usage

### Documentation
```bash
# Build docs from root directory
mkdocs build --config-file config/mkdocs.yml
mkdocs serve --config-file config/mkdocs.yml
```

### Development Commands
```bash
# Run make commands from root directory
make -f config/Makefile help
make -f config/Makefile install-dev
make -f config/Makefile test
```

Or use the modern alternatives:
```bash
# Modern Python development
pip install -e ".[dev]"
pytest
ruff check .
black .
```