# ADR-001: Project Structure Reorganization

## Status
Accepted

## Context
The project had grown organically with files scattered across the root directory and inconsistent module organization. This made it difficult to:
- Navigate the codebase
- Understand component relationships
- Maintain clean imports
- Follow Python packaging best practices

## Decision
Reorganize the project into a clean, predictable structure following Python packaging standards:

```
improved-local-assistant/
├── src/improved_local_assistant/    # Main package
│   ├── api/                         # FastAPI app & routes
│   ├── cli/                         # Typer CLI commands
│   ├── core/                        # Settings, logging, utils
│   ├── graph/                       # Graph management
│   ├── retrieval/                   # Hybrid retriever/router
│   ├── models/                      # Pydantic schemas
│   └── voice/                       # TTS/STT services
├── configs/                         # Config files (no code)
├── tools/                           # Ops helpers
├── scripts/                         # Dev/ops scripts
├── tests/                           # Test organization
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/adr/                        # Architecture decisions
```

## Consequences

### Positive
- Clear separation of concerns
- Predictable file locations
- Better import organization
- Follows Python packaging standards
- Easier onboarding for new developers

### Negative
- Requires import path updates
- Temporary disruption during migration
- Need to update documentation

## Implementation
- Use `git mv` to preserve history
- Update imports systematically
- Create new CLI with Typer
- Implement FastAPI factory pattern
- Add Makefile for common tasks
