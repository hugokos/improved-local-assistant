# Memory Fallback System

The Memory Fallback System automatically switches from the primary conversation model (hermes3:3b) to a lighter fallback model (tinyllama) when memory issues are detected.

## How It Works

### Proactive Memory Monitoring
The system continuously monitors memory usage and automatically switches to the fallback model when memory usage exceeds **98%**, preventing memory errors before they occur.

### Automatic Error Detection
The system also monitors for memory-related errors including:
- "model requires more system memory"
- "500 Internal Server Error"
- "out of memory"
- "insufficient memory"

### Graceful Degradation
When memory usage exceeds 98% OR a memory error is detected:
1. The system automatically switches to the lightweight model
2. Users see a contextual notification:
   - `[Using lightweight model - memory usage at 99.2%]` (proactive)
   - `[Using lightweight model due to memory error]` (reactive)
3. The fallback model (tinyllama) handles the conversation
4. The primary model is automatically retried when memory drops below threshold

### Status Management
The system tracks three states:
- `OPERATIONAL`: Primary model working normally
- `DEGRADED`: Memory issues detected, using fallback
- `FAILED`: Both models unavailable

## Configuration

Add to your `config.yaml`:

```yaml
memory_fallback:
  enabled: true
  primary_model: hermes3:3b
  fallback_model: tinyllama
  proactive_threshold_percent: 98  # Switch to fallback when memory exceeds this
  error_patterns:
    - "model requires more system memory"
    - "500 Internal Server Error"
    - "out of memory"
    - "insufficient memory"
  auto_reset_after_minutes: 10
```

## Manual Management

### Check Status
```bash
python cli/test_memory_fallback.py status
```

### Reset Model Status
```bash
python cli/test_memory_fallback.py reset
```

### Test Fallback System
```bash
python cli/test_memory_fallback.py test
```

## Benefits

1. **Proactive Prevention**: Prevents memory errors by switching models before they occur
2. **Uninterrupted Service**: Users can continue conversations even when memory is constrained
3. **Automatic Recovery**: System automatically retries primary model when memory drops
4. **Transparent Operation**: Users are informed why fallback is active (threshold vs error)
5. **Resource Efficiency**: Fallback model uses less memory and unloads after use
6. **Smart Thresholds**: Configurable memory threshold (default 98%) for optimal performance

## Technical Details

### Implementation
- Uses the existing `graceful_degradation.py` service
- Integrates with `LLMOrchestrator` for seamless switching
- Monitors error patterns in real-time
- Maintains conversation context across model switches

### Memory Management
- Primary model (hermes3:3b): ~1.2GB memory requirement
- Fallback model (tinyllama): ~0.6GB memory requirement
- Fallback model unloads immediately after response
- Primary model can be kept resident with `keep_alive`

### Error Handling
- Catches HTTP 500 errors from Ollama
- Parses error messages for memory indicators
- Falls back gracefully without losing conversation state
- Provides meaningful error messages to users

## Troubleshooting

### Fallback Not Working
1. Check if `memory_fallback.enabled` is `true` in config
2. Verify tinyllama model is available: `ollama list`
3. Check logs for error pattern matching

### Stuck in Fallback Mode
1. Use reset command: `python cli/test_memory_fallback.py reset`
2. Restart the application
3. Check available system memory

### Both Models Failing
1. Check Ollama service status
2. Verify model availability
3. Check system resources
4. Review Ollama logs

## Monitoring

The system provides detailed status information:
- Current model status (operational/degraded/failed)
- Whether fallback is currently active
- Model residency status
- Degradation manager component states

This ensures you always know which model is handling requests and why.
