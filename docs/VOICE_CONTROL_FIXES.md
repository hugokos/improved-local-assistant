# Voice Control Fixes - Binary Audio Frame Handling

## Problem Summary

The voice control system was experiencing issues where:
- Browser: Mic, VAD, and WebSocket connections worked fine, showing "🎙️ STT ready"
- Server: Vosk recognizers were created and TTS connected, but no audio bytes were being processed
- Root cause: Binary audio frames weren't reaching the STT handler properly

## Root Cause Analysis

1. **WebSocket Binary Frame Handling**: The server was using `receive()` instead of the robust `iter_bytes()` method
2. **Import Error**: `WebSocketDisconnect` was imported from the wrong module, causing crashes
3. **Client-Side Buffer Issues**: Audio frames might have been sending wrong buffer slices
4. **Missing RMS Logging**: No way to verify if audio data was actually reaching the server

## Fixes Applied

### 1. Server-Side WebSocket Handler (`app/ws/voice_stt.py`)

**Before:**
```python
from starlette.websockets import WebSocketDisconnect  # Wrong import
# Complex receive() loop with manual message handling
```

**After:**
```python
from fastapi import WebSocketDisconnect  # Correct import
# Robust iter_bytes() pattern with RMS logging

async def stt_websocket(websocket: WebSocket, session_id: str, app):
    await websocket.accept()
    
    try:
        # 1) Handshake (text frame)
        msg = await websocket.receive_json()
        if msg.get("type") == "stt_start":
            await voice_manager.create_voice_session(session_id)
            await websocket.send_json({"type": "stt_ready"})

        # 2) Audio loop (binary frames) - ROBUST PATTERN
        async for chunk in websocket.iter_bytes():
            if not chunk:
                continue
                
            # RMS logging to verify audio data
            r = rms16le(chunk)
            if r < 50:
                logger.debug("Audio RMS ~0 (likely silence); len=%d", len(chunk))
            else:
                logger.debug("Audio RMS=%.1f; len=%d", r, len(chunk))
            
            # Process audio
            result = await voice_manager.process_audio_chunk(session_id, chunk)
            
            if result.get("partial"):
                await websocket.send_json({"type": "stt_partial", "text": result["partial"]})
            if result.get("final"):
                await websocket.send_json({"type": "stt_final", "text": result["final"]})
                
    except WebSocketDisconnect:
        logger.info(f"STT WebSocket disconnected for session {session_id}")
```

### 2. Client-Side Binary Frame Handling (`app/static/js/voice-controller.js`)

**Key Changes:**
- Ensured `binaryType = 'arraybuffer'` is set on WebSocket
- Fixed audio frame handling to use exact ArrayBuffer from worklet
- Added proper frame size validation (640 bytes for 20ms frames)

**Before:**
```javascript
const view = data.data; // Ambiguous handling
const buf = view instanceof ArrayBuffer ? view : view.buffer;
const frame = buf.slice(0, validSizes.includes(view.byteLength) ? view.byteLength : 0);
```

**After:**
```javascript
handleAudioFrame(data) {
    if (this.halfDuplexMode === 'speaking' || this.micMuted) {
        return;
    }
    
    if ((this.state === 'listening' || this.state === 'utterance_active') && this.sttSocket) {
        // CRITICAL FIX: Properly handle the ArrayBuffer from worklet
        const frameBuffer = data.data; // This is an ArrayBuffer from the worklet
        
        if (frameBuffer && frameBuffer.byteLength === 640 && // Exact 20ms frame = 640 bytes
            this.sttSocket.readyState === WebSocket.OPEN && 
            this.sttSocket.bufferedAmount < this.maxBufferedAmount) {
            
            // CRITICAL FIX: Send the exact ArrayBuffer (already properly sliced by worklet)
            this.sttSocket.send(frameBuffer);
        }
    }
}
```

### 3. WebSocket Connection Setup

**Ensured proper binary type:**
```javascript
this.sttSocket = new WebSocket(wsUrl);
this.sttSocket.binaryType = 'arraybuffer'; // CRITICAL: Ensure binary frames are ArrayBuffer
```

### 4. RMS Calculation Function

**Added server-side RMS calculation for debugging:**
```python
def rms16le(b: bytes) -> float:
    """Calculate RMS of 16-bit little-endian PCM audio."""
    if not b: 
        return 0.0
    a = array.array('h')  # 16-bit signed
    a.frombytes(b)
    return math.sqrt(sum(x*x for x in a) / len(a))
```

## Audio Frame Specifications

### Expected Frame Format
- **Sample Rate**: 16 kHz (Vosk standard)
- **Bit Depth**: 16-bit signed PCM
- **Channels**: Mono (1 channel)
- **Endianness**: Little-endian
- **Frame Duration**: 20ms (optimal for WebRTC VAD)
- **Frame Size**: 320 samples = 640 bytes

### Frame Size Validation
```javascript
// Client-side validation
if (frameBuffer && frameBuffer.byteLength === 640) {
    this.sttSocket.send(frameBuffer);
}
```

```python
# Server-side validation
expected_frame_sizes = [320, 640, 960]  # 10ms, 20ms, 30ms at 16kHz
if len(audio_data) not in expected_frame_sizes:
    logger.debug(f"Non-standard frame size: {len(audio_data)} bytes")
```

## Testing

Created comprehensive test script (`scripts/test_voice_fixes.py`) that:
1. Tests binary frame handling functions
2. Creates synthetic audio frames with proper format
3. Verifies RMS calculations work correctly
4. Tests Vosk STT service with synthetic audio
5. Validates frame sizes and processing

**Test Results:**
```
✅ Binary frame handling test passed
✅ Recognizer created successfully  
✅ Test completed successfully
🎉 All tests passed! Voice control fixes should be working.
```

## Expected Behavior After Fixes

1. **Browser**: Mic, VAD, and WebSocket connections work as before
2. **Server**: Now logs audio RMS values, confirming binary data is received
3. **STT Processing**: Vosk receives proper audio frames and produces recognition results
4. **Real-time Response**: Partial and final transcription results flow back to client

## Debugging Commands

If issues persist, check these logs:

```bash
# Server logs should show:
# "Audio RMS=2319.3; len=640" (for speech)
# "Audio RMS ~0 (likely silence); len=640" (for silence)

# Client console should show:
# "📤 Sent audio frame: 640 bytes, RMS: 0.123"
# "📝 Partial result: 'hello world'"
# "✅ Final result: 'hello world'"
```

## Performance Considerations

1. **Frame Batching**: Using exact 20ms frames (640 bytes) prevents WebSocket backpressure
2. **RMS Calculation**: Lightweight client-side RMS for orb visualization
3. **Buffer Management**: `bufferedAmount` check prevents WebSocket overflow
4. **Memory Efficiency**: ArrayBuffer transfer with proper cleanup

## Browser Compatibility

- **Chrome/Edge**: Full support for AudioWorklet and ArrayBuffer WebSocket frames
- **Firefox**: Full support with proper `binaryType = 'arraybuffer'`
- **Safari**: Requires user gesture for AudioContext activation

## Next Steps

1. Test with real microphone input
2. Verify VAD integration works properly
3. Test half-duplex mode transitions
4. Validate TTS audio playback integration
5. Performance testing with extended conversations