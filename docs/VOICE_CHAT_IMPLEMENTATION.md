# Voice Chat Implementation Guide

## Overview

This document describes the implementation of voice chat functionality in the Improved Local AI Assistant. The voice system adds real-time speech-to-text (STT) and text-to-speech (TTS) capabilities while maintaining complete local processing and privacy.

## Architecture

### Backend Components

#### 1. Voice Manager (`services/voice_manager.py`)
- Central coordinator for all voice operations
- Manages voice sessions and state
- Provides metrics and monitoring
- Integrates with existing conversation flow

#### 2. Vosk STT Service (`services/vosk_stt_service.py`)
- Offline speech recognition using Vosk
- Real-time partial and final transcription
- Per-session recognizer management
- 16kHz audio processing

#### 3. Piper TTS Service (`services/piper_tts_service.py`)
- Offline text-to-speech using Piper
- Streaming audio synthesis
- Configurable voice models
- 22kHz audio output

#### 4. WebSocket Endpoints
- `/ws/stt/{session_id}`: Speech-to-text processing
- `/ws/tts/{session_id}`: Text-to-speech synthesis
- Integration with existing `/ws/{session_id}` chat endpoint

### Frontend Components

#### 1. Voice Controller (`app/static/js/voice-controller.js`)
- Manages voice mode state and UI
- Handles microphone capture and audio processing
- WebSocket communication for STT/TTS
- Integration with existing chat interface

#### 2. Audio Processing (`app/static/worklets/pcm-recorder.js`)
- AudioWorklet for real-time audio processing
- Float32 to Int16 PCM conversion
- RMS level calculation for visualization
- Off-main-thread processing

#### 3. UI Components
- Voice toggle button with keyboard shortcut (Shift+M)
- Microphone visualization orb with audio levels
- Live transcription display
- Voice mode theme adjustments

## Features

### Core Functionality
- **Real-time STT**: Continuous speech recognition with partial results
- **Streaming TTS**: Text-to-speech with audio streaming
- **Voice/Text Toggle**: Seamless switching between input modes
- **Session Integration**: Voice sessions tied to chat sessions
- **Local Processing**: All voice processing happens offline

### User Experience
- **Visual Feedback**: Mic orb animation based on audio levels
- **Live Transcription**: Real-time display of speech recognition
- **Keyboard Shortcuts**: Shift+M to toggle voice mode
- **Error Handling**: Graceful fallback to text mode
- **Accessibility**: Screen reader compatible controls

### Performance Features
- **Low Latency**: < 500ms STT, < 800ms TTS first audio
- **Memory Efficient**: < 600MB total memory usage
- **CPU Optimized**: Efficient processing on edge devices
- **Resource Monitoring**: Voice metrics in system status

## Installation and Setup

### 1. Install Dependencies
```bash
pip install vosk==0.3.47 piper-tts==1.0.0 sounddevice==0.4 webrtcvad==2.0.10
```

### 2. Download Voice Models
```bash
# Download recommended models
python scripts/download_voice_models.py --all

# Or download specific models
python scripts/download_voice_models.py --vosk small-en
python scripts/download_voice_models.py --piper en_US-lessac-medium
```

### 3. Configuration
Voice settings are configured in `config.yaml`:

```yaml
voice:
  enabled: true
  stt:
    enabled: true
    model_name: small-en
    sample_rate: 16000
  tts:
    enabled: true
    voice_name: en_US-lessac-medium
    sample_rate: 22050
    speed: 1.0
```

### 4. Test Installation
```bash
python scripts/test_voice_features.py
```

## Usage

### Starting Voice Mode
1. Click the "Voice" button or press Shift+M
2. Grant microphone permission when prompted
3. Speak naturally - partial transcription appears in real-time
4. AI responses are automatically spoken

### Voice Controls
- **Toggle Voice Mode**: Click voice button or Shift+M
- **Visual Feedback**: Mic orb shows audio levels and speaking state
- **Live Transcription**: See your speech converted to text in real-time
- **Error Recovery**: Automatic fallback to text mode on errors

## Technical Details

### Audio Processing Pipeline

#### Speech-to-Text Flow
1. **Microphone Capture**: Browser captures audio at 16kHz
2. **AudioWorklet Processing**: Convert Float32 to Int16 PCM
3. **WebSocket Streaming**: Send PCM chunks to `/ws/stt/{session_id}`
4. **Vosk Recognition**: Process audio with offline model
5. **Result Forwarding**: Send final transcript to conversation manager

#### Text-to-Speech Flow
1. **Text Generation**: Conversation manager generates response
2. **TTS Queue**: Text sent to `/ws/tts/{session_id}`
3. **Piper Synthesis**: Convert text to audio chunks
4. **Audio Streaming**: Stream PCM audio to browser
5. **Playback**: Web Audio API plays synthesized speech

### Integration Points

#### Conversation Manager Integration
- Voice transcripts processed same as text input
- Existing knowledge graph and citation systems work unchanged
- Voice sessions maintain conversation history and context

#### System Monitoring Integration
- Voice processing metrics in system status
- Resource usage monitoring for STT/TTS
- Health checks for voice services

#### WebSocket Protocol Extensions
New message types for voice functionality:
- `stt_partial`: Interim speech recognition results
- `stt_final`: Complete transcription
- `tts_start`: Beginning of speech synthesis
- `tts_complete`: End of speech synthesis

## Performance Characteristics

### Latency Targets
- **STT Latency**: < 500ms from speech to text
- **TTS Latency**: < 800ms from text to first audio
- **End-to-End**: < 2 seconds from speech to AI response

### Resource Usage
- **Memory**: < 600MB total (including voice models)
- **CPU**: Efficient processing on 4+ core systems
- **Storage**: ~100MB for recommended models

### Supported Platforms
- **Browsers**: Chrome 66+, Firefox 76+, Safari 14.1+
- **Operating Systems**: Windows 10+, macOS 10.15+, Linux
- **Hardware**: 4GB+ RAM, 4+ CPU cores recommended

## Troubleshooting

### Common Issues

#### Voice Features Not Available
- Check browser compatibility (requires AudioWorklet support)
- Verify microphone permissions granted
- Ensure voice dependencies installed

#### Poor Speech Recognition
- Check microphone quality and positioning
- Reduce background noise
- Speak clearly and at normal pace
- Consider upgrading to larger Vosk model

#### TTS Not Working
- Verify Piper voice models downloaded
- Check audio output device settings
- Test with different voice models

#### High Resource Usage
- Use smaller voice models for resource-constrained devices
- Adjust voice processing settings in config
- Monitor system resources during voice processing

### Debug Commands
```bash
# Test voice dependencies
python scripts/test_voice_features.py

# Check voice model availability
python scripts/download_voice_models.py --list

# System health with voice metrics
python scripts/system_health_check.py
```

## Development

### Adding New Voice Models

#### Vosk STT Models
1. Add model configuration to `scripts/download_voice_models.py`
2. Update model paths in voice configuration
3. Test with different languages/sizes

#### Piper TTS Voices
1. Add voice configuration to download script
2. Update voice selection in settings
3. Test voice quality and performance

### Extending Voice Features

#### Custom Audio Processing
- Modify AudioWorklet for advanced processing
- Add noise reduction or echo cancellation
- Implement voice activity detection

#### Voice Commands
- Add command recognition in STT processing
- Implement voice shortcuts and controls
- Create voice-specific conversation modes

## Security and Privacy

### Local Processing
- All voice processing happens on-device
- No audio data sent to external services
- Voice models run completely offline

### Data Handling
- Audio data not stored permanently
- Voice sessions isolated per user
- Transcripts follow same privacy as text chat

### Browser Security
- Requires secure context (HTTPS) for microphone access
- Respects browser permission model
- No persistent audio recording

## Phase 1 Enhancements (Implemented)

### Turn-taking and Barge-in
- **Barge-in Detection**: Immediate interruption when user speaks during TTS
- **Cooperative Cancellation**: TTS stops cleanly mid-synthesis
- **Natural Turn-taking**: < 150ms response time for interruptions
- **State Machine**: Enhanced with barge-in transitions

### Voice Command Grammar
- **Dual Recognition**: Separate recognizers for commands vs. dictation
- **Local Processing**: Commands never sent to LLM
- **Constrained Grammar**: Improved accuracy for control phrases
- **Immediate Feedback**: Visual confirmation of command execution

### Supported Voice Commands
```
Control: stop, cancel, mute, repeat
Speed: slower, faster, normal speed
Chat: new chat, summarize, cite sources, delete last
```

## Future Enhancements

### Phase 2 (Planned)
- Wake word detection ("Hey Assistant")
- Advanced VAD with webrtcvad integration
- Domain biasing for technical terms
- Wyoming protocol compatibility

### Phase 3+ (Roadmap)
- Multiple language support
- Voice model hot-swapping
- Advanced audio visualization
- Voice conversation analytics

### Performance Improvements
- GPU acceleration for voice processing
- Model quantization for smaller footprint
- Streaming optimizations for lower latency
- Adaptive quality based on system resources

## Support

For issues with voice functionality:
1. Run the voice test suite: `python scripts/test_voice_features.py`
2. Check system requirements and dependencies
3. Review browser console for JavaScript errors
4. Verify voice model downloads completed successfully
5. Test with different microphones/audio devices

The voice chat system is designed to enhance the existing chat experience while maintaining the privacy-first, local-only approach of the Improved Local AI Assistant.