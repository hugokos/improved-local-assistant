/**
 * Voice Controller for handling voice input/output in the chat interface.
 *
 * Implements proper voice state machine:
 * idle → listening → utterance_active → finalizing → waiting_for_bot → speaking → listening
 */

class VoiceController {
    constructor(chatInterface) {
        this.chatInterface = chatInterface;

        // Voice state machine
        this.state = 'idle';
        this.isVoiceMode = false;

        // WebSocket connections
        this.sttSocket = null;
        this.ttsSocket = null;

        // Audio processing
        this.audioContext = null;
        this.micStream = null;
        this.pcmRecorder = null;
        this.pcmPlayer = null;

        // Half-duplex control
        this.halfDuplexMode = 'idle';
        this.micMuted = false;
        this.ttsPlaying = false;
        this.holdOffUntil = 0;
        this.holdOffDurationMs = 250;
        this.bargeInEnabled = true;

        // Voice Activity Detection
        this.vadState = {
            isSpeaking: false,
            silenceMs: 0,
            silenceThreshold: 400,
            frameMs: 20,
            utteranceStartTime: 0,
            minUtteranceDuration: 300,
            speechFrameCount: 0,
            minSpeechFrames: 3,
            energyBaseline: 0.0,
            energyThreshold: 0.02
        };

        // UI elements
        this.voiceToggle = null;
        this.micOrb = null;
        this.liveTranscription = null;
        this.loadingDots = null;

        // Transcription state
        this.utteranceActive = false;
        this.partialBuf = '';
        this.finalText = '';
        this.lastCommittedText = '';
        this.audioLevel = 0.0;

        // Partial deduplication and throttling
        this.lastPartial = '';
        this.lastPartialTs = 0;
        this.partialThrottleMs = 100;

        // Command mode tracking
        this.isInCommandMode = false;

        // Legacy state (keep for compatibility)
        this.currentTranscript = '';
        this.partialTranscript = '';

        // Timeout management
        this.finalizationTimeout = null;

        // Endpoint deduplication
        this.lastEndpointAt = 0;
        this.currentSessionId = null;

        // WebSocket backpressure management
        this.maxBufferedAmount = 256000;

        // Audio format verification
        this.expectedSampleRate = 16000;
        this.audioFormatVerified = false;

        // Initialize
        this.init();
    }

    async init() {
        try {
            console.log('Starting VoiceController initialization...');

            // Find UI elements
            this.voiceToggle = document.getElementById('voiceToggle');
            this.micOrb = document.getElementById('micOrb');
            this.liveTranscription = document.getElementById('liveTranscription');

            // Add event listeners
            this.addEventListeners();

            // Check for voice support
            this.checkVoiceSupport();

            // Ensure required DOM elements exist
            this.ensurePartialTextElement();

            console.log('VoiceController initialized successfully');
        } catch (error) {
            console.error('Failed to initialize VoiceController:', error);
        }
    }

    addEventListeners() {
        if (this.voiceToggle) {
            this.voiceToggle.addEventListener('click', () => {
                this.toggleVoiceMode();
            });
        }

        document.addEventListener('keydown', (e) => {
            if (e.shiftKey && e.key.toLowerCase() === 'm') {
                e.preventDefault();
                this.toggleVoiceMode();
            }
        });
    }

    checkVoiceSupport() {
        try {
            if (typeof navigator === "undefined" || !navigator.mediaDevices ||
                typeof navigator.mediaDevices.getUserMedia !== "function") {
                console.warn('Voice features not supported');
                return false;
            }

            const hasAudioContext = !!(window.AudioContext || window.webkitAudioContext);
            if (!hasAudioContext) {
                console.warn('AudioContext not available');
                return false;
            }

            this.getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
            console.log('✅ All voice features are supported');
            return true;
        } catch (error) {
            console.error('Error in checkVoiceSupport:', error);
            return false;
        }
    }

    async toggleVoiceMode() {
        try {
            if (this.isVoiceMode) {
                await this.stopVoiceMode();
            } else {
                await this.ensureAudioReady();
                await this.startVoiceMode();
            }
        } catch (error) {
            console.error('Error toggling voice mode:', error);
            this.showVoiceError('Failed to toggle voice mode');
        }
    }

    async ensureAudioReady() {
        if (this.audioContext?.state !== 'running') {
            try {
                console.log('🔓 Ensuring AudioContext is ready:', this.audioContext?.state);
                await this.audioContext?.resume();
                console.log('✅ AudioContext unlocked:', this.audioContext?.state);
            } catch (error) {
                console.error('❌ Failed to unlock AudioContext:', error);
                throw new Error('Audio system not available. Please try again.');
            }
        }
    }

    async startVoiceMode() {
        try {
            console.log('🎤 Starting voice mode...');

            // CRITICAL: Unlock audio first (must be in user gesture call stack)
            const audioUnlocked = await this.unlockAudio();
            if (!audioUnlocked) {
                throw new Error('Failed to unlock audio - user gesture required');
            }

            // Request microphone access
            this.micStream = await this.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            console.log('🎤 Microphone access granted');

            // Load AudioWorklet for recording (reuse existing AudioContext)
            await this.audioContext.audioWorklet.addModule('/static/worklets/pcm-recorder.js');

            // Create worklet nodes (only recorder needed, TTS uses direct WebAudio)
            this.pcmRecorder = new AudioWorkletNode(this.audioContext, 'pcm-recorder');

            // Update state FIRST before connecting audio pipeline
            this.isVoiceMode = true;
            this.setState('listening');

            // Handle recorder messages
            this.pcmRecorder.port.onmessage = (event) => {
                this.handleRecorderMessage(event.data);
            };

            // Connect audio pipeline (recorder only, TTS uses separate path)
            const source = this.audioContext.createMediaStreamSource(this.micStream);
            source.connect(this.pcmRecorder);

            // Connect to WebSockets
            await this.connectSTT();
            await this.connectTTS();

            // Update UI
            this.updateVoiceUI();

            console.log('✅ Voice mode started successfully');

        } catch (error) {
            console.error('❌ Failed to start voice mode:', error);
            await this.stopVoiceMode();

            // Provide specific error messages
            if (error.name === 'NotAllowedError') {
                this.showVoiceError('Microphone access denied. Please allow microphone access and try again.');
            } else if (error.message.includes('audio')) {
                this.showVoiceError('Audio system not available. Please check your speakers and try again.');
            } else {
                this.showVoiceError('Failed to start voice mode. Please check your microphone and speakers.');
            }
        }
    }

    async stopVoiceMode() {
        try {
            console.log('Stopping voice mode...');

            if (this.finalizationTimeout) {
                clearTimeout(this.finalizationTimeout);
                this.finalizationTimeout = null;
            }

            if (this.sttSocket) {
                this.sttSocket.close();
                this.sttSocket = null;
            }

            if (this.ttsSocket) {
                this.ttsSocket.close();
                this.ttsSocket = null;
            }

            if (this.audioContext) {
                await this.audioContext.close();
                this.audioContext = null;
            }

            if (this.micStream) {
                this.micStream.getTracks().forEach(track => track.stop());
                this.micStream = null;
            }

            this.isVoiceMode = false;
            this.updateVoiceUI();

            console.log('Voice mode stopped');
        } catch (error) {
            console.error('Error stopping voice mode:', error);
        }
    }

    async connectSTT() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/stt/${this.chatInterface.sessionId}`;

            this.sttSocket = new WebSocket(wsUrl);
            this.sttSocket.binaryType = 'arraybuffer'; // CRITICAL: Ensure binary frames are ArrayBuffer

            this.sttSocket.onopen = () => {
                console.log('STT WebSocket connected');

                // 🔑 Handshake so server can create recognizer + validate format
                this.sttSocket.send(JSON.stringify({
                    type: 'stt_start',
                    session_id: this.chatInterface.sessionId,
                    sample_rate: 16000,
                    encoding: 'pcm16le',
                    channels: 1,
                    vad: 'server'
                }));

                resolve();
            };

            this.sttSocket.onmessage = (event) => {
                this.handleSTTMessage(JSON.parse(event.data));
            };

            this.sttSocket.onclose = () => {
                console.log('STT WebSocket disconnected');
            };

            this.sttSocket.onerror = (error) => {
                console.error('STT WebSocket error:', error);
                reject(error);
            };
        });
    }

    async connectTTS() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/tts/${this.chatInterface.sessionId}`;

            this.ttsSocket = new WebSocket(wsUrl);
            // CRITICAL: Set binaryType to 'arraybuffer' (default is 'blob')
            // Without this, binary frames arrive as Blob and must be converted
            this.ttsSocket.binaryType = 'arraybuffer';

            this.ttsSocket.onopen = () => {
                console.log('🔊 TTS WebSocket connected');
                resolve();
            };

            this.ttsSocket.addEventListener('message', async (event) => {
                try {
                    // Text frames carry control messages (tts_start, tts_end, errors)
                    if (typeof event.data === 'string') {
                        const msg = JSON.parse(event.data);

                        if (msg.type === 'tts_start') {
                            console.log('🔊 TTS stream starting...');
                            // CRITICAL: Ensure AudioContext is unlocked before first audio
                            await this.ensureAudioContext();
                            // Ensure we're routed to default destination (most reliable)
                            this.routeTtsToDefault();
                            this._ttsPlayhead = this.audioContext.currentTime + 0.03;
                            console.log('🔊 TTS START: context=' + this.audioContext.state);
                            this.setState('speaking');
                            this.setHalfDuplexMode('speaking');
                            return;
                        }

                        if (msg.type === 'tts_end') {
                            console.log('🔊 TTS stream ended');
                            this.setState('listening');
                            this.setHalfDuplexMode('hold_off');
                            return;
                        }

                        // Handle other control messages
                        this.handleTTSMessage(msg);
                        return;
                    }

                    // Binary frames: Should be ArrayBuffer due to binaryType setting
                    // If binaryType wasn't set, would need: await event.data.arrayBuffer()
                    const audioBuffer = event.data;

                    if (!(audioBuffer instanceof ArrayBuffer)) {
                        console.error('🔊 Expected ArrayBuffer, got:', typeof audioBuffer);
                        return;
                    }

                    if (audioBuffer.byteLength === 0) {
                        console.warn('🔊 Received empty TTS audio chunk');
                        return;
                    }

                    // CRITICAL: Log binary chunk reception (this should appear in console)
                    console.log(`🎧 TTS chunk bytes: ${audioBuffer.byteLength}`);
                    console.debug(`🎧 TTS chunk: ${(audioBuffer.byteLength/2)|0} samples (${audioBuffer.byteLength} bytes)`);

                    // Process PCM16LE audio data
                    this.enqueueTTSChunk(audioBuffer);

                } catch (error) {
                    console.error('🔊 Error processing TTS message:', error);

                    // Fallback: try to handle as Blob if ArrayBuffer failed
                    if (event.data instanceof Blob) {
                        try {
                            const arrayBuffer = await event.data.arrayBuffer();
                            this.enqueueTTSChunk(arrayBuffer);
                        } catch (fallbackError) {
                            console.error('🔊 Blob fallback failed:', fallbackError);
                        }
                    }
                }
            });

            this.ttsSocket.onclose = () => {
                console.log('TTS WebSocket disconnected');
            };

            this.ttsSocket.onerror = (error) => {
                console.error('TTS WebSocket error:', error);
                reject(error);
            };
        });
    }

    setState(newState, payload = {}) {
        console.log(`Voice state: ${this.state} → ${newState}`);
        this.state = newState;
        this.updateVoiceUI();

        // Update chat interface orb if available
        if (this.chatInterface && this.chatInterface.setMicState) {
            this.chatInterface.setMicState(newState, payload.level || this.audioLevel);
        }

        return this;
    }

    handleRecorderMessage(data) {
        if (data.type === 'vad_frame') {
            this.handleAudioFrame(data);
        } else if (data.type === 'rms') {
            this.audioLevel = data.level;
            this.updateMicVisualization();
        } else {
            console.warn('Unknown recorder message type:', data.type);
        }
    }

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

                // Compute RMS client-side for orb level (fast and simple)
                const dv = new DataView(frameBuffer);
                let sum = 0;
                for (let i = 0; i < dv.byteLength; i += 2) {
                    const s = dv.getInt16(i, true) / 32768; // little-endian
                    sum += s * s;
                }
                const rms = Math.sqrt(sum / (dv.byteLength / 2));
                const level = Math.min(1, rms * 2); // gentle boost

                // Update orb with current level
                if (this.chatInterface && this.chatInterface.setMicState) {
                    this.chatInterface.setMicState(this.state, level);
                }

                // CRITICAL FIX: Send the exact ArrayBuffer (already properly sliced by worklet)
                this.sttSocket.send(frameBuffer);

                // Debug logging occasionally
                if (Math.random() < 0.01) {
                    console.log(`📤 Sent audio frame: ${frameBuffer.byteLength} bytes, RMS: ${rms.toFixed(3)}`);
                }
            }
        }
    }

    handleSTTMessage(message) {
        switch (message.type) {
            case 'stt_ready':
                console.log('🎙️ STT ready');
                break;

            case 'stt_level':
                // Server-computed level (alternative to client-side RMS)
                if (this.chatInterface && this.chatInterface.setMicState) {
                    this.chatInterface.setMicState('listening', message.level);
                }
                break;

            case 'stt_partial':
                if (message.text && message.text.trim()) {
                    this.partialBuf = message.text.trim();
                    this.utteranceActive = true;
                    this.setState('utterance_active');
                    this.updateLiveTranscription();

                    // Set up auto-finalization timeout
                    this.setupAutoFinalization();
                }
                break;

            case 'stt_final':
                if (message.text && message.text.trim()) {
                    this.finalText = message.text.trim();
                    this.setState('finalizing');
                    this.commitTranscript();
                }
                break;

            default:
                console.debug('STT unknown message:', message);
        }
    }

    handleTTSMessage(message) {
        console.debug('🔊 TTS message:', message);
        switch (message.type) {
            case 'tts_start':
                console.log('🔊 TTS started');
                this.ensureAudioReady().catch(console.error);
                this.setState('speaking');
                this.setHalfDuplexMode('speaking');
                break;

            case 'tts_end':
                console.log('🔊 TTS ended');
                this.setState('listening');
                this.setHalfDuplexMode('hold_off');
                break;

            default:
                console.debug('🔊 Unknown TTS message type:', message.type);
        }
    }

    // CRITICAL: Must be called from user gesture (click/tap) for Edge/Chrome
    async ensureAudioContext() {
        try {
            // Create AudioContext if needed
            if (!this.audioContext) {
                const AudioContextClass = window.AudioContext || window.webkitAudioContext;
                if (!AudioContextClass) {
                    throw new Error('AudioContext not supported in this browser');
                }

                this.audioContext = new AudioContextClass({
                    latencyHint: 'interactive' // Optimize for real-time audio
                });
                console.log(`🔊 AudioContext created: state=${this.audioContext.state}, sampleRate=${this.audioContext.sampleRate}`);
            }

            // CRITICAL: Resume AudioContext (must be in user gesture call stack)
            if (this.audioContext.state === 'suspended') {
                console.log('🔓 Resuming suspended AudioContext (user gesture required)...');
                await this.audioContext.resume();
                console.log(`🔊 AudioContext resumed: state=${this.audioContext.state}`);
            }

            // Create TTS gain node if needed
            if (!this._ttsGain) {
                this._ttsGain = this.audioContext.createGain();
                this._ttsGain.gain.value = 1.0;
                // CRITICAL: Connect directly to destination for reliable audio
                this._ttsGain.connect(this.audioContext.destination);
                console.log('🔊 TTS gain node created and connected to destination');
            }

            // Initialize playhead for chunk scheduling
            this._ttsPlayhead = this.audioContext.currentTime + 0.03;

            console.log('✅ AudioContext ready for audio output');
            return this.audioContext;

        } catch (error) {
            console.error('❌ Failed to ensure AudioContext:', error);
            throw error;
        }
    }

    // Ensure TTS chain is properly set up
    ensureTtsChain() {
        if (!this.audioContext) {
            throw new Error('AudioContext not initialized');
        }

        if (!this._ttsGain) {
            this._ttsGain = this.audioContext.createGain();
            this._ttsGain.gain.value = 1.0;
        }
    }

    // Route A: DIRECT to default speakers (most reliable)
    routeTtsToDefault() {
        this.ensureTtsChain();

        try {
            this._ttsGain.disconnect();
        } catch (e) {
            // Ignore disconnect errors
        }

        this._ttsGain.connect(this.audioContext.destination);
        console.log('🔊 TTS routed to default destination');
    }

    // Route B: via <audio> element with setSinkId (for device selection)
    async routeTtsToMonitor() {
        this.ensureTtsChain();

        // Create MediaStreamDestination if needed
        if (!this._monitorDest) {
            this._monitorDest = this.audioContext.createMediaStreamDestination();
        }

        try {
            this._ttsGain.disconnect();
        } catch (e) {
            // Ignore disconnect errors
        }

        this._ttsGain.connect(this._monitorDest);

        // Create monitor audio element if needed
        if (!this._monitorEl) {
            this._monitorEl = document.createElement('audio');
            this._monitorEl.autoplay = true;
            this._monitorEl.playsInline = true;
            this._monitorEl.style.display = 'none';
            document.body.appendChild(this._monitorEl);
        }

        this._monitorEl.srcObject = this._monitorDest.stream;

        try {
            await this._monitorEl.play(); // MUST be triggered after a user gesture
            console.log('🔊 Monitor element playing');
        } catch (e) {
            console.error('❌ monitor.play() failed:', e);
            throw e;
        }
    }

    // Optional: switch speakers (requires HTTPS/localhost, and user gesture)
    async setOutputDevice(deviceId) {
        if (this._monitorEl && this._monitorEl.setSinkId) {
            await this._monitorEl.setSinkId(deviceId);
            console.log('🔊 Output device set via <audio>.setSinkId:', deviceId);
        } else {
            console.warn('⚠️ setSinkId not supported; using default output');
        }
    }

    // Step 4: Synthetic speech test (proves scheduler works without server)
    playSyntheticSpeech() {
        try {
            console.log('🔊 Playing synthetic speech test...');

            this.ensureTtsChain();
            this.routeTtsToDefault();

            const ctx = this.audioContext;
            this._ttsPlayhead = ctx.currentTime + 0.03;

            // 300ms of PCM16@16k "beep"
            const srIn = 16000;
            const duration = 0.3;
            const n = Math.round(srIn * duration);
            const pcm16 = new Int16Array(n);

            // Generate 660Hz tone
            for (let i = 0; i < n; i++) {
                const s = Math.sin(2 * Math.PI * 660 * (i / srIn));
                pcm16[i] = Math.max(-32767, Math.min(32767, s * 32767));
            }

            console.log(`🔊 Generated synthetic PCM: ${n} samples, ${pcm16.buffer.byteLength} bytes`);

            // Use existing enqueue function
            this.enqueueTTSChunk(pcm16.buffer);

            console.log('🔊 Synthetic speech test queued');

        } catch (error) {
            console.error('🔊 Synthetic speech test failed:', error);
        }
    }

    enqueueTTSChunk(arrayBuffer) {
        if (!this.audioContext || !this._ttsGain) {
            console.warn('🔊 AudioContext not ready for TTS chunk');
            return;
        }

        try {
            const ctx = this.audioContext;

            // Ensure AudioContext is running (Edge/Chrome requirement)
            if (ctx.state === 'suspended') {
                console.log('🔓 Resuming AudioContext for TTS chunk');
                ctx.resume().catch(console.error);
                return; // Skip this chunk, next one will work
            }

            // Convert PCM16LE to Float32Array
            const inSamples = new Int16Array(arrayBuffer, 0, arrayBuffer.byteLength / 2);
            console.debug(`🔊 Processing TTS chunk: ${inSamples.length} samples`);

            // Calculate upsampling ratio (16kHz -> context sample rate)
            const targetSampleRate = ctx.sampleRate;
            const sourceSampleRate = 16000;
            const upsampleRatio = Math.round(targetSampleRate / sourceSampleRate);

            // Convert and upsample
            const outSamples = new Float32Array(inSamples.length * upsampleRatio);
            for (let i = 0, j = 0; i < inSamples.length; i++) {
                const normalizedValue = Math.max(-1, Math.min(1, inSamples[i] / 32768));
                for (let k = 0; k < upsampleRatio; k++) {
                    outSamples[j++] = normalizedValue;
                }
            }

            // Create AudioBuffer
            const audioBuffer = ctx.createBuffer(1, outSamples.length, targetSampleRate);
            audioBuffer.copyToChannel(outSamples, 0);

            // Create and configure audio source
            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;

            // Add error handling for Edge compatibility
            source.onerror = (error) => {
                console.error('🔊 TTS audio source error:', error);
            };

            // Connect to gain node
            source.connect(this._ttsGain);

            // Schedule playback with proper timing
            const now = ctx.currentTime;
            if (!this._ttsPlayhead || this._ttsPlayhead < now + 0.01) {
                this._ttsPlayhead = now + 0.05; // Larger buffer for Edge
            }

            // Start playback
            source.start(this._ttsPlayhead);
            this._ttsPlayhead += audioBuffer.duration;

            console.debug(`🔊 TTS chunk scheduled: duration=${audioBuffer.duration.toFixed(3)}s, playhead=${this._ttsPlayhead.toFixed(3)}s`);

        } catch (error) {
            console.error('🔊 Failed to process TTS chunk:', error);

            // Fallback: try to recreate audio context
            if (error.name === 'InvalidStateError' || error.name === 'NotSupportedError') {
                console.log('🔄 Attempting to recreate AudioContext...');
                this.audioContext = null;
                this._ttsGain = null;
                this.ensureAudioContext().catch(console.error);
            }
        }
    }

    // Legacy method for compatibility - now redirects to new implementation
    handleTTSAudio(audioData) {
        console.debug(`🔊 Received TTS audio: ${audioData.byteLength} bytes`);
        this.enqueueTTSChunk(audioData);
    }

    // Enhanced TTS audio handling with browser compatibility
    async handleTTSAudioEnhanced(audioData) {
        try {
            // Ensure audio context is ready
            await this.ensureAudioContext();

            // Validate audio data
            if (!audioData || audioData.byteLength === 0) {
                console.warn('🔊 Empty TTS audio data received');
                return;
            }

            console.debug(`🔊 Processing TTS audio: ${audioData.byteLength} bytes`);

            // Check browser compatibility and use appropriate method
            const userAgent = navigator.userAgent;
            const isEdge = /Edg/.test(userAgent);
            const isChrome = /Chrome/.test(userAgent) && !/Edg/.test(userAgent);

            if (isEdge) {
                // Edge-specific handling with extra buffering
                await this.handleTTSAudioEdge(audioData);
            } else if (isChrome) {
                // Chrome-specific handling
                await this.handleTTSAudioChrome(audioData);
            } else {
                // Generic WebAudio handling
                this.enqueueTTSChunk(audioData);
            }

        } catch (error) {
            console.error('🔊 Enhanced TTS audio handling failed:', error);

            // Fallback to basic method
            try {
                this.enqueueTTSChunk(audioData);
            } catch (fallbackError) {
                console.error('🔊 Fallback TTS handling also failed:', fallbackError);
                this.showVoiceError('Audio playback failed. Please check your speakers.');
            }
        }
    }

    // Edge-specific TTS handling
    async handleTTSAudioEdge(audioData) {
        const ctx = this.audioContext;

        // Edge requires larger buffers and more conservative timing
        const bufferSize = Math.max(4096, audioData.byteLength / 2);
        const chunks = [];

        // Split into smaller chunks for Edge
        for (let offset = 0; offset < audioData.byteLength; offset += bufferSize * 2) {
            const chunkSize = Math.min(bufferSize * 2, audioData.byteLength - offset);
            const chunk = audioData.slice(offset, offset + chunkSize);
            chunks.push(chunk);
        }

        // Process chunks with delays for Edge stability
        for (let i = 0; i < chunks.length; i++) {
            this.enqueueTTSChunk(chunks[i]);

            // Small delay between chunks for Edge
            if (i < chunks.length - 1) {
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }
    }

    // Chrome-specific TTS handling
    async handleTTSAudioChrome(audioData) {
        // Chrome handles larger chunks better
        const maxChunkSize = 8192 * 2; // 8KB chunks

        if (audioData.byteLength <= maxChunkSize) {
            this.enqueueTTSChunk(audioData);
        } else {
            // Split large audio into chunks
            for (let offset = 0; offset < audioData.byteLength; offset += maxChunkSize) {
                const chunkSize = Math.min(maxChunkSize, audioData.byteLength - offset);
                const chunk = audioData.slice(offset, offset + chunkSize);
                this.enqueueTTSChunk(chunk);
            }
        }
    }

    // CRITICAL: Audio unlock method - MUST be called from user gesture (click/tap)
    async unlockAudio() {
        try {
            console.log('🔓 Unlocking audio (user gesture)...');

            // Ensure AudioContext is created and resumed
            await this.ensureAudioContext();

            // Play a brief test tone to activate the audio pipeline
            // This ensures the tab gets the speaker icon and audio actually works
            await this.playTestTone();

            console.log('✅ Audio unlocked successfully');
            return true;

        } catch (error) {
            console.error('❌ Failed to unlock audio:', error);
            return false;
        }
    }

    // Play test tone through the same path as TTS (proves audio works)
    async playTestTone() {
        if (!this.audioContext || !this._ttsGain) {
            throw new Error('AudioContext not ready');
        }

        const ctx = this.audioContext;

        // Create a 0.2s 440Hz tone using AudioBufferSourceNode (same as TTS)
        const sampleRate = ctx.sampleRate;
        const duration = 0.2;
        const samples = Math.floor(sampleRate * duration);
        const buffer = ctx.createBuffer(1, samples, sampleRate);
        const channelData = buffer.getChannelData(0);

        // Generate sine wave
        for (let i = 0; i < samples; i++) {
            channelData[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.3;
        }

        // Play through TTS path
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(this._ttsGain);

        const startTime = Math.max(ctx.currentTime + 0.02, this._ttsPlayhead || 0);
        source.start(startTime);
        this._ttsPlayhead = startTime + buffer.duration;

        console.log('🔊 Test tone played through TTS path');
    }

    // Diagnostic beep test (proves AudioContext → speakers path)
    async playBeep() {
        try {
            await this.ensureAudioContext();
            await this.playTestTone();
            console.log('🔊 Diagnostic beep completed');

        } catch (error) {
            console.error('🔊 Failed to play diagnostic beep:', error);

            // Fallback to HTML5 Audio
            try {
                const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT');
                audio.volume = 0.3;
                await audio.play();
                console.log('🔊 Fallback HTML5 beep played');
            } catch (fallbackError) {
                console.error('🔊 Fallback beep also failed:', fallbackError);
            }
        }
    }

    setHalfDuplexMode(mode) {
        const oldMode = this.halfDuplexMode;
        this.halfDuplexMode = mode;

        if (mode === 'speaking') {
            this.micMuted = true;
            this.ttsPlaying = true;
            this.setState('speaking');
            console.log('🔇 Half-duplex: SPEAKING (mic muted)');
        } else if (mode === 'listening') {
            this.micMuted = false;
            this.ttsPlaying = false;
            this.holdOffUntil = 0;
            this.setState('listening');
            console.log('🎤 Half-duplex: LISTENING (mic active)');
        } else if (mode === 'hold_off') {
            this.holdOffUntil = performance.now() + this.holdOffDurationMs;
            this.micMuted = true;
            this.ttsPlaying = false;
            this.setState('hold_off');
            console.log(`⏸️ Half-duplex: HOLD_OFF (${this.holdOffDurationMs}ms)`);

            setTimeout(() => {
                if (this.halfDuplexMode === 'hold_off') {
                    this.setHalfDuplexMode('listening');
                }
            }, this.holdOffDurationMs);
        }

        this.updateVoiceUI();
    }

    updateVoiceUI() {
        if (this.voiceToggle) {
            this.voiceToggle.classList.toggle('active', this.isVoiceMode);
            this.voiceToggle.setAttribute('aria-pressed', this.isVoiceMode.toString());
        }

        // Smooth mode transitions
        const messageInput = document.getElementById('messageInput');
        const voiceComponents = document.querySelector('.voice-components');

        if (this.isVoiceMode) {
            if (messageInput) {
                messageInput.classList.add('fade-out');
                setTimeout(() => {
                    messageInput.style.display = 'none';
                }, 200);
            }

            if (voiceComponents) {
                voiceComponents.classList.add('fade-in');
                voiceComponents.classList.remove('fade-out');
            }

            if (this.micOrb) {
                this.micOrb.style.display = 'block';
                this.showListeningLabel();
            }

            if (this.liveTranscription) {
                this.liveTranscription.style.display = 'block';
            }
        } else {
            if (voiceComponents) {
                voiceComponents.classList.add('fade-out');
                voiceComponents.classList.remove('fade-in');
            }

            setTimeout(() => {
                if (this.micOrb) {
                    this.micOrb.style.display = 'none';
                }

                if (this.liveTranscription) {
                    this.liveTranscription.style.display = 'none';
                }
            }, 200);

            if (messageInput) {
                messageInput.style.display = 'block';
                messageInput.classList.remove('fade-out');
            }
        }

        // Update orb visualization
        this.updateMicVisualization();
    }



    showListeningLabel() {
        if (!this.micOrb) return;

        const existingLabel = this.micOrb.querySelector('.listening-label');
        if (existingLabel) {
            existingLabel.remove();
        }

        const label = document.createElement('div');
        label.className = 'listening-label';
        label.textContent = 'Listening...';

        this.micOrb.style.position = 'relative';
        this.micOrb.appendChild(label);

        setTimeout(() => {
            if (label.parentNode) {
                label.remove();
            }
        }, 600);
    }

    setMuted(muted) {
        this.micMuted = muted;
        if (this.chatInterface && this.chatInterface.setMicState) {
            this.chatInterface.setMicState(muted ? 'muted' : this.state, this.audioLevel);
        }
    }

    setMicLevel(level) {
        const orb = document.getElementById('micOrb');
        if (orb) {
            orb.style.setProperty('--level', Math.max(0, Math.min(1, level)));
        }
    }

    updateMicVisualization() {
        this.setMicLevel(this.audioLevel);

        // Also update the chat interface orb with current level
        if (this.chatInterface && this.chatInterface.setMicState) {
            this.chatInterface.setMicState(this.state, this.audioLevel);
        }
    }

    ensurePartialTextElement() {
        const liveTranscription = this.liveTranscription || document.getElementById('liveTranscription');
        if (liveTranscription) {
            let partialText = document.getElementById('partialText') || liveTranscription.querySelector('.partial-text');
            if (!partialText) {
                partialText = document.createElement('span');
                partialText.className = 'partial-text';
                partialText.id = 'partialText';
                liveTranscription.appendChild(partialText);
                console.log('✅ Created missing partialText element');
            }
        } else {
            console.warn('⚠️ liveTranscription element not found - partialText cannot be created');
        }
    }

    updateLiveTranscription() {
        if (!this.liveTranscription) {
            this.liveTranscription = document.getElementById('liveTranscription');
        }

        if (this.liveTranscription) {
            let partialText = document.getElementById('partialText');
            if (!partialText) {
                partialText = this.liveTranscription.querySelector('.partial-text');
            }

            if (partialText) {
                if (this.utteranceActive && this.partialBuf) {
                    partialText.textContent = this.partialBuf;
                    partialText.style.opacity = '0.7';
                    partialText.style.color = '#666';
                } else if (this.currentTranscript) {
                    partialText.textContent = this.currentTranscript;
                    partialText.style.opacity = '1.0';
                    partialText.style.color = '#000';

                    const textToCheck = this.currentTranscript;
                    setTimeout(() => {
                        if (partialText.textContent === textToCheck && !this.utteranceActive) {
                            partialText.textContent = '';
                        }
                    }, 2000);
                } else {
                    partialText.textContent = '';
                }
            }

            this.liveTranscription.style.display = this.isVoiceMode ? 'block' : 'none';
        }
    }

    setupAutoFinalization() {
        // Clear any existing timeout
        if (this.finalizationTimeout) {
            clearTimeout(this.finalizationTimeout);
        }

        // Set up auto-finalization after 2 seconds of no new partials
        this.finalizationTimeout = setTimeout(() => {
            if (this.partialBuf && this.partialBuf.trim() && this.utteranceActive) {
                console.log('🏁 Auto-finalizing utterance:', this.partialBuf);

                // Treat the last partial as final
                this.finalText = this.partialBuf.trim();
                this.setState('finalizing');
                this.commitTranscript();
            }
        }, 2000); // 2 second timeout
    }

    commitTranscript() {
        // Clear finalization timeout
        if (this.finalizationTimeout) {
            clearTimeout(this.finalizationTimeout);
            this.finalizationTimeout = null;
        }

        const textToCommit = this.finalText || this.partialBuf;

        if (textToCommit && textToCommit.trim()) {
            this.currentTranscript = textToCommit.trim();
            this.lastCommittedText = this.currentTranscript;

            console.log('💬 Committing transcript:', this.currentTranscript);

            // ✅ Route through ChatInterface cleanly
            if (this.chatInterface?.sendMessage) {
                this.chatInterface.sendMessage(this.currentTranscript);
            }

            // Clear state
            this.finalText = '';
            this.partialBuf = '';
            this.utteranceActive = false;

            // Update UI
            this.updateLiveTranscription();

            // Transition to waiting for bot response
            this.setState('waiting_for_bot');
        }
    }

    showCommandHint(command) {
        if (this.liveTranscription) {
            let hint = this.liveTranscription.querySelector('#commandHint');
            if (!hint) {
                hint = document.createElement('span');
                hint.id = 'commandHint';
                hint.className = 'command-hint';
                this.liveTranscription.appendChild(hint);
            }
            hint.textContent = `🎤 ${command}`;
            hint.hidden = false;
            this.liveTranscription.classList.add('command-mode');
        }

        this.isInCommandMode = true;
    }

    showCommandFeedback(message) {
        if (this.liveTranscription) {
            let hint = this.liveTranscription.querySelector('#commandHint');
            if (!hint) {
                hint = document.createElement('span');
                hint.id = 'commandHint';
                hint.className = 'command-hint';
                this.liveTranscription.appendChild(hint);
            }
            hint.classList.add('command-feedback');
            hint.textContent = `✓ ${message}`;
            hint.hidden = false;
            this.liveTranscription.classList.add('command-executed');

            setTimeout(() => {
                if (hint) {
                    hint.hidden = true;
                    hint.classList.remove('command-feedback');
                }
                if (this.liveTranscription) {
                    this.liveTranscription.classList.remove('command-executed');
                }
            }, 2000);
        }

        this.isInCommandMode = false;
    }

    async speakText(text) {
        if (this.isVoiceMode && this.ttsSocket?.readyState === WebSocket.OPEN) {
            try {
                const message = {
                    type: 'synthesize',
                    text: text.trim()
                };
                console.log('🔊 Sending TTS request:', message);
                this.ttsSocket.send(JSON.stringify(message));
            } catch (error) {
                console.error('Error sending text to TTS:', error);
                this.setState('listening');
            }
        } else {
            console.warn('🔊 Cannot send TTS request:', {
                isVoiceMode: this.isVoiceMode,
                ttsSocketState: this.ttsSocket?.readyState,
                expectedState: WebSocket.OPEN
            });
        }
    }

    showVoiceError(message) {
        if (this.chatInterface && this.chatInterface.showError) {
            this.chatInterface.showError(`Voice: ${message}`);
        } else {
            console.error('Voice error:', message);
        }
    }

    showAudioUnlockPrompt() {
        // Only show once per session
        if (this.audioUnlockPromptShown) return;
        this.audioUnlockPromptShown = true;

        const prompt = document.createElement('div');
        prompt.className = 'audio-unlock-prompt';
        prompt.innerHTML = `
            <div class="prompt-content">
                <p>Audio is suspended. Click anywhere to enable audio.</p>
                <button onclick="this.parentElement.parentElement.remove()">Dismiss</button>
            </div>
        `;

        document.body.appendChild(prompt);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (prompt.parentNode) {
                prompt.remove();
            }
        }, 5000);

        // Remove on any click
        document.addEventListener('click', () => {
            if (prompt.parentNode) {
                prompt.remove();
            }
            if (this.audioContext && this.audioContext.state === 'suspended') {
                this.audioContext.resume();
            }
        }, { once: true });
    }

    getVoiceStatus() {
        return {
            isVoiceMode: this.isVoiceMode,
            state: this.state,
            audioLevel: this.audioLevel,
            vadState: this.vadState,
            hasSTTConnection: this.sttSocket && this.sttSocket.readyState === WebSocket.OPEN,
            hasTTSConnection: this.ttsSocket && this.ttsSocket.readyState === WebSocket.OPEN,
            currentTranscript: this.currentTranscript,
            partialTranscript: this.partialTranscript
        };
    }
}
