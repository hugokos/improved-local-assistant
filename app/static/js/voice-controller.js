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
            console.log('Starting voice mode...');
            
            this.micStream = await this.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            
            // Load AudioWorklets
            await this.audioContext.audioWorklet.addModule('/static/worklets/pcm-recorder.js');
            await this.audioContext.audioWorklet.addModule('/static/worklets/pcm-player.js');
            
            // Create worklet nodes
            this.pcmRecorder = new AudioWorkletNode(this.audioContext, 'pcm-recorder');
            this.pcmPlayer = new AudioWorkletNode(this.audioContext, 'pcm-player');
            
            // Handle recorder messages
            this.pcmRecorder.port.onmessage = (event) => {
                this.handleRecorderMessage(event.data);
            };
            
            // Connect audio pipeline
            const source = this.audioContext.createMediaStreamSource(this.micStream);
            source.connect(this.pcmRecorder);
            this.pcmPlayer.connect(this.audioContext.destination);
            
            // Connect to WebSockets
            await this.connectSTT();
            await this.connectTTS();
            
            // Update state and UI
            this.isVoiceMode = true;
            this.setState('listening');
            this.updateVoiceUI();
            
            console.log('✅ Voice mode started successfully');
        } catch (error) {
            console.error('❌ Failed to start voice mode:', error);
            await this.stopVoiceMode();
            this.showVoiceError('Failed to start voice mode. Please check your microphone.');
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
            this.sttSocket.binaryType = 'arraybuffer'; // Ensure binary frames
            
            this.sttSocket.onopen = () => {
                console.log('STT WebSocket connected');
                
                // 🔑 Handshake so server can create recognizer + validate format
                this.sttSocket.send(JSON.stringify({
                    type: 'stt_start',
                    session_id: this.chatInterface.sessionId,
                    sample_rate: 16000,
                    encoding: 'pcm16le',
                    channels: 1,
                    // hint server VAD policy if you like: 'server' | 'client' | 'both'
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
            this.ttsSocket.binaryType = 'arraybuffer';
            
            this.ttsSocket.onopen = () => {
                console.log('TTS WebSocket connected');
                resolve();
            };
            
            this.ttsSocket.onmessage = (event) => {
                if (event.data instanceof ArrayBuffer) {
                    this.handleTTSAudio(event.data);
                } else {
                    this.handleTTSMessage(JSON.parse(event.data));
                }
            };
            
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
        if (data.type === 'audio') {
            this.handleAudioFrame(data);
        } else if (data.type === 'rms') {
            this.audioLevel = data.level;
            this.updateMicVisualization();
        }
    }
    
    handleAudioFrame(data) {
        if (this.halfDuplexMode === 'speaking' || this.micMuted) {
            return;
        }
        
        if ((this.state === 'listening' || this.state === 'utterance_active') && this.sttSocket) {
            const validSizes = [320, 640, 960];
            const view = data.data; // Uint8Array or ArrayBuffer from worklet
            const buf = view instanceof ArrayBuffer ? view : view.buffer;
            const frame = buf.slice(0, validSizes.includes(view.byteLength) ? view.byteLength : 0);
            
            if (frame.byteLength && this.sttSocket.readyState === WebSocket.OPEN && 
                this.sttSocket.bufferedAmount < this.maxBufferedAmount) {
                
                // Compute RMS client-side for orb level (fast and simple)
                const dv = new DataView(frame);
                let sum = 0;
                for (let i = 0; i < dv.byteLength; i += 2) {
                    const s = dv.getInt16(i, true) / 32768;
                    sum += s * s;
                }
                const rms = Math.sqrt(sum / (dv.byteLength / 2));
                const level = Math.min(1, rms * 2); // gentle boost
                
                // Update orb with current level
                if (this.chatInterface && this.chatInterface.setMicState) {
                    this.chatInterface.setMicState(this.state, level);
                }
                
                // Send ArrayBuffer to server
                this.sttSocket.send(frame);
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
        }
    }
    
    handleTTSAudio(audioData) {
        if (this.pcmPlayer && audioData.byteLength > 0) {
            this.pcmPlayer.port.postMessage({
                type: 'audio',
                data: audioData
            });
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
    
    commitTranscript() {
        if (this.finalText && this.finalText.trim()) {
            this.currentTranscript = this.finalText.trim();
            this.lastCommittedText = this.currentTranscript;
            
            // Send to chat interface by simulating form submission
            if (this.chatInterface) {
                // Set the message input value and trigger send
                const messageInput = document.getElementById('messageInput');
                if (messageInput) {
                    messageInput.value = this.currentTranscript;
                    this.chatInterface.sendMessage();
                } else {
                    // Fallback: send directly via WebSocket if available
                    if (this.chatInterface.chatWebSocket && 
                        this.chatInterface.chatWebSocket.readyState === WebSocket.OPEN) {
                        this.chatInterface.chatWebSocket.send(this.currentTranscript);
                        this.chatInterface.addMessage('user', this.currentTranscript);
                        this.chatInterface.showTyping();
                        setTimeout(() => {
                            this.chatInterface.hideTyping();
                            this.chatInterface.startNewAssistantMessage();
                        }, 500);
                    }
                }
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
                this.ttsSocket.send(JSON.stringify({
                    type: 'synthesize',
                    text: text.trim()
                }));
            } catch (error) {
                console.error('Error sending text to TTS:', error);
                this.setState('listening');
            }
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