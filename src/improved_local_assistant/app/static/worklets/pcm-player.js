/**
 * PCM Player AudioWorklet for TTS audio playback.
 * 
 * Plays raw PCM audio chunks from TTS without decoding overhead.
 * Handles buffering and smooth playback.
 */

class PCMPlayerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        
        // Audio buffer (ring buffer)
        this.bufferSize = 8192; // ~0.5 seconds at 16kHz
        this.buffer = new Float32Array(this.bufferSize);
        this.writeIndex = 0;
        this.readIndex = 0;
        this.samplesAvailable = 0;
        
        // Playback state
        this.isPlaying = false;
        this.targetSampleRate = 22050; // Piper TTS output rate
        this.outputSampleRate = sampleRate; // Browser output rate
        
        // Resampling for output (22kHz -> 48kHz typically)
        this.resampleRatio = this.outputSampleRate / this.targetSampleRate;
        this.lastSample = 0;
        
        // Handle incoming audio data
        this.port.onmessage = (event) => {
            this.handleMessage(event.data);
        };
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'audio':
                this.enqueueAudio(data.buffer);
                break;
            case 'start':
                this.isPlaying = true;
                break;
            case 'stop':
                this.isPlaying = false;
                break;
            case 'clear':
                this.clearBuffer();
                break;
        }
    }
    
    enqueueAudio(audioBuffer) {
        // Convert Int16 PCM to Float32 with proper scaling
        const int16Array = new Int16Array(audioBuffer);
        const float32Array = new Float32Array(int16Array.length);
        
        for (let i = 0; i < int16Array.length; i++) {
            // Proper 16-bit to float conversion
            float32Array[i] = int16Array[i] / 32768.0;
        }
        
        // Add to ring buffer with overflow protection
        for (let i = 0; i < float32Array.length; i++) {
            if (this.samplesAvailable < this.bufferSize) {
                this.buffer[this.writeIndex] = float32Array[i];
                this.writeIndex = (this.writeIndex + 1) % this.bufferSize;
                this.samplesAvailable++;
            } else {
                // Buffer overflow - drop oldest samples (prevents memory issues)
                this.readIndex = (this.readIndex + 1) % this.bufferSize;
                this.buffer[this.writeIndex] = float32Array[i];
                this.writeIndex = (this.writeIndex + 1) % this.bufferSize;
                // samplesAvailable stays the same since we dropped one and added one
            }
        }
        
        // CRITICAL FIX: Start playback on first chunk for low latency
        if (!this.isPlaying && this.samplesAvailable > 0) {
            this.isPlaying = true;
            this.port.postMessage({ type: 'started' });
            console.log('🔊 PCM Player: Started playback with', this.samplesAvailable, 'samples buffered');
        }
    }
    
    clearBuffer() {
        this.writeIndex = 0;
        this.readIndex = 0;
        this.samplesAvailable = 0;
        this.isPlaying = false;
        this.buffer.fill(0);
    }
    
    process(inputs, outputs, parameters) {
        const output = outputs[0];
        
        if (!output || output.length === 0) {
            return true;
        }
        
        const outputChannel = output[0];
        
        if (!this.isPlaying || this.samplesAvailable === 0) {
            // Output silence
            outputChannel.fill(0);
            return true;
        }
        
        // Fill output buffer with resampled audio
        for (let i = 0; i < outputChannel.length; i++) {
            if (this.samplesAvailable > 0) {
                // Simple linear interpolation for resampling
                const sample = this.getSample();
                outputChannel[i] = sample;
            } else {
                outputChannel[i] = 0;
                // Stop playing when buffer is empty
                if (this.isPlaying) {
                    this.isPlaying = false;
                    this.port.postMessage({ type: 'ended' });
                }
            }
        }
        
        return true;
    }
    
    getSample() {
        if (this.samplesAvailable === 0) {
            return 0;
        }
        
        // Simple but effective resampling for Piper's 22kHz -> browser's 48kHz
        const sample = this.buffer[this.readIndex];
        
        // Linear interpolation resampling (better than probabilistic)
        // For 22kHz -> 48kHz: ratio ≈ 2.18, so we need to interpolate
        if (this.resampleRatio > 1.5) {
            // Upsampling: use linear interpolation
            const nextIndex = (this.readIndex + 1) % this.bufferSize;
            const nextSample = this.samplesAvailable > 1 ? this.buffer[nextIndex] : sample;
            
            // Simple linear interpolation (could be improved with better filters)
            const interpolated = sample * 0.7 + nextSample * 0.3;
            
            // Advance read pointer less frequently for upsampling
            this.lastSample = interpolated;
            
            // Advance based on ratio (consume input samples slower)
            if (Math.random() < (1.0 / this.resampleRatio)) {
                this.readIndex = (this.readIndex + 1) % this.bufferSize;
                this.samplesAvailable--;
            }
            
            return interpolated;
        } else {
            // Downsampling or 1:1: simple decimation
            this.readIndex = (this.readIndex + 1) % this.bufferSize;
            this.samplesAvailable--;
            return sample;
        }
    }
    
    // Get buffer status for monitoring
    getBufferStatus() {
        return {
            available: this.samplesAvailable,
            capacity: this.bufferSize,
            playing: this.isPlaying
        };
    }
}

registerProcessor('pcm-player', PCMPlayerProcessor);