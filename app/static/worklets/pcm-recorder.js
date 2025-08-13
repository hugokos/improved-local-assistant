/**
 * PCM Recorder AudioWorklet for voice processing.
 * 
 * Properly handles resampling from browser sample rate to 16kHz
 * and batches audio frames to prevent WebSocket backpressure.
 */

class PCMRecorderProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        
        // Resampling state - downsample to 16kHz for Vosk
        this.targetSampleRate = 16000; // Vosk standard rate
        this.inputSampleRate = sampleRate; // Browser's sample rate (usually 48kHz)
        this.resampleRatio = this.inputSampleRate / this.targetSampleRate; // e.g., 48000/16000 = 3
        this.resampleBuffer = new Float32Array(0);
        
        // WebRTC VAD requires exact frame timing: 10, 20, or 30ms frames
        this.vadFrameMs = 20; // Use 20ms frames (SWEET SPOT for most speech)
        this.vadFrameSize = Math.floor(this.targetSampleRate * this.vadFrameMs / 1000); // 320 samples = 20ms at 16kHz
        this.vadFrame = new Int16Array(this.vadFrameSize);
        this.vadFrameIndex = 0;
        
        // Remove batch system - use exact VAD frames only
        
        // RMS calculation for visualization
        this.rmsWindow = new Float32Array(Math.floor(this.targetSampleRate * 0.01)); // 10ms window at 16kHz
        this.rmsIndex = 0;
        
        console.log(`PCM Recorder initialized: ${this.inputSampleRate}Hz → ${this.targetSampleRate}Hz (ratio: ${this.resampleRatio.toFixed(2)})`);
    }
    
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        
        if (input.length > 0) {
            const inputChannel = input[0]; // Use first channel (mono)
            
            // Downsample to 16kHz using simple decimation
            this.downsampleAndProcess(inputChannel);
        }
        
        return true; // Keep processor alive
    }
    
    downsampleAndProcess(inputSamples) {
        // Append new samples to resample buffer
        const newBuffer = new Float32Array(this.resampleBuffer.length + inputSamples.length);
        newBuffer.set(this.resampleBuffer, 0);
        newBuffer.set(inputSamples, this.resampleBuffer.length);
        this.resampleBuffer = newBuffer;
        
        // Downsample using simple decimation (take every Nth sample)
        const outputLength = Math.floor(this.resampleBuffer.length / this.resampleRatio);
        
        for (let i = 0; i < outputLength; i++) {
            const inputIndex = Math.floor(i * this.resampleRatio);
            const sample = this.resampleBuffer[inputIndex];
            this.processSample(sample);
        }
        
        // Keep remaining samples for next process() call
        const samplesUsed = Math.floor(outputLength * this.resampleRatio);
        this.resampleBuffer = this.resampleBuffer.slice(samplesUsed);
    }
    
    processSample(sample) {
        // Clamp and convert to 16-bit PCM
        const clampedSample = Math.max(-1, Math.min(1, sample));
        const pcmSample = Math.round(clampedSample * 32767);
        
        // CRITICAL FIX: Accumulate samples into exact VAD frames
        this.vadFrame[this.vadFrameIndex] = pcmSample;
        this.vadFrameIndex++;
        
        // Update RMS window for visualization
        this.rmsWindow[this.rmsIndex] = clampedSample;
        this.rmsIndex = (this.rmsIndex + 1) % this.rmsWindow.length;
        
        // EXACT FRAME TIMING: Send VAD frame when complete (20ms = 320 samples = 640 bytes)
        if (this.vadFrameIndex >= this.vadFrameSize) {
            this.sendVADFrame();
            this.vadFrameIndex = 0;
        }
    }
    
    sendVADFrame() {
        // CRITICAL: Send exact 20ms VAD frame (320 samples = 640 bytes)
        const vadFrameData = this.vadFrame.slice(0, this.vadFrameSize); // Always full frame
        
        // STRICT VALIDATION: Ensure exact 640 bytes for WebRTC VAD
        const frameBytes = vadFrameData.length * 2; // Int16 = 2 bytes per sample
        if (frameBytes !== 640) {
            console.error(`❌ CRITICAL: VAD frame size ${frameBytes} bytes, expected 640 bytes`);
            return; // Drop invalid frames
        }
        
        // Debug frame sizes occasionally
        if (Math.random() < 0.01) { // Log ~1% of frames
            console.log(`✅ VAD frame: ${vadFrameData.length} samples = ${frameBytes} bytes (correct!)`);
        }
        
        this.port.postMessage({
            type: 'vad_frame',
            data: vadFrameData.buffer,
            sampleRate: this.targetSampleRate,
            samples: this.vadFrameSize,
            frameMs: this.vadFrameMs,
            frameBytes: frameBytes
        }, [vadFrameData.buffer]);
        
        // Calculate RMS for visualization from this frame
        let rmsSum = 0;
        for (let i = 0; i < this.vadFrameSize; i++) {
            const normalized = vadFrameData[i] / 32767;
            rmsSum += normalized * normalized;
        }
        const rms = Math.sqrt(rmsSum / this.vadFrameSize);
        
        // Send RMS level for visualization
        this.port.postMessage({
            type: 'rms',
            level: rms
        });
    }
    
    // Flush any remaining samples when stopping
    flush() {
        if (this.batchIndex > 0) {
            this.sendBatch();
            this.batchIndex = 0;
        }
    }
}

registerProcessor('pcm-recorder', PCMRecorderProcessor);