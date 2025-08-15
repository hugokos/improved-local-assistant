# Edge/Chrome Audio Fixes

## 🎯 **Root Cause Analysis**

The "no audio + no speaker icon" issue in Edge/Chrome is caused by two critical browser policies:

### 1. **Autoplay Policy (User Gesture Requirement)**
- **Problem**: AudioContext starts in `suspended` state and won't output audio until `resume()` is called from a user gesture
- **Symptom**: No speaker icon appears in browser tab, no audio plays
- **Solution**: Call `audioContext.resume()` inside click/tap event handlers

### 2. **WebSocket Binary Type Default**
- **Problem**: WebSocket binary frames arrive as `Blob` by default, not `ArrayBuffer`
- **Symptom**: TTS audio data processing fails silently
- **Solution**: Set `websocket.binaryType = 'arraybuffer'` before connecting

## 🔧 **Implemented Fixes**

### **1. Audio Unlock Pattern**

```javascript
// CRITICAL: Must be called from user gesture (click/tap)
async function unlockAudio() {
    // Create AudioContext if needed
    if (!this.audioContext) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ 
            latencyHint: 'interactive' 
        });
    }
    
    // CRITICAL: Resume AudioContext (must be in user gesture call stack)
    if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
    }
    
    // Play test tone to activate audio pipeline
    await this.playTestTone();
}
```

### **2. WebSocket Binary Configuration**

```javascript
// CRITICAL: Set binaryType before connecting
this.ttsSocket = new WebSocket(wsUrl);
this.ttsSocket.binaryType = 'arraybuffer'; // Default is 'blob'

this.ttsSocket.onmessage = (event) => {
    if (typeof event.data === 'string') {
        // Handle JSON control messages
        const msg = JSON.parse(event.data);
        // ...
    } else {
        // Binary data is now ArrayBuffer (not Blob)
        const audioBuffer = event.data; // ArrayBuffer
        this.enqueueTTSChunk(audioBuffer);
    }
};
```

### **3. Enhanced TTS Audio Processing**

```javascript
enqueueTTSChunk(arrayBuffer) {
    // Ensure AudioContext is running
    if (this.audioContext.state === 'suspended') {
        this.audioContext.resume().catch(console.error);
        return; // Skip this chunk, next one will work
    }
    
    // Convert PCM16LE to Float32Array with bounds checking
    const inSamples = new Int16Array(arrayBuffer);
    const upsampleRatio = Math.round(this.audioContext.sampleRate / 16000);
    const outSamples = new Float32Array(inSamples.length * upsampleRatio);
    
    for (let i = 0, j = 0; i < inSamples.length; i++) {
        const normalizedValue = Math.max(-1, Math.min(1, inSamples[i] / 32768));
        for (let k = 0; k < upsampleRatio; k++) {
            outSamples[j++] = normalizedValue;
        }
    }
    
    // Create and schedule AudioBuffer
    const audioBuffer = this.audioContext.createBuffer(1, outSamples.length, this.audioContext.sampleRate);
    audioBuffer.copyToChannel(outSamples, 0);
    
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this._ttsGain);
    
    // Schedule with proper timing
    const now = this.audioContext.currentTime;
    if (!this._ttsPlayhead || this._ttsPlayhead < now + 0.01) {
        this._ttsPlayhead = now + 0.03;
    }
    
    source.start(this._ttsPlayhead);
    this._ttsPlayhead += audioBuffer.duration;
}
```

## 🚀 **Integration Points**

### **Voice Mode Activation**
```javascript
async startVoiceMode() {
    // CRITICAL: Unlock audio first (in user gesture call stack)
    const audioUnlocked = await this.unlockAudio();
    if (!audioUnlocked) {
        throw new Error('Failed to unlock audio - user gesture required');
    }
    
    // Continue with microphone setup...
    this.micStream = await navigator.mediaDevices.getUserMedia({...});
    // ...
}
```

### **Speaker Test Button**
```javascript
// Called from click event (user gesture)
async function testSpeaker() {
    // Create AudioContext in user gesture call stack
    const testCtx = new AudioContext({ latencyHint: 'interactive' });
    
    // Resume immediately (still in user gesture)
    if (testCtx.state === 'suspended') {
        await testCtx.resume();
    }
    
    // Play test tone - this activates the audio pipeline
    // Browser will show speaker icon after this
    const oscillator = testCtx.createOscillator();
    // ... configure and play
}
```

## 🔍 **Debugging Checklist**

### **1. AudioContext State**
```javascript
console.log('AudioContext state:', audioContext.state); // Should be 'running'
```

### **2. WebSocket Binary Type**
```javascript
console.log('WebSocket binaryType:', ttsSocket.binaryType); // Should be 'arraybuffer'
```

### **3. Browser Tab Speaker Icon**
- Should appear after first successful audio output
- If missing, AudioContext is likely still suspended

### **4. Network Tab (DevTools)**
- WebSocket frames should show "Binary" entries
- Frame sizes should be > 0 bytes

## 🌐 **Browser-Specific Notes**

### **Microsoft Edge**
- Requires secure context (HTTPS or localhost) for full audio features
- May need autoplay policy set to "Allow" for development
- Prefers standard sample rates (44.1kHz, 48kHz)
- Benefits from smaller audio chunks with delays

### **Google Chrome**
- Strict autoplay policy enforcement
- Handles larger audio chunks efficiently
- Good WebAudio performance with proper unlocking

### **Development Settings**
- **Edge**: Settings → Cookies and site permissions → Media autoplay → Allow
- **Chrome**: chrome://settings/content/sound → Allow sites to play sound

## 📋 **Testing Procedure**

1. **Open debug tool**: `edge_audio_debug.html`
2. **Click "Test WebAudio Speaker"** (user gesture)
3. **Check console logs** for AudioContext state transitions
4. **Verify speaker icon** appears in browser tab
5. **Test TTS simulation** to verify audio pipeline
6. **Check WebSocket frames** in Network tab

## ✅ **Success Indicators**

- ✅ AudioContext state becomes "running"
- ✅ Browser tab shows speaker icon
- ✅ Test tones play audibly
- ✅ Console shows "audio unlocked" messages
- ✅ WebSocket binary frames process correctly
- ✅ TTS audio plays without gaps or distortion

## 🚨 **Common Issues**

### **No Speaker Icon**
- AudioContext still suspended (not resumed in user gesture)
- No actual audio output (check volume/mute)

### **Silent TTS**
- WebSocket binaryType not set to 'arraybuffer'
- AudioContext suspended during playback
- Audio chunks not properly converted/scheduled

### **Audio Gaps/Glitches**
- Insufficient buffering (increase playhead offset)
- Chunk processing errors (check bounds)
- Context state changes during playback

This comprehensive fix ensures reliable audio functionality across Edge, Chrome, and other modern browsers while respecting their security policies.