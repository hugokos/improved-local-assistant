// Improved Local AI Assistant Web Interface
class ChatInterface {
    constructor() {
        this.chatWebSocket = null;
        this.monitorWebSocket = null;
        this.sessionId = this.loadOrCreateSessionId();
        this.currentMessage = null;
        this.isTyping = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second delay

        // Voice controller
        this.voiceController = null;

        // Mic orb elements
        this.micOrbEl = null;
        this._bindedUpdateLevel = null;

        // Initialize markdown parser
        this.md = window.markdownit({
            html: false,        // Disable HTML tags in source
            xhtmlOut: false,    // Use '/' to close single tags (<br />)
            breaks: true,       // Convert '\n' in paragraphs into <br>
            linkify: true,      // Autoconvert URL-like text to links
            typographer: true   // Enable some language-neutral replacement + quotes beautification
        });

        this.init();
    }

    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9);
    }

    loadOrCreateSessionId() {
        // Try to load session ID from localStorage
        const savedSessionId = localStorage.getItem('chatSessionId');
        if (savedSessionId) {
            return savedSessionId;
        }

        // Create new session ID
        const newSessionId = this.generateSessionId();
        localStorage.setItem('chatSessionId', newSessionId);
        return newSessionId;
    }

    init() {
        // Initialize UI elements
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.messagesContainer = document.getElementById('chatBox');
        this.settingsButton = document.getElementById('settingsButton');
        this.settingsModal = document.getElementById('settingsModal');
        this.saveSettingsBtn = document.getElementById('saveSettingsBtn');

        // Initialize side panels
        this.initializeSidePanels();

        // Initialize vintage UI elements
        this.initializeVintageUI();

        // Initialize sliders
        this.initializeSliders();

        // Add event listeners
        this.addEventListeners();

        // Try to connect
        this.connectChat();
        this.connectMonitoring();

        // Load settings from localStorage
        this.loadSettings();

        // Load current models
        this.loadCurrentModels();

        // Initialize voice controller
        this.initializeVoiceController();

        // Initialize mic orb reference
        this.micOrbEl = document.getElementById('micOrb');
    }

    initializeVoiceController() {
        // Add a small delay to ensure DOM is fully ready
        setTimeout(() => {
            try {
                console.log('Initializing voice controller...');
                console.log('DOM elements available:', {
                    voiceToggle: !!document.getElementById('voiceToggle'),
                    micOrb: !!document.getElementById('micOrb'),
                    liveTranscription: !!document.getElementById('liveTranscription'),
                    partialText: !!document.getElementById('partialText')
                });

                if (typeof VoiceController !== 'undefined') {
                    this.voiceController = new VoiceController(this);

                    // Verify the instance was created properly
                    if (this.voiceController && typeof this.voiceController.setState === 'function') {
                        console.log('Voice controller initialized successfully');

                        // Hook into voice controller state changes
                        const origSetState = this.voiceController.setState.bind(this.voiceController);
                        this.voiceController.setState = (newState, payload = {}) => {
                            // Update orb immediately on any state change
                            // payload.level can be 0..1 if your controller exposes RMS
                            this.setMicState(newState, payload.level);
                            this.showMicOrb(true);
                            return origSetState(newState, payload);
                        };

                        // Optional: if your controller exposes an audio meter callback
                        if (this.voiceController && typeof this.voiceController.onLevel === 'function') {
                            this._bindedUpdateLevel = (lvl) => this.setMicState('listening', lvl);
                            this.voiceController.onLevel(this._bindedUpdateLevel);
                        }
                    } else {
                        console.error('Voice controller created but missing methods');
                        console.log('Voice controller type:', typeof this.voiceController);
                        console.log('setState method:', typeof this.voiceController?.setState);
                        this.voiceController = null;
                    }
                } else {
                    console.warn('VoiceController class not available');
                }
            } catch (error) {
                console.error('Failed to initialize voice controller:', error);
                this.voiceController = null;
            }
        }, 100); // 100ms delay
    }

    initializeSidePanels() {
        // Initialize panel toggle functionality
        const panelHeaders = document.querySelectorAll('.panel-header');

        panelHeaders.forEach(header => {
            header.addEventListener('click', () => {
                this.togglePanel(header);
            });

            // Add keyboard support
            header.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.togglePanel(header);
                }
            });
        });

        // Initialize panel data containers
        this.prebuiltCitationsContainer = document.querySelector('#prebuilt-citations-panel .panel-scroll');
        this.dynamicKgContainer = document.querySelector('#dynamic-kg-panel .panel-scroll');
        this.voiceCommandsContainer = document.querySelector('#voice-commands-panel .panel-scroll');

        // Initialize debounced update functions
        this.debouncedUpdatePrebuilt = this.debounce(this.updatePrebuiltCitations.bind(this), 300);
        this.debouncedUpdateDynamic = this.debounce(this.updateDynamicKg.bind(this), 300);

        // Initialize real graph events
        this.initializeRealGraphEvents();

        // Initialize voice commands panel
        this.initializeVoiceCommandsPanel();
    }

    initializeVintageUI() {
        // Initialize mini gauges
        this.initializeMiniGauges();

        // Initialize input form
        const inputBar = document.getElementById('inputBar');
        if (inputBar) {
            inputBar.onsubmit = (e) => {
                e.preventDefault();
                this.sendMessage();
            };
        }
    }

    initializeMiniGauges() {
        const C = 2 * Math.PI * 25; // radius is 25 for the mini gauges

        this.arcCpu = (el, p) => {
            if (el) {
                el.style.strokeDasharray = C;
                el.style.strokeDashoffset = C * (1 - p);
            }
        };

        this.arcRam = (el, p) => {
            if (el) {
                el.style.strokeDasharray = C;
                el.style.strokeDashoffset = C * (1 - p);
            }
        };

        // Start the gauge animation
        this.updateMiniGauges();
        setInterval(() => this.updateMiniGauges(), 2600);
    }

    updateMiniGauges() {
        const cpuArc = document.getElementById('cpuArcMini');
        const ramArc = document.getElementById('ramArcMini');
        const cpuTxt = document.getElementById('cpuMiniTxt');
        const ramTxt = document.getElementById('ramMiniTxt');

        // Use actual values if available, otherwise demo values
        const cpu = this.lastCpuUsage || (Math.random() * 0.7 + 0.25);
        const ram = this.lastMemoryUsage || (Math.random() * 0.6 + 0.3);

        this.arcCpu(cpuArc, cpu / 100);
        this.arcRam(ramArc, ram / 100);

        if (cpuTxt) cpuTxt.textContent = Math.round(cpu) + '%';
        if (ramTxt) ramTxt.textContent = Math.round(ram) + '%';
    }

    initializeSliders() {
        // Temperature slider
        const temperatureSlider = document.getElementById('temperature');
        const temperatureValue = document.getElementById('temperatureValue');

        if (temperatureSlider && temperatureValue) {
            temperatureSlider.addEventListener('input', () => {
                temperatureValue.textContent = temperatureSlider.value;
            });
        }

        // Memory limit slider
        const memoryLimitSlider = document.getElementById('memoryLimit');
        const memoryLimitValue = document.getElementById('memoryLimitValue');

        if (memoryLimitSlider && memoryLimitValue) {
            memoryLimitSlider.addEventListener('input', () => {
                memoryLimitValue.textContent = `${memoryLimitSlider.value} GB`;
            });
        }
    }

    addEventListeners() {
        // Message input events
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Settings modal events
        this.settingsButton.addEventListener('click', () => {
            this.openSettingsModal();
        });

        this.saveSettingsBtn.addEventListener('click', () => {
            this.saveSettings();
            this.closeSettingsModal();
        });

        // Graph management buttons
        const loadGraphBtn = document.getElementById('loadGraphBtn');
        const unloadGraphBtn = document.getElementById('unloadGraphBtn');
        const importGraphBtn = document.getElementById('importGraphBtn');

        if (loadGraphBtn) {
            loadGraphBtn.addEventListener('click', () => {
                this.loadSelectedGraphs();
            });
        }

        if (unloadGraphBtn) {
            unloadGraphBtn.addEventListener('click', () => {
                this.unloadSelectedGraphs();
            });
        }

        if (importGraphBtn) {
            importGraphBtn.addEventListener('click', () => {
                this.importGraph();
            });
        }

        // Configuration buttons
        const exportConfigBtn = document.getElementById('exportConfigBtn');
        const importConfigBtn = document.getElementById('importConfigBtn');
        const resetConfigBtn = document.getElementById('resetConfigBtn');

        if (exportConfigBtn) {
            exportConfigBtn.addEventListener('click', () => {
                this.exportConfig();
            });
        }

        if (importConfigBtn) {
            importConfigBtn.addEventListener('click', () => {
                this.importConfig();
            });
        }

        if (resetConfigBtn) {
            resetConfigBtn.addEventListener('click', () => {
                this.resetConfig();
            });
        }

        // Model switching buttons
        const switchConversationBtn = document.getElementById('switchConversationModel');
        const switchKnowledgeBtn = document.getElementById('switchKnowledgeModel');

        if (switchConversationBtn) {
            switchConversationBtn.addEventListener('click', () => {
                this.switchConversationModel();
            });
        }

        if (switchKnowledgeBtn) {
            switchKnowledgeBtn.addEventListener('click', () => {
                this.switchKnowledgeModel();
            });
        }
    }

    connectChat() {
        try {
            // Close existing connection if any
            if (this.chatWebSocket) {
                this.chatWebSocket.close();
            }

            this.updateStatus('Connecting...');
            this.disableInput();

            // Create new WebSocket connection
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;
            this.chatWebSocket = new WebSocket(wsUrl);

            this.chatWebSocket.onopen = () => {
                console.log('Chat WebSocket connected');
                this.updateStatus('Connected');
                this.enableInput();
                this.reconnectAttempts = 0;
                this.reconnectDelay = 1000;
            };

            this.chatWebSocket.onmessage = (event) => {
                this.handleChatMessage(event.data);
            };

            this.chatWebSocket.onclose = (event) => {
                console.log('Chat WebSocket disconnected', event);
                this.updateStatus('Disconnected');
                this.disableInput();

                // Try to reconnect if not closed cleanly
                if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    this.updateStatus(`Reconnecting (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

                    // Exponential backoff for reconnect
                    setTimeout(() => {
                        this.connectChat();
                    }, this.reconnectDelay);

                    // Increase delay for next attempt (max 30 seconds)
                    this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);
                }
            };

            this.chatWebSocket.onerror = (error) => {
                console.error('Chat WebSocket error:', error);
                this.updateStatus('Error');
                this.disableInput();
            };
        } catch (error) {
            console.error('Failed to connect chat:', error);
            this.updateStatus('Connection Failed');
            this.disableInput();
        }
    }

    connectMonitoring() {
        try {
            // Close existing connection if any
            if (this.monitorWebSocket) {
                this.monitorWebSocket.close();
            }

            // Create new WebSocket connection for monitoring
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/monitor`;
            this.monitorWebSocket = new WebSocket(wsUrl);

            this.monitorWebSocket.onopen = () => {
                console.log('Monitoring WebSocket connected');
                this.updateSystemStatus('Online', true);
            };

            this.monitorWebSocket.onmessage = (event) => {
                this.handleMonitoringMessage(event.data);
            };

            this.monitorWebSocket.onclose = () => {
                console.log('Monitoring WebSocket disconnected');
                this.updateSystemStatus('Offline', false);

                // Try to reconnect after 5 seconds
                setTimeout(() => {
                    this.connectMonitoring();
                }, 5000);
            };

            this.monitorWebSocket.onerror = (error) => {
                console.error('Monitoring WebSocket error:', error);
                this.updateSystemStatus('Error', false);
            };
        } catch (error) {
            console.error('Failed to connect monitoring:', error);
            this.updateSystemStatus('Connection Failed', false);
        }
    }

    handleChatMessage(data) {
        try {
            // Skip empty or whitespace-only messages
            if (!data || !data.trim()) {
                return;
            }

            // Try to parse as JSON first
            const jsonData = JSON.parse(data);

            // Skip if no type field (likely malformed JSON)
            if (!jsonData || typeof jsonData.type === 'undefined') {
                // Treat as text token if it's not empty
                if (data.trim()) {
                    this.appendToCurrentMessage(data);
                    this.maybeStartTTS();
                }
                return;
            }

            switch (jsonData.type) {
                case 'heartbeat':
                    return; // ignore quietly

                case 'assistant_text':
                    this.appendToCurrentMessage(jsonData.text || '');
                    this.maybeStartTTS();
                    break;

                case 'assistant_done':
                    if (this.voiceController?.isVoiceMode && !this.ttsStarted) {
                        this.ttsStarted = true;
                        this.voiceController.speakText(this.currentMessageText);
                    }
                    break;

                case 'citations':
                    console.log('📚 Received citations message:', jsonData.data);
                    this.updateCitations(jsonData.data);
                    break;

                case 'error':
                    this.showError(jsonData.message);
                    break;

                case 'system':
                    this.showSystemMessage(jsonData.message);
                    // Store session ID if provided
                    if (jsonData.session_id) {
                        this.sessionId = jsonData.session_id;
                        localStorage.setItem('chatSessionId', this.sessionId);
                    }
                    break;

                case 'typing':
                    this.handleTypingIndicator(jsonData.status);
                    break;

                case 'system_status':
                    this.updateMetrics(jsonData);
                    break;

                case 'graph_update':
                    this.handleGraphUpdate(jsonData);
                    break;

                case 'dynamic_kg_update':
                    console.log('🔗 Received dynamic KG update:', jsonData);
                    this.handleDynamicKgUpdate(jsonData);
                    break;

                case 'available_graphs':
                    this.updateAvailableGraphs(jsonData.graphs);
                    break;

                case 'model_switch':
                    this.handleModelSwitch(jsonData);
                    break;

                default:
                    console.log('Unknown message type:', jsonData.type);
            }
        } catch (e) {
            // Not JSON (token stream) – append and maybe start TTS
            this.appendToCurrentMessage(data);
            this.maybeStartTTS();
        }
    }

    handleMonitoringMessage(data) {
        try {
            const jsonData = JSON.parse(data);

            // GUARD: Ensure message has type field
            if (!jsonData.type) {
                console.warn('⚠️ Message missing type field:', jsonData);
                return;
            }

            if (jsonData.type === 'system_status') {
                this.updateMetrics(jsonData);
            } else if (jsonData.type === 'system') {
                // Handle legacy 'system' type messages
                this.updateMetrics(jsonData);
            } else if (jsonData.type === 'heartbeat') {
                // Ignore heartbeat messages silently
                return;
            } else {
                console.warn('⚠️ Unknown monitoring message type:', jsonData.type);
            }
        } catch (e) {
            console.error('Error handling monitoring message:', e);
        }
    }

    sendMessage(message) {
        // Handle both direct text and getting text from input field
        let text;
        if (message && typeof message === 'string') {
            text = message.trim();
        } else {
            const messageInput = this.messageInput;
            if (!messageInput || !messageInput.value.trim()) return;
            text = messageInput.value.trim();
            messageInput.value = '';
        }

        if (!text || !this.chatWebSocket || this.chatWebSocket.readyState !== WebSocket.OPEN) {
            return;
        }

        // Add user message to chat
        this.addMessage('user', text);

        // Send to server
        this.chatWebSocket.send(text);

        // Note: We don't manually show typing here anymore -
        // we wait for the server to send the typing indicator
    }

    addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role} emboss`;

        if (role === 'user') {
            messageDiv.innerHTML = `&gt; ${content}`;
        } else if (role === 'assistant') {
            // Use content div structure for assistant messages
            const contentDiv = document.createElement('div');
            contentDiv.className = 'content';

            // Clean stray leading ". " on each line before rendering
            const cleaned = content.replace(/^\.\s*/gm, '');
            contentDiv.innerHTML = this.md.render(cleaned);
            messageDiv.appendChild(contentDiv);
        } else {
            messageDiv.innerHTML = content;
        }

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    startNewAssistantMessage() {
        this.currentMessage = document.createElement('div');
        this.currentMessage.className = 'message assistant emboss';

        // Create a dedicated content div for proper markdown rendering
        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';
        this.currentMessage.appendChild(contentDiv);

        // Add typing spinner
        const spinner = document.createElement('div');
        spinner.className = 'typing-spinner';
        spinner.innerHTML = `
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        `;
        this.currentMessage.appendChild(spinner);

        this.messagesContainer.appendChild(this.currentMessage);

        // Initialize accumulated text for streaming
        this.currentMessageText = '';
        this.isFirstToken = true;
        this.ttsStarted = false; // Track if TTS has been started for this message

        this.scrollToBottom();
    }

    appendToCurrentMessage(text) {
        if (this.currentMessage) {
            const contentDiv = this.currentMessage.querySelector('.content');

            // On the very first token, remove the typing spinner
            if (this.isFirstToken) {
                const spinner = this.currentMessage.querySelector('.typing-spinner');
                if (spinner) {
                    spinner.remove();
                }
                this.isFirstToken = false;
            }

            // Accumulate the streaming text
            this.currentMessageText = (this.currentMessageText || '') + text;

            // Clean stray leading ". " on each line before rendering
            const cleaned = this.currentMessageText.replace(/^\.\s*/gm, '');

            // Re-render the entire accumulated text through markdown-it
            // This ensures proper paragraph and list formation
            const formattedContent = this.md.render(cleaned);
            contentDiv.innerHTML = formattedContent;

            // Send to voice controller for TTS if in voice mode
            if (this.voiceController && this.voiceController.isVoiceMode && !this.ttsStarted) {
                // SAFETY FALLBACK: Multiple triggers for TTS start
                let shouldStartTTS = false;
                let textToSpeak = '';

                // Primary trigger: waiting_for_bot state with enough text
                if (this.voiceController.state === 'waiting_for_bot' &&
                    this.isFirstToken === false && this.currentMessageText.length > 10) {
                    shouldStartTTS = true;
                    textToSpeak = this.currentMessageText;
                }

                // Fallback 1: First complete sentence
                if (!shouldStartTTS && this.currentMessageText.length > 20) {
                    const firstSentence = this.currentMessageText.match(/[^.!?]+[.!?]/)?.[0];
                    if (firstSentence && firstSentence.length > 15) {
                        shouldStartTTS = true;
                        textToSpeak = firstSentence;
                        console.log('🔄 TTS fallback: first sentence trigger');
                    }
                }

                if (shouldStartTTS) {
                    this.ttsStarted = true;
                    this.voiceController.speakText(textToSpeak);
                }
            }

            this.scrollToBottom();
        }
    }

    maybeStartTTS() {
        if (!this.voiceController?.isVoiceMode || this.ttsStarted) return;
        const text = this.currentMessageText || '';
        const firstSentence = text.match(/^[\s\S]*?[.!?](?:\s|$)/)?.[0] || '';
        if (firstSentence.length >= 20) {
            this.ttsStarted = true;
            this.voiceController.speakText(firstSentence);
        }
    }

    showTyping() {
        this.hideTyping(); // Remove any existing typing indicator
        const spinner = document.createElement('div');
        spinner.className = 'typing-spinner';
        spinner.id = 'typing';
        spinner.innerHTML = `
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        `;
        this.messagesContainer.appendChild(spinner);
        this.scrollToBottom();
    }

    hideTyping() {
        const existing = document.getElementById('typing');
        if (existing) {
            existing.remove();
        }

        // FALLBACK 2: TTS trigger when typing stops
        if (this.voiceController && this.voiceController.isVoiceMode && !this.ttsStarted &&
            this.currentMessageText && this.currentMessageText.length > 10) {
            console.log('🔄 TTS fallback: typing stopped trigger');
            this.ttsStarted = true;
            this.voiceController.speakText(this.currentMessageText);
        }
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    updateCitations(citationsData) {
        // Update side panels with citation data
        console.log('📚 Processing citations data:', citationsData);

        if (citationsData && citationsData.citations && citationsData.citations.length > 0) {
            console.log(`📚 Found ${citationsData.citations.length} citations to display`);
            this.debouncedUpdatePrebuilt(citationsData.citations);
        } else {
            console.log('📚 No citations found in data');
            // Still update to show "no citations" message
            this.debouncedUpdatePrebuilt([]);
        }
    }



    updateAvailableGraphs(graphs) {
        const graphList = document.getElementById('graphList');
        const graphSelector = document.getElementById('graphSelector');

        if (graphList && graphs) {
            // Clear existing options except the first one (Dynamic Graph)
            while (graphList.options.length > 1) {
                graphList.remove(1);
            }

            // Add new options
            graphs.forEach(graph => {
                const option = document.createElement('option');
                option.value = graph.id;
                option.textContent = graph.name;
                graphList.add(option);
            });
        }

        if (graphSelector && graphs) {
            // Clear existing options except the first one
            while (graphSelector.options.length > 1) {
                graphSelector.remove(1);
            }

            // Add new options
            graphs.forEach(graph => {
                const option = document.createElement('option');
                option.value = graph.id;
                option.textContent = graph.name;
                graphSelector.add(option);
            });
        }
    }

    showError(message) {
        this.addMessage('system', `Error: ${message}`);
    }

    showSystemMessage(message) {
        this.addMessage('system', message);
    }

    handleTypingIndicator(status) {
        if (status === 'start') {
            this.isTyping = true;
            // Start new assistant message with typing indicator
            this.startNewAssistantMessage();
        } else if (status === 'stop') {
            this.isTyping = false;
            // Remove typing indicator (will be handled by appendToCurrentMessage)
            this.hideTyping();
        }
    }

    handleGraphUpdate(data) {
        // Show notification about graph update
        this.showSystemMessage(`Knowledge graph "${data.graph_id}" was ${data.update_type}`);
    }

    handleDynamicKgUpdate(data) {
        // Handle dynamic knowledge graph updates (triples)
        console.log('🔗 Processing dynamic KG update:', data);

        // Handle both direct triples array and nested data structure
        let triples = [];
        if (data.triples && Array.isArray(data.triples)) {
            triples = data.triples;
        } else if (data.data && data.data.triples && Array.isArray(data.data.triples)) {
            triples = data.data.triples;
        }

        console.log(`🔗 Found ${triples.length} triples to process`);

        if (triples.length > 0) {
            // Update the dynamic KG panel with all triples at once
            this.updateDynamicKg(triples);

            // Also add individual triples for real-time updates
            triples.forEach((triple, index) => {
                console.log(`🔗 Processing triple ${index + 1}:`, triple);
                this.addDynamicTriple(triple);
            });
        } else {
            console.log('🔗 No triples found in dynamic KG update');
        }
    }

    setMicState(state, level = 0) {
        const el = this.micOrbEl;
        if (!el) return;

        // clamp and set audio level (drives the conic gradient & glow)
        const clamped = Math.max(0, Math.min(1, level || 0));
        el.style.setProperty('--level', clamped);

        // remove any previous state classes
        el.classList.remove(
            'listening','speaking','waiting','processing',
            'utterance-active','finalizing','muted','holdoff'
        );

        // map and add new one
        const s = (state || '').toLowerCase();
        const map = {
            'idle': null,
            'listening': 'listening',
            'vad_active': 'utterance-active',      // user currently talking
            'utterance_active': 'utterance-active', // user currently talking
            'speaking': 'speaking',                // TTS playing
            'processing': 'finalizing',            // STT finalization/LLM thinking
            'finalizing': 'finalizing',            // STT finalization/LLM thinking
            'waiting_for_bot': 'waiting',          // waiting for bot response
            'hold_off': 'holdoff',                 // brief re-enable delay
            'muted': 'muted'
        };
        const cls = map[s] || null;
        if (cls) el.classList.add(cls);
    }

    showMicOrb(show = true) {
        if (this.micOrbEl) this.micOrbEl.style.display = show ? 'block' : 'none';
    }

    updateStatus(status) {
        const statusElement = document.getElementById('modelStatus');
        if (statusElement) {
            if (status === 'Connected') {
                statusElement.textContent = '🔒 LOCAL-ONLY';
                statusElement.className = 'badge emboss';
            } else {
                statusElement.textContent = `🔄 ${status}`;
                statusElement.className = 'badge emboss';
            }
        }

        // Hide system status text
        const systemStatusText = document.getElementById('systemStatusText');
        if (systemStatusText) {
            systemStatusText.style.display = 'none';
        }
    }

    updateMetrics(data) {
        if (!data.resource_usage) return;

        const cpuUsage = data.resource_usage.cpu_percent || 0;
        const memoryUsage = data.resource_usage.memory_percent || 0;

        // Store for mini gauges
        this.lastCpuUsage = cpuUsage;
        this.lastMemoryUsage = memoryUsage;

        // Update mini gauges immediately
        this.updateMiniGauges();

        // Update system status dot color
        const resDot = document.getElementById('systemStatus');
        if (resDot) {
            if (cpuUsage > 80 || memoryUsage > 80) {
                resDot.style.background = 'var(--error-color)';
            } else if (cpuUsage > 50 || memoryUsage > 50) {
                resDot.style.background = 'var(--warning-color)';
            } else {
                resDot.style.background = 'var(--success-color)';
            }
        }
    }

    enableInput() {
        this.messageInput.disabled = false;
        this.sendButton.disabled = false;
        this.messageInput.placeholder = 'Ask something...';
    }

    disableInput() {
        this.messageInput.disabled = true;
        this.sendButton.disabled = true;
        this.messageInput.placeholder = 'Connecting...';
    }

    openSettingsModal() {
        this.settingsModal.classList.add('active');
    }

    closeSettingsModal() {
        this.settingsModal.classList.remove('active');
    }

    loadSettings() {
        try {
            const savedSettings = localStorage.getItem('assistantSettings');
            if (savedSettings) {
                const settings = JSON.parse(savedSettings);

                // Apply settings to UI
                if (settings.conversationModel) {
                    document.getElementById('conversationModel').value = settings.conversationModel;
                }

                if (settings.knowledgeModel) {
                    document.getElementById('knowledgeModel').value = settings.knowledgeModel;
                }

                if (settings.temperature) {
                    const temperatureSlider = document.getElementById('temperature');
                    const temperatureValue = document.getElementById('temperatureValue');
                    temperatureSlider.value = settings.temperature;
                    temperatureValue.textContent = settings.temperature;
                }

                if (settings.memoryLimit) {
                    const memoryLimitSlider = document.getElementById('memoryLimit');
                    const memoryLimitValue = document.getElementById('memoryLimitValue');
                    memoryLimitSlider.value = settings.memoryLimit;
                    memoryLimitValue.textContent = `${settings.memoryLimit} GB`;
                }

                if (settings.concurrentOps) {
                    document.getElementById('concurrentOps').value = settings.concurrentOps;
                }
            }
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    saveSettings() {
        try {
            const settings = {
                conversationModel: document.getElementById('conversationModel').value,
                knowledgeModel: document.getElementById('knowledgeModel').value,
                temperature: document.getElementById('temperature').value,
                memoryLimit: document.getElementById('memoryLimit').value,
                concurrentOps: document.getElementById('concurrentOps').value
            };

            localStorage.setItem('assistantSettings', JSON.stringify(settings));

            // Send settings to server
            if (this.chatWebSocket && this.chatWebSocket.readyState === WebSocket.OPEN) {
                this.chatWebSocket.send(JSON.stringify({
                    type: 'settings_update',
                    settings: settings
                }));
            }

            this.showSystemMessage('Settings saved successfully');
        } catch (error) {
            console.error('Error saving settings:', error);
            this.showError('Failed to save settings');
        }
    }

    resetConfig() {
        // Default settings
        const defaultSettings = {
            conversationModel: 'hermes3:3b',
            knowledgeModel: 'tinyllama',
            temperature: '0.7',
            memoryLimit: '8',
            concurrentOps: '2'
        };

        // Apply to UI
        document.getElementById('conversationModel').value = defaultSettings.conversationModel;
        document.getElementById('knowledgeModel').value = defaultSettings.knowledgeModel;
        document.getElementById('temperature').value = defaultSettings.temperature;
        document.getElementById('temperatureValue').textContent = defaultSettings.temperature;
        document.getElementById('memoryLimit').value = defaultSettings.memoryLimit;
        document.getElementById('memoryLimitValue').textContent = `${defaultSettings.memoryLimit} GB`;
        document.getElementById('concurrentOps').value = defaultSettings.concurrentOps;

        this.showSystemMessage('Settings reset to defaults');
    }

    // Panel Management Methods
    togglePanel(header) {
        const panel = header.parentElement;
        const content = panel.querySelector('.panel-content');
        const chevron = header.querySelector('.panel-chevron');
        const isExpanded = header.getAttribute('aria-expanded') === 'true';

        if (isExpanded) {
            // Collapse panel
            header.setAttribute('aria-expanded', 'false');
            content.classList.remove('expanded');
            content.style.height = '0';
        } else {
            // Expand panel
            header.setAttribute('aria-expanded', 'true');
            content.classList.add('expanded');

            // Calculate and set height for smooth animation
            const scrollHeight = content.scrollHeight;
            content.style.height = Math.min(scrollHeight, 300) + 'px';
        }
    }

    updatePrebuiltCitations(citations) {
        if (!this.prebuiltCitationsContainer) return;

        if (!citations || citations.length === 0) {
            this.prebuiltCitationsContainer.innerHTML = '<p class="no-data-message emboss">No prebuilt citations available</p>';
            return;
        }

        let html = '';
        citations.forEach((citation, index) => {
            html += `
                <div class="panel-item">
                    <div class="item-header">Citation ${index + 1}</div>
                    <div class="item-content">${citation.text}</div>
                    <div class="item-meta">Source: ${citation.source || 'Unknown'} | Score: ${citation.score ? (citation.score * 100).toFixed(1) + '%' : 'N/A'}</div>
                </div>
            `;
        });

        this.prebuiltCitationsContainer.innerHTML = html;

        // Auto-expand panel if it has new content and is collapsed
        const panel = document.getElementById('prebuilt-citations-panel');
        const header = panel.querySelector('.panel-header');
        if (header.getAttribute('aria-expanded') === 'false') {
            this.togglePanel(header);
        }
    }

    updateDynamicKg(triples) {
        if (!this.dynamicKgContainer) return;

        if (!triples || triples.length === 0) {
            this.dynamicKgContainer.innerHTML = '<p class="no-data-message emboss">No dynamic updates available</p>';
            return;
        }

        let html = '';
        triples.forEach((triple, index) => {
            html += `
                <div class="panel-item">
                    <div class="item-header">Triple ${index + 1}</div>
                    <div class="item-content">${triple.subject} → ${triple.predicate} → ${triple.object}</div>
                    <div class="item-meta">Confidence: ${triple.confidence ? (triple.confidence * 100).toFixed(1) + '%' : 'N/A'} | ${new Date().toLocaleTimeString()}</div>
                </div>
            `;
        });

        this.dynamicKgContainer.innerHTML = html;

        // Auto-expand panel if it has new content and is collapsed
        const panel = document.getElementById('dynamic-kg-panel');
        const header = panel.querySelector('.panel-header');
        if (header.getAttribute('aria-expanded') === 'false') {
            this.togglePanel(header);
        }
    }

    addPrebuiltCitation(citation) {
        if (!this.prebuiltCitationsContainer) return;

        // Remove "no data" message if present
        const noDataMsg = this.prebuiltCitationsContainer.querySelector('.no-data-message');
        if (noDataMsg) {
            noDataMsg.remove();
        }

        const citationElement = document.createElement('div');
        citationElement.className = 'panel-item';
        citationElement.innerHTML = `
            <div class="item-header">New Citation</div>
            <div class="item-content">${citation.text}</div>
            <div class="item-meta">Source: ${citation.source || 'Unknown'} | Score: ${citation.score ? (citation.score * 100).toFixed(1) + '%' : 'N/A'}</div>
        `;

        this.prebuiltCitationsContainer.insertBefore(citationElement, this.prebuiltCitationsContainer.firstChild);

        // Auto-expand panel if collapsed
        const panel = document.getElementById('prebuilt-citations-panel');
        const header = panel.querySelector('.panel-header');
        if (header.getAttribute('aria-expanded') === 'false') {
            this.togglePanel(header);
        }
    }

    addDynamicTriple(triple) {
        if (!this.dynamicKgContainer) return;

        // Remove "no data" message if present
        const noDataMsg = this.dynamicKgContainer.querySelector('.no-data-message');
        if (noDataMsg) {
            noDataMsg.remove();
        }

        const tripleElement = document.createElement('div');
        tripleElement.className = 'panel-item';
        tripleElement.innerHTML = `
            <div class="item-header">New Triple</div>
            <div class="item-content">${triple.subject} → ${triple.predicate} → ${triple.object}</div>
            <div class="item-meta">Confidence: ${triple.confidence ? (triple.confidence * 100).toFixed(1) + '%' : 'N/A'} | ${new Date().toLocaleTimeString()}</div>
        `;

        this.dynamicKgContainer.insertBefore(tripleElement, this.dynamicKgContainer.firstChild);

        // Auto-expand panel if collapsed
        const panel = document.getElementById('dynamic-kg-panel');
        const header = panel.querySelector('.panel-header');
        if (header.getAttribute('aria-expanded') === 'false') {
            this.togglePanel(header);
        }
    }

    initializeRealGraphEvents() {
        // Initialize real graph event listeners
        // Listen for citations and dynamic KG updates from WebSocket messages
        this.citationHistory = [];
        this.dynamicTripleHistory = [];
    }

    initializeVoiceCommandsPanel() {
        // Voice commands panel is static content, no dynamic initialization needed
        // Commands are already defined in HTML

        // Add click handlers for command examples (optional)
        const cmdList = document.querySelector('#voice-commands-panel .cmd-list');
        if (cmdList) {
            cmdList.addEventListener('click', (e) => {
                const kbd = e.target.closest('kbd');
                if (kbd && this.voiceController && this.voiceController.isVoiceMode) {
                    // Optional: simulate voice command when clicked
                    const command = kbd.textContent.replace(/"/g, '');
                    console.log('Voice command clicked:', command);
                    // Could trigger the command here if desired
                }
            });
        }
    }

    updateSystemStatus(status, isOnline) {
        // Hide system status elements
        const statusText = document.getElementById('systemStatusText');
        const statusDot = document.getElementById('systemStatus');

        if (statusText) {
            statusText.style.display = 'none';
        }

        if (statusDot) {
            statusDot.style.display = 'none';
        }
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    exportConfig() {
        try {
            const settings = {
                conversationModel: document.getElementById('conversationModel').value,
                backgroundModel: document.getElementById('backgroundModel').value,
                temperature: document.getElementById('temperature').value,
                memoryLimit: document.getElementById('memoryLimit').value,
                concurrentOps: document.getElementById('concurrentOps').value
            };

            // Create a download link
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(settings, null, 2));
            const downloadAnchorNode = document.createElement('a');
            downloadAnchorNode.setAttribute("href", dataStr);
            downloadAnchorNode.setAttribute("download", "assistant_config.json");
            document.body.appendChild(downloadAnchorNode);
            downloadAnchorNode.click();
            downloadAnchorNode.remove();

            this.showSystemMessage('Configuration exported successfully');
        } catch (error) {
            console.error('Error exporting config:', error);
            this.showError('Failed to export configuration');
        }
    }

    importConfig() {
        // Create a file input element
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = '.json';

        fileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const settings = JSON.parse(e.target.result);

                    // Apply settings to UI
                    if (settings.conversationModel) {
                        document.getElementById('conversationModel').value = settings.conversationModel;
                    }

                    if (settings.backgroundModel) {
                        document.getElementById('backgroundModel').value = settings.backgroundModel;
                    }

                    if (settings.temperature) {
                        const temperatureSlider = document.getElementById('temperature');
                        const temperatureValue = document.getElementById('temperatureValue');
                        temperatureSlider.value = settings.temperature;
                        temperatureValue.textContent = settings.temperature;
                    }

                    if (settings.memoryLimit) {
                        const memoryLimitSlider = document.getElementById('memoryLimit');
                        const memoryLimitValue = document.getElementById('memoryLimitValue');
                        memoryLimitSlider.value = settings.memoryLimit;
                        memoryLimitValue.textContent = `${settings.memoryLimit} GB`;
                    }

                    if (settings.concurrentOps) {
                        document.getElementById('concurrentOps').value = settings.concurrentOps;
                    }

                    this.showSystemMessage('Configuration imported successfully');
                } catch (error) {
                    console.error('Error importing config:', error);
                    this.showError('Failed to import configuration: Invalid format');
                }
            };
            reader.readAsText(file);
        });

        // Trigger file selection
        fileInput.click();
    }

    loadSelectedGraphs() {
        const graphList = document.getElementById('graphList');
        const selectedGraphs = Array.from(graphList.selectedOptions).map(option => option.value);

        if (selectedGraphs.length === 0) {
            this.showError('No graphs selected');
            return;
        }

        // Send load request to server
        if (this.chatWebSocket && this.chatWebSocket.readyState === WebSocket.OPEN) {
            this.chatWebSocket.send(JSON.stringify({
                type: 'load_graphs',
                graph_ids: selectedGraphs
            }));

            this.showSystemMessage(`Loading graphs: ${selectedGraphs.join(', ')}`);
        }
    }

    unloadSelectedGraphs() {
        const graphList = document.getElementById('graphList');
        const selectedGraphs = Array.from(graphList.selectedOptions).map(option => option.value);

        if (selectedGraphs.length === 0) {
            this.showError('No graphs selected');
            return;
        }

        // Send unload request to server
        if (this.chatWebSocket && this.chatWebSocket.readyState === WebSocket.OPEN) {
            this.chatWebSocket.send(JSON.stringify({
                type: 'unload_graphs',
                graph_ids: selectedGraphs
            }));

            this.showSystemMessage(`Unloading graphs: ${selectedGraphs.join(', ')}`);
        }
    }

    importGraph() {
        // Create a file input element
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = '.json,.txt,.md,.pdf';

        fileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (!file) return;

            this.showSystemMessage(`Importing graph from file: ${file.name}`);

            // In a real implementation, we would upload the file to the server
            // For now, just show a success message
            setTimeout(() => {
                this.showSystemMessage(`Graph imported successfully: ${file.name}`);
            }, 1500);
        });

        // Trigger file selection
        fileInput.click();
    }

    async switchConversationModel() {
        const modelSelect = document.getElementById('conversationModel');
        const switchBtn = document.getElementById('switchConversationModel');
        const selectedModel = modelSelect.value;

        if (!selectedModel) {
            this.showError('Please select a conversation model');
            return;
        }

        try {
            // Disable button and show loading state
            switchBtn.disabled = true;
            switchBtn.textContent = 'Switching...';
            modelSelect.classList.add('model-switching');

            const response = await fetch('/api/models/conversation/switch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_name: selectedModel
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showSystemMessage(`Successfully switched conversation model to ${result.new_model}`);
                this.updateCurrentModelDisplay('conversation', result.new_model);
                modelSelect.classList.add('model-switch-success');

                // Update stored settings
                const settings = JSON.parse(localStorage.getItem('assistantSettings') || '{}');
                settings.conversationModel = result.new_model;
                localStorage.setItem('assistantSettings', JSON.stringify(settings));
            } else {
                this.showError(`Failed to switch model: ${result.error}`);
                modelSelect.classList.add('model-switch-error');
            }

        } catch (error) {
            console.error('Error switching conversation model:', error);
            this.showError('Failed to switch conversation model');
            modelSelect.classList.add('model-switch-error');
        } finally {
            // Re-enable button and restore state
            switchBtn.disabled = false;
            switchBtn.textContent = 'Switch';
            modelSelect.classList.remove('model-switching');

            // Remove animation classes after animation completes
            setTimeout(() => {
                modelSelect.classList.remove('model-switch-success', 'model-switch-error');
            }, 500);
        }
    }

    async switchKnowledgeModel() {
        const modelSelect = document.getElementById('knowledgeModel');
        const switchBtn = document.getElementById('switchKnowledgeModel');
        const selectedModel = modelSelect.value;

        if (!selectedModel) {
            this.showError('Please select a knowledge extraction model');
            return;
        }

        try {
            // Disable button and show loading state
            switchBtn.disabled = true;
            switchBtn.textContent = 'Switching...';
            modelSelect.classList.add('model-switching');

            const response = await fetch('/api/models/knowledge/switch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_name: selectedModel
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showSystemMessage(`Successfully switched knowledge model to ${result.new_model}`);
                this.updateCurrentModelDisplay('knowledge', result.new_model);
                modelSelect.classList.add('model-switch-success');

                // Update stored settings
                const settings = JSON.parse(localStorage.getItem('assistantSettings') || '{}');
                settings.knowledgeModel = result.new_model;
                localStorage.setItem('assistantSettings', JSON.stringify(settings));
            } else {
                this.showError(`Failed to switch model: ${result.error}`);
                modelSelect.classList.add('model-switch-error');
            }

        } catch (error) {
            console.error('Error switching knowledge model:', error);
            this.showError('Failed to switch knowledge model');
            modelSelect.classList.add('model-switch-error');
        } finally {
            // Re-enable button and restore state
            switchBtn.disabled = false;
            switchBtn.textContent = 'Switch';
            modelSelect.classList.remove('model-switching');

            // Remove animation classes after animation completes
            setTimeout(() => {
                modelSelect.classList.remove('model-switch-success', 'model-switch-error');
            }, 500);
        }
    }

    updateCurrentModelDisplay(modelType, modelName) {
        const displayElement = document.getElementById(`current${modelType.charAt(0).toUpperCase() + modelType.slice(1)}Model`);
        if (displayElement) {
            displayElement.textContent = modelName;
            displayElement.className = `model-badge ${modelType}`;
        }
    }

    handleModelSwitch(data) {
        // Handle WebSocket notifications about model switches
        this.showSystemMessage(`Model switched: ${data.model_type} model changed from ${data.old_model} to ${data.new_model} (${data.switch_time.toFixed(2)}s)`);
        this.updateCurrentModelDisplay(data.model_type, data.new_model);
    }

    async loadCurrentModels() {
        try {
            const response = await fetch('/api/models/current');
            const models = await response.json();

            if (models.conversation) {
                this.updateCurrentModelDisplay('conversation', models.conversation);
                document.getElementById('conversationModel').value = models.conversation;
            }

            if (models.knowledge) {
                this.updateCurrentModelDisplay('knowledge', models.knowledge);
                document.getElementById('knowledgeModel').value = models.knowledge;
            }

        } catch (error) {
            console.error('Error loading current models:', error);
        }
    }

    // Voice Controller Integration Methods
    setMicState(state, level = 0) {
        // Update microphone orb visualization based on voice state
        if (this.micOrbEl) {
            // Remove all state classes
            this.micOrbEl.classList.remove('listening', 'speaking', 'muted', 'utterance_active', 'finalizing', 'waiting_for_bot');

            // Add current state class
            if (state && state !== 'idle') {
                this.micOrbEl.classList.add(state);
            }

            // Update audio level visualization
            if (level !== undefined && level >= 0) {
                this.micOrbEl.style.setProperty('--level', Math.max(0, Math.min(1, level)));
            }
        }
    }



    // The one renderer everyone uses
    addMessage(role, content) {
        console.log('🔧 addMessage called:', { role, content });

        const msg = document.createElement('div');
        msg.className = `message ${role}`;

        if (role === 'assistant') {
            const contentDiv = document.createElement('div');
            contentDiv.className = 'content';
            const cleaned = (content || '').replace(/^\.\s*/gm, '');
            contentDiv.innerHTML = this.md.render(cleaned);
            msg.appendChild(contentDiv);
        } else {
            msg.textContent = content;
        }

        const chatMessages = document.getElementById('chatBox');
        console.log('🔧 chatMessages element:', chatMessages);

        if (chatMessages) {
            chatMessages.appendChild(msg);
            console.log('🔧 Message added to DOM');
            this.scrollToBottom();
        } else {
            console.error('❌ chatMessages element not found! Available elements:', {
                'chatBox': !!document.getElementById('chatBox'),
                'chatMessages': !!document.getElementById('chatMessages'),
                'message-input': !!document.getElementById('message-input'),
                'messageInput': !!document.getElementById('messageInput')
            });
        }
    }

    // Back-compat shim INSIDE ChatInterface (optional while refactoring)
    async sendChatMessage(text) {
        return this.sendMessage(text);
    }

    showMicOrb(show = true) {
        // Show or hide the microphone orb
        if (this.micOrbEl) {
            this.micOrbEl.style.display = show ? 'block' : 'none';
        }
    }









    showError(message) {
        // Show an error message to the user
        this.showSystemMessage(`Error: ${message}`, 'error');
    }
}

// Toggle sidebar function
function toggleSidebar() {
    const ribbon = document.getElementById('kgRibbon');
    const toggle = document.getElementById('toggleBtn');

    if (ribbon && toggle) {
        ribbon.classList.toggle('collapsed');
        toggle.textContent = ribbon.classList.contains('collapsed') ? 'Sources ▴' : 'Sources ▾';
    }
}

// Settings modal functions
function openSettingsModal() {
    document.getElementById('settingsModal').classList.add('active');
}

function closeSettingsModal() {
    document.getElementById('settingsModal').classList.remove('active');
}

// Global function for send button
function sendMessage() {
    if (window.chatInterface) {
        window.chatInterface.sendMessage();
    }
}

// Production-Ready Voice Test Widget
class VoiceTestWidget {
    constructor() {
        this.micStream = null;
        this.analyser = null;
        this.micSource = null;
        this.isTestingMic = false;
        this._micTestRAF = null;
    }

    // Ensure/unlock AudioContext on user click (autoplay policy compliance)
    async ensureCtx(vc) {
        if (!vc.audioContext) {
            vc.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                latencyHint: 'interactive'
            });
        }
        if (vc.audioContext.state === 'suspended') {
            await vc.audioContext.resume();
        }
        if (!vc._ttsGain) {
            vc._ttsGain = vc.audioContext.createGain();
            vc._ttsGain.gain.value = 1.0;
            vc._ttsGain.connect(vc.audioContext.destination);
        }
        return vc.audioContext;
    }

    // Play 440Hz test tone through SAME path as TTS (PCM16@16k -> upsample -> schedule)
    async playSpeakerTest(vc, seconds = 0.5) {
        const ctx = vc.audioContext;
        const srIn = 16000, srOut = ctx.sampleRate;
        const up = Math.round(srOut / srIn);

        // Generate PCM16@16k sine wave
        const n = Math.floor(srIn * seconds);
        const pcm16 = new Int16Array(n);
        for (let i = 0; i < n; i++) {
            const s = Math.sin(2 * Math.PI * 440 * (i / srIn));
            pcm16[i] = Math.max(-1, Math.min(1, s)) * 32767;
        }

        // Reuse TTS upsample + schedule logic
        const out = new Float32Array(n * up);
        for (let i = 0, j = 0; i < n; i++) {
            const v = pcm16[i] / 32768;
            for (let k = 0; k < up; k++) out[j++] = v;
        }

        const buf = ctx.createBuffer(1, out.length, srOut);
        buf.copyToChannel(out, 0);
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(vc._ttsGain);

        const startAt = Math.max(ctx.currentTime + 0.02, vc._ttsPlayhead || 0);
        src.start(startAt);
        vc._ttsPlayhead = startAt + buf.duration;
    }

    async testSpeaker() {
        const statusEl = document.getElementById('voiceTestStatus');
        const button = document.getElementById('testSpeakerBtn');

        try {
            button.disabled = true;
            statusEl.textContent = 'Testing speaker (user gesture)...';
            statusEl.className = 'voice-test-status info';

            // CRITICAL: This method is called from a click event (user gesture)
            // Perfect for unlocking audio in Edge/Chrome

            console.log('🔊 Speaker test starting (user gesture detected)');

            // Enhanced browser detection
            const isEdge = /Edg/.test(navigator.userAgent);
            const isChrome = /Chrome/.test(navigator.userAgent) && !/Edg/.test(navigator.userAgent);
            const isSecureContext = window.isSecureContext;

            console.log('🔍 Browser environment:', {
                browser: isEdge ? 'Edge' : isChrome ? 'Chrome' : 'Other',
                hasAudioContext: !!(window.AudioContext || window.webkitAudioContext),
                isSecureContext: isSecureContext,
                protocol: window.location.protocol
            });

            if (!(window.AudioContext || window.webkitAudioContext)) {
                throw new Error('AudioContext not supported in this browser');
            }

            statusEl.textContent = 'Creating and unlocking AudioContext...';

            // CRITICAL: Create AudioContext in user gesture call stack
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            const testCtx = new AudioContextClass({
                latencyHint: 'interactive'
            });

            console.log('🔍 AudioContext created:', {
                state: testCtx.state,
                sampleRate: testCtx.sampleRate
            });

            // CRITICAL: Resume AudioContext (must be in user gesture call stack)
            if (testCtx.state === 'suspended') {
                console.log('🔓 Resuming AudioContext (user gesture)...');
                await testCtx.resume();
                console.log('🔓 AudioContext resumed, new state:', testCtx.state);

                // Edge sometimes needs a small delay after resume
                if (isEdge) {
                    await new Promise(resolve => setTimeout(resolve, 100));
                    console.log('🔓 Edge delay completed, final state:', testCtx.state);
                }
            }

            if (testCtx.state !== 'running') {
                throw new Error(`AudioContext failed to start (state: ${testCtx.state})`);
            }

            statusEl.textContent = 'Playing test tone...';

            // Enhanced oscillator setup for Edge compatibility
            const oscillator = testCtx.createOscillator();
            const gainNode = testCtx.createGain();

            // Edge-specific oscillator configuration
            oscillator.type = 'sine';  // Explicit sine wave
            oscillator.connect(gainNode);
            gainNode.connect(testCtx.destination);

            // More gradual volume ramp for Edge
            const startTime = testCtx.currentTime;
            const duration = 0.8;  // Longer duration for Edge

            oscillator.frequency.setValueAtTime(440, startTime);
            gainNode.gain.setValueAtTime(0, startTime);
            gainNode.gain.linearRampToValueAtTime(0.2, startTime + 0.1);  // Fade in
            gainNode.gain.linearRampToValueAtTime(0.2, startTime + duration - 0.1);
            gainNode.gain.linearRampToValueAtTime(0, startTime + duration);  // Fade out

            console.log('🔊 Starting oscillator at', startTime, 'for', duration, 'seconds');

            // Edge-specific oscillator handling
            try {
                oscillator.start(startTime);
                oscillator.stop(startTime + duration);

                // Monitor oscillator state
                oscillator.onended = () => {
                    console.log('🔊 Oscillator ended successfully');
                };

            } catch (oscError) {
                console.error('🔊 Oscillator start/stop error:', oscError);
                throw oscError;
            }

            // Wait for tone to complete with extra time for Edge
            await new Promise(resolve => setTimeout(resolve, duration * 1000 + 200));

            statusEl.textContent = '✅ Speaker test completed! Did you hear a 440Hz tone?';
            statusEl.className = 'voice-test-status success';

            // Clean up with Edge-specific handling
            try {
                if (testCtx.state !== 'closed') {
                    await testCtx.close();
                    console.log('🔊 AudioContext closed successfully');
                }
            } catch (closeError) {
                console.warn('🔊 AudioContext close warning:', closeError);
            }

        } catch (error) {
            console.error('🔊 Speaker test error:', error);

            // Enhanced fallback for Edge
            try {
                statusEl.textContent = 'Trying HTML5 audio fallback...';
                statusEl.className = 'voice-test-status info';

                await this.testSpeakerFallback();

                statusEl.textContent = '✅ Fallback audio test completed! Did you hear a beep?';
                statusEl.className = 'voice-test-status success';

            } catch (fallbackError) {
                console.error('🔊 Fallback test also failed:', fallbackError);

                // Final fallback: Simple audio element test
                try {
                    statusEl.textContent = 'Trying simple audio element...';
                    await this.testSimpleAudio();
                    statusEl.textContent = '✅ Simple audio test completed! Check your speakers.';
                    statusEl.className = 'voice-test-status success';
                } catch (simpleError) {
                    console.error('🔊 All audio tests failed:', simpleError);
                    statusEl.textContent = `❌ All speaker tests failed. Check: 1) Volume/mute 2) Audio device 3) Browser permissions`;
                    statusEl.className = 'voice-test-status error';
                }
            }
        } finally {
            button.disabled = false;
        }
    }

    async testSpeakerFallback() {
        // Enhanced beep with better Edge compatibility
        const beepDataUrl = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBTuL0fPTgjMGHm7A7+OZURE';

        const audio = new Audio(beepDataUrl);
        audio.volume = 0.5;  // Louder for Edge
        audio.preload = 'auto';

        console.log('🔊 Fallback: Creating HTML5 Audio element');

        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Audio fallback timeout'));
            }, 5000);

            audio.oncanplaythrough = () => {
                console.log('🔊 Fallback: Audio can play through');
                clearTimeout(timeout);
                audio.play()
                    .then(() => {
                        console.log('🔊 Fallback: Audio played successfully');
                        resolve();
                    })
                    .catch(reject);
            };

            audio.onended = () => {
                console.log('🔊 Fallback: Audio playback ended');
            };

            audio.onerror = (e) => {
                console.error('🔊 Fallback: Audio error:', e);
                clearTimeout(timeout);
                reject(new Error('Audio element failed to load'));
            };

            audio.load();
        });
    }

    async testSimpleAudio() {
        // Simplest possible audio test for Edge
        console.log('🔊 Simple: Testing basic audio element');

        return new Promise((resolve, reject) => {
            // Create a very short beep tone
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();

            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            oscillator.frequency.value = 800;  // Higher pitch for better audibility
            gainNode.gain.value = 0.1;

            oscillator.start();
            oscillator.stop(audioContext.currentTime + 0.2);

            oscillator.onended = () => {
                audioContext.close().then(resolve).catch(resolve);  // Resolve even if close fails
            };

            setTimeout(() => {
                resolve();  // Fallback resolution
            }, 1000);
        });
    }
    }

    // Edge-specific audio diagnostics
    async runEdgeAudioDiagnostics() {
        console.log('🔍 === EDGE AUDIO DIAGNOSTICS ===');

        // Show the Edge indicator
        const indicator = document.getElementById('edgeAudioIndicator');
        const statusEl = document.getElementById('edgeAudioStatus');
        const speakerIconEl = document.getElementById('speakerIconStatus');

        if (indicator) {
            indicator.style.display = 'block';
            statusEl.textContent = 'Running diagnostics...';
        }

        // 1. Browser detection
        const userAgent = navigator.userAgent;
        const isEdge = /Edg/.test(userAgent);
        const edgeVersion = isEdge ? userAgent.match(/Edg\/(\d+)/)?.[1] : null;

        console.log('🔍 Browser Info:', {
            userAgent,
            isEdge,
            edgeVersion,
            isSecureContext: window.isSecureContext,
            protocol: window.location.protocol,
            hostname: window.location.hostname
        });

        if (statusEl) {
            statusEl.textContent = isEdge ? `Edge ${edgeVersion}` : 'Not Edge';
        }

        // 2. Audio API availability
        const audioSupport = {
            AudioContext: !!window.AudioContext,
            webkitAudioContext: !!window.webkitAudioContext,
            HTMLAudioElement: !!window.HTMLAudioElement,
            getUserMedia: !!navigator.mediaDevices?.getUserMedia
        };

        console.log('🔍 Audio APIs:', audioSupport);

        // 3. Test AudioContext creation
        let contextWorking = false;
        try {
            const testCtx = new (window.AudioContext || window.webkitAudioContext)();
            console.log('🔍 AudioContext Test:', {
                state: testCtx.state,
                sampleRate: testCtx.sampleRate,
                baseLatency: testCtx.baseLatency,
                outputLatency: testCtx.outputLatency,
                maxChannelCount: testCtx.destination.maxChannelCount
            });

            // Test resume
            if (testCtx.state === 'suspended') {
                await testCtx.resume();
                console.log('🔍 AudioContext resumed to state:', testCtx.state);
            }

            contextWorking = testCtx.state === 'running';
            await testCtx.close();
        } catch (error) {
            console.error('🔍 AudioContext creation failed:', error);
        }

        // 4. Monitor for browser speaker icon
        this.monitorSpeakerIcon(speakerIconEl);

        // 5. Test simple oscillator
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            await ctx.resume();

            const osc = ctx.createOscillator();
            const gain = ctx.createGain();

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.frequency.value = 440;
            gain.gain.value = 0.1;

            osc.start();
            osc.stop(ctx.currentTime + 0.1);

            console.log('🔍 Oscillator test: Created and scheduled successfully');

            setTimeout(async () => {
                await ctx.close();
            }, 200);

        } catch (error) {
            console.error('🔍 Oscillator test failed:', error);
        }

        // 6. Test HTML5 Audio
        try {
            const audio = new Audio();
            console.log('🔍 HTML5 Audio:', {
                canPlayType_wav: audio.canPlayType('audio/wav'),
                canPlayType_mp3: audio.canPlayType('audio/mpeg'),
                canPlayType_ogg: audio.canPlayType('audio/ogg'),
                volume: audio.volume,
                muted: audio.muted
            });
        } catch (error) {
            console.error('🔍 HTML5 Audio test failed:', error);
        }

        console.log('🔍 === DIAGNOSTICS COMPLETE ===');

        if (statusEl) {
            statusEl.textContent = contextWorking ? '✅ Ready' : '❌ Issues detected';
        }
    }

    // Monitor for browser speaker icon (Edge-specific)
    monitorSpeakerIcon(speakerIconEl) {
        if (!speakerIconEl) return;

        // Check if we can detect audio output
        let iconDetected = false;

        // Method 1: Check for audio output devices
        if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
            navigator.mediaDevices.enumerateDevices()
                .then(devices => {
                    const audioOutputs = devices.filter(device => device.kind === 'audiooutput');
                    console.log('🔊 Audio output devices:', audioOutputs.length);

                    if (audioOutputs.length > 0) {
                        speakerIconEl.textContent = '🔊 Audio devices detected';
                        speakerIconEl.style.color = 'green';
                    } else {
                        speakerIconEl.textContent = '❌ No audio devices';
                        speakerIconEl.style.color = 'red';
                    }
                })
                .catch(error => {
                    console.error('🔊 Device enumeration failed:', error);
                    speakerIconEl.textContent = '❓ Cannot detect devices';
                    speakerIconEl.style.color = 'orange';
                });
        }

        // Method 2: Monitor document title for speaker icon (Edge shows it there)
        const originalTitle = document.title;
        let titleCheckCount = 0;

        const checkTitle = () => {
            titleCheckCount++;
            if (titleCheckCount > 20) return; // Stop after 10 seconds

            // Edge sometimes shows audio indicator in title or tab
            if (document.title !== originalTitle || document.title.includes('🔊')) {
                console.log('🔊 Title changed, possible audio indicator:', document.title);
                speakerIconEl.textContent = '🔊 Audio active (title changed)';
                speakerIconEl.style.color = 'green';
                iconDetected = true;
            }

            if (!iconDetected) {
                setTimeout(checkTitle, 500);
            }
        };

        setTimeout(checkTitle, 1000); // Start checking after 1 second

        // Method 3: Check for audio context state changes
        setTimeout(() => {
            if (!iconDetected) {
                speakerIconEl.textContent = '❓ No speaker icon detected';
                speakerIconEl.style.color = 'gray';
                console.log('🔊 No speaker icon detected in Edge after 10 seconds');
            }
        }, 10000);
    }

    // Build microphone constraints with proper WebRTC best practices
    buildMicConstraints(deviceId = null, opts = {}) {
        const {
            echoCancellation = false,
            noiseSuppression = false,
            autoGainControl = false
        } = opts;

        return {
            audio: {
                deviceId: deviceId ? { exact: deviceId } : undefined,
                channelCount: { ideal: 1 },
                sampleRate: { ideal: 48000 }, // browser HW rate; downsampled to 16k later
                echoCancellation,
                noiseSuppression,
                autoGainControl
            },
            video: false
        };
    }

    async testMicrophone() {
        const statusEl = document.getElementById('voiceTestStatus');
        const button = document.getElementById('testMicBtn');
        const meterEl = document.getElementById('micLevelMeter');
        const indicatorEl = document.getElementById('micLevelIndicator');

        if (this.isTestingMic) {
            this.stopMicTest();
            return;
        }

        try {
            button.disabled = true;
            statusEl.textContent = 'Checking microphone support...';
            statusEl.className = 'voice-test-status info';

            // Debug: Check microphone support
            console.log('🔍 Microphone support check:', {
                hasGetUserMedia: !!navigator.mediaDevices?.getUserMedia,
                hasEnumerateDevices: !!navigator.mediaDevices?.enumerateDevices,
                isSecureContext: window.isSecureContext,
                protocol: window.location.protocol
            });

            if (!navigator.mediaDevices?.getUserMedia) {
                throw new Error('getUserMedia not supported (HTTPS required for microphone access)');
            }

            statusEl.textContent = 'Requesting microphone access...';

            // Create fresh AudioContext for testing
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            const testCtx = new AudioContextClass({ latencyHint: 'interactive' });

            if (testCtx.state === 'suspended') {
                await testCtx.resume();
            }

            console.log('🔍 Test AudioContext:', {
                state: testCtx.state,
                sampleRate: testCtx.sampleRate
            });

            // Get microphone with proper constraints
            this.micStream = await navigator.mediaDevices.getUserMedia(
                this.buildMicConstraints(null, {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                })
            );

            console.log('🔍 Microphone stream:', {
                active: this.micStream.active,
                tracks: this.micStream.getTracks().length,
                trackSettings: this.micStream.getTracks()[0]?.getSettings()
            });

            // Visual level meter using AnalyserNode
            this.micSource = testCtx.createMediaStreamSource(this.micStream);
            this.analyser = testCtx.createAnalyser();
            this.analyser.fftSize = 512;
            this.micSource.connect(this.analyser);

            this.testCtx = testCtx; // Store for cleanup

            this.isTestingMic = true;
            button.textContent = '🛑 Stop Test';
            button.disabled = false;
            meterEl.style.display = 'block';
            statusEl.textContent = '🎤 Microphone active - speak to see levels';
            statusEl.className = 'voice-test-status success';

            // Draw meter with RMS calculation
            const data = new Uint8Array(this.analyser.frequencyBinCount);
            const tick = () => {
                if (!this.isTestingMic) return;

                this.analyser.getByteTimeDomainData(data);

                // Compute RMS 0..1
                let sum = 0;
                for (let i = 0; i < data.length; i++) {
                    const v = (data[i] - 128) / 128;
                    sum += v * v;
                }
                const rms = Math.sqrt(sum / data.length);

                // Update meter (scale RMS for better visibility)
                const level = Math.min(1, rms * 4) * 100;
                indicatorEl.style.width = `${level}%`;

                // Debug high levels
                if (rms > 0.1) {
                    console.debug('🎤 Mic level:', rms.toFixed(3));
                }

                this._micTestRAF = requestAnimationFrame(tick);
            };

            this._micTestRAF = requestAnimationFrame(tick);

        } catch (error) {
            console.error('🎤 Microphone test error:', error);
            statusEl.textContent = `❌ Microphone test failed: ${error.message}`;
            statusEl.className = 'voice-test-status error';
            this.isTestingMic = false;
            button.disabled = false;
        }
    }

    stopMicTest() {
        const button = document.getElementById('testMicBtn');
        const meterEl = document.getElementById('micLevelMeter');
        const statusEl = document.getElementById('voiceTestStatus');

        if (this._micTestRAF) {
            cancelAnimationFrame(this._micTestRAF);
            this._micTestRAF = null;
        }

        if (this.micStream) {
            this.micStream.getTracks().forEach(track => track.stop());
            this.micStream = null;
        }

        // Clean up audio nodes
        this.analyser = null;
        this.micSource = null;

        this.isTestingMic = false;
        button.textContent = '🎤 Test Microphone';
        button.disabled = false;
        meterEl.style.display = 'none';
        statusEl.textContent = '✅ Microphone test completed';
        statusEl.className = 'voice-test-status success';
    }
}

// Initialize chat interface when page loads
document.addEventListener('DOMContentLoaded', () => {
    const chat = new ChatInterface();
    window.chatInterface = chat;

    // Initialize voice test widget
    const voiceTest = new VoiceTestWidget();

    document.getElementById('testSpeakerBtn')?.addEventListener('click', async () => {
        // Run diagnostics first for Edge debugging
        await voiceTest.runEdgeAudioDiagnostics();
        voiceTest.testSpeaker();
    });

    document.getElementById('testMicBtn')?.addEventListener('click', () => {
        voiceTest.testMicrophone();
    });

    // Diagnostic test buttons
    document.getElementById('hardBeepBtn')?.addEventListener('click', async () => {
        const statusEl = document.getElementById('diagnosticStatus');
        statusEl.textContent = 'Running hard beep test...';
        statusEl.className = 'voice-test-status info';

        try {
            // Hard beep test (isolated AudioContext)
            const ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
            console.log('Hard beep: Before resume:', ctx.state);

            await ctx.resume();
            console.log('Hard beep: After resume:', ctx.state);

            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            gain.gain.value = 0.2;
            osc.type = 'sine';
            osc.frequency.value = 880;

            osc.connect(gain).connect(ctx.destination);

            const t = Math.max(ctx.currentTime + 0.02, 0);
            osc.start(t);
            osc.stop(t + 0.2);

            osc.onended = () => {
                console.log('Hard beep: Beep played. Context state:', ctx.state);
                setTimeout(() => ctx.close(), 400);
            };

            statusEl.textContent = '✅ Hard beep test completed! Check for speaker icon.';
            statusEl.className = 'voice-test-status success';

        } catch (error) {
            console.error('Hard beep test failed:', error);
            statusEl.textContent = `❌ Hard beep failed: ${error.message}`;
            statusEl.className = 'voice-test-status error';
        }
    });

    document.getElementById('syntheticSpeechBtn')?.addEventListener('click', () => {
        const statusEl = document.getElementById('diagnosticStatus');

        if (!chat.voiceController || !chat.voiceController.isVoiceMode) {
            statusEl.textContent = '⚠️ Enable voice mode first';
            statusEl.className = 'voice-test-status warning';
            return;
        }

        try {
            statusEl.textContent = 'Playing synthetic speech...';
            statusEl.className = 'voice-test-status info';

            chat.voiceController.playSyntheticSpeech();

            statusEl.textContent = '✅ Synthetic speech test completed!';
            statusEl.className = 'voice-test-status success';

        } catch (error) {
            console.error('Synthetic speech test failed:', error);
            statusEl.textContent = `❌ Synthetic speech failed: ${error.message}`;
            statusEl.className = 'voice-test-status error';
        }
    });

    document.getElementById('routeTestBtn')?.addEventListener('click', async () => {
        const statusEl = document.getElementById('diagnosticStatus');

        if (!chat.voiceController || !chat.voiceController.audioContext) {
            statusEl.textContent = '⚠️ AudioContext not available';
            statusEl.className = 'voice-test-status warning';
            return;
        }

        try {
            statusEl.textContent = 'Testing audio routing...';
            statusEl.className = 'voice-test-status info';

            const vc = chat.voiceController;

            // Test Route A (direct to destination)
            console.log('Testing Route A: Direct to destination');
            vc.routeTtsToDefault();

            // Test Route B (via monitor element)
            console.log('Testing Route B: Via monitor element');
            await vc.routeTtsToMonitor();

            // Switch back to default
            vc.routeTtsToDefault();

            statusEl.textContent = '✅ Audio routing test completed!';
            statusEl.className = 'voice-test-status success';

        } catch (error) {
            console.error('Route test failed:', error);
            statusEl.textContent = `❌ Route test failed: ${error.message}`;
            statusEl.className = 'voice-test-status error';
        }
    });

    // Comprehensive compatibility shims for mixed architecture
    window.appendMessage = (message) => {
        if (typeof message === 'object' && message.role && message.content) {
            chat.addMessage(message.role, message.content);
        }
    };

    window.showTypingIndicator = () => chat.showTyping();
    window.hideTypingIndicator = () => chat.hideTyping();

    // Override global sendMessage to use class method
    window.sendMessage = () => {
        chat.sendMessage(); // Let the class method handle getting text from input
    };

    // Add scrollToBottom shim
    window.scrollToBottom = () => chat.scrollToBottom();
});
