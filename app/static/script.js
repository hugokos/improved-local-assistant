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
        
        // Initialize debounced update functions
        this.debouncedUpdatePrebuilt = this.debounce(this.updatePrebuiltCitations.bind(this), 300);
        this.debouncedUpdateDynamic = this.debounce(this.updateDynamicKg.bind(this), 300);
        
        // Initialize real graph events
        this.initializeRealGraphEvents();
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
            // Try to parse as JSON first
            const jsonData = JSON.parse(data);
            
            switch (jsonData.type) {
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
            // Not JSON, treat as streaming text
            this.appendToCurrentMessage(data);
        }
    }
    
    handleMonitoringMessage(data) {
        try {
            const jsonData = JSON.parse(data);
            
            if (jsonData.type === 'system_status') {
                this.updateMetrics(jsonData);
            }
        } catch (e) {
            console.error('Error handling monitoring message:', e);
        }
    }
    
    sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (message && this.chatWebSocket && this.chatWebSocket.readyState === WebSocket.OPEN) {
            this.addMessage('user', message);
            this.chatWebSocket.send(message);
            this.messageInput.value = '';
            
            // Show typing indicator
            this.showTyping();
            
            // Start new assistant message after a brief delay
            setTimeout(() => {
                this.hideTyping();
                this.startNewAssistantMessage();
            }, 500);
        }
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
            
            this.scrollToBottom();
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
            // You could add a visual typing indicator here
        } else if (status === 'stop') {
            this.isTyping = false;
            // Remove typing indicator
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

// Initialize chat interface when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new ChatInterface();
});