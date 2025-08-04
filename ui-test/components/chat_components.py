"""
Reusable Streamlit components for chat interface
"""

import streamlit as st
from datetime import datetime
import time

def render_message(role, content, timestamp=None, show_timestamp=True):
    """Render a single chat message"""
    with st.chat_message(role):
        st.markdown(content)
        if show_timestamp and timestamp:
            st.caption(f"*{timestamp}*")

def typing_effect(text, placeholder, delay=0.05):
    """Simulate typing effect for assistant responses"""
    full_response = ""
    for chunk in text.split():
        full_response += chunk + " "
        time.sleep(delay)
        placeholder.markdown(full_response + "▌")
    placeholder.markdown(full_response)
    return full_response

def model_selector(models, default_index=0):
    """Render model selection dropdown"""
    return st.selectbox(
        "Select Model",
        models,
        index=default_index,
        help="Choose the language model for responses"
    )

def graph_selector(graphs, default_index=0):
    """Render knowledge graph selection dropdown"""
    return st.selectbox(
        "Knowledge Graph", 
        graphs,
        index=default_index,
        help="Select the knowledge graph to query"
    )

def system_status_panel(model_status="Ready", graph_name="general", show_metrics=True):
    """Render system status panel"""
    st.subheader("System Status")
    
    # Model status
    if model_status == "Ready":
        st.success(f"Model: {model_status}")
    elif model_status == "Loading":
        st.warning(f"Model: {model_status}")
    else:
        st.error(f"Model: {model_status}")
    
    # Graph status
    st.info(f"Graph: {graph_name}")
    
    if show_metrics:
        st.metric("Response Time", "2.1s")
        st.metric("Memory Usage", "3.2GB")

def chat_settings_sidebar():
    """Render chat settings in sidebar"""
    with st.sidebar:
        st.header("Settings")
        
        # Model settings
        model = model_selector(["llama3.2:3b", "llama3.2:1b", "qwen2.5:3b"])
        
        # Graph settings
        graph = graph_selector(["survivalist", "general", "tech"])
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
            max_tokens = st.number_input("Max Tokens", 100, 4000, 2000)
            context_length = st.number_input("Context Length", 1, 10, 5)
        
        # System status
        system_status_panel()
        
        # Actions
        st.subheader("Actions")
        clear_chat = st.button("Clear Chat", type="secondary")
        export_chat = st.button("Export Chat", type="secondary")
        
        return {
            "model": model,
            "graph": graph, 
            "temperature": temperature,
            "max_tokens": max_tokens,
            "context_length": context_length,
            "clear_chat": clear_chat,
            "export_chat": export_chat
        }

def message_input_area(placeholder="Ask me anything..."):
    """Render message input area with send button"""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        message = st.text_input("Message", placeholder=placeholder, label_visibility="collapsed")
    
    with col2:
        send_button = st.button("Send", type="primary")
    
    return message, send_button

def conversation_history_sidebar(conversations):
    """Render conversation history in sidebar"""
    with st.sidebar:
        st.subheader("Recent Conversations")
        
        for conv in conversations[:5]:  # Show last 5 conversations
            if st.button(conv["title"], key=f"conv_{conv['id']}"):
                return conv["id"]
        
        if st.button("View All Conversations"):
            return "view_all"
    
    return None