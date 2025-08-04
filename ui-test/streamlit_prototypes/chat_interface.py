import streamlit as st
import time
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Chat Interface Prototype",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_status" not in st.session_state:
    st.session_state.model_status = "Ready"

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        ["llama3.2:3b", "llama3.2:1b", "qwen2.5:3b"],
        index=0
    )
    
    # Graph selection
    graph = st.selectbox(
        "Knowledge Graph",
        ["survivalist", "general", "tech"],
        index=0
    )
    
    # System status
    st.subheader("System Status")
    st.success(f"Model: {st.session_state.model_status}")
    st.info(f"Graph: {graph}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("💬 Local Assistant Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "timestamp" in message:
            st.caption(f"*{message['timestamp']}*")

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"*{timestamp}*")
    
    # Simulate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate typing effect
        assistant_response = f"This is a mock response to: '{prompt}'. In the real app, this would connect to your local LLM and knowledge graph."
        
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        response_timestamp = datetime.now().strftime("%H:%M:%S")
        st.caption(f"*{response_timestamp}*")
    
    # Add assistant message to session state
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "timestamp": response_timestamp
    })

# Footer
st.markdown("---")
st.markdown("*UI Prototype - Not connected to actual backend*")