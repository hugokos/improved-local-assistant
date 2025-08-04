import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mock_data.sample_conversations import sample_models

st.set_page_config(
    page_title="Model Manager",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Model Manager")

# Model status overview
st.subheader("Model Status")

for model in sample_models:
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    with col1:
        status_icon = "🟢" if model["status"] == "active" else "⚪"
        st.write(f"{status_icon} **{model['name']}**")
    
    with col2:
        st.write(f"{model['response_time']}s")
    
    with col3:
        st.write(model["memory_usage"])
    
    with col4:
        st.write(model["status"].title())
    
    with col5:
        if model["status"] == "active":
            if st.button("Stop", key=f"stop_{model['name']}"):
                st.success(f"Stopped {model['name']}")
        else:
            if st.button("Start", key=f"start_{model['name']}"):
                st.success(f"Started {model['name']}")

# Model configuration
st.subheader("Model Configuration")

selected_model = st.selectbox("Select Model to Configure", [m["name"] for m in sample_models])

col1, col2 = st.columns(2)

with col1:
    st.write("**Performance Settings**")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.number_input("Max Tokens", 100, 4000, 2000)
    context_window = st.number_input("Context Window", 1000, 8000, 4000)

with col2:
    st.write("**Resource Limits**")
    max_memory = st.slider("Max Memory (GB)", 1, 16, 8)
    cpu_threads = st.number_input("CPU Threads", 1, 16, 4)
    gpu_layers = st.number_input("GPU Layers", 0, 50, 20)

if st.button("Apply Configuration"):
    st.success(f"Configuration applied to {selected_model}")

# Model installation
st.subheader("Install New Models")

col1, col2 = st.columns([3, 1])

with col1:
    new_model = st.text_input("Model Name (e.g., llama3.2:7b)", placeholder="Enter model name...")

with col2:
    if st.button("Install", type="primary"):
        if new_model:
            with st.spinner(f"Installing {new_model}..."):
                st.success(f"Successfully installed {new_model}")
        else:
            st.error("Please enter a model name")

# Available models
st.subheader("Available Models")
available_models = [
    "llama3.2:1b", "llama3.2:3b", "llama3.2:7b",
    "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b",
    "mistral:7b", "codellama:7b", "phi3:3.8b"
]

cols = st.columns(3)
for i, model in enumerate(available_models):
    with cols[i % 3]:
        if st.button(f"Install {model}", key=f"install_{model}"):
            st.info(f"Installing {model}...")

st.markdown("---")
st.markdown("*Model Manager Prototype - Mock functionality for UI testing*")