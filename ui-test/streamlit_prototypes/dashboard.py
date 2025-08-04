import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Assistant Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Local Assistant Dashboard")

# Generate mock data
@st.cache_data
def generate_mock_data():
    # Chat activity data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    chat_data = pd.DataFrame({
        'date': dates,
        'messages': np.random.poisson(15, len(dates)),
        'queries': np.random.poisson(8, len(dates))
    })
    
    # Model performance data
    models = ['llama3.2:3b', 'llama3.2:1b', 'qwen2.5:3b']
    perf_data = pd.DataFrame({
        'model': models,
        'avg_response_time': [2.3, 1.8, 2.1],
        'success_rate': [0.95, 0.92, 0.94],
        'usage_count': [150, 89, 112]
    })
    
    return chat_data, perf_data

chat_data, perf_data = generate_mock_data()

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Conversations", "1,234", "+12%")

with col2:
    st.metric("Avg Response Time", "2.1s", "-0.3s")

with col3:
    st.metric("Knowledge Graphs", "3", "+1")

with col4:
    st.metric("System Uptime", "99.2%", "+0.5%")

# Charts row
col1, col2 = st.columns(2)

with col1:
    st.subheader("Chat Activity (Last 30 Days)")
    fig = px.line(chat_data, x='date', y=['messages', 'queries'], 
                  title="Daily Messages and Queries")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Model Performance")
    fig = px.bar(perf_data, x='model', y='avg_response_time',
                 title="Average Response Time by Model")
    st.plotly_chart(fig, use_container_width=True)

# System status
st.subheader("System Status")

col1, col2 = st.columns(2)

with col1:
    st.write("**Active Models:**")
    for _, row in perf_data.iterrows():
        status = "🟢" if row['success_rate'] > 0.9 else "🟡"
        st.write(f"{status} {row['model']} - {row['success_rate']:.1%} success rate")

with col2:
    st.write("**Knowledge Graphs:**")
    graphs = [
        {"name": "survivalist", "status": "🟢", "size": "2.3MB"},
        {"name": "general", "status": "🟢", "size": "1.8MB"},
        {"name": "tech", "status": "🟡", "size": "3.1MB"}
    ]
    
    for graph in graphs:
        st.write(f"{graph['status']} {graph['name']} - {graph['size']}")

# Recent activity
st.subheader("Recent Activity")
recent_activity = [
    {"time": "2 min ago", "event": "User query processed", "model": "llama3.2:3b"},
    {"time": "5 min ago", "event": "Knowledge graph updated", "model": "survivalist"},
    {"time": "12 min ago", "event": "Model switched", "model": "qwen2.5:3b"},
    {"time": "18 min ago", "event": "Chat session started", "model": "llama3.2:3b"},
]

for activity in recent_activity:
    st.write(f"**{activity['time']}** - {activity['event']} ({activity['model']})")

st.markdown("---")
st.markdown("*Dashboard Prototype - Mock data for UI testing*")