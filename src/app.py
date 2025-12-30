"""
Streamlit frontend for the Advanced Reasoning Agent.
Hybrid Design: Original aesthetic with real-time reasoning trace.
"""
import sys
import os
import time
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

from src.agent.graph import create_agent, get_final_answer
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

load_dotenv()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="ARA • Advanced Reasoning Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS (Premium Hybrid Aesthetic)
# ============================================================================

st.markdown("""
<style>
    /* Premium Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
    }
    
    /* Chat message styling */
    .reasoning-step {
        background: rgba(20, 20, 40, 0.6);
        border: 1px solid rgba(100, 100, 255, 0.2);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.85em;
    }
    
    .tool-call-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75em;
        font-weight: 700;
        margin-right: 8px;
        text-transform: uppercase;
    }
    
    .badge-calculator { background: #00d2ff; color: #000; }
    .badge-search { background: #00ff96; color: #000; }
    .badge-code { background: #ff9f00; color: #000; }
    .badge-reflect { background: #ff00ff; color: #fff; }
    
    .tool-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #444;
    }
    
    .final-answer {
        background: linear-gradient(135deg, rgba(100, 255, 150, 0.15), rgba(100, 150, 255, 0.15));
        border: 2px solid rgba(100, 255, 200, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-top: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(90deg, #00ff96, #6496ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2em;
    }
    
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2em;
        font-size: 0.9em;
        letter-spacing: 1px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 15, 0.95);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR (Restoring Detailed Sidebar)
# ============================================================================

with st.sidebar:
    st.markdown('<p style="font-size:1.5em; font-weight:700; color:#00ff96;">⚙️ Configuration</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🔧 Available Tools")
    st.markdown("""
    - **🐍 Python Executor** - E2B Sandbox
    - **🔍 Web Search** - Tavily API
    - **🧮 Calculator** - Math engine
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Session Stats")
    if "query_count" not in st.session_state: st.session_state.query_count = 0
    if "tool_calls" not in st.session_state: st.session_state.tool_calls = 0
    
    col1, col2 = st.columns(2)
    with col1: st.metric("Queries", st.session_state.query_count)
    with col2: st.metric("Tool Calls", st.session_state.tool_calls)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.tool_calls = 0
        st.rerun()

# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown('<p class="main-header">🧠 ARA</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Reasoning Agent with Self-Reflection</p>', unsafe_allow_html=True)

# ============================================================================
# CHAT INTERFACE
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "trace" in message:
            with st.expander("🔍 View Reasoning Steps", expanded=False):
                for step in message["trace"]:
                    st.markdown(step, unsafe_allow_html=True)
        st.markdown(message["content"], unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Ask me anything... (math, code, research)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.query_count += 1
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        trace_steps = []
        step_count = 0
        tool_call_count = 0
        
        with st.status("🧠 Engine Starting...", expanded=True) as status:
            agent = create_agent()
            
            # Start streaming
            for event in agent.stream(
                {"messages": [HumanMessage(content=prompt)], "reflection_count": 0},
                stream_mode="updates"
            ):
                step_count += 1
                node_name = list(event.keys())[0]
                data = event[node_name]
                
                if node_name == "reason":
                    status.update(label=f"🧠 Step {step_count}: Reasoning...")
                    msg = data["messages"][-1]
                    
                    if msg.content:
                        thought_html = f'<div class="reasoning-step"><strong>💭 Thought:</strong><br>{msg.content}</div>'
                        st.markdown(thought_html, unsafe_allow_html=True)
                        trace_steps.append(thought_html)
                    
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_call_count += 1
                            badge_class = "badge-calculator" if "calculator" in tc["name"] else "badge-search" if "search" in tc["name"] else "badge-code"
                            action_html = f"""
                            <div class="tool-container">
                                <span class="tool-call-badge {badge_class}">{tc["name"]}</span> <code>{str(tc["args"])}</code>
                            </div>
                            """
                            st.markdown(action_html, unsafe_allow_html=True)
                            trace_steps.append(action_html)

                elif node_name == "tools":
                    status.update(label=f"📊 Step {step_count}: Tool Result...")
                    for tool_msg in data["messages"]:
                        obs_html = f'<div class="tool-container" style="color: #888;"><strong>📤 Observation:</strong> {tool_msg.content[:500]}...</div>'
                        st.markdown(obs_html, unsafe_allow_html=True)
                        trace_steps.append(obs_html)

                elif node_name == "reflect":
                    status.update(label=f"🔄 Step {step_count}: Self-Reflection...")
                    msg = data["messages"][-1]
                    reflect_html = f"""
                    <div class="reasoning-step" style="border-left: 4px solid #ff00ff;">
                        <span class="tool-call-badge badge-reflect">Reflection</span><br>
                        {msg.content}
                    </div>
                    """
                    st.markdown(reflect_html, unsafe_allow_html=True)
                    trace_steps.append(reflect_html)

            status.update(label="✅ Finalizing Answer", state="complete", expanded=False)
            
            # Get final result for history
            # (Note: In streaming mode updates, the final result is in the last event)
            # But let's just re-run get_final_answer on the messages we have for consistency
            final_content = get_final_answer({"messages": data["messages"] if "messages" in data else []})
        
        st.session_state.tool_calls += tool_call_count
        
        # Display Final Answer in themed container
        final_answer_html = f'<div class="final-answer">{final_content}</div>'
        st.markdown(final_answer_html, unsafe_allow_html=True)
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer_html,
            "trace": trace_steps
        })

# ============================================================================
# EXAMPLE QUERIES (Restoring original style)
# ============================================================================

st.markdown("---")
st.subheader("💡 Example Queries")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🧮 Math Problem", use_container_width=True):
        st.info("Try: *If a train travels at 60 mph for 2.5 hours, then 45 mph for 1.5 hours, what is the total distance?*")
with col2:
    if st.button("🐍 Code Task", use_container_width=True):
        st.info("Try: *Write Python code to calculate the first 20 Fibonacci numbers.*")
with col3:
    if st.button("🔍 Research", use_container_width=True):
        st.info("Try: *What are the latest developments in quantum computing?*")
