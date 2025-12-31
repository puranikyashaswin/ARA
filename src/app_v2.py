"""
Streamlit frontend for ARA v2 Multi-Agent Orchestrator.
Enhanced UI with agent activity monitoring, side-by-side comparison, and execution flow visualization.
"""
import sys
import os
import time
import re
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

from src.orchestrator.graph import run_v2_agent, create_orchestrator_graph
from src.agent.graph import run_agent, get_final_answer
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

load_dotenv()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="ARA v2 • Multi-Agent Orchestrator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS (Premium Hybrid Aesthetic - Enhanced for v2)
# ============================================================================

st.markdown("""
<style>
    /* Premium Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
    }
    
    /* Agent Activity Cards */
    .agent-card {
        background: rgba(20, 20, 40, 0.7);
        border: 1px solid rgba(100, 100, 255, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
    .agent-card.active {
        border-color: #00ff96;
        box-shadow: 0 0 20px rgba(0, 255, 150, 0.2);
    }
    
    .agent-card.complete {
        border-color: rgba(100, 255, 150, 0.5);
        background: rgba(0, 255, 100, 0.05);
    }
    
    .agent-icon {
        font-size: 1.5em;
        margin-right: 8px;
    }
    
    .agent-name {
        font-weight: 700;
        color: #fff;
        font-size: 1.1em;
    }
    
    .agent-status {
        color: #888;
        font-size: 0.85em;
    }
    
    /* Execution Flow */
    .flow-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 20px 0;
        padding: 20px;
        background: rgba(0,0,0,0.2);
        border-radius: 12px;
    }
    
    .flow-step {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 8px;
        margin: 0 8px;
        font-weight: 600;
    }
    
    .flow-step.planning { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
    .flow-step.execution { background: linear-gradient(135deg, #0891b2, #0ea5e9); }
    .flow-step.verification { background: linear-gradient(135deg, #c026d3, #e879f9); }
    .flow-step.synthesis { background: linear-gradient(135deg, #059669, #10b981); }
    .flow-step.direct { background: linear-gradient(135deg, #f59e0b, #fbbf24); color: #000; }
    
    .flow-arrow {
        color: #555;
        font-size: 1.5em;
    }
    
    /* Confidence Meter */
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.1);
        margin-top: 8px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    .confidence-high { background: linear-gradient(90deg, #10b981, #34d399); }
    .confidence-medium { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .confidence-low { background: linear-gradient(90deg, #ef4444, #f87171); }
    
    /* Comparison Mode */
    .compare-header {
        text-align: center;
        font-size: 1.2em;
        font-weight: 700;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    
    .v1-header { background: linear-gradient(135deg, #2563eb, #3b82f6); }
    .v2-header { background: linear-gradient(135deg, #059669, #10b981); }
    
    /* Headers */
    .main-header {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(90deg, #00ff96, #6496ff, #ff00ff);
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
    
    .final-answer {
        background: linear-gradient(135deg, rgba(100, 255, 150, 0.15), rgba(100, 150, 255, 0.15));
        border: 2px solid rgba(100, 255, 200, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-top: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 15, 0.95);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Subtask list */
    .subtask-item {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 10px;
        margin: 6px 0;
        border-left: 3px solid #6366f1;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown('<p style="font-size:1.5em; font-weight:700; color:#00ff96;">⚙️ v2 Configuration</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mode Selection
    st.subheader("🎯 Mode")
    mode = st.radio(
        "Select Mode",
        ["v2 Orchestrator", "v1 vs v2 Comparison", "v1 Only"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # v2 Options
    if mode != "v1 Only":
        st.subheader("🔧 v2 Options")
        use_v1_execution = st.checkbox("Use v1 for sub-tasks", value=True, 
                                       help="Use v1 agent with tools for complex sub-tasks")
        enable_verification = st.checkbox("Always verify", value=False,
                                         help="Force verification even for simple problems")
    
    st.markdown("---")
    
    # Agent Overview
    st.subheader("🤖 v2 Agents")
    st.markdown("""
    - **📋 Planning** - Breaks down problems
    - **⚡ Execution** - Solves sub-tasks
    - **✅ Verification** - Cross-checks
    - **🔗 Synthesis** - Combines results
    """)
    
    st.markdown("---")
    
    # Session Stats
    st.subheader("📊 Session Stats")
    if "query_count" not in st.session_state: st.session_state.query_count = 0
    if "v2_calls" not in st.session_state: st.session_state.v2_calls = 0
    
    col1, col2 = st.columns(2)
    with col1: st.metric("Queries", st.session_state.query_count)
    with col2: st.metric("v2 Calls", st.session_state.v2_calls)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.v2_calls = 0
        st.rerun()

# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown('<p class="main-header">🧠 ARA v2</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Agent Orchestrator with Verification</p>', unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_answer(text: str) -> str:
    """Extract numeric answer from response."""
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', str(text))
    if match:
        return match.group(1)
    return str(text)

def render_execution_flow(path: list[str]):
    """Render the execution flow visualization."""
    flow_html = '<div class="flow-container">'
    
    step_icons = {
        "planning": "📋",
        "execution": "⚡",
        "verification": "✅",
        "synthesis": "🔗",
        "direct_v1": "🚀"
    }
    
    for i, step in enumerate(path):
        if i > 0:
            flow_html += '<span class="flow-arrow">→</span>'
        icon = step_icons.get(step, "•")
        css_class = step.replace("_v1", "").replace("direct", "direct")
        flow_html += f'<span class="flow-step {css_class}">{icon} {step.title()}</span>'
    
    flow_html += '</div>'
    st.markdown(flow_html, unsafe_allow_html=True)

def render_confidence(confidence: float):
    """Render confidence bar."""
    pct = int(confidence * 100)
    color_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"
    
    st.markdown(f"""
    <div>
        <span style="color:#888;">Confidence: <strong>{pct}%</strong></span>
        <div class="confidence-bar">
            <div class="confidence-fill {color_class}" style="width: {pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_subtasks(subtasks: list):
    """Render subtask list."""
    if not subtasks:
        return
    
    st.markdown("**📋 Sub-tasks:**")
    for st_item in subtasks:
        st.markdown(f"""
        <div class="subtask-item">
            <strong>#{st_item['id']}</strong> {st_item['description'][:60]}...
            <br><span style="color:#00ff96;">→ {st_item.get('result', 'Pending')}</span>
        </div>
        """, unsafe_allow_html=True)

def run_v1_query(query: str) -> dict:
    """Run v1 agent and return structured result."""
    start = time.time()
    result = run_agent(query)
    elapsed = time.time() - start
    
    answer = get_final_answer(result)
    numeric = extract_answer(answer)
    
    tool_calls = sum(
        1 for m in result.get("messages", [])
        if hasattr(m, "tool_calls") and m.tool_calls
    )
    
    return {
        "answer": numeric,
        "full_answer": answer,
        "time": elapsed,
        "tool_calls": tool_calls,
        "messages": result.get("messages", [])
    }

# ============================================================================
# CHAT INTERFACE
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Ask me anything... (math, code, research)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.query_count += 1
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        
        # COMPARISON MODE
        if mode == "v1 vs v2 Comparison":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="compare-header v1-header">🔵 v1 Agent</div>', unsafe_allow_html=True)
                with st.spinner("Running v1..."):
                    v1_result = run_v1_query(prompt)
                
                st.markdown(f"**Answer:** {v1_result['answer']}")
                st.markdown(f"⏱️ Time: {v1_result['time']:.2f}s | 🔧 Tools: {v1_result['tool_calls']}")
                
                with st.expander("Full Response"):
                    st.markdown(v1_result['full_answer'])
            
            with col2:
                st.markdown('<div class="compare-header v2-header">🟢 v2 Orchestrator</div>', unsafe_allow_html=True)
                with st.spinner("Running v2..."):
                    start = time.time()
                    v2_result = run_v2_agent(prompt)
                    v2_time = time.time() - start
                    st.session_state.v2_calls += 1
                
                st.markdown(f"**Answer:** {v2_result['answer']}")
                st.markdown(f"⏱️ Time: {v2_time:.2f}s | 📊 Complexity: {v2_result.get('complexity', 'N/A')}")
                render_confidence(v2_result.get('confidence', 0))
                
                render_execution_flow(v2_result.get('execution_path', []))
                
                if v2_result.get('verification'):
                    v = v2_result['verification']
                    status = "✅ Valid" if v['is_valid'] else "⚠️ Issues Found"
                    st.markdown(f"**Verification:** {status}")
            
            # Store combined result
            combined = f"""
            <div class="final-answer">
                <strong>v1:</strong> {v1_result['answer']} ({v1_result['time']:.2f}s)<br>
                <strong>v2:</strong> {v2_result['answer']} ({v2_time:.2f}s)
            </div>
            """
            st.session_state.messages.append({"role": "assistant", "content": combined})
        
        # V1 ONLY MODE
        elif mode == "v1 Only":
            with st.spinner("Running v1 agent..."):
                v1_result = run_v1_query(prompt)
            
            st.markdown(f'<div class="final-answer">{v1_result["full_answer"]}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": v1_result["full_answer"]})
        
        # V2 ORCHESTRATOR MODE (Default)
        else:
            with st.status("🧠 v2 Orchestrator Active...", expanded=True) as status:
                
                # Show agent activity placeholders
                agent_status = st.empty()
                
                status.update(label="📋 Planning Agent analyzing problem...")
                agent_status.markdown("""
                <div class="agent-card active">
                    <span class="agent-icon">📋</span>
                    <span class="agent-name">Planning Agent</span>
                    <div class="agent-status">Breaking down the problem...</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Run v2 orchestrator
                start = time.time()
                v2_result = run_v2_agent(prompt, use_v1_for_execution=use_v1_execution)
                elapsed = time.time() - start
                st.session_state.v2_calls += 1
                
                status.update(label="✅ Complete!", state="complete")
                agent_status.empty()
            
            # Show execution path
            st.markdown("### 🔄 Execution Flow")
            render_execution_flow(v2_result.get('execution_path', []))
            
            # Show details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Complexity", v2_result.get('complexity', 'unknown').title())
            with col2:
                st.metric("Time", f"{elapsed:.2f}s")
            with col3:
                render_confidence(v2_result.get('confidence', 0))
            
            # Show subtasks if any
            if v2_result.get('subtasks') and len(v2_result['subtasks']) > 1:
                with st.expander("📋 Sub-tasks", expanded=False):
                    render_subtasks(v2_result['subtasks'])
            
            # Show verification if performed
            if v2_result.get('verification'):
                v = v2_result['verification']
                status_text = "✅ Verified" if v['is_valid'] else "⚠️ Issues Found"
                st.info(f"**Verification:** {status_text} ({v['confidence']*100:.0f}% confident)")
            
            # Final answer
            answer_html = f'<div class="final-answer"><strong>Answer:</strong> {v2_result["answer"]}</div>'
            st.markdown(answer_html, unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "assistant", "content": answer_html})

# ============================================================================
# EXAMPLE QUERIES
# ============================================================================

st.markdown("---")
st.subheader("💡 Try These Examples")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🧮 Multi-Step Math**")
    st.code("A store has 120 apples. They sell 40% on Monday and 25% of the remainder on Tuesday. How many are left?", language=None)

with col2:
    st.markdown("**🔢 Simple Calculation**")
    st.code("What is 25% of 200?", language=None)

with col3:
    st.markdown("**📊 Complex Reasoning**")
    st.code("If John has 50 apples and gives 20% to Mary, then 30% of his remaining apples to Bob, how many does John have?", language=None)
