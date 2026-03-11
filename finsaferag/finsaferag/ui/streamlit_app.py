"""
Streamlit Chatbot UI for RAG with Privacy Protection
Financial Q&A Chatbot Interface - ChatGPT-style Design
"""
import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any
import time
import html  # dùng để escape nội dung chat


# Configuration
API_BASE_URL = "http://localhost:8000"
DEFAULT_SESSION_ID = "streamlit_session"


# Page configuration
st.set_page_config(
    page_title="Financial Q&A Chatbot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>

    /* Nền chung */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
    }

    /* Container chính */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 8rem !important;
        max-width: 900px !important;
        margin: 0 auto !important;
        background: #ffffff !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb !important;
    }

    /* Sidebar title */
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }

    /* Normal text in sidebar */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #374151 !important;
        font-size: 0.95rem !important;
    }

    /* Status block (API Online, Federated...) */
    .sidebar-status-block {
        background: #e8fdf2 !important;
        color: #0f5132 !important;
        border: 1px solid #a7e3c3 !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        margin-bottom: 0.7rem !important;
        font-weight: 600;
    }

    /* Highlight block (mode info) */
    .sidebar-info-block {
        background: #e0f0ff !important;
        color: #1e3a8a !important;
        border: 1px solid #93c5fd !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        margin-top: 0.8rem !important;
        font-weight: 600;
    }

    /* Radio button container */
    .stRadio > div > label {
        background: #f9fafb !important;
        border: 2px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        color: #1f2937 !important;
        transition: all 0.25s ease;
    }

    /* Hover effect */
    .stRadio > div > label:hover {
        border-color: #3b82f6 !important;
        background: #eff6ff !important;
    }

    /* Selected radio */
    input:checked + div {
        border: 2px solid #3b82f6 !important;
        background: #dbeafe !important;
    }

    /* Checkbox text */
    .stCheckbox > label {
        color: #1f2937 !important;
    }

    /* Section divider spacing */
    [data-testid="stSidebar"] hr {
        margin-top: 1.2rem !important;
        margin-bottom: 1.2rem !important;
        border-color: #e5e7eb !important;
    }

    /* Hiển thị và style nút thu gọn sidebar */
    button[data-testid="collapseSidebarButton"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        position: fixed !important;
        top: 1rem !important;
        left: 1rem !important;
        z-index: 999 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    button[data-testid="collapseSidebarButton"]:hover {
        background: var(--primary-dark) !important;
        transform: scale(1.1) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }
    
    /* Đảm bảo sidebar luôn có thể hiển thị */
    [data-testid="stSidebar"] {
        visibility: visible !important;
    }
    
    /* Nút khôi phục sidebar (backup) */
    .sidebar-restore-btn {
        position: fixed !important;
        top: 1rem !important;
        left: 1rem !important;
        z-index: 1000 !important;
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.2rem !important;
        cursor: pointer !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .sidebar-restore-btn:hover {
        background: var(--primary-dark) !important;
        transform: scale(1.1) !important;
    }

    /* 🎨 BLUE THEME */
    :root {
        --primary: #0ea5e9;
        --primary-light: #e0f2fe;
        --primary-dark: #0284c7;
        --secondary: #06b6d4;
        --success: #3b82f6;
        --warning: #f59e0b;
        --danger: #ef4444;
        --neutral-50: #f9fafb;
        --neutral-100: #f3f4f6;
        --neutral-200: #e5e7eb;
        --neutral-300: #d1d5db;
        --neutral-600: #4b5563;
        --neutral-700: #374151;
        --neutral-800: #1f2937;
    }

    .main {
        background: #f8fafc !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 8rem !important;
        max-width: 900px !important;
        background: #ffffff !important;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Message bubbles */
    .message {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .message.user {
        flex-direction: row-reverse;
    }
    
    .message.user .message-content {
        align-items: flex-end !important;
    }
    
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .avatar.user {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        color: white;
    }
    
    .avatar.bot {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    .message-content {
        flex: 1;
        max-width: 85%;
        display: flex;
        flex-direction: column;
    }
    
    .message-bubble {
        padding: 1rem 1.25rem;
        border-radius: 1.25rem;
        line-height: 1.6;
        word-wrap: break-word;
        font-size: 0.95rem;
        color: #374151;
        background: #ffffff !important;
        border: 1px solid #e5e7eb;
    }
    
    .message.user .message-bubble {
        background: #ffffff !important;
        border: 1px solid #d1d5db;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        color: #374151;
    }
    
    .message.bot .message-bubble {
        background: #ffffff !important;
        border: 1px solid #d1d5db;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        color: #374151;
    }
    
    /* Privacy badge */
    .privacy-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.4rem 0.75rem;
        border-radius: 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.75rem;
        width: fit-content;
    }
    
    .privacy-badge.protected {
        background: #dbeafe;
        color: #075985;
        border: 1px solid #93c5fd;
    }
    
    .privacy-badge.warning {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde047;
    }
    
    /* Input container - fixed at bottom */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        border-top: 2px solid var(--neutral-200);
        padding: 1.5rem;
        z-index: 1000;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.08);
    }
    
    @media (min-width: 768px) {
        .input-container {
            left: 21rem;
            width: calc(100% - 21rem);
        }
    }
    
    .input-wrapper {
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 1.25rem !important;
        border: 2px solid var(--neutral-200) !important;
        padding: 1rem 1.5rem !important;
        font-size: 0.95rem !important;
        resize: none !important;
        transition: all 0.3s !important;
        background: white !important;
        color: #374151 !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.1) !important;
        outline: none !important;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 1.25rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: none !important;
        transition: all 0.3s !important;
        cursor: pointer !important;
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
    }
    
    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.3) !important;
    }
    
    .stButton button[kind="primary"]:active {
        transform: translateY(0) !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f0f8ff;
        border-right: 2px solid #0ea5e9;
    }
    
    .sidebar-header {
        background: #0ea5e9;
        color: white;
        padding: 1rem;
        border-radius: 0.75rem;
    }
    
    .sidebar .stMarkdown h2 {
        color: var(--neutral-800) !important;
        font-size: 1.25rem !important;
        margin-bottom: 1rem !important;
    }
    
    .sidebar .stMarkdown h3 {
        color: var(--neutral-700) !important;
        font-size: 0.95rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    .stRadio > label {
        font-size: 0.95rem !important;
        color: var(--neutral-700) !important;
    }
    
    .stRadio > div > label {
        background: var(--neutral-50) !important;
        border: 2px solid var(--neutral-200) !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 1rem !important;
        margin: 0.5rem 0 !important;
        transition: all 0.3s !important;
    }
    
    .stRadio > div > label:hover {
        border-color: var(--primary) !important;
        background: var(--primary-light) !important;
    }
    
    .stCheckbox > label {
        color: var(--neutral-700) !important;
        font-size: 0.95rem !important;
    }
    
    .stSelectbox > div > div {
        border: 2px solid var(--neutral-200) !important;
        border-radius: 0.75rem !important;
    }
    
    .stSuccess {
        background: #dbeafe !important;
        border: 1px solid #93c5fd !important;
        border-radius: 0.75rem !important;
    }
    
    .stError {
        background: #fee2e2 !important;
        border: 1px solid #fca5a5 !important;
        border-radius: 0.75rem !important;
    }
    
    .stWarning {
        background: #fef3c7 !important;
        border: 1px solid #fde047 !important;
        border-radius: 0.75rem !important;
    }
    
    .stInfo {
        background: #dbeafe !important;
        border: 1px solid #93c5fd !important;
        border-radius: 0.75rem !important;
    }
    
    .stMetric {
        background: white !important;
        padding: 1rem !important;
        border-radius: 0.75rem !important;
        border: 1px solid var(--neutral-200) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    .stMetric > div > div {
        color: var(--neutral-700) !important;
    }
    
    .stMetric > div > div > div > div {
        color: var(--primary) !important;
        font-weight: 700 !important;
    }
    
    .streamlit-expanderHeader {
        background: var(--neutral-50) !important;
        border-radius: 0.75rem !important;
        border: 1px solid var(--neutral-200) !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: white !important;
        border-color: var(--primary) !important;
    }

    [data-testid="stExpander"] .stMarkdown,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span,
    [data-testid="stExpander"] li {
        color: #1f2937 !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stExpander"] .stText pre {
        color: #1f2937 !important;
        background: #f8fafc !important;
        border: 1px solid #e5e7eb !important;
        padding: 0.75rem 1rem !important;
        border-radius: 0.75rem !important;
        line-height: 1.5 !important;
    }

    [data-testid="stExpander"] strong {
        color: #111827 !important;
        font-weight: 600 !important;
    }

    [data-testid="stExpander"] hr {
        border-color: #e5e7eb !important;
        margin: 1rem 0 !important;
    }
    
    .welcome-screen {
        max-width: 700px;
        margin: 4rem auto;
        text-align: center;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .welcome-subtitle {
        font-size: 1.1rem;
        color: var(--neutral-600);
        margin-bottom: 2rem;
    }
    
    .example-card {
        background: white;
        border: 2px solid var(--neutral-200);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.3s;
        text-align: left;
    }
    
    .example-card:hover {
        border-color: var(--primary);
        background: var(--primary-light);
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.15);
        transform: translateY(-4px);
    }
    
    .example-icon {
        font-size: 1.75rem;
        margin-bottom: 0.75rem;
    }
    
    .example-text {
        color: var(--neutral-700);
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .typing-indicator {
        display: inline-flex;
        gap: 0.3rem;
        padding: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--primary);
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    
    .streamlit-hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, var(--neutral-200), transparent) !important;
        margin: 1.5rem 0 !important;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--neutral-100);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary) !important;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark) !important;
    }
    
    .stMarkdown {
        color: var(--neutral-800) !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--neutral-800) !important;
    }
    
    .stSpinner > div {
        border-color: var(--primary) !important;
    }
    
    [data-testid="stExpander"] * {
        color: #111827 !important;
        opacity: 1 !important;
    }

    [data-testid="stExpander"] pre,
    [data-testid="stExpander"] code {
        color: #111827 !important;
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 1rem !important;
        line-height: 1.5 !important;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Dict[str, Any]:
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "error": "Cannot connect to API"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def fetch_llm_settings() -> Optional[Dict[str, Any]]:
    """Fetch current LLM settings from API"""
    try:
        r = requests.get(f"{API_BASE_URL}/api/settings", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def update_llm_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update LLM settings via API"""
    try:
        r = requests.post(f"{API_BASE_URL}/api/settings", json=payload, timeout=10)
        return {"success": r.status_code == 200, "data": r.json() if r.status_code == 200 else None, "error": r.text if r.status_code != 200 else None}
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_federated_health() -> Dict[str, Any]:
    """Check if Federated Server is ready"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("federated_ready"):
                return {
                    "status": "ready",
                    "num_clients": data.get("num_clients", 0),
                    "grid_status": data.get("flower_grid_status", "unknown")
                }
            else:
                return {
                    "status": "waiting_clients",
                    "num_clients": data.get("num_clients", 0),
                    "grid_status": data.get("flower_grid_status", "unknown")
                }
        return {"status": "unhealthy"}
    except Exception:
        return {"status": "offline"}


def send_query(
    question: str,
    apply_privacy: bool = True,
    retriever_type: Optional[str] = None,
    use_federated: bool = False,
) -> Dict[str, Any]:
    """Send query to API"""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
            "apply_privacy": apply_privacy,
            "retriever_type": retriever_type,
            "use_federated": use_federated
        }
        
        endpoint = "/api/query/federated" if use_federated else "/api/query"
        
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=None
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timeout (>60s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def display_message(role: str, content: str, metadata: Optional[Dict] = None, timestamp: str = None):
    """Display a chat message with ChatGPT-style design (❌ không hiển thị giờ nữa)"""
    
    # Avatar
    avatar_emoji = "👤" if role == "user" else "🤖"
    avatar_class = "user" if role == "user" else "bot"
    message_class = "user" if role == "user" else "bot"
    
    # Privacy badge
    privacy_html = ""
    if metadata and metadata.get("privacy_applied"):
        privacy_html = '<div class="privacy-badge protected">🔒 Privacy Protected</div>'
    elif metadata and metadata.get("privacy_applied") is False and role == "assistant":
        privacy_html = '<div class="privacy-badge warning">⚠️ No Privacy</div>'

    # 🔥 BRUTE-FORCE: xoá mọi pattern rác liên quan message-time / </div>
    raw = content or ""
    raw = raw.replace("</div>", "")
    raw = raw.replace("<div class=\"message-time\">", "")
    raw = raw.replace("<div class='message-time'>", "")

    # Escape nội dung → không render HTML/markdown
    safe_content = html.escape(raw)
    safe_content = safe_content.replace("`", "&#96;")
    safe_content = safe_content.replace("\n", "<br>")

    # KHÔNG còn .message-time trong HTML
    st.markdown(f"""
    <div class="message {message_class}">
        <div class="avatar {avatar_class}">{avatar_emoji}</div>
        <div class="message-content">
            <div class="message-bubble">{safe_content}</div>
            {privacy_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Privacy stats
    if metadata and metadata.get("privacy_stats") and role == "assistant":
        with st.expander("🔍 Privacy Details"):
            stats = metadata["privacy_stats"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("PII Detected", stats.get("pii_detected", 0))
            with col2:
                st.metric("Removed", stats.get("sentences_removed", 0))
            with col3:
                st.metric("PII Density", f"{stats.get('pii_density', 0):.1%}")
            with col4:
                st.metric("Risk", f"{stats.get('average_risk', 0):.2f}")
    
    # Sources
    if metadata and metadata.get("source_nodes") and role == "assistant":
        with st.expander("📄 Sources"):
            for i, node in enumerate(metadata["source_nodes"][:3], 1):
                st.markdown(f"**Source {i}** (Score: {node.get('score', 0):.3f})")
                src_text = html.escape(node.get("text", "")[:200]) + "..."
                st.markdown(f"<pre>{src_text}</pre>", unsafe_allow_html=True)
                if i < len(metadata["source_nodes"][:3]):
                    st.divider()


def sidebar_config():
    """Sidebar configuration"""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.subheader("API Status")
        health = check_api_health()
        
        if health["status"] == "healthy":
            st.success("✅ API Online")
            data = health.get("data", {})
            if data.get("privacy_enabled"):
                st.info("🔒 Privacy Enabled")
            
            if data.get("federated_ready"):
                st.success(f"🌐 Federated Server: {data.get('num_clients', 0)} clients")
            elif data.get("flower_grid_status") == "no_clients":
                st.warning("🌐 Federated Server: Waiting for clients")
            else:
                st.info("🌐 Federated Server: Not connected")
                
        elif health["status"] == "offline":
            st.error("❌ API Offline")
            st.warning("Start API:\n```bash\npython -m uvicorn api.main:app --port 8000\n```")
        else:
            st.error(f"❌ {health.get('error')}")
        
        st.divider()
        
        st.subheader("RAG Mode")
        rag_mode = st.radio(
            "Select RAG mode:",
            ["Single Machine", "Federated (Multi-Client)"],
            help="Single: Fast, local. Federated: Slower but ensemble answer from multiple clients",
            key="rag_mode_radio",   # 🔑 thêm key cho an toàn
        )
        use_federated = rag_mode == "Federated (Multi-Client)"

        if use_federated:
            fed_health = check_federated_health()
            if fed_health["status"] == "ready":
                st.success(f"🌐 Federated Ready: {fed_health['num_clients']} clients connected ✓")
            elif fed_health["status"] == "waiting_clients":
                st.warning(f"🌐 Server waiting for clients... ({fed_health['num_clients']} connected)")
                st.info("Start clients with: `python -m flwr run . --node-config '{\"node-id\": 1}'`")
            else:
                st.error("🌐 Federated Server offline")
                st.error("Start server with: `python -m flwr run .`")
        else:
            st.info("🖥️ Single Mode: Queries local RAG only")
        
        st.divider()
        
        st.subheader("Privacy Settings")
        apply_privacy = st.checkbox(
            "Enable Privacy Protection",
            value=True,
            help="Protect PII in responses"
        )
        
        st.divider()
        
        st.subheader("Retriever")
        st.info("MOCK Mode: All queries to default")
        retriever_type = st.selectbox(
            "Type",
            ["default", "financial", "general", "technical", "legal"]
        )

        st.divider()

        st.subheader("LLM Settings")
        llm_settings = fetch_llm_settings()
        if llm_settings:
            provider = st.selectbox(
                "Provider",
                ["nvidia", "openai"],
                index=0 if llm_settings.get("llm_provider") == "nvidia" else 1,
                help="NVIDIA: open source models via API. OpenAI: GPT models",
                key="llm_provider_select"
            )
            if provider == "nvidia":
                nvidia_key = st.text_input(
                    "NVIDIA API Key",
                    value="",
                    type="password",
                    placeholder="nvapi-xxx" if not llm_settings.get("nvidia_api_key_set") else "****",
                    help="Get key at build.nvidia.com",
                    key="nvidia_api_key"
                )
                
                current_model = llm_settings.get("nvidia_model") or "meta/llama-3.1-8b-instruct"
                
                # Text input for model - user can type ANY model
                nvidia_model = st.text_input(
                    "Model ID",
                    value=current_model,
                    placeholder="e.g., meta/llama-3.1-8b-instruct",
                    help="Enter model ID from build.nvidia.com",
                    key="nvidia_model_input"
                )
                
                # Quick select buttons for common models
                st.caption("Quick select:")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("llama-3.1-8b", key="q1", use_container_width=True):
                        st.session_state["nvidia_model_input"] = "meta/llama-3.1-8b-instruct"
                        st.rerun()
                    if st.button("mistral-7b", key="q3", use_container_width=True):
                        st.session_state["nvidia_model_input"] = "mistralai/mistral-7b-instruct-v0.3"
                        st.rerun()
                with col2:
                    if st.button("gemma-2-9b", key="q2", use_container_width=True):
                        st.session_state["nvidia_model_input"] = "google/gemma-2-9b-it"
                        st.rerun()
                    if st.button("deepseek-r1", key="q4", use_container_width=True):
                        st.session_state["nvidia_model_input"] = "deepseek-ai/deepseek-r1"
                        st.rerun()
                
                if st.button("Save NVIDIA Settings", type="primary", use_container_width=True, key="save_nvidia"):
                    if not nvidia_model or not nvidia_model.strip():
                        st.error("Please enter a model ID")
                    else:
                        payload = {"llm_provider": "nvidia", "llm": "nvidia", "nvidia_model": nvidia_model.strip()}
                        if nvidia_key:
                            payload["nvidia_api_key"] = nvidia_key
                        result = update_llm_settings(payload)
                        if result.get("success"):
                            st.success(f"Saved: {nvidia_model}")
                        else:
                            st.error(result.get("error", "Failed"))
            else:
                openai_key = st.text_input(
                    "OpenAI API Key",
                    value="",
                    type="password",
                    placeholder="sk-xxx" if not llm_settings.get("api_key_set") else "****",
                    key="openai_api_key"
                )
                api_name = st.text_input(
                    "Model",
                    value=llm_settings.get("api_name") or "gpt-4o-mini",
                    placeholder="gpt-4o-mini, gpt-4o, etc.",
                    key="openai_model"
                )
                if st.button("Save OpenAI Settings", type="primary", use_container_width=True, key="save_openai"):
                    payload = {"llm_provider": "openai", "llm": "openai", "api_name": api_name}
                    if openai_key:
                        payload["api_key"] = openai_key
                    result = update_llm_settings(payload)
                    if result.get("success"):
                        st.success("Saved. LLM reloaded.")
                    else:
                        st.error(result.get("error", "Failed"))
            st.caption("Changes apply to single-machine mode. Flower mode may need server restart.")
        else:
            st.warning("API offline - cannot load settings")
        
        st.divider()
        
        st.subheader("Chat History")
        message_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.write(f"📝 {message_count} messages")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    return {
        "apply_privacy": apply_privacy,
        "retriever_type": retriever_type if retriever_type != "default" else None,
        "api_healthy": health["status"] == "healthy",
        "use_federated": use_federated
    }


def show_welcome_screen():
    """Show welcome screen with example questions"""
    st.markdown("""
    <div class="welcome-screen">
        <div class="welcome-title">💰 Financial Q&A Chatbot</div>
        <div class="welcome-subtitle">
            Ask me anything about financial reports, earnings, and market data
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Try asking:")
    
    examples = [
        {"icon": "📊", "text": "What is 3M's revenue in 2019?"},
        {"icon": "👔", "text": "Who is the CEO of Apple?"},
        {"icon": "💹", "text": "Show me Tesla's profit margin"},
        {"icon": "📈", "text": "What are the main products of Microsoft?"},
    ]
    
    cols = st.columns(2)
    for idx, example in enumerate(examples):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="example-card">
                <div class="example-icon">{example['icon']}</div>
                <div class="example-text">{example['text']}</div>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Inject JavaScript mạnh hơn để restore sidebar
    st.components.v1.html("""
    <script>
        window.parent.document.addEventListener('DOMContentLoaded', function() {
            function forceSidebarVisible() {
                const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
                const collapseBtn = window.parent.document.querySelector('button[data-testid="collapseSidebarButton"]');
                
                if (sidebar) {
                    sidebar.style.display = '';
                    sidebar.style.visibility = 'visible';
                    sidebar.style.opacity = '1';
                    sidebar.setAttribute('aria-expanded', 'true');
                    sidebar.classList.remove('st-emotion-cache-hidden');
                }
                
                if (collapseBtn) {
                    collapseBtn.style.display = 'flex';
                    collapseBtn.style.visibility = 'visible';
                    collapseBtn.style.opacity = '1';
                }
            }
            
            // Chạy ngay và lặp lại
            forceSidebarVisible();
            setInterval(forceSidebarVisible, 500);
            
            // Tạo nút restore
            const btn = window.parent.document.createElement('button');
            btn.innerHTML = '☰';
            btn.style.cssText = 'position: fixed; top: 1rem; left: 1rem; z-index: 9999; background: #0ea5e9; color: white; border: none; border-radius: 50%; width: 45px; height: 45px; cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.3); font-size: 1.5rem; font-weight: bold; transition: all 0.3s;';
            btn.onmouseover = () => { btn.style.background = '#0284c7'; btn.style.transform = 'scale(1.1)'; };
            btn.onmouseout = () => { btn.style.background = '#0ea5e9'; btn.style.transform = 'scale(1)'; };
            btn.onclick = forceSidebarVisible;
            
            // Xóa nút cũ nếu có
            const oldBtn = window.parent.document.querySelector('.sidebar-restore-button');
            if (oldBtn) oldBtn.remove();
            
            btn.className = 'sidebar-restore-button';
            window.parent.document.body.appendChild(btn);
        });
    </script>
    """, height=0)
    
    # Init session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"{DEFAULT_SESSION_ID}_{int(time.time())}"

    # 🧹 DỌN RÁC: xoá mọi message có </div> hoặc message-time
    cleaned_messages = []
    for m in st.session_state.messages:
        content = m.get("content", "")
        if "</div>" in content or 'class="message-time"' in content:
            continue
        cleaned_messages.append(m)
    st.session_state.messages = cleaned_messages

    # ⭕️ GỌI SIDEBAR CHỈ 1 LẦN
    config = sidebar_config()
    
    # Welcome screen
    if len(st.session_state.messages) == 0:
        show_welcome_screen()
    
    # Hiển thị history
    for message in st.session_state.messages:
        display_message(
            role=message["role"],
            content=message["content"],
            metadata=message.get("metadata"),
            timestamp=message.get("timestamp")
        )
    
    # Form input
    input_container = st.container()
    with input_container:
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Message",
                placeholder="Type your question here... (Shift+Enter for new line)",
                label_visibility="collapsed",
                height=80,
                key="chat_input"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submit = st.form_submit_button("Send 🚀", type="primary", use_container_width=True)
    
    # Xử lý khi user gửi
    if submit and user_input:
        if not config["api_healthy"]:
            st.error("❌ API is not available. Please start the FastAPI server.")
            return
        
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        st.rerun()
    
    # Call backend nếu last message là user
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        with st.spinner(""):
            st.markdown("""
            <div class="message bot">
                <div class="avatar bot">🤖</div>
                <div class="message-content">
                    <div class="message-bubble">
                        <div class="typing-indicator">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            result = send_query(
                question=st.session_state.messages[-1]["content"],
                apply_privacy=config["apply_privacy"],
                retriever_type=config["retriever_type"],
                use_federated=config.get("use_federated", False)
            )
        
        if result["success"]:
            data = result["data"]
            answer = data.get("answer", "").strip()
            
            if answer == "</div>":
                answer = ""
            
            if not answer or answer == "...":
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ No answer generated. Mode: {data.get('routed_to', 'unknown')}",
                    "metadata": {
                        "privacy_applied": False,
                        "privacy_stats": None,
                        "source_nodes": [],
                        "response_time": data.get("response_time", 0)
                    },
                    "timestamp": datetime.now().isoformat()
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": {
                        "privacy_applied": data.get("privacy_applied", False),
                        "privacy_stats": data.get("privacy_stats"),
                        "source_nodes": data.get("source_nodes", []),
                        "response_time": data.get("response_time", 0)
                    },
                    "timestamp": datetime.now().isoformat()
                })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {result['error']}",
                "timestamp": datetime.now().isoformat()
            })
        
        st.rerun()



if __name__ == "__main__":
    main()
