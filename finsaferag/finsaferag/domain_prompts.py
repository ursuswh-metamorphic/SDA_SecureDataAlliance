"""
Domain-aware prompts for multi-domain RAG support.
Supports: financial, medical, legal, technical, general
"""
from typing import Dict, Optional

# Domain configurations: system role, ensemble role, QA template hints
DOMAIN_CONFIG: Dict[str, Dict[str, str]] = {
    "financial": {
        "system_role": "You are a senior financial analyst specializing in synthesizing answers from multiple retrieval sources.",
        "ensemble_role": "You are a senior financial analyst. Below are answers from different financial data sources.",
        "qa_instruction": "Provide answers relating to financial performance, reports, earnings, and market data.",
        "tone": "Maintain a professional analytical tone consistent with financial reporting.",
        "examples": "Q: What was Apple's revenue growth in 2023? A: 2.8%.\nQ: When did the Federal Reserve raise interest rates last? A: 2023-07.",
        "welcome_title": "Financial Q&A Chatbot",
        "welcome_subtitle": "Ask me anything about financial reports, earnings, and market data",
    },
    "medical": {
        "system_role": "You are a medical information specialist synthesizing answers from multiple retrieval sources.",
        "ensemble_role": "You are a medical information specialist. Below are answers from different medical data sources.",
        "qa_instruction": "Provide answers relating to medical conditions, treatments, clinical data, and healthcare.",
        "tone": "Maintain an accurate, evidence-based tone. Do not provide medical advice; cite sources.",
        "examples": "Q: What are common symptoms of diabetes? A: Increased thirst, frequent urination, fatigue.\nQ: What is the typical dosage of aspirin? A: Consult a healthcare provider.",
        "welcome_title": "Medical Q&A Chatbot",
        "welcome_subtitle": "Ask me anything about medical information, treatments, and clinical data",
    },
    "legal": {
        "system_role": "You are a legal research specialist synthesizing answers from multiple retrieval sources.",
        "ensemble_role": "You are a legal research specialist. Below are answers from different legal data sources.",
        "qa_instruction": "Provide answers relating to contracts, regulations, compliance, and legal matters.",
        "tone": "Maintain a precise, citation-focused tone. Do not provide legal advice.",
        "examples": "Q: What are key terms in a standard NDA? A: Confidentiality, exclusions, term.\nQ: When does GDPR apply? A: When processing EU residents' data.",
        "welcome_title": "Legal Q&A Chatbot",
        "welcome_subtitle": "Ask me anything about legal documents, regulations, and compliance",
    },
    "technical": {
        "system_role": "You are a technical documentation specialist synthesizing answers from multiple retrieval sources.",
        "ensemble_role": "You are a technical documentation specialist. Below are answers from different technical data sources.",
        "qa_instruction": "Provide answers relating to APIs, code, architecture, and technical documentation.",
        "tone": "Maintain a clear, precise technical tone with code examples when relevant.",
        "examples": "Q: How do I authenticate with OAuth2? A: Use the authorization code flow.\nQ: What is the REST API endpoint for users? A: GET /api/v1/users.",
        "welcome_title": "Technical Q&A Chatbot",
        "welcome_subtitle": "Ask me anything about APIs, documentation, and technical topics",
    },
    "general": {
        "system_role": "You are a helpful assistant synthesizing answers from multiple retrieval sources.",
        "ensemble_role": "You are a helpful assistant. Below are answers from different data sources.",
        "qa_instruction": "Provide accurate, concise answers based on the retrieved context.",
        "tone": "Maintain a clear, factual tone.",
        "examples": "Q: What is the capital of France? A: Paris.\nQ: When was the UN founded? A: 1945.",
        "welcome_title": "RAG Q&A Chatbot",
        "welcome_subtitle": "Ask me anything based on the indexed documents",
    },
}


def get_domain(domain: Optional[str] = None) -> str:
    """Get validated domain string. Falls back to config or 'general'."""
    if domain and domain.lower() in DOMAIN_CONFIG:
        return domain.lower()
    try:
        from config import Config
        cfg = Config()
        d = getattr(cfg, "domain", "financial") or "financial"
        return d.lower() if d.lower() in DOMAIN_CONFIG else "general"
    except Exception:
        return "general"


def get_domain_config(domain: Optional[str] = None) -> Dict[str, str]:
    """Get full config for a domain."""
    d = get_domain(domain)
    return DOMAIN_CONFIG.get(d, DOMAIN_CONFIG["general"]).copy()


def get_ensemble_prompt_prefix(domain: Optional[str] = None) -> str:
    """Prefix for ensemble synthesis (server_app, llm_querier)."""
    cfg = get_domain_config(domain)
    return cfg["ensemble_role"] + " Synthesize them into ONE comprehensive, accurate final answer.\n\n"


def get_ensemble_system_instruction(domain: Optional[str] = None) -> str:
    """System instruction for ensemble synthesis (llm_querier)."""
    cfg = get_domain_config(domain)
    return (
        f"{cfg['system_role']} "
        "Provide a brief reasoning step (1–2 sentences) before the final answer. "
        "Focus on factual correctness, consistency across sources, and eliminate contradictions. "
        f"{cfg['tone']}"
    )


def get_text_qa_template(domain: Optional[str] = None) -> str:
    """Full TEXT_QA_TEMPLATE string for client_runner."""
    cfg = get_domain_config(domain)
    return (
        f"Below are some examples of concise question answering:\n{cfg['examples']}\n"
        "---------------------\n"
        "Below is the context information.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        f"Based solely on the above context, and without using ANY prior knowledge, "
        f"answer the following question as concisely as possible: {{query_str}}\n"
        "Instructions:\n"
        "*   Answer EXTREMELY briefly in a factual tone.\n"
        "*   Use ONLY information appearing in the context.\n"
        "*   If units are required (e.g., %, USD, dates), include them.\n"
        "*   Do NOT speculate or use outside knowledge — rely strictly on the provided context."
    )


def get_welcome_title(domain: Optional[str] = None) -> str:
    """Welcome title for UI."""
    return get_domain_config(domain)["welcome_title"]


def get_welcome_subtitle(domain: Optional[str] = None) -> str:
    """Welcome subtitle for UI."""
    return get_domain_config(domain)["welcome_subtitle"]


def get_example_questions(domain: Optional[str] = None) -> list:
    """Example questions for UI (list of dicts with icon, text)."""
    examples_by_domain = {
        "financial": [
            {"icon": "📊", "text": "What is 3M's revenue in 2019?"},
            {"icon": "👔", "text": "Who is the CEO of Apple?"},
            {"icon": "💹", "text": "Show me Tesla's profit margin"},
            {"icon": "📈", "text": "What are the main products of Microsoft?"},
        ],
        "medical": [
            {"icon": "🩺", "text": "What are symptoms of hypertension?"},
            {"icon": "💊", "text": "What is the mechanism of aspirin?"},
            {"icon": "🏥", "text": "Describe common diabetes treatments"},
            {"icon": "📋", "text": "What are contraindications for ibuprofen?"},
        ],
        "legal": [
            {"icon": "📜", "text": "What are key terms in an NDA?"},
            {"icon": "⚖️", "text": "When does GDPR apply?"},
            {"icon": "📑", "text": "What is required in a contract?"},
            {"icon": "🔒", "text": "Explain data retention requirements"},
        ],
        "technical": [
            {"icon": "🔧", "text": "How do I use the REST API?"},
            {"icon": "📦", "text": "What is the deployment process?"},
            {"icon": "🔐", "text": "How does OAuth2 authentication work?"},
            {"icon": "📚", "text": "Where is the API documentation?"},
        ],
        "general": [
            {"icon": "❓", "text": "What does this document say about X?"},
            {"icon": "📄", "text": "Summarize the key points"},
            {"icon": "🔍", "text": "Find information about Y"},
            {"icon": "📌", "text": "List the main topics"},
        ],
    }
    d = get_domain(domain)
    return examples_by_domain.get(d, examples_by_domain["general"])
