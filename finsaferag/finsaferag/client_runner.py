"""
Client runner for federated RAG with optional query transformation & synthesis.

Features:
  - Mode 1 (Retrieval-only): Fast, lightweight, returns documents + scores
  - Mode 2 (Full RAG): Slow, heavyweight, returns documents + scores + LLM-generated answer
  - Query Transform: Optional HyDE, Stepback, SubQuestion to improve retrieval quality
"""

import os
import logging
from typing import Dict, Any

from config import Config
from embs.embedding import get_embedding
from llms.llm import get_llm
from index import get_index
from retriever import get_retriver, response_synthesizer
from process.postprocess_rerank import get_postprocessor
from process.query_transform import transform_and_query, transform  
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings, PromptTemplate
from data.fedrag_data_loader import map_node_to_pid

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Cache per-corpus engines: { corpus_name: { "index": index, "query_engine": ..., ... } }
_engines: Dict[str, Dict[str, Any]] = {}

# Cache embedding models to avoid reloading from HuggingFace
_embedding_cache: Dict[str, Any] = {}

# Cache indexes to avoid reloading from disk
_index_cache: Dict[str, Any] = {}

# ============= Custom Prompt Template (từ main_response.py) =============
TEXT_QA_TEMPLATE_STR = (
    "Below are some examples of concise financial question answering:\n"
    "Q: What was Apple's revenue growth in 2023? A: 2.8%.\n"
    "Q: When did the Federal Reserve raise interest rates last? A: 2023-07.\n"
    "---------------------\n"
    "Below is the financial context information.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Based solely on the above context, and without using ANY prior knowledge, "
    "answer the following financial question as concisely as possible: {query_str}\n"
    "Instructions:\n"
    "*   Answer EXTREMELY briefly in a factual, finance-oriented tone.\n"
    "*   Use ONLY numbers, dates, metrics, and facts appearing in the context.\n"
    "*   If units are required (e.g., %, USD, billion, YoY/ QoQ), include them.\n"
    "*   Format your answer precisely as requested, e.g., \"X.X%\", \"USD X.XB\", \"YYYY-MM-DD\".\n"
    "*   If a calculation is needed and the units are clear, compute the result and round appropriately.\n"
    "*   Do NOT speculate, generalize, or use outside financial knowledge — rely strictly on the provided context."
)


TEXT_QA_TEMPLATE = PromptTemplate(TEXT_QA_TEMPLATE_STR)
# ========================================================================


def _ensure_engine(
    node_id: str,
    embedding_model_path: str = None,
    include_synthesis: bool = False,
    query_transform_mode: str = "none",
):
    """
    Khởi tạo hoặc lấy cached engine cho client (node_id).
    Mỗi client có index riêng, dữ liệu riêng (FDB partition).
    """
    
    pid = map_node_to_pid(node_id)

    # Cache key chỉ theo node + chế độ
    cache_key = f"client={pid}:synthesis={include_synthesis}:transform={query_transform_mode}"
    if cache_key in _engines:
        log.info(f"Using cached engine for '{cache_key}'")
        return _engines[cache_key]

    cfg = Config()

    # (A) Query transform
    if query_transform_mode != "none":
        cfg.query_transform = query_transform_mode

    # (B) Persist dir theo client
    persist_dir = os.path.join(cfg.persist_dir, f"client_{pid}")
    
    # (C) Load embedding model with caching
    embedding_key = embedding_model_path or getattr(cfg, "default_embedding_path", cfg.model_path)
    
    if embedding_key in _embedding_cache:
        log.info(f"Using cached embedding model: {embedding_key}")
        embeddings = _embedding_cache[embedding_key]
        last_dir = os.path.basename(embedding_model_path) if embedding_model_path else "default"
    else:
        if embedding_model_path:
            embeddings = get_embedding(embedding_model_path)
            last_dir = os.path.basename(embedding_model_path)
        else:
            embeddings = get_embedding(
                getattr(cfg, "default_embedding_path", cfg.model_path)
            )
            last_dir = "default"
            
        # Cache it for future use
        _embedding_cache[embedding_key] = embeddings
        log.info(f"Cached embedding model: {embedding_key}")

    # Apply global settings
    Settings.chunk_size = cfg.chunk_size
    Settings.embed_model = embeddings

    # (D) Only load LLM if synthesis is required
    if include_synthesis:
        llm = get_llm(cfg.llm)
        Settings.llm = llm
        log.info(f"Initialized LLM for node {node_id} (synthesis mode)")
    else:
        log.info(f"Skipping LLM for node {node_id} (retrieval-only)")

    # (E) Build/load index — now passed node_id as required
    index_cache_key = persist_dir
    
    if index_cache_key in _index_cache:
        log.info(f"Using cached index for: {index_cache_key}")
        index = _index_cache[index_cache_key]["index"]
        hierarchical_storage_context = _index_cache[index_cache_key]["hierarchical_storage_context"]
    else:
        log.info(f"Loading/building index from: {persist_dir}")
        index, hierarchical_storage_context = get_index(
            node_id=node_id,
            persist_dir=persist_dir,
            split_type="sentence",
            chunk_size=cfg.chunk_size,
        )
        
        # Cache it for future use
        _index_cache[index_cache_key] = {
            "index": index,
            "hierarchical_storage_context": hierarchical_storage_context
        }
        log.info(f"Cached index: {index_cache_key}")

    # (F) Build retriever
    retriever = get_retriver(
        cfg.retriever,
        index,
        hierarchical_storage_context=hierarchical_storage_context,
        cfg=cfg,
    )

    # (G) Build query engine
    if include_synthesis:
        response_s = response_synthesizer(
            getattr(cfg, "response_mode", 0)
        )
        postproc = [get_postprocessor(cfg)]

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_s,
            node_postprocessors=postproc,
        )

        # Template override
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": TEXT_QA_TEMPLATE}
        )

        log.info(f"Created QueryEngine with synthesis for node {node_id}")

    else:
        query_engine = retriever
        log.info(f"Created lightweight retriever for node {node_id}")

    # (H) Cache engine theo client
    _engines[cache_key] = {
        "cfg": cfg,
        "index": index,
        "hierarchical_storage_context": hierarchical_storage_context,
        "query_engine": query_engine,
        "retriever": retriever,
        "embeddings": embeddings,
        "last_dir": last_dir,
        "persist_dir": persist_dir,
        "include_synthesis": include_synthesis,
        "query_transform_mode": query_transform_mode,
        "node_id": node_id,
        "pid": pid, 
    }

    log.info(
        f"Initialized engine for node={node_id} "
        f"(synthesis={include_synthesis}, transform={query_transform_mode})"
    )

    return _engines[cache_key]



def query_faiss_index(
    node_id: str,
    question: str,
    knn: int = 10,
    embedding_model_path: str = None,
    include_synthesis: bool = False,
    query_transform_mode: str = "none",
):
    """
    Query local corpus (partition) của một client (node_id) và trả về
    các tài liệu được retrieve ± answer synthesis.

    Args:
        node_id: ID của client / Flower node (dùng để chọn partition + index)
        question: Query string
        knn: Top-k documents to retrieve
        embedding_model_path: Custom embedding path (optional)
        include_synthesis: 
            - False: Return only documents + scores (Mode 1 — Retrieval-only)
            - True: Return documents + scores + generated answer (Mode 2 — Full RAG)
        query_transform_mode: Query transformation mode
            - "none": không biến đổi (default)
            - "hyde_zeroshot", "hyde_fewshot": HyDE transform
            - "stepback_zeroshot", "stepback_fewshot": Stepback transform
            - "subquery_zeroshot", "subquery_fewshot": SubQuestion transform
    
    Returns:
        dict: Format depends on mode:
        - Mode 1 (include_synthesis=False):
            {doc_id: {"rank": int, "score": float, "title": str, "content": str}, ...}
        - Mode 2 (include_synthesis=True):
            {
                "answer": str,  # ← LLM-generated answer
                "documents": {doc_id: {"rank": int, "score": float, "title": str, "content": str}, ...}
            }
    """

    # CHANGED: gọi _ensure_engine với node_id, không còn corpus_name
    engine = _ensure_engine(
        node_id=node_id,
        embedding_model_path=embedding_model_path,
        include_synthesis=include_synthesis,
        query_transform_mode=query_transform_mode,
    )

    cfg = engine["cfg"]
    query_engine = engine["query_engine"]

    # Apply query transformation if needed
    transformed_question = question
    response_nodes = None
    answer_text = None

    if query_transform_mode != "none":
        try:
            # SubQuestion transform: trả về response trực tiếp
            if "subquery" in query_transform_mode:
                response = transform_and_query(transformed_question, cfg, query_engine)
                answer_text = getattr(response, "response", "") or ""
                response_nodes = getattr(response, "source_nodes", []) or []
            else:
                # HyDE / Stepback: chỉ transform sau đó query
                transformed_question = transform(transformed_question, cfg)
                log.info(f"[node {node_id}] Transformed question: {transformed_question}")
        except Exception as e:
            log.warning(f"[node {node_id}] Query transform failed: {e}. Using original question.")
            transformed_question = question

    # Execute query based on mode (nếu chưa có response từ transform)
    if response_nodes is None:
        if include_synthesis:
            # Mode 2: Full RAG — query() trả answer + source nodes
            try:
                response = query_engine.query(transformed_question)
                answer_text = getattr(response, "response", "") or ""
                response_nodes = getattr(response, "source_nodes", []) or []
                log.info(f"[node {node_id}] Generated answer (first 100 chars): {answer_text[:100]}")
            except Exception as e:
                log.error(f"[node {node_id}] Query execution failed: {e}")
                return {"answer": "", "documents": {}}
        else:
            # Mode 1: Retrieval-only — retrieve() trả nodes
            try:
                response_nodes = query_engine.retrieve(transformed_question)
            except Exception as e:
                log.error(f"[node {node_id}] Retrieval failed: {e}")
                return {}

    if not response_nodes:
        if include_synthesis:
            return {"answer": answer_text or "", "documents": {}}
        else:
            return {}

    # Sort nodes by score and pick top-knn
    sorted_nodes = sorted(
        response_nodes,
        key=lambda x: getattr(x, "score", 0.0),
        reverse=True,
    )[:knn]

    # Format documents output
    documents_dict = {}
    for rank, node in enumerate(sorted_nodes, 1):
        # doc_id
        doc_id = None
        try:
            if hasattr(node, "metadata") and isinstance(node.metadata, dict):
                doc_id = node.metadata.get("id", None)
            if doc_id is None and hasattr(node, "doc_id"):
                doc_id = node.doc_id
        except Exception:
            doc_id = None

        # content
        content = node.get_content() if hasattr(node, "get_content") else str(node)

        # score
        score = float(getattr(node, "score", 0.0))

        # title
        if hasattr(node, "metadata") and isinstance(node.metadata, dict):
            title = node.metadata.get("title", "")
        else:
            title = ""

        key = str(doc_id) if doc_id is not None else f"node_{len(documents_dict)}"
        documents_dict[key] = {
            "rank": rank,
            "score": score,
            "title": title,
            "content": content,
        }

    # Return based on mode
    if include_synthesis:
        return {
            "answer": answer_text or "",
            "documents": documents_dict,
        }
    else:
        return documents_dict

# Cache management functions
def clear_embedding_cache(model_name: str = None):
    """Clear cached embedding models to free memory."""
    global _embedding_cache
    if model_name:
        if model_name in _embedding_cache:
            del _embedding_cache[model_name]
            log.info(f"Cleared embedding cache: {model_name}")
    else:
        _embedding_cache.clear()
        log.info("Cleared all embedding caches")


def clear_index_cache(node_id: str = None):
    """Clear cached indexes to free memory."""
    global _index_cache
    if node_id:
        persist_dir_pattern = f"client_{node_id}"
        keys_to_remove = [k for k in _index_cache.keys() if persist_dir_pattern in k]
        for k in keys_to_remove:
            del _index_cache[k]
        log.info(f"Cleared index cache for node: {node_id}")
    else:
        _index_cache.clear()
        log.info("Cleared all index caches")


def clear_all_caches():
    """Clear all caches (engines + embeddings + indexes)."""
    global _engines, _embedding_cache, _index_cache
    _engines.clear()
    _embedding_cache.clear()
    _index_cache.clear()
    log.info("Cleared all caches (engines + embeddings + indexes)")
