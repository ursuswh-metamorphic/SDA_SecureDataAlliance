"""fedrag: A Flower Federated RAG Server (inference only)."""

import hashlib
import logging
import numpy as np
from collections import defaultdict
from time import sleep
import traceback

from flwr.app import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp
from llms.llm import get_llm

import threading
import uvicorn
from config import Config

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# -------------------------------------------------------
# Run FastAPI inside SAME Flower process
# -------------------------------------------------------
def start_fastapi():
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


# -------------------------------------------------------
# Helper funcs
# -------------------------------------------------------

def node_online_loop(grid: Grid):
    node_ids = []
    while not node_ids:
        node_ids = grid.get_node_ids()
        sleep(1)
    return node_ids


def get_hash(doc: str):
    return hashlib.sha256(doc.encode()).hexdigest()


def merge_documents(documents, scores, knn, k_rrf=0):
    sorted_idx = np.argsort(scores)[::-1]
    sorted_docs = [documents[i] for i in sorted_idx]

    if k_rrf == 0:
        return sorted_docs[:knn]

    rrf_store = defaultdict(dict)
    for rank, doc in enumerate(sorted_docs):
        doc_hash = get_hash(doc)
        rrf_store[doc_hash]["rank"] = 1 / (k_rrf + rank + 1)
        rrf_store[doc_hash]["doc"] = doc

    sorted_rrf = sorted(rrf_store.values(), key=lambda x: x["rank"], reverse=True)
    return [item["doc"] for item in sorted_rrf[:knn]]


def ensemble_answers(client_answers, llm_querier):
    """
    Ensemble multiple client answers into one final synthesized answer.
    
    Args:
        client_answers: List of answers from different clients
        llm_querier: LLM object with .complete() or .query() method
    
    Returns:
        str: Synthesized final answer
    """
    
    if not client_answers:
        log.warning("[ENSEMBLE] No client answers to ensemble")
        return ""
    
    if len(client_answers) == 1:
        log.info(f"[ENSEMBLE] Only 1 answer, returning as-is: {client_answers[0][:80]}...")
        return client_answers[0]

    # ⭐ Build ensemble prompt
    ensemble_prompt = (
        "You are a senior financial analyst. Below are answers from different financial data sources. "
        "Synthesize them into ONE comprehensive, accurate final answer.\n\n"
    )
    
    for i, ans in enumerate(client_answers, 1):
        ensemble_prompt += f"[Source {i}] {ans}\n\n"
    
    ensemble_prompt += (
        "Instructions:\n"
        "1. Combine key information from all sources\n"
        "2. Use specific numbers, dates, metrics from sources\n"
        "3. Resolve any conflicts by preferring most recent data\n"
        "4. Be concise but comprehensive\n\n"
        "FINAL SYNTHESIZED ANSWER:\n"
    )

    try:
        log.info(f"[ENSEMBLE] Starting ensemble with {len(client_answers)} answers...")
        log.info(f"[ENSEMBLE] Prompt:\n{ensemble_prompt[:200]}...")
        
        if llm_querier is None:
            log.error("[ENSEMBLE] LLM is None! Cannot synthesize")
            return " | ".join(client_answers[:2])
        
        # ⭐ Call LLM to synthesize
        log.info(f"[ENSEMBLE] Calling LLM synthesis...")
        result = llm_querier.complete(ensemble_prompt)
        
        # Extract result text
        if hasattr(result, "text"):
            result_str = result.text.strip()
        elif hasattr(result, "message"):
            result_str = result.message.strip()
        else:
            result_str = str(result).strip()
        
        log.info(f"[ENSEMBLE] LLM output: {result_str[:100]}...")
        
        # Validate result
        if result_str and len(result_str) > 30:
            log.info(f"[OK] [ENSEMBLE] Success: {result_str[:100]}...")
            return result_str
        else:
            log.warning(f"[ENSEMBLE] LLM returned too short result: '{result_str}'")
            # Fallback: combine top 2 answers
            fallback = f"{client_answers[0]} | {client_answers[1]}" if len(client_answers) > 1 else client_answers[0]
            log.info(f"[ENSEMBLE] Using fallback answer")
            return fallback
    
    except Exception as e:
        log.error(f"[ENSEMBLE] Error during synthesis: {e}")
        log.error(traceback.format_exc())
        # Fallback: return first answer
        return client_answers[0] if client_answers else ""


# -------------------------------------------------------
# ServerApp ENTRY (Flower calls this)
# -------------------------------------------------------

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    log.info("[START] Flower Federated Server starting...")
    """Flower Server Main Entry Point"""
    log.info("=" * 60)
    log.info("[START] Flower Federated RAG Server Starting...")
    log.info("=" * 60)

    global _llm_querier, _grid
    
    # ⭐ Step 1: Load config
    try:
        cfg = Config()
        log.info(f"[OK] Config loaded")
    except Exception as e:
        log.error(f"Failed to load config: {e}")
        cfg = None

    # ⭐ Step 2: Initialize LLM
    llm_querier = None
    try:
        if cfg and hasattr(cfg, 'llm_config'):
            llm_model_name = getattr(cfg.llm_config, 'model_name', getattr(cfg, 'llm', 'nvidia'))
        else:
            llm_model_name = getattr(cfg, 'llm', 'nvidia') if cfg else "nvidia"
        
        log.info(f"[LLM] Loading model: {llm_model_name}")
        llm_querier = get_llm(llm_model_name)  # ← ⭐ USE get_llm() FROM llm.py
        
        log.info(f"[OK] [LLM] Successfully initialized: {llm_model_name}")
        log.info(f"[OK] [LLM] Type: {type(llm_querier).__name__}")
    
    except Exception as e:
        log.error(f"[FAIL] [LLM] Failed to initialize: {e}")
        log.error(traceback.format_exc())
        llm_querier = None
    
    # ⭐ Step 3: Store global references
    _llm_querier = llm_querier
    _grid = grid
    
    # ⭐ FIX 2: Pass LLMQuerier đến API
    from api.main import set_flower_grid
    set_flower_grid(grid, llm_querier)  # ← THÊM llm_querier
    log.info("[OK] Flower Grid + LLMQuerier connected to FastAPI")

    # Start FastAPI
    fastapi_thread = threading.Thread(target=start_fastapi, daemon=True)
    fastapi_thread.start()
    log.info("[OK] FastAPI started inside Flower process")
    
    import time
    while True:
        time.sleep(1)

    return


# -------------------------------------------------------
# Exported for FastAPI federated endpoint
# -------------------------------------------------------

def submit_question(grid, question, qid, knn, node_ids, use_synthesis, qmode):
    messages = []
    for node_id in node_ids:
        cfg = ConfigRecord()
        cfg["question"] = question
        cfg["question_id"] = qid
        cfg["knn"] = knn
        cfg["use_synthesis"] = use_synthesis
        cfg["query_transform_mode"] = qmode

        rd = RecordDict({"config": cfg})
        msg = Message(
            content=rd,
            message_type=MessageType.QUERY,
            dst_node_id=node_id,
            group_id=str(qid),
        )
        messages.append(msg)

    replies = grid.send_and_receive(messages)
    log.info(f"Received {len(replies)}/{len(messages)} replies")

    documents, scores, answers = [], [], []

    for idx, reply in enumerate(replies):
        if not reply.has_content():
            log.warning(f"Client {idx} reply empty")
            continue

        ds = reply.content["docs_n_scores"]
        documents.extend(ds["documents"])
        scores.extend(ds["scores"])

        if "answer" in ds and ds["answer"]:
            answers.append(ds["answer"])

    return documents, scores, answers


__all__ = [
    "submit_question",
    "merge_documents",
    "ensemble_answers",
    "node_online_loop",
    "get_hash",
]
