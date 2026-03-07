"""fedrag: A Flower Federated RAG app.

Federated RAG with 2 modes + optional query transformation:
  - Mode 1 (Retrieval-only): Fast, lightweight, returns documents + scores
  - Mode 2 (Full RAG): Slow, heavyweight, returns documents + scores + LLM-generated answer
  - Query Transform: Optional HyDE, Stepback, SubQuestion to improve retrieval
"""

import logging
from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

from client_runner import query_faiss_index
from process.query_transform import choose_query_transform_mode

log = logging.getLogger(__name__)

app = ClientApp()

# ============= Configuration =============
USE_LLM_SYNTHESIS = True

# Query transformation mode:
#   "none" → no transformation (default)
#   "hyde_zeroshot", "hyde_fewshot" → HyDE transform
#   "stepback_zeroshot", "stepback_fewshot" → Stepback transform
#   "subquery_zeroshot", "subquery_fewshot" → SubQuestion transform
DEFAULT_QUERY_TRANSFORM_MODE = "none"
# =========================================


@app.query()
def query(msg: Message, context: Context):
    """
    Handle query from Flower server with 2 processing modes + query transformation.
    
    Expected msg.content["config"]:
    {
        "question": str,
        "question_id": str,
        "knn": int (default 5),
        "embedding_model_path": str (optional),
        "use_synthesis": bool (optional, overrides USE_LLM_SYNTHESIS),
        "query_transform_mode": str (optional, overrides DEFAULT_QUERY_TRANSFORM_MODE)
    }
    
    Response format:
    - Mode 1 (retrieval-only):
        {
            "documents": [...],
            "scores": [...],
            "titles": [...]
        }
    
    - Mode 2 (full RAG):
        {
            "answer": str,
            "documents": [...],
            "scores": [...],
            "titles": [...]
        }
    """
    
    # Extract parameters from request
    node_id = context.node_id
    question = str(msg.content["config"]["question"])
    question_id = str(msg.content["config"]["question_id"])
    knn = int(msg.content["config"].get("knn", 10))
    embedding_model_path = msg.content["config"].get("embedding_model_path", None)
    
    # Allow per-request override of synthesis mode
    use_synthesis = msg.content["config"].get("use_synthesis", USE_LLM_SYNTHESIS)
    
    # Allow per-request override of query transform mode
    query_transform_mode = msg.content["config"].get(
        "query_transform_mode",
        DEFAULT_QUERY_TRANSFORM_MODE,
    )

    if query_transform_mode == "auto":
        query_transform_mode = choose_query_transform_mode(question)

    # Query local partition (per-node) with chosen mode + transformation
    result = query_faiss_index(
        node_id=node_id,
        question=question,
        knn=knn,
        embedding_model_path=embedding_model_path,
        include_synthesis=use_synthesis,
        query_transform_mode=query_transform_mode,
    )

    # Extract components based on mode
    if use_synthesis:
        # Mode 2: Result has "answer" + "documents"
        answer_text = result.get("answer", "")
        documents_dict = result.get("documents", {})
    else:
        # Mode 1: Result is documents_dict directly
        answer_text = None
        documents_dict = result

    # Format output
    scores = [doc["score"] for _, doc in documents_dict.items()]
    documents = [doc["content"] for _, doc in documents_dict.items()]
    titles = [doc.get("title", "") for _, doc in documents_dict.items()]

    # node_id + số docs
    log.info(
        f"ClientApp [Node {node_id}]: QID={question_id}, "
        f"Retrieved={len(documents)} docs, "
        f"Mode={'Full RAG' if use_synthesis else 'Retrieval-only'}, "
        f"Transform={query_transform_mode}"
    )

    # Build reply record
    if use_synthesis:
        # Mode 2: Include answer
        docs_n_scores = ConfigRecord({
            "answer": answer_text,
            "documents": documents,
            "scores": scores,
            "titles": titles,
        })
    else:
        # Mode 1: Documents only
        docs_n_scores = ConfigRecord({
            "documents": documents,
            "scores": scores,
            "titles": titles,
        })
    
    reply_record = RecordDict({"docs_n_scores": docs_n_scores})

    # Return message
    return Message(reply_record, reply_to=msg)
