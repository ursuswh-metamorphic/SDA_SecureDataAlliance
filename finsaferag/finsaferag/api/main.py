"""
FastAPI backend for RAG Chatbot with Privacy-Aware Summary
"""
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root (RAGTest) to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from api.models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    ErrorResponse,
    PrivacyStats,
    SourceNode,
)
from api.router import get_router
from config import Config

# Import RAG components
from llama_index.core import Settings
from index import get_index
from retriever import get_retriver, response_synthesizer
from process.postprocess_rerank import get_postprocessor
from llms.llm import get_llm
from embs.embedding import get_embedding
from llama_index.core.query_engine import RetrieverQueryEngine
from process.query_transform import transform_and_query
from flwr.app import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.serverapp import Grid
import time as time_module  # avoid shadowing time above

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Privacy module (optional)
# -------------------------------------------------------------------

PRIVACY_AVAILABLE = True
from privacy import apply_privacy_to_response




# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(
    title="RAG Chatbot with Privacy Protection",
    description="Financial Q&A chatbot with privacy-aware summary module",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG components
cfg = None
index = None
query_engine = None
router = None

# Global Flower Grid (sẽ được set bởi server)
_flower_grid: Optional[Grid] = None
_llm_querier: Optional[object] = None  # ← THÊM


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global cfg, index, query_engine, router, response_synthesizer

    try:
        logger.info("Starting RAG Chatbot API...")
        logger.info("Loading configuration...")
        cfg = Config()  # gán vào global

        logger.info("Initializing query router...")
        router = get_router()  # gán vào global

        logger.info("Setting up embeddings and LLM...")

        # Check embeddings path
        embeddings_path = getattr(cfg, "embeddings", None)
        if not embeddings_path or embeddings_path == "embedding path":
            logger.warning("No valid embeddings path in config, using default")
            embeddings_path = "BAAI/bge-base-en"

        embeddings = get_embedding(embeddings_path)
        llm = get_llm(cfg.llm)

        Settings.llm = llm
        Settings.embed_model = embeddings
        Settings.chunk_size = cfg.chunk_size

        logger.info("Building/loading index for API...")

        # dùng client_0 (hoặc client khác nếu bạn muốn)
        base_persist_root = getattr(cfg, "persist_dir", "storage")
        node_id = "0"
        persist_dir = os.path.join(base_persist_root, f"client_{node_id}")
        logger.info(f"Building/loading index for API from: {persist_dir}")

        index, hierarchical_storage_context = get_index(
            node_id=node_id,
            persist_dir=persist_dir,
            split_type=getattr(cfg, "split_type", "sentence"),
            chunk_size=getattr(cfg, "chunk_size", 512),
        )

        logger.info(f"✓ Index loaded successfully for node: {node_id}")

        # Create query engine
        logger.info("Creating query engine...")
        node_postprocessors = [get_postprocessor(cfg)]

        query_engine = RetrieverQueryEngine(
            retriever=get_retriver(
                cfg.retriever,
                index,
                hierarchical_storage_context=hierarchical_storage_context,
                cfg=cfg,
            ),
            response_synthesizer=response_synthesizer(0),
            node_postprocessors=node_postprocessors,
        )

        # --- an toàn khi cfg không có .privacy ---
        privacy_cfg = getattr(cfg, "privacy", {})  # ← Thay None thành {}
        # DEBUG CODE
        print(f"\n{'='*60}")
        print(f"DEBUG: Config Privacy Section")
        print(f"{'='*60}")
        print(f"privacy_cfg type: {type(privacy_cfg)}")
        print(f"privacy_cfg value: {privacy_cfg}")
        print(f"enable_privacy_summary: {privacy_cfg.get('enable_privacy_summary', 'NOT FOUND')}")
        print(f"presidio_entities: {privacy_cfg.get('presidio_entities', 'NOT FOUND')}")
        print(f"{'='*60}\n")

        if isinstance(privacy_cfg, dict):
            privacy_enabled = privacy_cfg.get("enable_privacy_summary", False)
        else:
            privacy_enabled = False

        logger.info("✓ RAG Chatbot API started successfully!")
        logger.info(
            f"  Privacy module: {'ENABLED' if (privacy_enabled and PRIVACY_AVAILABLE) else 'DISABLED'}"
        )
        logger.info(f"  Available retrievers: {router.get_available_retrievers()}")

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        logger.error(traceback.format_exc())
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat(),
        ).dict(),
    )


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs",
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    global cfg, query_engine, router, _flower_grid

    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    privacy_enabled = False
    if cfg:
        privacy_cfg = getattr(cfg, "privacy", {})
        if isinstance(privacy_cfg, dict):
            privacy_enabled = privacy_cfg.get("enable_privacy_summary", False)

    available_retrievers = []
    if router:
        available_retrievers = router.get_available_retrievers()

    # 🔹 THÊM: Check Flower grid status
    federated_ready = False
    num_clients = 0
    flower_grid_status = "not_connected"

    if _flower_grid is not None:
        try:
            num_clients = len(_flower_grid.get_node_ids())
            if num_clients > 0:
                federated_ready = True
                flower_grid_status = "connected"
            else:
                flower_grid_status = "no_clients"
        except Exception as e:
            logger.warning(f"Cannot check Flower grid: {e}")
            flower_grid_status = "error"

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        privacy_enabled=privacy_enabled and PRIVACY_AVAILABLE,
        available_retrievers=available_retrievers,
        federated_ready=federated_ready,
        num_clients=num_clients,
        flower_grid_status=flower_grid_status,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process user query through RAG pipeline with privacy protection
    """
    global cfg, query_engine, router

    if query_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    start_time = time.time()

    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        logger.info(f"Processing query: '{question[:100]}...'")

        # Route query
        routing_decision = router.route_query(
            query=question, user_preference=request.retriever_type
        )
        routed_to = routing_decision["retriever_type"]
        logger.info(
            f"Routed to: {routed_to} (reasoning: {routing_decision['reasoning']})"
        )

        # Execute query through RAG pipeline
        logger.info("Executing RAG query...")
        response = transform_and_query(question, cfg, query_engine)

        # Store original response
        original_answer = response.response

        # Apply privacy protection if requested
        privacy_stats_obj = None
        privacy_applied = False

        privacy_cfg = getattr(cfg, "privacy", {})
        if (
            request.apply_privacy
            and PRIVACY_AVAILABLE
            and isinstance(privacy_cfg, dict)
            and privacy_cfg.get("enable_privacy_summary", False)
        ):
            logger.info("Applying privacy protection...")
            response, privacy_metadata = apply_privacy_to_response(
                response, question, cfg
            )
            privacy_applied = True

            # Build privacy stats
            if privacy_metadata:
                privacy_stats_obj = PrivacyStats(
                    pii_detected=len(privacy_metadata.get("pii_entities", [])),
                    pii_density=privacy_metadata.get("pii_density", 0.0),
                    sentences_removed=privacy_metadata.get("eraser", {}).get(
                        "removed_count", 0
                    ),
                    average_risk=privacy_metadata.get("eraser", {}).get(
                        "average_risk", 0.0
                    ),
                    encrypted=privacy_metadata.get("encryption", {}).get(
                        "enabled", False
                    ),
                    entities=[
                        e.get("entity_type", "")
                        for e in privacy_metadata.get("pii_entities", [])
                    ],
                )

                logger.info(
                    f"Privacy stats: PII={privacy_stats_obj.pii_detected}, "
                    f"Removed={privacy_stats_obj.sentences_removed}"
                )
            else:
                privacy_applied = False
                privacy_stats_obj = None

        # Extract source nodes
        source_nodes = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for node in response.source_nodes[:5]:  # Top 5 sources
                source_nodes.append(
                    SourceNode(
                        text=node.node.get_content()[:500],  # Truncate long text
                        score=node.score if hasattr(node, "score") else 0.0,
                        metadata=node.node.metadata
                        if hasattr(node.node, "metadata")
                        else {},
                    )
                )

        # Calculate response time
        response_time = time.time() - start_time

        # Build response
        result = QueryResponse(
            question=question,
            answer=response.response,
            original_answer=original_answer if privacy_applied else None,
            privacy_applied=privacy_applied,
            privacy_stats=privacy_stats_obj,
            source_nodes=source_nodes,
            response_time=response_time,
            routed_to=routed_to,
        )

        logger.info(f"Query completed in {response_time:.2f}s")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}"
        )


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    global cfg

    stats = {
        "privacy_enabled": False,
        "total_queries": 0,  # Would track in production
        "avg_response_time": 0.0,  # Would track in production
    }

    privacy_cfg = getattr(cfg, "privacy", {})
    if cfg and isinstance(privacy_cfg, dict):
        stats["privacy_enabled"] = privacy_cfg.get("enable_privacy_summary", False)
        stats["privacy_config"] = {
            "presidio_entities": privacy_cfg.get("presidio_entities", []),
            "eraser_enabled": privacy_cfg.get("eraser_drop_high_risk", False),
            "tenseal_enabled": privacy_cfg.get("tenseal_enabled", False),
            "flower_enabled": privacy_cfg.get("flower_enabled", False),
        }

    return stats


# ========== FLOWER BRIDGE HELPERS ==========


def set_flower_grid(grid: Grid, llm_querier: object = None):
    """Set global Flower grid + LLMQuerier for federated queries"""
    global _flower_grid, _llm_querier
    _flower_grid = grid
    _llm_querier = llm_querier  # ← SAVE
    logger.info(
        f"Federated RAG bridge initialized. "
        f"LLMQuerier: {'Available' if llm_querier else 'NOT available'}"
    )


def submit_question_to_federated(
    question: str,
    knn: int = 3,
    use_synthesis: bool = True,
    query_transform_mode: str = "none",
    node_ids: list | None = None,
) -> Dict[str, Any]:
    """
    Submit question to Federated Server and get ensemble answer
    
    ✅ Collect answers từ tất cả clients
    ❌ ĐỢI: ensemble sẽ làm ở /api/query/federated
    """

    try:
        from server_app import submit_question, merge_documents
    except ImportError as e:
        logger.error(f"Cannot import from server_app: {e}")
        raise HTTPException(
            status_code=500,
            detail="Federated system not available",
        )

    global _flower_grid

    if _flower_grid is None:
        raise HTTPException(
            status_code=503,
            detail="Federated RAG system not initialized",
        )

    if node_ids is None:
        node_ids = _flower_grid.get_node_ids()
        if not node_ids:
            raise HTTPException(
                status_code=503,
                detail="No clients connected",
            )

    logger.info(f"[FEDERATED] Submitting to {len(node_ids)} clients...")

    try:
        import uuid
        qid = str(uuid.uuid4())[:8]
        
        # Step 1: Get responses từ tất cả clients
        documents, scores, client_answers = submit_question(
            grid=_flower_grid,
            question=question,
            qid=qid,
            knn=knn,
            node_ids=node_ids,
            use_synthesis=use_synthesis,
            qmode=query_transform_mode,
        )

        logger.info(
            f"[FEDERATED] Collected: {len(client_answers)} answers, "
            f"{len(documents)} docs from {len(node_ids)} clients"
        )

        if not documents:
            return {
                "answer": "",
                "documents": [],
                "scores": [],
                "client_answers": client_answers,  # ✅ Return raw answers
                "num_clients": len(node_ids),
            }

        # Step 2: Merge documents (RRF)
        merged_docs = merge_documents(documents, scores, knn, k_rrf=60)

        return {
            "answer": "",  # ⭐ ĐỀ TRỐNG - ensemble sẽ làm sau
            "documents": merged_docs,
            "scores": scores,
            "client_answers": client_answers,  # ✅ Return raw answers để ensemble
            "num_clients": len(node_ids),
            "documents_count": len(merged_docs),
        }

    except Exception as e:
        logger.error(f"[FEDERATED] Submit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== FEDERATED ENDPOINT ==========


@app.post("/api/query/federated", response_model=QueryResponse)
async def query_federated(request: QueryRequest):
    """Query through Federated RAG system"""
    
    global _flower_grid, _llm_querier, cfg
    
    if _flower_grid is None:
        raise HTTPException(status_code=503, detail="Federated RAG not ready")

    start_time = time_module.time()

    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        logger.info(f"[FEDERATED] Processing: '{question[:80]}...'")

        # ⭐ FIX: Apply privacy BEFORE sending question to federated system
        privacy_cfg = getattr(cfg, "privacy", {})
        privacy_applied = False
        privacy_metadata_combined = None
        original_question = question
        
        if (
            request.apply_privacy
            and PRIVACY_AVAILABLE
            and isinstance(privacy_cfg, dict)
            and privacy_cfg.get("enable_privacy_summary", False)
        ):
            logger.info("[FEDERATED] Applying privacy to QUESTION...")
            try:
                # Create mock response with question
                class MockResponse:
                    def __init__(self, text: str):
                        self.response = text

                mock_question = MockResponse(question)
                sanitized_question, privacy_metadata_q = apply_privacy_to_response(
                    mock_question, question, cfg
                )
                question = sanitized_question.response
                privacy_applied = True
                privacy_metadata_combined = privacy_metadata_q
                
                logger.info(
                    f"[FEDERATED] Question sanitized: "
                    f"{privacy_metadata_q.get('pii_entities', []) if privacy_metadata_q else []}"
                )
            except Exception as e:
                logger.warning(f"[FEDERATED] Question privacy failed: {e}")
                privacy_applied = False

        # Call federated system with sanitized question
        fed_result = submit_question_to_federated(
            question=question,  # ← Now sanitized
            knn=3,
            use_synthesis=True,
            query_transform_mode="none",
            node_ids=None,
        )

        # Extract data
        answer = fed_result["answer"]
        merged_docs = fed_result["documents"]
        client_answers = fed_result["client_answers"]
        num_clients = fed_result["num_clients"]

        logger.info(f"[FEDERATED] Got {len(client_answers)} answers, {len(merged_docs)} docs")

        # Ensemble answers
        if client_answers:
            logger.info(f"[FEDERATED] Client answers collected: {len(client_answers)}")
            
            if _llm_querier:
                logger.info(f"[FEDERATED] Ensembling {len(client_answers)} answers via LLM...")
                try:
                    from server_app import ensemble_answers as ensemble_fn
                    
                    final_answer = ensemble_fn(client_answers, _llm_querier)
                    
                    if final_answer and final_answer.strip():
                        answer = final_answer
                        logger.info(f"✓ [FEDERATED] Ensemble success: {answer[:100]}...")
                    else:
                        logger.warning("[FEDERATED] Ensemble returned empty")
                        answer = client_answers[0] if client_answers else ""
                
                except Exception as e:
                    logger.error(f"[FEDERATED] Ensemble failed: {e}")
                    answer = client_answers[0] if client_answers else ""
            else:
                logger.error("[FEDERATED] ⚠️ LLMQuerier is None! Cannot ensemble")
                answer = client_answers[0] if client_answers else ""
        else:
            logger.error("[FEDERATED] No client answers collected!")
            answer = ""

        # ⭐ FIX: Apply privacy ALSO to ANSWER
        privacy_stats_obj = None
        privacy_metadata_a = None

        if (
            request.apply_privacy
            and PRIVACY_AVAILABLE
            and isinstance(privacy_cfg, dict)
            and privacy_cfg.get("enable_privacy_summary", False)
        ):
            try:
                logger.info("[FEDERATED] Applying privacy protection to ANSWER...")

                class MockResponse:
                    def __init__(self, text: str):
                        self.response = text

                mock_resp = MockResponse(answer)
                processed_resp, privacy_metadata_a = apply_privacy_to_response(
                    mock_resp, original_question, cfg  # ← Use original question for context
                )
                answer = processed_resp.response

                if privacy_metadata_a:
                    logger.info(
                        f"[FEDERATED] Answer Privacy: {len(privacy_metadata_a.get('pii_entities', []))} PII detected"
                    )
                    
                    # MERGE metadata từ question + answer
                    if privacy_metadata_combined and privacy_metadata_a:
                        combined_pii = (
                            privacy_metadata_combined.get("pii_entities", []) +
                            privacy_metadata_a.get("pii_entities", [])
                        )
                        privacy_metadata_combined = {
                            **privacy_metadata_a,
                            "pii_entities": combined_pii,  # ← Combined PII from Q + A
                            "pii_density": (
                                privacy_metadata_combined.get("pii_density", 0.0) +
                                privacy_metadata_a.get("pii_density", 0.0)
                            ) / 2
                        }
                    else:
                        privacy_metadata_combined = privacy_metadata_a

            except Exception as e:
                logger.warning(f"[FEDERATED] Answer privacy failed: {e}")

        # Build privacy stats from combined metadata
        if privacy_metadata_combined:
            privacy_stats_obj = PrivacyStats(
                pii_detected=len(privacy_metadata_combined.get("pii_entities", [])),
                pii_density=privacy_metadata_combined.get("pii_density", 0.0),
                sentences_removed=privacy_metadata_combined.get("eraser", {}).get(
                    "removed_count", 0
                ),
                average_risk=privacy_metadata_combined.get("eraser", {}).get(
                    "average_risk", 0.0
                ),
                encrypted=privacy_metadata_combined.get("encryption", {}).get(
                    "enabled", False
                ),
                entities=[
                    e.get("entity_type", "")
                    for e in privacy_metadata_combined.get("pii_entities", [])
                ],
            )
            logger.info(
                f"[FEDERATED] Privacy stats: {privacy_stats_obj.pii_detected} PII from Q+A"
            )

        # Format source nodes
        source_nodes = []
        for i, doc in enumerate(merged_docs[:5]):
            source_nodes.append(
                SourceNode(
                    text=doc[:300] if isinstance(doc, str) else str(doc)[:300],
                    score=0.9 - (i * 0.05),
                    metadata={"rank": i + 1, "num_clients": num_clients},
                )
            )

        response_time = time_module.time() - start_time

        logger.info(f"[FEDERATED] Query completed in {response_time:.2f}s")

        return QueryResponse(
            question=original_question,  # ← Return ORIGINAL question (unsanitized for reference)
            answer=answer,  # ✅ Sanitized answer
            original_answer=None,
            privacy_applied=privacy_applied,
            privacy_stats=privacy_stats_obj,  # ← Combined stats from Q + A
            source_nodes=source_nodes,
            response_time=response_time,
            routed_to=f"federated_ensemble_{num_clients}_clients",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[FEDERATED] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )

