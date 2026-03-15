# fedrag_data_loader.py

from typing import List
from pathlib import Path

from llama_index.core import Document
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# ================================
# CONFIG
# ================================
NUM_CLIENTS = 10

# BASE_DIR = .../finsaferag/finsaferag
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _resolve_rag_corpus_path() -> Path:
    """Resolve rag_corpus path from config or default locations (domain-aware)."""
    try:
        from config import Config
        cfg = Config()
        data_dir = getattr(cfg, "data_dir", ".") or "."
        corpus_file = getattr(cfg, "corpus_file", "rag_corpus.json") or "rag_corpus.json"
        if data_dir in (".", ""):
            candidates = [
                BASE_DIR / corpus_file,
                DATA_DIR / corpus_file,
            ]
        else:
            candidates = [
                DATA_DIR / data_dir / corpus_file,
                BASE_DIR / data_dir / corpus_file,
            ]
    except Exception:
        candidates = [
            BASE_DIR / "rag_corpus.json",
            DATA_DIR / "rag_corpus.json",
        ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Không tìm thấy corpus file. "
        f"Đã thử các path: {[str(p) for p in candidates]}"
    )


RAG_CORPUS_FILE = _resolve_rag_corpus_path()
RAG_CORPUS_PATH = str(RAG_CORPUS_FILE)

# ================================
# BUILD FEDERATED DATASET (global)
# ================================
fds = FederatedDataset(
    dataset="json",
    partitioners={"train": IidPartitioner(num_partitions=NUM_CLIENTS)},
    shuffle=True,
    seed=42,
    data_files={"train": RAG_CORPUS_PATH},
)

# ================================
# UTILS
# ================================

def map_node_to_pid(node_id: str) -> int:
    """
    Map node_id -> partition id.
    Flower thường dùng node_id = '0', '1', ...
    """
    pid = int(node_id) % NUM_CLIENTS
    return pid


def load_partition(pid: int):
    """Load partition raw từ FederatedDataset."""
    return fds.load_partition(pid, "train")


def partition_to_documents(partition) -> List[Document]:
    """Convert partition (HF Dataset) → List[Document]."""
    docs = [
        Document(
            text=row["text"],
            metadata={"title": row["title"], "id": row["id"]},
            doc_id=str(row["id"]),
        )
        for row in partition
    ]
    return docs


# ================================
# MAIN API (GỌI TỪ CLIENT_APP)
# ================================
def get_client_documents(node_id: str) -> List[Document]:
    """
    Hàm duy nhất bạn cần gọi ở client_app.py

    node_id -> partition_id -> HuggingFace Dataset -> List[Document]
    """
    pid = map_node_to_pid(node_id)
    partition = load_partition(pid)
    docs = partition_to_documents(partition)
    return docs
