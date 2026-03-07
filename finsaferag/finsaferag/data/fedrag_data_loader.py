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

# Xác định đường dẫn tuyệt đối tới rag_corpus.json
# BASE_DIR = .../finsaferag/finsaferag
BASE_DIR = Path(__file__).resolve().parent.parent

# Thử 2 vị trí phổ biến:
# 1) finsaferag/finsaferag/rag_corpus.json
# 2) finsaferag/finsaferag/data/rag_corpus.json
CANDIDATE_PATHS = [
    BASE_DIR / "rag_corpus.json",
    BASE_DIR / "data" / "rag_corpus.json",
]

RAG_CORPUS_FILE = None
for p in CANDIDATE_PATHS:
    if p.is_file():
        RAG_CORPUS_FILE = p
        break

if RAG_CORPUS_FILE is None:
    raise FileNotFoundError(
        "Không tìm thấy 'rag_corpus.json'. "
        f"Đã thử các path: {[str(p) for p in CANDIDATE_PATHS]}"
    )

# Dùng đường dẫn tuyệt đối cho FederatedDataset
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
