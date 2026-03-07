"""fedrag: A Flower Federated RAG app.

Retriever dùng lại LlamaIndex (cùng index/embedding với RAG local),
chỉ wrap thêm cho phù hợp với federated.
"""

import os
from collections import OrderedDict
from typing import Optional

from config import Config
from index import get_index               # hàm build/load index bạn đã có
from retriever import vector_retriever    # hàm tạo retriever bạn đã có (local RAG)

# Đọc config một lần (singleton)
cfg = Config()

# Giá trị mặc định lấy từ config.toml
DEFAULT_STORAGE_ROOT = cfg.persist_dir        
DEFAULT_SPLIT_TYPE = cfg.split_type          
DEFAULT_CHUNK_SIZE = cfg.chunk_size          
DEFAULT_TOP_K = 8                            


class Retriever:
    def __init__(
        self,
        storage_root: Optional[str] = None,
        split_type: Optional[str] = None,
        chunk_size: Optional[int] = None,
        default_top_k: Optional[int] = None,
    ) -> None:
        self.storage_root = storage_root or DEFAULT_STORAGE_ROOT
        self.split_type = split_type or DEFAULT_SPLIT_TYPE
        self.chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        self.default_top_k = default_top_k or DEFAULT_TOP_K

    def _resolve_persist_dir(self, corpus_name: str) -> str:
        """Suy ra persist_dir từ corpus_name."""
        if os.path.isabs(corpus_name) or "/" in corpus_name or "\\" in corpus_name:
            return corpus_name
        # Ngược lại, ghép với storage_root (ví dụ: storage/<corpus_name>)
        return os.path.join(self.storage_root, corpus_name)

    def query_index(self, corpus_name: str, query: str, knn: Optional[int] = None):
        """Query index của corpus_name, trả về OrderedDict giống kiểu FAISS cũ.

        Format:
        {
            doc_id: {
                "rank": int,
                "score": float,
                "title": str,
                "content": str,
            },
            ...
        }
        """
        if knn is None:
            knn = self.default_top_k

        persist_dir = self._resolve_persist_dir(corpus_name)

        # Dùng lại get_index: tự build hoặc load từ persist_dir, backend FAISS
        index, hierarchical_storage_context = get_index(
            sources=None,            # hiện tại bạn không dùng tham số này
            persist_dir=persist_dir,
            split_type=self.split_type,
            chunk_size=self.chunk_size,
        )

        # Dùng lại vector_retriever của RAG local
        retriever = vector_retriever(
            index=index,
            similarity_top_k=knn,
            show_progress=False,
            store_nodes_override=True,
        )

        # LlamaIndex retriever trả về list[NodeWithScore]
        results = retriever.retrieve(query)

        final_res = OrderedDict()
        for i, nws in enumerate(results):
            node = nws.node
            score = float(nws.score)

            # id bạn đã set trong Document(metadata['id']), fallback sang node_id nếu thiếu
            doc_id = str(node.metadata.get("id", getattr(node, "node_id", f"node_{i}")))
            title = str(node.metadata.get("title", ""))

            try:
                content = node.get_content()
            except Exception:
                content = getattr(node, "text", "")

            final_res[doc_id] = {
                "rank": i + 1,
                "score": score,
                "title": title,
                "content": content,
            }

        return final_res