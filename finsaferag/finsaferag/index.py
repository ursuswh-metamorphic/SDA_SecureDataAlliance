# Thực hiện gọi fedrag_data_loader nhằm: 
#     - Tạo ra dataset riêng dựa trên node_id
# Sau đó nó sẽ thực hiện tạo faiss vector store để lưu dataset nhằm retrieve về sau

import os
import faiss
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import LangchainNodeParser, HierarchicalNodeParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from llama_index.vector_stores.faiss import FaissVectorStore
except ModuleNotFoundError as e:
    try:
        from llama_index.integrations.vector_stores.faiss import FaissVectorStore
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "FaissVectorStore not found. Install: pip install llama-index-vector-stores-faiss"
        ) from e

# Thay loader cũ bằng loader federated (module riêng tùy chỉnh để có thể chia dataset cho nhiều bên dựa trên node_id của từng client)
from data.fedrag_data_loader import get_client_documents


def get_index(node_id, persist_dir, split_type="sentence", chunk_size=1024):
    """
    Build hoặc load FAISS index cho một client (node_id).
    Dữ liệu được lấy qua get_client_documents(node_id).
    """
    hierarchical_storage_context = None

    # ==============================
    # CASE 1: index chưa tồn tại → build mới
    # ==============================
    if not os.path.exists(persist_dir):

        # 1. Load document partition cho client tương ứng
        documents = get_client_documents(node_id)

        # 2. Split thành nodes
        if split_type == "sentence":
            parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)

        elif split_type == "character":
            parser = LangchainNodeParser(RecursiveCharacterTextSplitter())
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)

        elif split_type == "hierarchical":
            parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 512, 128]
            )
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)

        else:
            raise ValueError(f"split_type {split_type} not supported.")

        print("nodes:", len(nodes))

        # 3. Create FAISS vector store
        embedding_dim = len(nodes[0].embedding) if nodes[0].embedding else 768
        faiss_index = faiss.IndexFlatL2(embedding_dim)

        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 4. Build index (VectorStoreIndex backend = FAISS)
        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

        # 5. Persist cho client
        if split_type == "hierarchical":
            docstore = SimpleDocumentStore()
            docstore.add_documents(nodes)
            hierarchical_storage_context = StorageContext.from_defaults(docstore=docstore)
            hierarchical_storage_context.persist(persist_dir=persist_dir + "-hierarchical")

        storage_context.persist(persist_dir=persist_dir)

    # ==============================
    # CASE 2: index đã tồn tại → load lại
    # ==============================
    else:
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, 
            persist_dir=persist_dir
        )

        if split_type == "hierarchical":
            hierarchical_storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir + "-hierarchical"
            )

        index = load_index_from_storage(storage_context)

    return index, hierarchical_storage_context
