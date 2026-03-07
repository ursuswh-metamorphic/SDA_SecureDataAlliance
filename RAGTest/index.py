from llama_index.core import VectorStoreIndex
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore

from data.loader import get_documents
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore

def get_index(sources, persist_dir, split_type="sentence", chunk_size=1024):
    hierarchical_storage_context = None
    if not os.path.exists(persist_dir):
        # load the documents and create the index
        documents = get_documents()
        if split_type == "sentence":
            parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            print("nodes: " + str(nodes.__len__()))
        elif split_type == "character":
            parser = LangchainNodeParser(RecursiveCharacterTextSplitter())
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            print("nodes: " + str(nodes.__len__()))
        elif split_type == "hierarchical":
            parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 512, 128]
            )
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            print("nodes: " + str(nodes.__len__()))
        else:
            raise ValueError(f"split_type {split_type} not supported.")
        #Create FAISS vector store + storage context
        vector_store = FaissVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # VectorStoreIndex with FAISS => Backend
        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
        # store it for later
        if split_type == "hierarchical":
            docstore = SimpleDocumentStore()
            docstore.add_documents(nodes)
            hierarchical_storage_context = StorageContext.from_defaults(docstore=docstore)
            # save
            hierarchical_storage_context.persist(persist_dir=persist_dir+"-hierarchical")

        storage_context.persist(persist_dir=persist_dir)
    else:
        # load the existing index
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if split_type == "hierarchical":
            hierarchical_storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir + "-hierarchical"
            )

        # 8. Load index từ storage context (đã gắn FAISS)
        index = load_index_from_storage(storage_context)
    return index, hierarchical_storage_context
