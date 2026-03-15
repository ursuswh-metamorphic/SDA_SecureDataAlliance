"""fedrag: A Flower Federated RAG app."""

import argparse
import os
import sys

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(DIR_PATH)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.download import DownloadCorpora
from fedrag.mirage_qa import MirageQA
from fedrag.retriever import Retriever


VALID_DATASETS = ["pubmed", "statpearls", "textbooks", "wikipedia"]

if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Datasets to download and index for the FedRAG workflow."
    )

    # Add an optional positional argument for number of partitions
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["textbooks"],
        help="List of dataset names as strings. Valid names are: `pubmed`, `textbooks`, `statpearls`, `wikipedia`",
    )

    # Add an optional positional argument for number of partitions
    parser.add_argument(
        "--index_num_chunks",
        nargs="?",
        type=int,
        default=0,
        help="How many chunks to consider when building the index for each corpus.",
    )

    args = parser.parse_args()
    dataset_names = sorted(
        {dataset_name.lower() for dataset_name in args.datasets}
    )  # make sure each dataset appears once in the collection
    num_chunks = (
        None if args.index_num_chunks == 0 else args.index_num_chunks
    )  # set to None if 0 else int value

    # use the default configurations for the FAISS based retrieval system
    retriever = Retriever()

    sample_query = "What are the complications of a cardiovascular disease?"
    for dataset_name in dataset_names:
        if dataset_name not in VALID_DATASETS:
            print("Not a valid dataset name: ", dataset_name)
            continue

        print(f"Downloading corpus: {dataset_name}")
        DownloadCorpora.download(corpus=dataset_name)
        print(f"Downloaded corpus: {dataset_name}")

        print(f"Rebuilding MedCPT FAISS index for corpus: {dataset_name}")
        retriever.build_faiss_index(
            dataset_name=dataset_name, batch_size=32, num_chunks=num_chunks
        )
        print(f"Built MedCPT FAISS index for corpus: {dataset_name}")

        print(f"Querying FAISS indexer of corpus: {dataset_name}")
        res = retriever.query_faiss_index(dataset_name, sample_query, knn=2)
        print(
            f"Query: {sample_query} and Corpus: {dataset_name}, returned the following results: {res}"
        )

    print("Downloading MIRAGE QA benchmark data.")
    mirage_file = os.path.join(DIR_PATH, "mirage.json")
    MirageQA.download(mirage_file)
    print("Downloaded MIRAGE QA benchmark data.")
