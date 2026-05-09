"""
download_paper_data.py — fetch FedE4RAG paper's test data from HuggingFace.

Downloads 3 files from `DocAILab/FedE4RAG_Dataset` into `FedE/paper_test_data/`:
  * RAG4FIN/val_qa/data_50.json    -> val_qa_data_50.json    (50 validation queries)
  * RAG4FIN/test_qa/data_100.json  -> test_qa_data_100.json  (100 test queries)
  * RAG4FIN/test_corpus.json       -> test_corpus.json       (~6,656 doc pages)

These are the EXACT data files used in the paper's Tables II/III. Required for
paper-faithful evaluation (eval_paper_faithful.py).

Run from repo root: python FedE/scripts/download_paper_data.py
"""
import os
import shutil
import sys

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "DocAILab/FedE4RAG_Dataset"
REPO_TYPE = "dataset"

# (remote filename, local filename in paper_test_data/)
FILES = [
    ("RAG4FIN/val_qa/data_50.json", "val_qa_data_50.json"),
    ("RAG4FIN/test_qa/data_100.json", "test_qa_data_100.json"),
    ("RAG4FIN/test_corpus.json", "test_corpus.json"),
]


def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "paper_test_data",
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[download_paper_data] Target dir: {out_dir}")
    print(f"[download_paper_data] Repo:       {REPO_ID} ({REPO_TYPE})")
    print()

    # huggingface_hub creates a cache-tree (subfolders); we want flat layout.
    # Strategy: download into a temp cache dir, then copy to flat target paths.
    for remote, local in FILES:
        try:
            print(f"  Downloading {remote} ...")
            cached = hf_hub_download(
                repo_id=REPO_ID,
                filename=remote,
                repo_type=REPO_TYPE,
            )
        except HfHubHTTPError as e:
            print(f"  [FAIL] HTTP error: {e}")
            print(
                f"  Hint: set HF_TOKEN env-var if hitting rate limits "
                f"(unauthenticated downloads have 100/day limit)."
            )
            sys.exit(1)

        dest = os.path.join(out_dir, local)
        shutil.copy2(cached, dest)
        size = os.path.getsize(dest)
        print(f"    -> {dest}  ({size / 1024:.1f} KB)")

    print()
    print(f"[download_paper_data] All 3 files downloaded successfully.")
    print(f"    Next: python FedE/scripts/inspect_paper_data.py")


if __name__ == "__main__":
    main()
