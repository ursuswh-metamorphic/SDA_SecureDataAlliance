# prepare_rag_corpus.py
import json
from pathlib import Path


def _get_config_paths():
    """Get data paths from config (domain-aware)."""
    try:
        from config import Config
        cfg = Config()
        data_dir = getattr(cfg, "data_dir", ".") or "."
        base = Path(__file__).resolve().parent
        if data_dir not in (".", ""):
            base = base / data_dir
        return {
            "test_corpus": base / (getattr(cfg, "test_corpus_file", "test_corpus_backup.json") or "test_corpus_backup.json"),
            "val_qa": base / (getattr(cfg, "val_qa_file", "test_retrieval.json") or "test_retrieval.json"),
            "out": base / (getattr(cfg, "corpus_file", "rag_corpus.json") or "rag_corpus.json"),
        }
    except Exception:
        base = Path(__file__).resolve().parent
        return {
            "test_corpus": base / "test_corpus_backup.json",
            "val_qa": base / "test_retrieval.json",
            "out": base / "rag_corpus.json",
        }


def build_rag_corpus(
    test_corpus_path: str = None,
    val_qa_path: str = None,
    out_path: str = None,
    max_corpus_docs: int = 6066,
) -> None:
    """
    Gộp 2 file:
      - test_corpus_backup.json (hoặc test_corpus_file từ config)
      - test_retrieval.json (hoặc val_qa_file từ config)
    thành 1 file JSON phẳng: [{"id", "title", "text"}, ...]
    để dùng với HuggingFace + Flower.
    Hỗ trợ multi-domain qua config: data_dir, corpus_file, val_qa_file, test_corpus_file.
    """
    cfg_paths = _get_config_paths()
    test_corpus_path = Path(test_corpus_path) if test_corpus_path else cfg_paths["test_corpus"]
    val_qa_path = Path(val_qa_path) if val_qa_path else cfg_paths["val_qa"]
    out_path = Path(out_path) if out_path else cfg_paths["out"]

    docs = []

    # 1) Phần corpus
    test_corpus_path = Path(test_corpus_path)
    if not test_corpus_path.exists():
        test_corpus_path.parent.mkdir(parents=True, exist_ok=True)
        test_corpus_path.write_text("{}")
        print(f"✓ Created empty: {test_corpus_path}")
    with test_corpus_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for _, entry in data.items():
        for _, passage in entry.items():
            title = ""  # nếu chưa có title cụ thể
            text = passage["page_content"]
            idx = passage["index"]

            docs.append(
                {
                    "id": idx,
                    "title": title,
                    "text": text,
                }
            )
            if len(docs) >= max_corpus_docs:
                break
        if len(docs) >= max_corpus_docs:
            break

    print("len(corpus docs):", len(docs))

    # 2) Phần val_qa
    with val_qa_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        title = entry["other_info"]["doc_name"]
        refs = entry["key_content"]["reference"]
        ids = entry["key_content"]["reference_idx"]

        for reference, idx in zip(refs, ids):
            docs.append(
                {
                    "id": idx,
                    "title": title,
                    "text": reference,
                }
            )

    print("Total docs:", len(docs))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved to {out_path}")


if __name__ == "__main__":
    build_rag_corpus()
