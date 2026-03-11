# prepare_rag_corpus.py
import json
from pathlib import Path


def build_rag_corpus(
    test_corpus_path: str = "test_corpus_backup.json",
    val_qa_path: str = "data/data_50.json",
    out_path: str = "rag_corpus.json",
    max_corpus_docs: int = 6066,
) -> None:
    """
    Gộp 2 file:
      - test_corpus_backup.json
      - data_50.json
    thành 1 file JSON phẳng: [{"id", "title", "text"}, ...]
    để dùng với HuggingFace + Flower.
    """
    docs = []

    # 1) Phần corpus: test_corpus_backup.json
    test_corpus_path = Path(test_corpus_path)
    if not test_corpus_path.exists():
        test_corpus_path.parent.mkdir(parents=True, exist_ok=True)
        test_corpus_path.write_text("{}")
        print(f"[OK] Created empty: {test_corpus_path}")
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

    # 2) Phần val_qa: data_50.json
    val_qa_path = Path(val_qa_path)
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

    print(f"[OK] Saved to {out_path}")


if __name__ == "__main__":
    build_rag_corpus()
