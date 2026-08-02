"""
tools/clean_training_data.py — Filter data leakage & map canonical company names.

1. Loads `selected_data.json` (33,649 records).
2. Normalizes company names via `tools/company_map.json` (43 raw spellings -> 41 canonical).
3. Excludes 96 records that point to restored golden pages (preventing train-test leakage).
4. Saves `selected_data_clean.json`.
"""
import json
import os
import re

FEDE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTED_DATA_PATH = os.path.join(FEDE_DIR, "selected_data.json")
RESTORED_PAGES_PATH = os.path.join(FEDE_DIR, "..", "artifacts", "data", "restored_pages_v1.jsonl")
MAP_PATH = os.path.join(FEDE_DIR, "tools", "company_map.json")
OUTPUT_PATH = os.path.join(FEDE_DIR, "selected_data_clean.json")


def norm_company(raw: str, aliases: dict) -> str:
    cleaned = re.sub(r'[^A-Z0-9]', '', raw.strip().upper())
    return aliases.get(cleaned, cleaned)


def main():
    with open(MAP_PATH, 'r', encoding='utf-8') as f:
        map_info = json.load(f)
    aliases = map_info.get('aliases', {})

    # Load 168 restored golden doc+page pairs
    restored_pairs = set()
    if os.path.exists(RESTORED_PAGES_PATH):
        with open(RESTORED_PAGES_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                restored_pairs.add((row['doc_name'].strip(), int(row['page_num'])))
        print(f"[clean_data] Loaded {len(restored_pairs)} restored golden (doc, page) targets.")

    with open(SELECTED_DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"[clean_data] Original records in selected_data.json: {len(raw_data):,}")

    clean_records = []
    dropped_leak = 0

    for item in raw_data:
        comp = norm_company(item.get('company', ''), aliases)
        item['company'] = comp

        # Check page provenance if available e.g. "PEPSICO_2020_10K#p459"
        page_str = item.get('page', '')
        is_leak = False
        if '#' in page_str:
            doc, pstr = page_str.split('#', 1)
            pnum_match = re.search(r'\d+', pstr)
            if pnum_match:
                pnum = int(pnum_match.group(0))
                if (doc.strip(), pnum) in restored_pairs:
                    is_leak = True

        if is_leak:
            dropped_leak += 1
        else:
            clean_records.append(item)

    print(f"[clean_data] Dropped {dropped_leak} records that leaked restored golden pages.")
    print(f"[clean_data] Final clean training records: {len(clean_records):,}")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(clean_records, f, ensure_ascii=False, indent=2)
    print(f"[clean_data] Saved clean training dataset to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
