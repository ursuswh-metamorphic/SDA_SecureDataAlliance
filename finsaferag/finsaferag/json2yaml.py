import json
import yaml

# 1. Đọc JSON gốc
with open("feeds.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Ghi ra YAML
with open("feeds.yaml", "w", encoding="utf-8") as f:
    # default_flow_style=False -> dùng dạng block (nhiều dòng, dễ đọc)
    # allow_unicode=True -> giữ nguyên ký tự như "Ö", "Rösti", v.v.
    yaml.dump(
        data,
        f,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )

print("Converted feeds.json -> feeds.yaml")