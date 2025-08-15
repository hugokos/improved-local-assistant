"""
summarise_survivalist_graph.py
Run with:  python summarise_survivalist_graph.py
(or copy/paste into a notebook / REPL)
"""

import json
from pathlib import Path

PERSIST_DIR = Path("./data/prebuilt_graphs/survivalist")  # adjust if needed


def pretty(obj, indent=2):
    return json.dumps(obj, indent=indent, ensure_ascii=False)[:800] + " ..."


def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def main():
    if not PERSIST_DIR.exists():
        raise SystemExit(f"❌ Directory not found: {PERSIST_DIR.resolve()}")

    print(f"\n📂 Inspecting graph at: {PERSIST_DIR.resolve()}\n")

    # ---------- quick summaries ----------
    graph_store = load_json(PERSIST_DIR / "graph_store.json")
    index_store = load_json(PERSIST_DIR / "index_store.json")
    vector_store = load_json(PERSIST_DIR / "default__vector_store.json")

    print("🔑   graph_store keys          :", list(graph_store.keys())[:10])
    print("📏   # triples (approx)        :", sum(len(v) for v in graph_store.values()))
    print("📦   index_store top-level keys:", list(index_store.keys())[:10])
    print(
        "📊   vector rows               :",
        len(vector_store.get("collections", [{}])[0].get("data", [])),
    )

    print("\n—" * 40)

    # ---------- sample payloads ----------
    print(
        "Sample graph_store entry:\n",
        pretty({k: graph_store[k] for k in list(graph_store.keys())[:1]}),
    )

    print(
        "\nSample index_store entry:\n",
        pretty({k: index_store[k] for k in list(index_store.keys())[:1]}),
    )

    print(
        "\nSample vector row (truncated):\n",
        pretty(vector_store.get("collections", [{}])[0].get("data", [])[:1]),
    )


if __name__ == "__main__":
    main()
