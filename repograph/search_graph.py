# retrieve code graph scripts

import pickle
import sys
import json
import traceback
from pathlib import Path

DEFAULT_GRAPH_PATH = "/data/zxl/Search2026/outputData/devEvalRepoGraph/mrjob/graph.pkl"
DEFAULT_TAGS_PATH = "/data/zxl/Search2026/outputData/devEvalRepoGraph/mrjob/tags.json"

def _load_tags(tags_path: str):
    with open(tags_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(query_name: str, graph_path: str, tags_path: str):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    tags = _load_tags(tags_path)

    try:
        if query_name not in G:
            raise KeyError(
                f"Node {query_name!r} not found in graph. "
                f"(loaded graph has {G.number_of_nodes()} nodes)"
            )

        # 1-hop neighborhood: outgoing + incoming
        successors = list(G.successors(query_name))
        predecessors = list(G.predecessors(query_name))
        print(len(successors), len(predecessors))

        # tags.json may contain duplicate names across files; keep all
        tags_by_name = {}
        for tag in tags:
            tags_by_name.setdefault(tag["name"], []).append(tag)

        returned = []
        for name in successors + [query_name] + predecessors:
            for tag in tags_by_name.get(name, []):
                if "test" in str(tag.get("fname", "")):
                    continue
                returned.append(
                    {
                        "fname": tag.get("fname"),
                        "line": tag.get("line"),
                        "name": tag.get("name"),
                        "kind": tag.get("kind"),
                        "category": tag.get("category"),
                        "info": tag.get("info"),
                    }
                )

        return {
            "query": query_name,
            "successors": successors,
            "predecessors": predecessors,
            "matches": returned,
        }
    except Exception:
        return {"query": query_name, "error": traceback.format_exc()}

if __name__ == '__main__':
    query = sys.argv[1] if len(sys.argv) > 1 else None
    graph_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GRAPH_PATH
    tags_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_TAGS_PATH

    if not query:
        print(
            "Usage: python repograph/search_graph.py <name> [graph.pkl] [tags.json]"
        )
        raise SystemExit(2)

    if not Path(graph_path).exists():
        print(f"Graph file not found: {graph_path}")
        raise SystemExit(2)
    if not Path(tags_path).exists():
        print(f"Tags file not found: {tags_path}")
        raise SystemExit(2)

    result = main(query, graph_path, tags_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))