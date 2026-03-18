# retrieve code graph scripts

import pickle
import sys
import json
import traceback
from pathlib import Path
import re
from difflib import SequenceMatcher

DEFAULT_GRAPH_PATH = "/data/zxl/Search2026/outputData/devEvalRepoGraph/mrjob/graph.pkl"
DEFAULT_TAGS_PATH = "/data/zxl/Search2026/outputData/devEvalRepoGraph/mrjob/tags.json"

def _load_tags(tags_path: str):
    with open(tags_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _one_hop(G, center: str):
    # 1-hop neighborhood: outgoing + incoming
    return list(G.successors(center)), list(G.predecessors(center))


def _collect_matches(tags, names):
    # tags.json may contain duplicate names across files; keep all
    tags_by_name = {}
    for tag in tags:
        tags_by_name.setdefault(tag["name"], []).append(tag)

    returned = []
    for name in names:
        for tag in tags_by_name.get(name, []):
            if "test" in str(tag.get("fname", "")):
                continue
            returned.append(
                {
                    "fname": tag.get("fname"),
                    "line": tag.get("line"),
                    "name": tag.get("name"),
                    "qualified_name": tag.get("qualified_name"),
                    "kind": tag.get("kind"),
                    "category": tag.get("category"),
                    "info": tag.get("info"),
                }
            )
    return returned


def _normalize_name(s: str) -> str:
    # Lowercase and remove non-alphanumerics so that:
    # - case changes don't matter
    # - underscores/dashes/spaces don't matter
    # - minor punctuation differences don't matter
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _fuzzy_topk_nodes(G, query_name: str, k: int = 5):
    qn = _normalize_name(query_name)
    if not qn:
        return []

    scored = []
    for node in G.nodes:
        node_str = str(node)
        nn = _normalize_name(node_str)
        if not nn:
            continue

        # blend raw + normalized similarities (helps keep case-sensitive exact-ish matches)
        score_norm = _similarity(qn, nn)
        score_raw = _similarity(query_name, node_str)
        score = 0.7 * score_norm + 0.3 * score_raw
        scored.append((score, node_str))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = []
    seen = set()
    for score, node_str in scored:
        if node_str in seen:
            continue
        seen.add(node_str)
        top.append({"name": node_str, "score": round(float(score), 4)})
        if len(top) >= k:
            break
    return top


def main_exact(query_name: str, graph_path: str, tags_path: str):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    tags = _load_tags(tags_path)

    try:
        if query_name not in G:
            raise KeyError(
                f"Node {query_name!r} not found in graph. "
                f"(loaded graph has {G.number_of_nodes()} nodes)"
            )

        successors, predecessors = _one_hop(G, query_name)
        returned = _collect_matches(tags, successors + [query_name] + predecessors)

        return {
            "query": query_name,
            "successors": successors,
            "predecessors": predecessors,
            "matches": returned,
        }
    except Exception:
        return {"query": query_name, "error": traceback.format_exc()}


def main_fuzzy(query_name: str, graph_path: str, tags_path: str, k: int = 5):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    tags = _load_tags(tags_path)

    try:
        if query_name in G:
            successors, predecessors = _one_hop(G, query_name)
            returned = _collect_matches(tags, successors + [query_name] + predecessors)
            return {
                "query": query_name,
                "resolved": query_name,
                "fuzzy_candidates": [{"name": query_name, "score": 1.0}],
                "successors": successors,
                "predecessors": predecessors,
                "matches": returned,
            }

        fuzzy_candidates = _fuzzy_topk_nodes(G, query_name, k=k)
        if not fuzzy_candidates:
            raise KeyError(
                f"Node {query_name!r} not found in graph (and fuzzy match failed). "
                f"Loaded graph has {G.number_of_nodes()} nodes."
            )

        resolved = fuzzy_candidates[0]["name"]
        successors, predecessors = _one_hop(G, resolved)
        returned = _collect_matches(tags, successors + [resolved] + predecessors)
        return {
            "query": query_name,
            "resolved": resolved,
            "fuzzy_candidates": fuzzy_candidates,
            "successors": successors,
            "predecessors": predecessors,
            "matches": returned,
        }
    except Exception:
        return {"query": query_name, "error": traceback.format_exc()}


# Backward-compatible default entrypoint: exact match
def main(query_name: str, graph_path: str, tags_path: str):
    return main_exact(query_name, graph_path, tags_path)


if __name__ == '__main__':
    query = sys.argv[1] if len(sys.argv) > 1 else None

    # CLI supports either:
    # - python search_graph.py <name> [graph.pkl] [tags.json]                (exact, default)
    # - python search_graph.py <name> <exact|fuzzy> [graph.pkl] [tags.json]  (explicit mode)
    mode = "fuzzy"
    arg2 = sys.argv[2] if len(sys.argv) > 2 else None
    if arg2 in ("exact", "fuzzy"):
        mode = arg2
        graph_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_GRAPH_PATH
        tags_path = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_TAGS_PATH
    else:
        graph_path = arg2 if arg2 else DEFAULT_GRAPH_PATH
        tags_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_TAGS_PATH

    if not query:
        print(
            "Usage:\n"
            "  python repograph/search_graph.py <name> [graph.pkl] [tags.json]\n"
            "  python repograph/search_graph.py <name> <exact|fuzzy> [graph.pkl] [tags.json]\n"
        )
        raise SystemExit(2)

    if not Path(graph_path).exists():
        print(f"Graph file not found: {graph_path}")
        raise SystemExit(2)
    if not Path(tags_path).exists():
        print(f"Tags file not found: {tags_path}")
        raise SystemExit(2)

    if mode == "fuzzy":
        result = main_fuzzy(query, graph_path, tags_path)
    else:
        result = main_exact(query, graph_path, tags_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))