import argparse
import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.dev_eval_task import DevEvalTask, parse_task
from utils.jsonl_io import iter_jsonl, write_jsonl_line
from utils.llm_clients import BackendName, make_client
from utils.source_code_utils import read_line_range, resolve_signature
from utils.task_recall import compute_task_recall


# 默认路径参数，仅用于命令行直接运行本脚本时的便捷入口；
# 实际批量实验时应通过函数参数传入这些路径。
SOURCE_CODE_DIR = "/data/lowcode_public/DevEval_zxl/Source_Code"
REPO_DIR = "/data/lowcode_public/DevEval_zxl/Source_Code/Database/alembic/alembic"
FILTERED_PATH = "/data/zxl/Search2026/outputData/devEvalSearchOut/0316_batch_workflow/alembic/filtered.jsonl"
GRAPH_PKL = "/data/zxl/Search2026/outputData/devEvalRepoGraph/alembic/graph.pkl"
TAGS_JSON = "/data/zxl/Search2026/outputData/devEvalRepoGraph/alembic/tags.json"
OUTPUT_JSONL = "/data/zxl/Search2026/outputData/devEvalRepoGraph/alembic/diagnostic_graph_context.jsonl"

MODEL_BACKEND_CHOICE: BackendName = "openai"
MODEL_NAME = "deepseek-v3"


SEARCH_TERMS_PROMPT_TEMPLATE = """You are helping retrieve relevant code dependencies from a Python repository graph.

Given a task requirement and the target function signature to implement, propose up to {max_terms} search terms.
These terms should be likely function/class names or module-qualified symbols that the implementation may need to call or reference.

Rules:
- Output MUST be valid JSON.
- Output MUST be a JSON array of strings.
- Array length MUST be between 1 and {max_terms}.
- Each string should be short (<= 80 chars).
- Prefer symbols (function/class/module names) over natural language.
- Do NOT include explanations.

Task requirement:
{requirement_text}

Target signature:
{signature}
"""


def _normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


@dataclass(frozen=True)
class ResolvedNode:
    query: str
    resolved: str
    # top candidates for debugging
    candidates: List[Dict[str, Any]]


class GraphContextSearcher:
    def __init__(self, graph_pkl: str, tags_json: str):
        with open(graph_pkl, "rb") as f:
            self.G = pickle.load(f)
        with open(tags_json, "r", encoding="utf-8") as f:
            self.tags = [json.loads(line) for line in f if line.strip()]

        # Build a fallback index by name and qualified_name
        self.tags_by_name: Dict[str, List[dict]] = {}
        self.tags_by_qname: Dict[str, List[dict]] = {}
        for t in self.tags:
            n = t.get("name")
            qn = t.get("qualified_name")
            if isinstance(n, str) and n:
                self.tags_by_name.setdefault(n, []).append(t)
            if isinstance(qn, str) and qn:
                self.tags_by_qname.setdefault(qn, []).append(t)

        # Pre-collect node qualified_name map (for exact/fuzzy resolution)
        self.node_qname_to_node: Dict[str, str] = {}
        for n, data in self.G.nodes(data=True):
            qn = (data or {}).get("qualified_name")
            if isinstance(qn, str) and qn and qn not in self.node_qname_to_node:
                self.node_qname_to_node[qn] = str(n)

    def resolve_node_fuzzy(self, query_name: str, *, k: int = 5) -> ResolvedNode:
        # exact by node key
        if query_name in self.G:
            return ResolvedNode(query=query_name, resolved=query_name, candidates=[{"name": query_name, "score": 1.0}])

        # exact by qualified_name attribute
        if query_name in self.node_qname_to_node:
            n = self.node_qname_to_node[query_name]
            return ResolvedNode(query=query_name, resolved=n, candidates=[{"name": n, "score": 1.0, "qualified_name": query_name}])

        qn = _normalize_name(query_name)
        if not qn:
            return ResolvedNode(query=query_name, resolved="", candidates=[])

        scored: List[Tuple[float, str, Optional[str]]] = []
        for node, data in self.G.nodes(data=True):
            node_str = str(node)
            nn = _normalize_name(node_str)
            if nn:
                score = 0.7 * _similarity(qn, nn) + 0.3 * _similarity(query_name, node_str)
                scored.append((score, node_str, None))
            qname_attr = (data or {}).get("qualified_name")
            if isinstance(qname_attr, str) and qname_attr:
                qna = _normalize_name(qname_attr)
                if qna:
                    score = 0.7 * _similarity(qn, qna) + 0.3 * _similarity(query_name, qname_attr)
                    scored.append((score, node_str, qname_attr))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidates: List[Dict[str, Any]] = []
        seen = set()
        for score, node_str, qname_attr in scored:
            key = (node_str, qname_attr)
            if key in seen:
                continue
            seen.add(key)
            rec: Dict[str, Any] = {"name": node_str, "score": round(float(score), 4)}
            if qname_attr:
                rec["qualified_name"] = qname_attr
            candidates.append(rec)
            if len(candidates) >= k:
                break

        resolved = candidates[0]["name"] if candidates else ""
        return ResolvedNode(query=query_name, resolved=resolved, candidates=candidates)

    def one_hop(self, node: str) -> Tuple[List[str], List[str]]:
        return list(self.G.successors(node)), list(self.G.predecessors(node))

    def node_kind(self, node: str) -> str:
        data = self.G.nodes.get(node, {}) or {}
        kind = data.get("kind")
        return str(kind) if kind is not None else ""

    def node_to_context(self, node: str) -> Dict[str, Any]:
        data = self.G.nodes.get(node, {}) or {}
        fname = data.get("fname")
        line = data.get("line")
        qname = data.get("qualified_name") or data.get("name") or node

        code = ""
        if isinstance(fname, str) and fname and isinstance(line, (list, tuple)) and len(line) == 2:
            try:
                start, end = int(line[0]), int(line[1])
                if start > 0 and end > 0 and end >= start:
                    code = read_line_range(fname, start, end)
            except Exception:
                code = ""

        return {
            "sig": qname,
            "method_signature": qname,
            "method_code": code,
            "node": {
                "name": node,
                "qualified_name": data.get("qualified_name"),
                "kind": data.get("kind"),
                "category": data.get("category"),
                "fname": fname,
                "rel_fname": data.get("rel_fname"),
                "line": line,
            },
        }


def llm_generate_search_terms(
    client,
    *,
    requirement_text: str,
    signature: str,
    max_terms: int = 5,
) -> List[str]:
    prompt = SEARCH_TERMS_PROMPT_TEMPLATE.format(
        max_terms=max_terms,
        requirement_text=requirement_text.strip(),
        signature=signature.rstrip("\n"),
    )
    raw = client.generate(prompt)
    raw = (raw or "").strip()

    # DEBUG
    print(f"DEBUG: raw: {raw}")

    # strict JSON array preferred
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            terms = [str(x).strip() for x in obj if str(x).strip()]
            return terms[:max_terms] or []
    except Exception:
        pass

    # fallback: extract first JSON array in text
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                terms = [str(x).strip() for x in obj if str(x).strip()]
                return terms[:max_terms] or []
        except Exception:
            pass

    # last resort: split lines
    lines = [ln.strip(" \t-•") for ln in raw.splitlines() if ln.strip()]


    return [ln for ln in lines[:max_terms] if ln]


def rank_aggregate_one_task(
    searcher: GraphContextSearcher,
    search_terms: List[str],
) -> Dict[str, Any]:
    """
    Ranking rules:
    - preserve search term order
    - within each term: resolved center first
    - then 1-hop neighbors: kind=def before kind=ref
    - deduplicate by node key, keep earliest/best rank
    """
    term_results: List[Dict[str, Any]] = []
    ranked: List[Dict[str, Any]] = []
    seen = set()

    def kind_priority(k: str) -> int:
        return 0 if k == "def" else (1 if k == "ref" else 2)

    for ti, term in enumerate(search_terms):
        resolved = searcher.resolve_node_fuzzy(term, k=5)
        if not resolved.resolved:
            term_results.append(
                {
                    "term": term,
                    "resolved": "",
                    "candidates": resolved.candidates,
                    "successors": [],
                    "predecessors": [],
                }
            )
            continue

        succ, pred = searcher.one_hop(resolved.resolved)
        neighbors = succ + pred
        # sort neighbors by kind priority (def before ref) but keep stable within same kind
        neighbors_sorted = sorted(
            neighbors,
            key=lambda n: (kind_priority(searcher.node_kind(n)),),
        )

        # center first
        for pos, n in enumerate([resolved.resolved] + neighbors_sorted):
            if n in seen:
                continue
            seen.add(n)
            is_center = (pos == 0)
            ranked.append(
                {
                    "node": n,
                    "from_term": term,
                    "term_index": ti,
                    "role": "center" if is_center else "neighbor",
                    "kind": searcher.node_kind(n),
                }
            )

        term_results.append(
            {
                "term": term,
                "resolved": resolved.resolved,
                "candidates": resolved.candidates,
                "successors": succ,
                "predecessors": pred,
            }
        )

    return {"term_results": term_results, "ranked_nodes": ranked}


def analyze_project(
    source_code_dir: str,
    *,
    repo_dir: str,
    filtered_path: str,
    graph_pkl: str,
    tags_json: str,
    output_jsonl: str,
    backend: BackendName,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    timeout_s: float,
    max_tasks: Optional[int],
    sleep_s: float,
    max_terms: int = 5,
    recall_ks: Tuple[int, ...] = (10, 15, 20),
) -> Dict[str, Any]:
    """
    For each task in filtered_path:
    - use LLM to generate <= max_terms search terms
    - fuzzy search graph and expand 1-hop for each term
    - aggregate + rank results deterministically
    - build context_code_list from graph node attributes (fname/line/qualified_name)
    - compute recall@K, precision@K, f1@K
    - save per-task record to output_jsonl for later RAG
    """
    # clear output file
    with open(output_jsonl, "w", encoding="utf-8"):
        pass

    client = make_client(
        backend=backend,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )

    searcher = GraphContextSearcher(graph_pkl, tags_json)

    processed = 0
    agg = {k: {"pred": 0, "match": 0, "gt": 0} for k in recall_ks}
    tasks_with_dep = 0

    for record in iter_jsonl(filtered_path):
        task: DevEvalTask = parse_task(record)

        abs_file, signature = resolve_signature(
            source_code_dir, task.completion_path, task.signature_position
        )
        requirement_text = task.requirement_text
        print(f"DEBUG: requirement_text: {requirement_text}")
        print(f"DEBUG: signature: {signature}")

        search_terms = llm_generate_search_terms(
            client,
            requirement_text=requirement_text,
            signature=signature,
            max_terms=max_terms,
        )
        print(f"DEBUG: search_terms: {search_terms}")
        print("=======================================")
        if not search_terms:
            # fallback: use namespace short name and signature function name
            fallback = task.namespace.split(".")[-1]
            search_terms = [fallback] if fallback else []

        agg_info = rank_aggregate_one_task(searcher, search_terms)
        ranked_nodes: List[Dict[str, Any]] = agg_info["ranked_nodes"]

        # Requirement: move all centers to the very front (across all terms),
        # while keeping neighbors' relative order as currently produced.
        centers = [x for x in ranked_nodes if x.get("role") == "center"]
        neighbors = [x for x in ranked_nodes if x.get("role") != "center"]
        ranked_nodes = centers + neighbors

        # Build context list in ranked order
        context_code_list = []
        for item in ranked_nodes:
            n = item["node"]
            ctx = searcher.node_to_context(n)
            ctx["rank_info"] = item
            context_code_list.append(ctx)

        # Compute metrics at K
        metrics_by_k: Dict[str, Any] = {}
        for k in recall_ks:
            topk = context_code_list[:k]
            recall_info = compute_task_recall(task.dependency, topk)
            num_match = int(recall_info["dependency_hit"])
            num_gt = int(recall_info["dependency_total"])
            num_pred = len(topk)
            precision = (num_match / num_pred) if num_pred > 0 else 0.0
            recall = float(recall_info["recall"]) if recall_info["recall"] is not None else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Diagnostic readability: predictions should not include method_code.
            predictions_no_code = []
            for ctx in topk:
                ctx2 = dict(ctx)
                ctx2.pop("method_code", None)
                predictions_no_code.append(ctx2)

            metrics_by_k[f"top{k}"] = {
                "metrics": {
                    "P": precision,
                    "R": recall,
                    "F1": f1,
                    "pred": num_pred,
                    "match": num_match,
                    "gt": num_gt,
                },
                "predictions": predictions_no_code,
            }

            agg[k]["pred"] += num_pred
            agg[k]["match"] += num_match
            agg[k]["gt"] += num_gt

        if task.dependency:
            tasks_with_dep += 1

        out_rec = {
            "idx": processed,
            "namespace": task.namespace,
            "file": abs_file,
            "signature": signature,
            "requirement_text": requirement_text,
            "dependency": task.dependency,
            "search_terms": search_terms,
            "per_term": agg_info["term_results"],
            "ranked_nodes": ranked_nodes,
            "results": metrics_by_k,
        }
        write_jsonl_line(output_jsonl, out_rec)

        processed += 1
        print(f"DEBUG: processed: {processed}")
        if sleep_s > 0:
            time.sleep(sleep_s)
        if max_tasks is not None and processed >= max_tasks:
            break

    def safe_div(a: float, b: float) -> float:
        return (a / b) if b else 0.0

    project_metrics: Dict[str, Any] = {
        "source_code_dir": source_code_dir,
        "repo_dir": repo_dir,
        "filtered_path": filtered_path,
        "graph_pkl": graph_pkl,
        "tags_json": tags_json,
        "output_jsonl": output_jsonl,
        "num_tasks": processed,
        "tasks_with_dependency": tasks_with_dep,
        "agg": {},
    }

    for k in recall_ks:
        m = agg[k]["match"]
        p = agg[k]["pred"]
        gt = agg[k]["gt"]
        P = safe_div(m, p)
        R = safe_div(m, gt)
        F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
        project_metrics["agg"][k] = {"match": m, "pred": p, "gt": gt, "P": P, "R": R, "F1": F1}

    return project_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--source_code_dir", default=SOURCE_CODE_DIR)
    p.add_argument("--repo_dir", default=REPO_DIR)
    p.add_argument("--filtered_path", default=FILTERED_PATH)
    p.add_argument("--graph_pkl", default=GRAPH_PKL)
    p.add_argument("--tags_json", default=TAGS_JSON)
    p.add_argument("--output_jsonl", default=OUTPUT_JSONL)
    p.add_argument("--backend", choices=["openai", "ollama", "mock"], default=MODEL_BACKEND_CHOICE)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--timeout_s", type=float, default=120.0)
    p.add_argument("--max_tasks", type=int, default=0)
    p.add_argument("--sleep_s", type=float, default=0.0)
    p.add_argument("--max_terms", type=int, default=5)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    max_tasks = args.max_tasks if args.max_tasks and args.max_tasks > 0 else None

    if not Path(args.filtered_path).exists():
        raise SystemExit(f"filtered_path not found: {args.filtered_path}")
    if not Path(args.graph_pkl).exists():
        raise SystemExit(f"graph_pkl not found: {args.graph_pkl}")
    if not Path(args.tags_json).exists():
        raise SystemExit(f"tags_json not found: {args.tags_json}")

    metrics = analyze_project(
        args.source_code_dir,
        repo_dir=args.repo_dir,
        filtered_path=args.filtered_path,
        graph_pkl=args.graph_pkl,
        tags_json=args.tags_json,
        output_jsonl=args.output_jsonl,
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
        timeout_s=args.timeout_s,
        max_tasks=max_tasks,
        sleep_s=args.sleep_s,
        max_terms=args.max_terms,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
