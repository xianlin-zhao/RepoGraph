import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from context_search import MODEL_BACKEND_CHOICE, MODEL_NAME, analyze_project


SOURCE_CODE_DIR = "/data/lowcode_public/DevEval_no_targetMehod/Source_Code"
DEFAULT_EXCEL_PATH = "/data/zxl/Search2026/RepoGraph/docs/0405_5projects_repograph.xlsx"

# Per-project input/output path pattern:
#   <base_dir>/<project_name>/<file_name>
DEFAULT_FILTERED_BASE_DIR = "/data/zxl/Search2026/outputData/devEvalSearchOut/0316_batch_workflow"
DEFAULT_GRAPH_BASE_DIR = "/data/zxl/Search2026/outputData/devEvalRepoGraph"
DEFAULT_OUTPUT_BASE_DIR = "/data/zxl/Search2026/outputData/devEvalRepoGraph"

DEFAULT_FILTERED_FILE_NAME = "filtered.jsonl"
DEFAULT_GRAPH_FILE_NAME = "graph.pkl"
DEFAULT_TAGS_FILE_NAME = "tags.json"
DEFAULT_OUTPUT_JSONL_FILE_NAME = "diagnostic_graph_context_gpt.jsonl"

REPORT_CSV_FILE_NAME = "context_search_report_gpt.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch run context_search for projects listed in Excel."
    )
    parser.add_argument("--excel_path", default=DEFAULT_EXCEL_PATH)
    parser.add_argument("--sheet_name", default=0, help="Excel sheet name or index.")
    parser.add_argument(
        "--source_code_dir",
        default=SOURCE_CODE_DIR,
        help="Fixed SOURCE_CODE_DIR used for all projects.",
    )

    parser.add_argument("--filtered_base_dir", default=DEFAULT_FILTERED_BASE_DIR)
    parser.add_argument("--graph_base_dir", default=DEFAULT_GRAPH_BASE_DIR)
    parser.add_argument("--output_base_dir", default=DEFAULT_OUTPUT_BASE_DIR)

    parser.add_argument("--filtered_file_name", default=DEFAULT_FILTERED_FILE_NAME)
    parser.add_argument("--graph_file_name", default=DEFAULT_GRAPH_FILE_NAME)
    parser.add_argument("--tags_file_name", default=DEFAULT_TAGS_FILE_NAME)
    parser.add_argument("--output_jsonl_file_name", default=DEFAULT_OUTPUT_JSONL_FILE_NAME)

    parser.add_argument(
        "--report_csv",
        default=REPORT_CSV_FILE_NAME,
        help="Path to output CSV report for per-project and overall metrics.",
    )

    parser.add_argument("--backend", choices=["openai", "ollama", "mock"], default=MODEL_BACKEND_CHOICE)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=0)
    parser.add_argument("--timeout_s", type=float, default=120.0)
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--sleep_s", type=float, default=0.0)
    parser.add_argument("--max_terms", type=int, default=5)
    return parser


def _load_projects(excel_path: str, sheet_name: str | int) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    required = {"project_name", "project_root"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Excel is missing required columns: {sorted(missing)}. "
            f"Expected columns: {sorted(required)}"
        )
    return df


def _build_project_path(base_dir: str, project_name: str, file_name: str) -> str:
    return str(Path(base_dir) / project_name / file_name)


def _to_csv_rows(project_name: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tasks = int(metrics.get("num_tasks", 0))
    token_usage = metrics.get("token_usage") or {}
    prompt_tokens = int(token_usage.get("prompt_tokens", 0))
    completion_tokens = int(token_usage.get("completion_tokens", 0))
    total_tokens = int(token_usage.get("total_tokens", 0))
    for k in (10, 15, 20):
        agg_k = (metrics.get("agg") or {}).get(k, {})
        rows.append(
            {
                "project_name": project_name,
                "k": k,
                "P": float(agg_k.get("P", 0.0)),
                "R": float(agg_k.get("R", 0.0)),
                "F1": float(agg_k.get("F1", 0.0)),
                "match": int(agg_k.get("match", 0)),
                "pred": int(agg_k.get("pred", 0)),
                "gt": int(agg_k.get("gt", 0)),
                "tasks": tasks,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )
    return rows


def _build_overall_rows(project_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    overall_rows: List[Dict[str, Any]] = []
    for k in (10, 15, 20):
        rows_k = [r for r in project_rows if int(r["k"]) == k]
        tasks_total = sum(int(r["tasks"]) for r in rows_k)
        match_total = sum(int(r["match"]) for r in rows_k)
        pred_total = sum(int(r["pred"]) for r in rows_k)
        gt_total = sum(int(r["gt"]) for r in rows_k)
        prompt_tokens_total = sum(int(r["prompt_tokens"]) for r in rows_k)
        completion_tokens_total = sum(int(r["completion_tokens"]) for r in rows_k)
        total_tokens_total = sum(int(r["total_tokens"]) for r in rows_k)

        if tasks_total > 0:
            p_weighted = sum(float(r["P"]) * int(r["tasks"]) for r in rows_k) / tasks_total
            r_weighted = sum(float(r["R"]) * int(r["tasks"]) for r in rows_k) / tasks_total
            f1_weighted = sum(float(r["F1"]) * int(r["tasks"]) for r in rows_k) / tasks_total
        else:
            p_weighted = 0.0
            r_weighted = 0.0
            f1_weighted = 0.0

        overall_rows.append(
            {
                "project_name": "ALL",
                "k": k,
                "P": p_weighted,
                "R": r_weighted,
                "F1": f1_weighted,
                "match": match_total,
                "pred": pred_total,
                "gt": gt_total,
                "tasks": tasks_total,
                "prompt_tokens": prompt_tokens_total,
                "completion_tokens": completion_tokens_total,
                "total_tokens": total_tokens_total,
            }
        )
    return overall_rows


def main() -> None:
    args = build_arg_parser().parse_args()

    source_code_dir = str(args.source_code_dir).strip()
    if not source_code_dir:
        raise ValueError("source_code_dir is empty.")
    if not Path(source_code_dir).exists():
        raise ValueError(f"source_code_dir does not exist: {source_code_dir}")
    if not Path(source_code_dir).is_dir():
        raise ValueError(f"source_code_dir is not a directory: {source_code_dir}")

    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    max_tasks = args.max_tasks if args.max_tasks and args.max_tasks > 0 else None

    df = _load_projects(args.excel_path, args.sheet_name)
    project_rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        row_index = idx + 2
        try:
            project_name = str(row.get("project_name", "")).strip()
            repo_dir = str(row.get("project_root", "")).strip()
            if not project_name:
                raise ValueError(f"Row {row_index}: project_name is empty.")
            if not repo_dir:
                raise ValueError(f"Row {row_index}: project_root is empty.")
            if not Path(repo_dir).exists():
                raise ValueError(f"Row {row_index}: project_root does not exist: {repo_dir}")
            if not Path(repo_dir).is_dir():
                raise ValueError(f"Row {row_index}: project_root is not a directory: {repo_dir}")

            filtered_path = _build_project_path(
                args.filtered_base_dir, project_name, args.filtered_file_name
            )
            graph_pkl = _build_project_path(
                args.graph_base_dir, project_name, args.graph_file_name
            )
            tags_json = _build_project_path(
                args.graph_base_dir, project_name, args.tags_file_name
            )
            output_jsonl = _build_project_path(
                args.output_base_dir, project_name, args.output_jsonl_file_name
            )

            if not Path(filtered_path).exists():
                raise ValueError(f"filtered_path not found: {filtered_path}")
            if not Path(graph_pkl).exists():
                raise ValueError(f"graph_pkl not found: {graph_pkl}")
            if not Path(tags_json).exists():
                raise ValueError(f"tags_json not found: {tags_json}")

            Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

            print("=================================")
            print(f"Running project: {project_name}")
            print(f"repo_dir:       {repo_dir}")
            print(f"filtered_path:  {filtered_path}")
            print(f"graph_pkl:      {graph_pkl}")
            print(f"tags_json:      {tags_json}")
            print(f"output_jsonl:   {output_jsonl}")
            print("=================================")

            metrics = analyze_project(
                source_code_dir,
                repo_dir=repo_dir,
                filtered_path=filtered_path,
                graph_pkl=graph_pkl,
                tags_json=tags_json,
                output_jsonl=output_jsonl,
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
            project_rows.extend(_to_csv_rows(project_name, metrics))
        except Exception as exc:
            print(f"[Failed] Row {row_index}: {exc}")

    overall_rows = _build_overall_rows(project_rows)
    all_rows = project_rows + overall_rows

    report_csv = str(args.report_csv).strip()
    if not report_csv:
        raise ValueError("report_csv is empty.")
    Path(report_csv).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "project_name",
        "k",
        "P",
        "R",
        "F1",
        "match",
        "pred",
        "gt",
        "tasks",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]
    with open(report_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("=================================")
    print(f"CSV report saved to: {report_csv}")
    print(f"Rows written: {len(all_rows)}")
    print("=================================")


if __name__ == "__main__":
    main()
