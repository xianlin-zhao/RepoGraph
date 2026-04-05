import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd

from construct_graph import CodeGraph


DEFAULT_EXCEL_PATH = "/data/zxl/Search2026/CodeContextSearch/docs/EvoCodeBench_5projects_clean.xlsx"
DEFAULT_OUTPUT_BASE_DIR = "/data/zxl/Search2026/outputData/EvoCodeBenchRepoGraph"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch construct repo graphs from an Excel config."
    )
    parser.add_argument(
        "--excel_path",
        default=DEFAULT_EXCEL_PATH,
        help="Path to Excel file containing project_name and project_root columns.",
    )
    parser.add_argument(
        "--output_base_dir",
        default=DEFAULT_OUTPUT_BASE_DIR,
        help="Base output directory. Each project's outputs go under <base>/<project_name>/.",
    )
    parser.add_argument(
        "--sheet_name",
        default=0,
        help="Excel sheet name or index. Default is first sheet.",
    )
    return parser


def _validate_project_row(row: pd.Series, row_index: int) -> tuple[str, str]:
    project_name = str(row.get("project_name", "")).strip()
    project_root = str(row.get("project_root", "")).strip()

    if not project_name:
        raise ValueError(f"Row {row_index}: project_name is empty.")
    if not project_root:
        raise ValueError(f"Row {row_index}: project_root is empty.")
    if not Path(project_root).exists():
        raise ValueError(f"Row {row_index}: project_root does not exist: {project_root}")
    if not Path(project_root).is_dir():
        raise ValueError(f"Row {row_index}: project_root is not a directory: {project_root}")
    return project_name, project_root


def construct_single_project_graph(project_name: str, project_root: str, output_base_dir: str) -> None:
    output_dir = Path(output_base_dir) / project_name
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / "graph.pkl"
    tags_path = output_dir / "tags.json"

    code_graph = CodeGraph(root=project_root)
    py_files = code_graph.find_files([project_root])
    tags, graph = code_graph.get_code_graph(py_files)

    with open(graph_path, "wb") as graph_file:
        pickle.dump(graph, graph_file)

    with open(tags_path, "w", encoding="utf-8") as tags_file:
        for tag in tags:
            tags_file.write(
                json.dumps(
                    {
                        "fname": tag.fname,
                        "rel_fname": tag.rel_fname,
                        "line": tag.line,
                        "name": tag.name,
                        "qualified_name": tag.qualified_name,
                        "kind": tag.kind,
                        "category": tag.category,
                        "info": tag.info,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("---------------------------------")
    print(f"Project: {project_name}")
    print(f"Root: {project_root}")
    print(f"Graph saved to: {graph_path}")
    print(f"Tags saved to:  {tags_path}")
    print(f"Nodes: {len(graph.nodes)} | Edges: {len(graph.edges)}")
    print("---------------------------------")


def load_projects_from_excel(excel_path: str, sheet_name: str | int) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    required_cols = {"project_name", "project_root"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Excel is missing required columns: {sorted(missing_cols)}. "
            f"Expected columns: {sorted(required_cols)}"
        )
    return df


def main() -> None:
    args = build_arg_parser().parse_args()

    excel_path = str(args.excel_path).strip()
    output_base_dir = str(args.output_base_dir).strip()

    if not excel_path:
        raise ValueError("excel_path is empty.")
    if not output_base_dir:
        raise ValueError("output_base_dir is empty.")

    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    df = load_projects_from_excel(excel_path, args.sheet_name)

    success = 0
    failed = 0

    for idx, row in df.iterrows():
        row_index = idx + 2  # Data rows in Excel start from row 2 (header is row 1)
        try:
            project_name, project_root = _validate_project_row(row, row_index)
            construct_single_project_graph(project_name, project_root, output_base_dir)
            success += 1
        except Exception as exc:
            failed += 1
            print(f"[Failed] Row {row_index}: {exc}")

    print("=================================")
    print(f"Batch finished. Success: {success}, Failed: {failed}")
    print("=================================")


if __name__ == "__main__":
    main()
