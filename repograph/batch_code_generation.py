import argparse
from pathlib import Path

import pandas as pd

from code_generation import generate_completions


SOURCE_CODE_DIR = "/data/lowcode_public/DevEval_zxl/Source_Code"
DEFAULT_EXCEL_PATH = "/data/zxl/Search2026/RepoGraph/docs/0311_5projects_repograph.xlsx"

# Each per-project path is built as:
#   <base_dir>/<project_name>/<file_name>
DEFAULT_FILTERED_BASE_DIR = "/data/zxl/Search2026/outputData/devEvalSearchOut/0316_batch_workflow"
DEFAULT_DIAGNOSTIC_BASE_DIR = "/data/zxl/Search2026/outputData/devEvalRepoGraph"
DEFAULT_OUTPUT_BASE_DIR = "/data/zxl/Search2026/outputData/devEvalCompletionOut/0318_batch_workflow"

DEFAULT_FILTERED_FILE_NAME = "filtered.jsonl"
DEFAULT_DIAGNOSTIC_FILE_NAME = "diagnostic_graph_context.jsonl"
DEFAULT_OUTPUT_FILE_NAME = "repograph_completion.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch run code generation from Excel project list."
    )
    parser.add_argument("--excel_path", default=DEFAULT_EXCEL_PATH)
    parser.add_argument("--sheet_name", default=0, help="Excel sheet name or index.")

    parser.add_argument("--filtered_base_dir", default=DEFAULT_FILTERED_BASE_DIR)
    parser.add_argument("--diagnostic_base_dir", default=DEFAULT_DIAGNOSTIC_BASE_DIR)
    parser.add_argument("--output_base_dir", default=DEFAULT_OUTPUT_BASE_DIR)

    parser.add_argument("--filtered_file_name", default=DEFAULT_FILTERED_FILE_NAME)
    parser.add_argument("--diagnostic_file_name", default=DEFAULT_DIAGNOSTIC_FILE_NAME)
    parser.add_argument("--output_file_name", default=DEFAULT_OUTPUT_FILE_NAME)
    parser.add_argument(
        "--source_code_dir",
        default=SOURCE_CODE_DIR,
        help="Fixed source code root, default aligned with code_generation.py SOURCE_CODE_DIR.",
    )

    # Generation parameters (aligned with code_generation.py)
    parser.add_argument("--backend", choices=["openai", "ollama", "mock"], default="openai")
    parser.add_argument("--model", default="deepseek-v3")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=0)
    parser.add_argument("--timeout_s", type=float, default=120.0)
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--sleep_s", type=float, default=0.0)
    parser.add_argument(
        "--debug_log_name",
        default="repograph_completion_debug.log",
        help="Per-project debug log file name under <output_base_dir>/<project_name>/",
    )
    return parser


def _validate_row(row: pd.Series, row_index: int) -> str:
    project_name = str(row.get("project_name", "")).strip()
    if not project_name:
        raise ValueError(f"Row {row_index}: project_name is empty.")
    return project_name


def _load_projects(excel_path: str, sheet_name: str | int) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    required = {"project_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Excel is missing required columns: {sorted(missing)}. "
            f"Expected columns: {sorted(required)}"
        )
    return df


def _build_project_path(base_dir: str, project_name: str, file_name: str) -> str:
    return str(Path(base_dir) / project_name / file_name)


def main() -> None:
    args = build_arg_parser().parse_args()

    source_code_dir = str(args.source_code_dir).strip()
    if not source_code_dir:
        raise ValueError("source_code_dir is empty.")
    if not Path(source_code_dir).exists():
        raise ValueError(f"source_code_dir does not exist: {source_code_dir}")
    if not Path(source_code_dir).is_dir():
        raise ValueError(f"source_code_dir is not a directory: {source_code_dir}")

    df = _load_projects(args.excel_path, args.sheet_name)
    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    max_tasks = args.max_tasks if args.max_tasks and args.max_tasks > 0 else None

    success = 0
    failed = 0

    for idx, row in df.iterrows():
        row_index = idx + 2  # Excel data starts from row 2
        try:
            project_name = _validate_row(row, row_index)

            filtered_path = _build_project_path(
                args.filtered_base_dir, project_name, args.filtered_file_name
            )
            diagnostic_jsonl = _build_project_path(
                args.diagnostic_base_dir, project_name, args.diagnostic_file_name
            )
            output_jsonl = _build_project_path(
                args.output_base_dir, project_name, args.output_file_name
            )
            debug_log_path = _build_project_path(
                args.output_base_dir, project_name, args.debug_log_name
            )

            Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

            print("=================================")
            print(f"Running project: {project_name}")
            print(f"source_code_dir: {source_code_dir}")
            print(f"filtered_path:   {filtered_path}")
            print(f"diagnostic_jsonl:{diagnostic_jsonl}")
            print(f"output_jsonl:    {output_jsonl}")
            print("=================================")

            generate_completions(
                filtered_path=filtered_path,
                source_code_dir=source_code_dir,
                diagnostic_jsonl=diagnostic_jsonl,
                output_jsonl=output_jsonl,
                debug_log_path_override=debug_log_path,
                backend=args.backend,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_tokens,
                timeout_s=args.timeout_s,
                max_tasks=max_tasks,
                sleep_s=args.sleep_s,
            )
            success += 1
        except Exception as exc:
            failed += 1
            print(f"[Failed] Row {row_index}: {exc}")

    print("=================================")
    print(f"Batch finished. Success: {success}, Failed: {failed}")
    print("=================================")


if __name__ == "__main__":
    main()
