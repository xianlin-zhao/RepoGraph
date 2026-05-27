import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple
import pandas as pd

from utils.llm_clients import BackendName, make_client
from utils.completion_postprocess import (
    extract_code_from_markdown,
    keep_only_completion,
    preview_text,
)
from utils.dev_eval_task import DevEvalTask, parse_task
from utils.jsonl_io import iter_jsonl, write_jsonl_line
from utils.source_code_utils import resolve_signature, read_line_range, get_class_skeleton
from utils.task_recall import compute_task_recall


SOURCE_CODE_DIR = "/data/lowcode_public/DevEval_no_targetMehod/Source_Code"
FILTERED_PATH = "/data/zxl/Search2026/outputData/devEvalSearchOut/0316_batch_workflow/mingus/filtered.jsonl"
DIAGNOSTIC_JSONL = "/data/zxl/Search2026/outputData/devEvalRepoGraph/mingus/diagnostic_graph_context_gpt.jsonl"
OUTPUT_COMPLETION_PATH = "/data/zxl/Search2026/outputData/devEvalCompletionOut/0405_repograph/mingus/repograph_completion.jsonl"

# 代码生成使用的大模型
MODEL_NAME = "gpt-5-mini"
MODEL_BACKEND_CHOICE = "openai"

DEBUG = True  # 是否打印调试信息到控制台
DEBUG_LOG_FULL = True  # DEBUG 为 True 时，是否将完整 prompt、补全结果等写入日志文件
GENERATION_FLAG = True  # 是否做代码生成，默认True，如果只是统计context recall，则设置为False

PROMPT_TEMPLATE = (
    "Please complete the function in the given Python code"
    "located at the end of the instuction based on relevant repository information.\n\n"
    "Constraints:\n"
    "- Output only the completion that should follow the given signature!\n"
    "- Do not repeat the signature!\n"
    "- Do not repeat the requirement comment!\n"
    "- You can reference the code fragments from the repo to help you complete the function!\n\n"
    "Here are some relevant code fragments from the repo:\n"
    "{{context_code_in_prompt}}\n\n\n\n"
    "Input Code (You should complete):\n"
    "```Python\n"
    "{{signature}}\n\n"
    "{{requirement_comment}}\n\n"
    "```\n\n"
    "Completed Code:\n"
)


# 读入之前搜索的结果（包含是否match等指标）
def load_diagnostic_result(diagnostic_jsonl: str) -> list[Dict[str, Any]]:
    print("Loading diagnostic_jsonl...")
    diag_records = []
    
    with open(diagnostic_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                diag_records.append(json.loads(line))
    
    return diag_records


# 得到相应任务的搜索结果，作为之后给LLM的context
def get_searched_context_code(
    task: DevEvalTask, diag_record: Dict[str, Any]
) -> list[Dict[str, Any]]:
    task_namespace = task.namespace
    preds = diag_record["results"]["top15"]["predictions"]
    context_code_list = []

    for pred in preds:
        node = pred['node']
        category = node['category']
        line = node['line']
        fname = node['fname']
        method_signature = pred['sig']
        if isinstance(fname, str) and fname and isinstance(line, (list, tuple)) and len(line) == 2:
            if category == 'function' or node['kind'] == 'ref':
                start_line = int(line[0])
                end_line = int(line[1])
                method_code = read_line_range(fname, start_line, end_line)
            elif category == 'class' and node['kind'] == 'def':
                method_code = get_class_skeleton(fname, method_signature)
            context_code_list.append({
                'method_signature': method_signature,
                'func_file': fname,
                'method_code': method_code,
            })
        else:
            continue
    
    return context_code_list


# 将搜索到的代码片段拼接起来，作为prompt中的context
def assemble_context_code_into_prompt(context_code_list: list[Dict[str, Any]]) -> str:
    context_code_in_prompt = ""
    max_len = 3000
    for context_code in context_code_list:
        method_code = str(context_code.get("method_code", ""))
        if len(method_code) > max_len:
            method_code = method_code[:max_len] + "..."
        context_code_in_prompt += (
            f"{context_code.get('func_file', '')}\n"
            f"{method_code}\n\n"
        )
    return context_code_in_prompt


# 将需求文本格式化为多行注释，用于拼接在函数签名下面
def format_requirement_as_comment(requirement_text: str) -> str:
    if not requirement_text:
        return ""
    lines = requirement_text.splitlines()
    indent_str = "    "  # 每个缩进4个空格
    delimiter = '"""'  # 多行注释用"""表示
    escaped_delimiter = '\\"\\"\\"'
    lines = [line.replace(delimiter, escaped_delimiter) for line in lines]

    content = "\n".join(indent_str + line if line else indent_str for line in lines)
    return f"{indent_str}{delimiter}\n{content}\n{indent_str}{delimiter}\n"


def build_prompt(signature: str, requirement_comment: str, context_code_in_prompt: str) -> str:
    return (
        PROMPT_TEMPLATE.replace("{{signature}}", signature.rstrip("\n"))
        .replace("{{requirement_comment}}", requirement_comment)
        .replace("{{context_code_in_prompt}}", context_code_in_prompt)
    )


def generate_completions(
    *,
    filtered_path: str,
    source_code_dir: str,
    diagnostic_jsonl: str,
    output_jsonl: str,
    debug_log_path_override: Optional[str] = None,
    backend: BackendName,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    timeout_s: float,
    max_tasks: Optional[int],
    sleep_s: float,
) -> None:

    # 清空输出结果的jsonl文件
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

    diag_records = load_diagnostic_result(diagnostic_jsonl)

    debug_log_path = None
    if DEBUG and DEBUG_LOG_FULL:
        debug_log_path = debug_log_path_override or (os.path.splitext(output_jsonl)[0] + "_debug.log")
        with open(debug_log_path, "w", encoding="utf-8") as _:
            pass

    processed = 0
    recall_sum = 0.0
    recall_count = 0
    recall_none_count = 0
    for record in iter_jsonl(filtered_path):
        task = parse_task(record)

        diag_record = diag_records[processed]

        abs_file, signature = resolve_signature(
            source_code_dir, task.completion_path, task.signature_position
        )

        # 获取该任务对应的代码搜索结果
        searched_context_code_list = get_searched_context_code(task, diag_record)
        print(f"code len: {len(searched_context_code_list)}")
        context_code_in_prompt = assemble_context_code_into_prompt(searched_context_code_list)

        # 计算该任务的context recall
        recall_info = compute_task_recall(task.dependency, searched_context_code_list)
        if recall_info["recall"] is None:
            recall_none_count += 1
        else:
            recall_sum += float(recall_info["recall"])
            recall_count += 1

        requirement_comment = format_requirement_as_comment(task.requirement_text)
        prompt = build_prompt(signature=signature, requirement_comment=requirement_comment,
                                context_code_in_prompt=context_code_in_prompt)

        if DEBUG:
            print(f"[debug] namespace={task.namespace}", file=sys.stderr)
            print(f"[debug] file={abs_file}", file=sys.stderr)
            print(f"[debug] signature_position={task.signature_position}", file=sys.stderr)
            print("[debug] signature:\n" + preview_text(signature), file=sys.stderr)
            print("[debug] requirement_comment:\n" + preview_text(requirement_comment), file=sys.stderr)
            print("[debug] prompt:\n" + preview_text(prompt), file=sys.stderr)
            if DEBUG_LOG_FULL and debug_log_path:
                sep = "=" * 80
                with open(debug_log_path, "a", encoding="utf-8") as logf:
                    logf.write(f"\n{sep}\n")
                    logf.write(f"Task idx={processed}  namespace={task.namespace}\n")
                    logf.write(f"file={abs_file}\n")
                    logf.write(f"{sep}\n\n")
                    logf.write("--- Full prompt (complete) ---\n\n")
                    logf.write(prompt)
                    logf.write("\n\n")
                    logf.write("--- Context only (context_code_in_prompt) ---\n\n")
                    logf.write(context_code_in_prompt)
                    logf.write("\n\n")
                    logf.flush()

        if GENERATION_FLAG:
            raw_completion = client.generate(prompt)
            extracted_completion = extract_code_from_markdown(raw_completion)
            completion = keep_only_completion(
                extracted_completion,
                signature=signature,
                requirement_comment=requirement_comment,
                requirement_text=task.requirement_text,
            )

            if DEBUG:
                print("[debug] raw_completion:\n" + preview_text(raw_completion), file=sys.stderr)
                print("[debug] final_completion:\n" + preview_text(completion), file=sys.stderr)
                if DEBUG_LOG_FULL and debug_log_path:
                    with open(debug_log_path, "a", encoding="utf-8") as logf:
                        logf.write("--- Raw completion from LLM ---\n\n")
                        logf.write(raw_completion)
                        logf.write("\n\n--- Final completion (after postprocess) ---\n\n")
                        logf.write(completion)
                        logf.write("\n\n")
                        logf.flush()
        else:
            completion = ""

        if GENERATION_FLAG:
            write_jsonl_line(output_jsonl, {
                "namespace": task.namespace,
                "completion": completion,
                "idx": processed,
                "dependency": task.dependency,
                "recall": recall_info,
            })

        processed += 1
        if sleep_s > 0:
            time.sleep(sleep_s)
        if max_tasks is not None and processed >= max_tasks:
            break
    
    mean_recall = (recall_sum / recall_count) if recall_count > 0 else None
    print(
        json.dumps(
            {
                "recall_mean": mean_recall,
                "tasks_with_dependency": recall_count,
                "tasks_without_dependency": recall_none_count,
                "tasks_total": processed,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--filtered_path", default=FILTERED_PATH)
    p.add_argument("--source_code_dir", default=SOURCE_CODE_DIR)
    p.add_argument("--diagnostic_jsonl", default=DIAGNOSTIC_JSONL, help="检索结果 jsonl 路径")
    p.add_argument("--output", default=OUTPUT_COMPLETION_PATH)
    p.add_argument("--debug_log", default="", help="DEBUG 且 DEBUG_LOG_FULL 时的全量日志路径，默认 <output>_debug.log")
    p.add_argument("--backend", choices=["openai", "ollama", "mock"], default=MODEL_BACKEND_CHOICE)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--timeout_s", type=float, default=120.0)
    p.add_argument("--max_tasks", type=int, default=0)
    p.add_argument("--sleep_s", type=float, default=0.0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    max_tasks = args.max_tasks if args.max_tasks and args.max_tasks > 0 else None
    generate_completions(
        filtered_path=args.filtered_path,
        source_code_dir=args.source_code_dir,
        diagnostic_jsonl=args.diagnostic_jsonl,
        output_jsonl=args.output,
        debug_log_path_override=(args.debug_log.strip() or None),
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
        timeout_s=args.timeout_s,
        max_tasks=max_tasks,
        sleep_s=args.sleep_s,
    )


if __name__ == "__main__":
    main()
