import re


def extract_code_from_markdown(text: str) -> str:
    if not text:
        return ""
    pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n([\s\S]*?)\n?```", re.MULTILINE)
    matches = pattern.findall(text)
    if matches:
        candidate = max(matches, key=lambda s: len(s.strip()))
        return candidate.strip("\n")
    return text.strip("\n")


# 打印内容，方便调试
def preview_text(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, total_chars={len(text)})"


def remove_triple_quoted_block_containing_anchor(text: str, anchor: str) -> str:
    if not text or not anchor:
        return text
    out = text
    while True:
        anchor_idx = out.find(anchor)
        if anchor_idx == -1:
            break
        start_sq = out.rfind("'''", 0, anchor_idx)
        start_dq = out.rfind('"""', 0, anchor_idx)
        start = max(start_sq, start_dq)
        if start == -1:
            break
        delimiter = out[start : start + 3]
        end = out.find(delimiter, anchor_idx)
        if end == -1:
            break
        out = out[:start] + out[end + 3 :]
    return out


# 从大模型返回的结果中，提取出补全的代码，去掉可能的函数签名、多行注释
def keep_only_completion(
    completion_text: str,
    *,
    signature: str,
    requirement_comment: str,
    requirement_text: str,
) -> str:
    if not completion_text:
        return ""

    text = completion_text.replace("\r\n", "\n").replace("\r", "\n")

    # 如果补全结果里直接就包含需求注释，删掉
    req_comment_norm = (
        (requirement_comment or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    )
    if req_comment_norm:
        text = text.replace(req_comment_norm, "")

    # 删除可能的多行注释
    req_text_norm = (requirement_text or "").strip()
    if req_text_norm:
        anchor = req_text_norm[:30]
        text = remove_triple_quoted_block_containing_anchor(text, anchor)

    # 删除可能的函数签名
    sig_norm = (signature or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if sig_norm:
        sig_idx = text.rfind(sig_norm)
        if sig_idx != -1:
            text = text[sig_idx + len(sig_norm) :]

    return text.lstrip("\n")

