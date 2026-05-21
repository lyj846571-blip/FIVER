from __future__ import annotations

import re
from typing import Dict, Iterable


def strip_think(text: str) -> str:
    if not text:
        return ""
    if re.search(r"<think>", text, re.IGNORECASE) and not re.search(r"</think>", text, re.IGNORECASE):
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_final_answer(output: str) -> str:
    text = strip_think(output)
    match = re.search(r"\**\s*final\s*answer\s*\**\s*:?\s*\**\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().strip("*").strip()
    return text.strip()


def canonical_answer(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:/(?:\d+(?:\.\d+)?|\.\d+))?", value)
    return numbers[-1] if numbers else value


def summarize(records: Iterable[Dict[str, str]]) -> Dict[str, float]:
    rows = list(records)
    if not rows:
        return {"count": 0.0, "accuracy": 0.0, "avg_total_tokens": 0.0, "avg_formal_tokens": 0.0}
    correct = sum(1 for row in rows if row.get("correct") == "1")
    total_tokens = sum(float(row.get("tokens") or 0) for row in rows)
    formal_tokens = sum(float(row.get("formal_tokens") or 0) for row in rows)
    count = len(rows)
    return {
        "count": float(count),
        "accuracy": correct / count,
        "avg_total_tokens": total_tokens / count,
        "avg_formal_tokens": formal_tokens / count,
    }
