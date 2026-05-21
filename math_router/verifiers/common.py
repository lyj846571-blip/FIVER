from __future__ import annotations

import json
import math
import queue
import re
import subprocess
import time
from typing import Any, Dict, List


def usage_tokens(response: object) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    return int(getattr(usage, "total_tokens", 0) or 0)


def feedback(status: str, reason: str) -> str:
    if status == "CORRECT":
        return f"CORRECT: {reason}"
    if status == "INCORRECT":
        return f"INCORRECT: {reason}"
    return f"UNVERIFIED: {reason}"


def format_memory(memory: List[Dict[str, Any]]) -> str:
    if not memory:
        return "(none)"
    return "\n\n".join(json.dumps(item, ensure_ascii=False) for item in memory)


def strip_think(text: str) -> str:
    if not text:
        return ""
    if re.search(r"<think>", text, re.IGNORECASE) and not re.search(r"</think>", text, re.IGNORECASE):
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    raw = strip_think(text)
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Expected a JSON object, got: {raw[:300]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_lean_code(text: str) -> str:
    blocks = re.findall(r"```(?:lean4?|lean)?\s*\n(.*?)\n```", text or "", re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    return (text or "").strip()


def enqueue_output(pipe: Any, output_queue: "queue.Queue[str]") -> None:
    for line in iter(pipe.readline, ""):
        output_queue.put(line)
    pipe.close()


def send_repl_command(process: subprocess.Popen[str], payload: Dict[str, Any]) -> None:
    if process.stdin is None:
        raise RuntimeError("Lean REPL stdin is closed.")
    process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n\n")
    process.stdin.flush()


def read_repl_json(
    process: subprocess.Popen[str],
    stdout_queue: "queue.Queue[str]",
    stderr_queue: "queue.Queue[str]",
    timeout_s: int,
) -> Dict[str, Any]:
    buffer = ""
    start = time.time()
    decoder = json.JSONDecoder()
    while True:
        try:
            buffer += stdout_queue.get(timeout=0.2)
            try:
                parsed, _ = decoder.raw_decode(buffer)
                if isinstance(parsed, dict):
                    return parsed
                raise RuntimeError("Lean REPL returned non-object JSON.")
            except json.JSONDecodeError:
                pass
        except queue.Empty:
            pass
        if process.poll() is not None:
            stderr = []
            while not stderr_queue.empty():
                stderr.append(stderr_queue.get_nowait())
            raise RuntimeError("Lean REPL exited early. " + "".join(stderr))
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Lean REPL timed out after {timeout_s} seconds.")
