from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset


TOOL_ALIASES = {
    "lean4": "lean4",
    "deepseek": "deepseek",
}


@dataclass
class RouterSample:
    sample_id: str
    question_id: str
    route_text: str
    problem: str
    step_text: str
    tool_results: Dict[str, Dict[str, float]]


class RouterDataset(Dataset):
    def __init__(
        self,
        samples: List[RouterSample],
        tool_names: List[str],
        embeddings: np.ndarray,
        latency_norm_stats: Dict[str, Dict[str, float]],
    ):
        self.samples = samples
        self.tool_names = tool_names
        self.embeddings = embeddings
        self.latency_norm_stats = latency_norm_stats

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        success = []
        cost_target = []
        latency_mask = []
        for tool_name in self.tool_names:
            result = sample.tool_results.get(tool_name)
            if result is None:
                success.append(0.0)
                cost_target.append(0.0)
                latency_mask.append(0.0)
                continue
            latency = max(float(result.get("latency_s", 0.0)), 1e-6)
            success.append(float(result.get("success", 0.0)))
            cost_target.append(latency_to_cost_target(latency, self.latency_norm_stats))
            latency_mask.append(1.0)
        return {
            "embedding": torch.tensor(self.embeddings[idx], dtype=torch.float32),
            "success": torch.tensor(success, dtype=torch.float32),
            "cost_target": torch.tensor(cost_target, dtype=torch.float32),
            "latency_mask": torch.tensor(latency_mask, dtype=torch.float32),
        }


def load_router_jsonl(paths: Iterable[str | Path], tool_names: List[str]) -> List[RouterSample]:
    allowed_tools = set(tool_names)
    grouped: Dict[str, RouterSample] = {}
    skipped = 0
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                sample_id = _required_text(record, "sid")
                route_text = _required_text(record, "step_text")
                tool_name = _normalize_tool_name(record.get("tool_name"))
                if not sample_id or not route_text or tool_name not in allowed_tools:
                    skipped += 1
                    continue
                if sample_id not in grouped:
                    grouped[sample_id] = RouterSample(
                        sample_id=sample_id,
                        question_id=str(record.get("question_id") or record.get("qid") or ""),
                        route_text=route_text,
                        problem=str(record.get("problem") or record.get("question") or ""),
                        step_text=str(record.get("step_text") or record.get("candidate_step_nl") or ""),
                        tool_results={},
                    )
                success = _to_int01(_required_value(record, "success"))
                latency = _positive_float(_required_value(record, "latency_s"), "latency_s")
                existing = grouped[sample_id].tool_results.get(tool_name)
                if existing is None or success > int(existing["success"]) or (
                    success == int(existing["success"]) and latency < float(existing["latency_s"])
                ):
                    grouped[sample_id].tool_results[tool_name] = {
                        "success": float(success),
                        "latency_s": float(latency),
                    }
    samples = list(grouped.values())
    if not samples:
        raise RuntimeError(f"No usable router samples were loaded. Skipped records: {skipped}")
    return samples


def build_latency_norm_stats(samples: List[RouterSample], tool_names: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    global_values: List[float] = []
    for tool_name in tool_names:
        values = []
        for sample in samples:
            result = sample.tool_results.get(tool_name)
            if result is None:
                continue
            latency = float(result.get("latency_s", 0.0) or 0.0)
            if latency > 0:
                value = float(np.log1p(latency))
                values.append(value)
                global_values.append(value)
        stats[tool_name] = _log_stats(values)
    stats["global"] = _log_stats(global_values)
    return stats


def latency_to_cost_target(latency: float, latency_norm_stats: Dict[str, Dict[str, float]]) -> float:
    value = np.log1p(max(float(latency), 1e-6))
    global_stats = latency_norm_stats["global"]
    normalized = (value - global_stats["min_log_latency"]) / global_stats["log_range"]
    return float(np.clip(normalized, 0.0, 1.0))


def samples_to_metadata(samples: List[RouterSample]) -> Dict[str, Any]:
    tool_counts: Dict[str, int] = {}
    for sample in samples:
        for tool_name in sample.tool_results:
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
    return {
        "samples": len(samples),
        "tool_result_counts": tool_counts,
    }


def _log_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "min_log_latency": 0.0,
            "max_log_latency": 1.0,
            "log_range": 1.0,
            "avg_latency": 1.0,
        }
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    return {
        "min_log_latency": min_value,
        "max_log_latency": max_value,
        "log_range": max(max_value - min_value, 1e-6),
        "avg_latency": float(np.mean(np.expm1(values))),
    }


def _required_value(record: Dict[str, Any], key: str) -> Any:
    if key not in record or record[key] is None:
        raise KeyError(f"Router training JSONL row is missing required field: {key}")
    return record[key]


def _required_text(record: Dict[str, Any], key: str) -> str:
    value = str(_required_value(record, key)).strip()
    if not value:
        raise ValueError(f"Router training JSONL field is empty: {key}")
    return value


def _normalize_tool_name(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return TOOL_ALIASES.get(raw, raw)


def _to_int01(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "ok", "correct"}:
        return 1
    if text in {"0", "false", "no", "n", "incorrect", "unverified"}:
        return 0
    raise ValueError(f"Cannot parse success as 0/1: {value!r}")


def _positive_float(value: Any, key: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"Cannot parse {key} as float: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{key} must be positive, got {parsed}.")
    return parsed
