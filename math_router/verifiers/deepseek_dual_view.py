from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..llm import GenerationConfig, ModelEndpoint, chat_completion, embed
from ..prompts import render_prompt
from .base import BaseVerifier, VerificationResult
from .common import cosine, feedback, parse_json_object, usage_tokens


class DeepSeekDualViewVerifier(BaseVerifier):
    def __init__(
        self,
        endpoint: ModelEndpoint,
        embedding_endpoint: ModelEndpoint,
        generation: GenerationConfig,
        prompt_dir: str | Path,
        samples_per_view: int,
        memory_top_k: int,
    ):
        self.endpoint = endpoint
        self.embedding_endpoint = embedding_endpoint
        self.generation = generation
        self.prompt_dir = prompt_dir
        self.samples_per_view = samples_per_view
        self.memory_top_k = memory_top_k

    def verify(self, problem: str, proof_step: str, memory: List[Dict[str, Any]]) -> VerificationResult:
        retrieved = self._retrieve_memory(proof_step, memory)
        support_runs = [
            self._call_view("deepseek_support.txt", problem, proof_step, retrieved, mode="support")
            for _ in range(self.samples_per_view)
        ]
        attack_runs = [
            self._call_view("deepseek_attack.txt", problem, proof_step, retrieved, mode="attack")
            for _ in range(self.samples_per_view)
        ]
        support_summary = _summarize_support(support_runs)
        attack_summary = _summarize_attack(attack_runs)
        status, reason = _aggregate_dual_view(support_summary, attack_summary)
        total_tokens = sum(int(item.get("tokens", 0)) for item in support_runs + attack_runs)
        extra = {
            "support_runs": support_runs,
            "attack_runs": attack_runs,
            "support_summary": support_summary,
            "attack_summary": attack_summary,
        }
        if status == "CORRECT":
            extra["memory_item"] = self._memory_item(proof_step, status="CORRECT")
        return VerificationResult(status=status, feedback=feedback(status, reason), tokens=total_tokens, extra=extra)

    def _call_view(
        self,
        prompt_name: str,
        problem: str,
        proof_step: str,
        memory: List[Dict[str, Any]],
        mode: str,
    ) -> Dict[str, Any]:
        prompt = render_prompt(
            prompt_name,
            self.prompt_dir,
            problem=problem,
            proof_step=proof_step,
            memory=self._format_memory_section(memory),
        )
        response = chat_completion(
            self.endpoint,
            [
                {"role": "system", "content": render_prompt("deepseek_system.txt", self.prompt_dir)},
                {"role": "user", "content": prompt},
            ],
            self.generation,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = parse_json_object(raw)
        normalized = _normalize_support(parsed, raw) if mode == "support" else _normalize_attack(parsed, raw)
        normalized["tokens"] = usage_tokens(response)
        return normalized

    def _retrieve_memory(self, proof_step: str, memory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not memory:
            return []
        query_embedding = embed(self.embedding_endpoint, proof_step)
        scored = []
        for item in memory:
            if item.get("status") != "CORRECT":
                continue
            item_embedding = item.get("embedding")
            if item_embedding is None:
                continue
            scored.append((cosine(query_embedding, item_embedding), item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _score, item in scored[: self.memory_top_k]]

    def _memory_item(self, proof_step: str, status: str) -> Dict[str, Any]:
        return {
            "step_nl": proof_step,
            "lean_statement": "",
            "lean_proof": "",
            "status": status,
            "embedding": embed(self.embedding_endpoint, proof_step),
        }

    @staticmethod
    def _format_memory_section(memory: List[Dict[str, Any]]) -> str:
        if not memory:
            return ""
        lines = ["Previously verified correct steps (for reference):"]
        for index, item in enumerate(memory, 1):
            lines.append(
                f"{index}. step_nl: {item.get('step_nl', '')}\n"
                f"   lean_statement: {item.get('lean_statement', '')}\n"
                f"   lean_proof: {item.get('lean_proof', '')}"
            )
        return "\n".join(lines)


def _normalize_support(parsed: Dict[str, Any], raw: str) -> Dict[str, Any]:
    return {
        "verdict": _normalize_verdict(parsed.get("verdict")),
        "confidence": _clamp(parsed.get("confidence")),
        "claim": str(parsed.get("claim") or ""),
        "assumptions_used": _string_list(parsed.get("assumptions_used")),
        "critical_checks": _string_list(parsed.get("critical_checks")),
        "remaining_risks": _string_list(parsed.get("remaining_risks")),
        "short_rationale": str(parsed.get("short_rationale") or ""),
        "raw_text": raw,
    }


def _normalize_attack(parsed: Dict[str, Any], raw: str) -> Dict[str, Any]:
    attack_type = str(parsed.get("attack_type") or "none").strip().lower()
    if attack_type not in {"domain", "logic", "algebra", "missing_assumption", "counterexample", "none"}:
        raise ValueError(f"Invalid attack_type: {attack_type}")
    return {
        "verdict": _normalize_verdict(parsed.get("verdict")),
        "confidence": _clamp(parsed.get("confidence")),
        "attack_type": attack_type,
        "critical_issue": str(parsed.get("critical_issue") or ""),
        "counterexample_attempt": str(parsed.get("counterexample_attempt") or ""),
        "remaining_risks": _string_list(parsed.get("remaining_risks")),
        "short_rationale": str(parsed.get("short_rationale") or ""),
        "raw_text": raw,
    }


def _normalize_verdict(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"valid", "true", "correct", "supported"}:
        return "valid"
    if text in {"invalid", "false", "incorrect", "unsupported"}:
        return "invalid"
    if text == "uncertain":
        return "uncertain"
    raise ValueError(f"Invalid verdict: {value!r}")


def _clamp(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"confidence must be numeric, got {value!r}") from exc
    return max(0.0, min(1.0, parsed))


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = [str(item) for item in value]
    else:
        items = [str(value)]
    return [item.strip() for item in items if item.strip() and item.strip().lower() not in {"none", "n/a", "null"}]


def _summarize_support(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = _summarize_views(outputs)
    claims = []
    risks = []
    for item in outputs:
        claim = str(item.get("claim") or "").strip()
        if claim:
            claims.append(claim)
        risks.extend(_string_list(item.get("remaining_risks")))
    summary["claims"] = sorted(set(claims))
    summary["remaining_risks"] = sorted(set(risks))
    return summary


def _summarize_attack(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = _summarize_views(outputs)
    issues = []
    risks = []
    counterexamples = []
    attack_types = []
    for item in outputs:
        if item.get("critical_issue"):
            issues.append(str(item["critical_issue"]))
        risks.extend(_string_list(item.get("remaining_risks")))
        counterexample = str(item.get("counterexample_attempt") or "").strip()
        if counterexample and counterexample.lower() not in {"none", "n/a", "null"}:
            counterexamples.append(counterexample)
        attack_type = str(item.get("attack_type") or "none").strip().lower()
        if attack_type:
            attack_types.append(attack_type)
    summary["critical_issues"] = sorted(set(issues))
    summary["remaining_risks"] = sorted(set(risks))
    summary["counterexample_attempts"] = sorted(set(counterexamples))
    summary["attack_types"] = sorted(set(attack_types))
    return summary


def _summarize_views(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"valid": 0, "invalid": 0, "uncertain": 0}
    confidence = {"valid": 0.0, "invalid": 0.0, "uncertain": 0.0}
    for item in outputs:
        verdict = _normalize_verdict(item.get("verdict"))
        counts[verdict] += 1
        confidence[verdict] += _clamp(item.get("confidence"))
    best = max(counts, key=lambda key: (counts[key], confidence[key]))
    avg_confidence = confidence[best] / max(counts[best], 1)
    return {"verdict": best, "confidence": avg_confidence, "counts": counts}


def _aggregate_dual_view(support: Dict[str, Any], attack: Dict[str, Any]) -> Tuple[str, str]:
    support_verdict = support["verdict"]
    attack_verdict = attack["verdict"]
    support_confidence = _clamp(support.get("confidence", 0.5))
    attack_confidence = _clamp(attack.get("confidence", 0.5))
    issues = attack.get("critical_issues", [])
    risks = attack.get("remaining_risks", [])
    counterexamples = attack.get("counterexample_attempts", [])
    if attack_verdict == "invalid" and attack_confidence >= 0.55 and (issues or counterexamples):
        return "INCORRECT", "Attack view found a clear mathematical issue or counterexample."
    if support_verdict == "invalid" and support_confidence >= 0.75 and attack_verdict in {"invalid", "uncertain"}:
        return "INCORRECT", "Support view could not justify the step and judged it invalid under the available information."
    if (
        support_verdict == "valid"
        and support_confidence >= 0.65
        and attack_verdict == "valid"
        and attack_confidence >= 0.50
        and not issues
        and not risks
    ):
        return "CORRECT", "Support and attack views agree that the step is justified and no unresolved risk remains."
    if support_verdict == "valid" and (issues or risks):
        return "UNVERIFIED", "Attack view exposed unresolved risks or missing conditions."
    if support_verdict != attack_verdict:
        return "UNVERIFIED", "Support and attack views disagree, so the verifier abstains."
    return "UNVERIFIED", "Dual-view verifier did not reach a stable rigorous consensus."
