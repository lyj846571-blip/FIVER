from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..llm import GenerationConfig, ModelEndpoint, chat_completion, embed
from ..prompts import render_prompt
from .base import BaseVerifier, VerificationResult
from .common import (
    cosine,
    enqueue_output,
    extract_lean_code,
    parse_json_object,
    read_repl_json,
    send_repl_command,
    usage_tokens,
)

MATH500_CATEGORIES = [
    "Prealgebra",
    "Algebra",
    "Number Theory",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Precalculus",
]
META_MARKER = "__lean4_dual_memory_meta__"


class Lean4MemoryVerifier(BaseVerifier):
    def __init__(
        self,
        formalizer_endpoint: ModelEndpoint,
        backtranslate_endpoint: ModelEndpoint,
        prover_endpoint: ModelEndpoint,
        memory_embedding_endpoint: ModelEndpoint,
        formalizer_generation: GenerationConfig,
        backtranslate_generation: GenerationConfig,
        prover_generation: GenerationConfig,
        prompt_dir: str | Path,
        lean_repl: str,
        lean_lake_path: str,
        lean_project_root: str,
        repl_timeout_s: int,
        translator_attempts: int,
        prover_attempts: int,
        memory_top_k: int,
        inner_memory_top_k: int,
        outer_memory_top_k: int,
    ):
        self.formalizer_endpoint = formalizer_endpoint
        self.backtranslate_endpoint = backtranslate_endpoint
        self.prover_endpoint = prover_endpoint
        self.memory_embedding_endpoint = memory_embedding_endpoint
        self.formalizer_generation = formalizer_generation
        self.backtranslate_generation = backtranslate_generation
        self.prover_generation = prover_generation
        self.prompt_dir = prompt_dir
        self.lean_repl = lean_repl
        self.lean_lake_path = lean_lake_path
        self.lean_project_root = lean_project_root
        self.repl_timeout_s = repl_timeout_s
        self.translator_attempts = translator_attempts
        self.prover_attempts = prover_attempts
        self.memory_top_k = memory_top_k
        self.inner_memory_top_k = inner_memory_top_k
        self.outer_memory_top_k = outer_memory_top_k

    def verify(self, problem: str, proof_step: str, memory: List[Dict[str, Any]]) -> VerificationResult:
        total_tokens = 0
        meta = self._get_or_init_meta(memory)
        last_error = ""
        lean_statement = ""
        step_category = ""

        for _ in range(self.translator_attempts):
            retrieved = self._retrieve_verified_memory(proof_step, memory)
            translator_errors = self._retrieve_top_k(
                proof_step,
                meta["TRANSLATOR_ERROR_STORE"],
                self.inner_memory_top_k,
            )
            category_failures = self._retrieve_category_errors(
                meta,
                stage="translator",
                query_text=proof_step,
                category=step_category,
            )
            response = chat_completion(
                self.formalizer_endpoint,
                [
                    {
                        "role": "user",
                        "content": render_prompt(
                            "lean_formalize.txt",
                            self.prompt_dir,
                            problem=problem,
                            proof_step=proof_step,
                            memory=self._format_verified_memory(retrieved),
                            bad_memory=self._format_translator_error_memory(translator_errors),
                            category_memory=self._format_category_memory(category_failures, include_proof_branch=False),
                            step_category=step_category or "unclassified",
                        ),
                    }
                ],
                self.formalizer_generation,
            )
            total_tokens += usage_tokens(response)
            lean_statement = extract_lean_code(response.choices[0].message.content or "")
            if not lean_statement:
                last_error = "Formalizer returned no Lean statement."
                self._write_translator_error(meta, proof_step, "", "EMPTY_TRANSLATION", last_error, step_category)
                continue
            try:
                statement_check = self._check_lean(lean_statement)
            except Exception as exc:
                statement_check = {"ok": False, "error": str(exc)}
            if not statement_check["ok"]:
                last_error = f"Lean statement failed to compile: {statement_check['error']}"
                self._write_translator_error(
                    meta,
                    proof_step,
                    lean_statement,
                    "LEAN_STATEMENT_COMPILE_ERROR",
                    last_error,
                    step_category,
                )
                continue
            try:
                semantic_ok, category, semantic_tokens, semantic_error = self._backtranslate_check(
                    proof_step,
                    lean_statement,
                )
                total_tokens += semantic_tokens
                step_category = category
            except Exception as exc:
                last_error = str(exc)
                self._write_translator_error(
                    meta,
                    proof_step,
                    lean_statement,
                    "BACKTRANSLATE_EXCEPTION",
                    last_error,
                    step_category,
                )
                continue
            if semantic_ok:
                break
            last_error = semantic_error or f"Lean statement semantic check failed. category={category}"
            self._write_translator_error(
                meta,
                proof_step,
                lean_statement,
                "SEMANTIC_MISMATCH",
                last_error,
                step_category,
            )
        else:
            return VerificationResult("UNVERIFIED", f"UNVERIFIED: {last_error}", tokens=total_tokens)

        for _ in range(self.prover_attempts):
            retrieved = self._retrieve_verified_memory(proof_step, memory)
            prover_errors_goal = self._retrieve_prover_errors(
                meta,
                proof_step,
                lean_statement,
                proof_branch="goal",
            )
            prover_errors_neg = self._retrieve_prover_errors(
                meta,
                proof_step,
                lean_statement,
                proof_branch="neg",
            )
            category_failures_goal = self._retrieve_category_errors(
                meta,
                stage="prover",
                query_text=f"{proof_step}\n{lean_statement}",
                category=step_category,
                proof_branch="goal",
            )
            category_failures_neg = self._retrieve_category_errors(
                meta,
                stage="prover",
                query_text=f"{proof_step}\n{lean_statement}",
                category=step_category,
                proof_branch="neg",
            )
            goal_response = chat_completion(
                self.prover_endpoint,
                [
                    {
                        "role": "user",
                        "content": render_prompt(
                            "lean_prove.txt",
                            self.prompt_dir,
                            proof_step=proof_step,
                            lean_statement=lean_statement,
                            memory=self._format_verified_memory(retrieved),
                            prover_error_memory=self._format_prover_error_memory(prover_errors_goal),
                            category_memory=self._format_category_memory(category_failures_goal, include_proof_branch=True),
                            step_category=step_category or "unclassified",
                        ),
                    }
                ],
                self.prover_generation,
            )
            total_tokens += usage_tokens(goal_response)
            goal_proof = extract_lean_code(goal_response.choices[0].message.content or "")
            if goal_proof:
                try:
                    goal_check = self._check_lean(goal_proof)
                except Exception as exc:
                    goal_check = {"ok": False, "error": str(exc)}
                if goal_check["ok"]:
                    memory_item = self._memory_item(proof_step, lean_statement, goal_proof)
                    return VerificationResult(
                        "CORRECT",
                        "CORRECT: Lean proof compiled successfully.",
                        tokens=total_tokens,
                        extra={
                            "lean_statement": lean_statement,
                            "lean_proof": goal_proof,
                            "memory_item": memory_item,
                        },
                    )
                last_error = f"Goal proof failed: {goal_check['error']}"
                self._write_prover_error(
                    meta,
                    proof_step,
                    lean_statement,
                    goal_proof,
                    "",
                    "GOAL_PROOF_COMPILE_ERROR",
                    last_error,
                    "goal",
                    step_category,
                )
            else:
                last_error = "Goal prover returned no Lean proof."
                self._write_prover_error(
                    meta,
                    proof_step,
                    lean_statement,
                    "",
                    "",
                    "EMPTY_GOAL_PROOF",
                    last_error,
                    "goal",
                    step_category,
                )

            neg_response = chat_completion(
                self.prover_endpoint,
                [
                    {
                        "role": "user",
                        "content": render_prompt(
                            "lean_disprove.txt",
                            self.prompt_dir,
                            proof_step=proof_step,
                            lean_statement=lean_statement,
                            memory=self._format_verified_memory(retrieved),
                            prover_error_memory=self._format_prover_error_memory(prover_errors_neg),
                            category_memory=self._format_category_memory(category_failures_neg, include_proof_branch=True),
                            step_category=step_category or "unclassified",
                        ),
                    }
                ],
                self.prover_generation,
            )
            total_tokens += usage_tokens(neg_response)
            neg_proof = extract_lean_code(neg_response.choices[0].message.content or "")
            if neg_proof:
                try:
                    neg_check = self._check_lean(neg_proof)
                except Exception as exc:
                    neg_check = {"ok": False, "error": str(exc)}
                if neg_check["ok"]:
                    return VerificationResult(
                        "INCORRECT",
                        "INCORRECT: Lean proof of the negation compiled successfully.",
                        tokens=total_tokens,
                        extra={"lean_statement": lean_statement, "lean_neg_proof": neg_proof},
                    )
                last_error = f"Negated proof failed: {neg_check['error']}"
                self._write_prover_error(
                    meta,
                    proof_step,
                    lean_statement,
                    "",
                    neg_proof,
                    "NEG_PROOF_COMPILE_ERROR",
                    last_error,
                    "neg",
                    step_category,
                )
            else:
                last_error = "Negation prover returned no Lean proof."
                self._write_prover_error(
                    meta,
                    proof_step,
                    lean_statement,
                    "",
                    "",
                    "EMPTY_NEG_PROOF",
                    last_error,
                    "neg",
                    step_category,
                )

        return VerificationResult("UNVERIFIED", f"UNVERIFIED: {last_error}", tokens=total_tokens)

    def _backtranslate_check(self, proof_step: str, lean_statement: str) -> Tuple[bool, str, int, str]:
        response = chat_completion(
            self.backtranslate_endpoint,
            [
                {
                    "role": "user",
                    "content": render_prompt(
                        "lean_backtranslate.txt",
                        self.prompt_dir,
                        proof_step=proof_step,
                        lean_statement=lean_statement,
                        allowed_categories=", ".join(MATH500_CATEGORIES),
                    ),
                }
            ],
            self.backtranslate_generation,
        )
        parsed = parse_json_object(response.choices[0].message.content or "")
        same_meaning = str(parsed.get("same_meaning") or "").strip().lower()
        category = str(parsed.get("step_category") or "").strip()
        if same_meaning not in {"yes", "no"}:
            raise ValueError("Backtranslation JSON missing same_meaning=yes|no.")
        if category not in MATH500_CATEGORIES:
            raise ValueError(f"Backtranslation JSON step_category must be one of {MATH500_CATEGORIES}, got {category!r}.")
        return same_meaning == "yes", category, usage_tokens(response), ""

    def _check_lean(self, code: str) -> Dict[str, Any]:
        process = subprocess.Popen(
            [self.lean_lake_path, "env", self.lean_repl],
            cwd=self.lean_project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        try:
            if process.stdin is None or process.stdout is None or process.stderr is None:
                raise RuntimeError("Lean REPL process did not expose stdin/stdout/stderr.")
            stdout_queue: "queue.Queue[str]" = queue.Queue()
            stderr_queue: "queue.Queue[str]" = queue.Queue()
            threading.Thread(target=enqueue_output, args=(process.stdout, stdout_queue), daemon=True).start()
            threading.Thread(target=enqueue_output, args=(process.stderr, stderr_queue), daemon=True).start()

            send_repl_command(process, {"cmd": "import Mathlib"})
            init_response = read_repl_json(process, stdout_queue, stderr_queue, self.repl_timeout_s)
            env = init_response.get("env")
            if env is None:
                return {"ok": False, "error": "Lean REPL did not return an environment id."}

            send_repl_command(process, {"cmd": code, "env": env})
            response = read_repl_json(process, stdout_queue, stderr_queue, self.repl_timeout_s)
            messages = response.get("messages", []) if isinstance(response, dict) else []
            errors = [str(item.get("data") or item.get("message") or item) for item in messages if item.get("severity") == "error"]
            return {"ok": len(errors) == 0, "error": "\n".join(errors)}
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()

    def _retrieve_verified_memory(self, proof_step: str, memory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        proved = [
            item
            for item in memory
            if item.get("status") in {"PROVED", "CORRECT"}
            and item.get("embedding") is not None
            and item.get("marker") != META_MARKER
        ]
        if not proved:
            return []
        return self._retrieve_top_k(proof_step, proved, self.memory_top_k)

    def _memory_item(self, proof_step: str, lean_statement: str, lean_proof: str) -> Dict[str, Any]:
        memory_text = f"{proof_step}\n{lean_statement}\n{lean_proof}"
        return {
            "step_nl": proof_step,
            "lean_statement": lean_statement,
            "lean_proof": lean_proof,
            "status": "PROVED",
            "embedding": embed(self.memory_embedding_endpoint, memory_text),
        }

    def _get_or_init_meta(self, memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        for item in memory:
            if item.get("marker") == META_MARKER:
                item.setdefault("TRANSLATOR_ERROR_STORE", [])
                item.setdefault("PROVER_ERROR_STORE", [])
                item.setdefault("CATEGORY_ERROR_STORE", [])
                return item
        meta = {
            "marker": META_MARKER,
            "TRANSLATOR_ERROR_STORE": [],
            "PROVER_ERROR_STORE": [],
            "CATEGORY_ERROR_STORE": [],
        }
        memory.append(meta)
        return meta

    def _retrieve_top_k(self, query_text: str, store: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not query_text or not store or top_k <= 0:
            return []
        query_embedding = embed(self.memory_embedding_endpoint, query_text)
        scored = []
        for item in store:
            item_embedding = item.get("embedding")
            if item_embedding is None:
                continue
            scored.append((cosine(query_embedding, item_embedding), item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _score, item in scored[:top_k]]

    def _retrieve_category_errors(
        self,
        meta: Dict[str, Any],
        stage: str,
        query_text: str,
        category: str,
        proof_branch: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not category:
            return []
        scoped = []
        for item in meta["CATEGORY_ERROR_STORE"]:
            if item.get("category") != category:
                continue
            if item.get("stage") not in {stage, f"{stage}_final"}:
                continue
            if proof_branch is not None and item.get("proof_branch", "") != proof_branch:
                continue
            scoped.append(item)
        return self._retrieve_top_k(query_text, scoped, self.outer_memory_top_k)

    def _retrieve_prover_errors(
        self,
        meta: Dict[str, Any],
        proof_step: str,
        lean_statement: str,
        proof_branch: str,
    ) -> List[Dict[str, Any]]:
        query = f"{proof_step}\n{lean_statement}".strip()
        scoped = [item for item in meta["PROVER_ERROR_STORE"] if item.get("proof_branch") == proof_branch]
        return self._retrieve_top_k(query, scoped, self.inner_memory_top_k)

    def _write_translator_error(
        self,
        meta: Dict[str, Any],
        proof_step: str,
        wrong_lean_statement: str,
        error_type: str,
        error_msg: str,
        category: str,
    ) -> None:
        if not proof_step or not error_msg:
            return
        memory_text = f"{proof_step}\n{wrong_lean_statement}\n{error_type}\n{error_msg}".strip()
        item = {
            "step_nl": proof_step,
            "wrong_lean_statement": wrong_lean_statement,
            "error_type": error_type,
            "error_msg": error_msg,
            "embedding": embed(self.memory_embedding_endpoint, memory_text),
        }
        self._dedup_append(meta["TRANSLATOR_ERROR_STORE"], item, ("step_nl", "wrong_lean_statement", "error_type", "error_msg"))
        self._write_category_error(meta, proof_step, "translator", error_type, error_msg, category)

    def _write_prover_error(
        self,
        meta: Dict[str, Any],
        proof_step: str,
        lean_statement: str,
        lean_proof_goal: str,
        lean_proof_neg: str,
        error_type: str,
        error_msg: str,
        proof_branch: str,
        category: str,
    ) -> None:
        if not proof_step or not error_msg:
            return
        memory_text = (
            f"{proof_branch}\n{proof_step}\n{lean_statement}\n{lean_proof_goal}\n"
            f"{lean_proof_neg}\n{error_type}\n{error_msg}"
        ).strip()
        item = {
            "proof_branch": proof_branch,
            "step_nl": proof_step,
            "lean_statement": lean_statement,
            "lean_proof_goal": lean_proof_goal,
            "lean_proof_neg_goal": lean_proof_neg,
            "error_type": error_type,
            "error_msg": error_msg,
            "embedding": embed(self.memory_embedding_endpoint, memory_text),
        }
        self._dedup_append(meta["PROVER_ERROR_STORE"], item, ("proof_branch", "step_nl", "lean_statement", "error_type", "error_msg"))
        self._write_category_error(meta, proof_step, "prover", error_type, error_msg, category, proof_branch)

    def _write_category_error(
        self,
        meta: Dict[str, Any],
        proof_step: str,
        stage: str,
        error_type: str,
        error_msg: str,
        category: str,
        proof_branch: str = "",
    ) -> None:
        if not category or not proof_step or not error_msg:
            return
        memory_text = f"{category}\n{stage}\n{proof_branch}\n{proof_step}\n{error_type}\n{error_msg}".strip()
        item = {
            "category": category,
            "stage": stage,
            "proof_branch": proof_branch,
            "step_nl": proof_step,
            "error_type": error_type,
            "error_msg": error_msg,
            "hint": self._error_hint(error_type, error_msg),
            "embedding": embed(self.memory_embedding_endpoint, memory_text),
        }
        self._dedup_append(
            meta["CATEGORY_ERROR_STORE"],
            item,
            ("category", "stage", "proof_branch", "step_nl", "error_type", "error_msg"),
        )

    @staticmethod
    def _dedup_append(store: List[Dict[str, Any]], item: Dict[str, Any], keys: Tuple[str, ...]) -> None:
        for old in store:
            if all(old.get(key) == item.get(key) for key in keys):
                return
        store.append(item)

    @staticmethod
    def _format_verified_memory(memory: List[Dict[str, Any]]) -> str:
        if not memory:
            return "(none)"
        parts = []
        for index, item in enumerate(memory, 1):
            parts.append(
                f"[Memory {index}]\n"
                f"step_nl: {item.get('step_nl', '')}\n"
                f"lean_statement: {item.get('lean_statement', '')}\n"
                f"lean_proof: {item.get('lean_proof', '')}\n"
                f"status: {item.get('status', '')}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_translator_error_memory(memory: List[Dict[str, Any]]) -> str:
        if not memory:
            return "(none)"
        parts = []
        for index, item in enumerate(memory, 1):
            parts.append(
                f"[Translator Error Memory {index}]\n"
                f"step_nl: {item.get('step_nl', '')}\n"
                f"wrong_lean_statement: {item.get('wrong_lean_statement', '')}\n"
                f"error_type: {item.get('error_type', '')}\n"
                f"error_msg: {item.get('error_msg', '')}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_prover_error_memory(memory: List[Dict[str, Any]]) -> str:
        if not memory:
            return "(none)"
        parts = []
        for index, item in enumerate(memory, 1):
            parts.append(
                f"[Prover Error Memory {index}]\n"
                f"proof_branch: {item.get('proof_branch', '')}\n"
                f"step_nl: {item.get('step_nl', '')}\n"
                f"lean_statement: {item.get('lean_statement', '')}\n"
                f"lean_proof_goal: {item.get('lean_proof_goal', '')}\n"
                f"lean_proof_neg_goal: {item.get('lean_proof_neg_goal', '')}\n"
                f"error_type: {item.get('error_type', '')}\n"
                f"error_msg: {item.get('error_msg', '')}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_category_memory(memory: List[Dict[str, Any]], include_proof_branch: bool) -> str:
        if not memory:
            return "(none)"
        parts = []
        for index, item in enumerate(memory, 1):
            proof_branch = f"proof_branch: {item.get('proof_branch', '')}\n" if include_proof_branch else ""
            parts.append(
                f"[Category Failure Reminder {index}]\n"
                f"category: {item.get('category', '')}\n"
                f"stage: {item.get('stage', '')}\n"
                f"{proof_branch}"
                f"past_step: {item.get('step_nl', '')}\n"
                f"error_type: {item.get('error_type', '')}\n"
                f"error_msg: {item.get('error_msg', '')}\n"
                f"hint: {item.get('hint', '')}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _error_hint(error_type: str, error_msg: str) -> str:
        text = f"{error_type} {error_msg}".lower()
        if "semantic" in text or "meaning" in text:
            return "Check that the Lean statement preserves the natural-language meaning before proving."
        if "timeout" in text:
            return "Prefer shorter Lean proofs and avoid tactics that may search too broadly."
        if "unknown" in text or "failed" in text or "compile" in text:
            return "Avoid repeating the same Lean syntax or proof pattern that produced this error."
        return "Use this previous failure as a negative example."
