from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from math_router.llm import GenerationConfig, ModelEndpoint, chat_completion
from math_router.metrics import canonical_answer, extract_final_answer, summarize
from math_router.prompts import DEFAULT_PROMPT_DIR, render_prompt
from math_router.router import RouterConfig, ToolRouter
from math_router.verifiers import DeepSeekDualViewVerifier, Lean4MemoryVerifier
from math_router.reason import WorkflowConfig, run_workflow


def main() -> None:
    args = build_parser().parse_args()
    prompt_dir = str(Path(args.prompt_dir))

    outer_generation = GenerationConfig(
        temperature=args.outer_temperature,
        top_p=args.outer_top_p,
        presence_penalty=args.outer_presence_penalty,
        frequency_penalty=args.outer_frequency_penalty,
        seed=args.seed,
        extra_body=_json_arg(args.outer_extra_body),
    )
    outer_endpoint = ModelEndpoint(args.outer_base_url, args.outer_model, args.outer_api_key_env, args.outer_api_key)
    embedding_endpoint = ModelEndpoint(
        args.embedding_base_url,
        args.embedding_model,
        args.embedding_api_key_env,
        args.embedding_api_key,
    )
    router = ToolRouter(
        RouterConfig(
            model_path=Path(args.router_model_path),
            embedding_endpoint=embedding_endpoint,
            weight_performance=args.weight_performance,
            difficulty_epsilon=args.router_difficulty_epsilon,
            cost_hidden_dim=args.router_cost_hidden_dim,
        )
    )
    verifiers = build_verifiers(args, prompt_dir)
    workflow_config = WorkflowConfig(
        outer_endpoint=outer_endpoint,
        outer_generation=outer_generation,
        prompt_dir=prompt_dir,
        max_steps=args.max_steps,
        memory_top_k=args.memory_top_k,
    )

    rows = list(load_rows(args))
    results: List[Dict[str, str]] = []
    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    for row in rows:
        final, state = run_workflow(row["problem"], workflow_config, router, verifiers)
        pred = extract_final_answer(final or "")
        record: Dict[str, str] = {
            "index": str(row.get("index", "")),
            "problem": row["problem"],
            "pred": pred,
            "pred_canonical": canonical_answer(pred),
            "tools_used": json.dumps(state.get("tool_trace", []), ensure_ascii=False),
            "tokens": str(state.get("total_tokens", 0)),
            "formal_tokens": str(state.get("formal_tokens", 0)),
            "reasoning_model_tokens": str(state.get("reasoning_model_tokens", 0)),
        }
        if row.get("answer") is not None:
            record["gt"] = str(row["answer"])
            judge = judge_answer(args, prompt_dir, row["problem"], pred, str(row["answer"]))
            record["judge"] = judge
            record["correct"] = "1" if judge.lower() == "true" else "0"
        results.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if output_path is not None:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if any("correct" in item for item in results):
        print(json.dumps(summarize(results), ensure_ascii=False))


def build_verifiers(args: argparse.Namespace, prompt_dir: str):
    generation = GenerationConfig(
        temperature=args.verifier_temperature,
        top_p=args.verifier_top_p,
        seed=args.seed,
        extra_body=_json_arg(args.verifier_extra_body),
    )
    _require_args(
        args,
        "memory-embedding-base-url",
        "memory-embedding-model",
        "deepseek-base-url",
        "deepseek-model",
        "deepseek-verifier-attempts",
        "lean-formalizer-base-url",
        "lean-formalizer-model",
        "lean-backtranslate-base-url",
        "lean-backtranslate-model",
        "lean-prover-base-url",
        "lean-prover-model",
        "lean-repl",
        "lean-lake-path",
        "lean-project-root",
        "lean-repl-timeout-s",
        "lean-translator-attempts",
        "lean-prover-attempts",
        "lean-inner-memory-top-k",
        "lean-outer-memory-top-k",
    )
    memory_embedding_endpoint = ModelEndpoint(
        args.memory_embedding_base_url,
        args.memory_embedding_model,
        args.memory_embedding_api_key_env,
        args.memory_embedding_api_key,
    )
    return {
        "deepseek": DeepSeekDualViewVerifier(
            endpoint=ModelEndpoint(args.deepseek_base_url, args.deepseek_model, args.deepseek_api_key_env, args.deepseek_api_key),
            embedding_endpoint=memory_embedding_endpoint,
            generation=generation,
            prompt_dir=prompt_dir,
            samples_per_view=args.deepseek_verifier_attempts,
            memory_top_k=args.memory_top_k,
        ),
        "lean4": Lean4MemoryVerifier(
            formalizer_endpoint=ModelEndpoint(
                args.lean_formalizer_base_url,
                args.lean_formalizer_model,
                args.lean_formalizer_api_key_env,
                args.lean_formalizer_api_key,
            ),
            backtranslate_endpoint=ModelEndpoint(
                args.lean_backtranslate_base_url,
                args.lean_backtranslate_model,
                args.lean_backtranslate_api_key_env,
                args.lean_backtranslate_api_key,
            ),
            prover_endpoint=ModelEndpoint(
                args.lean_prover_base_url,
                args.lean_prover_model,
                args.lean_prover_api_key_env,
                args.lean_prover_api_key,
            ),
            memory_embedding_endpoint=memory_embedding_endpoint,
            formalizer_generation=generation,
            backtranslate_generation=generation,
            prover_generation=generation,
            prompt_dir=prompt_dir,
            lean_repl=args.lean_repl,
            lean_lake_path=args.lean_lake_path,
            lean_project_root=args.lean_project_root,
            repl_timeout_s=args.lean_repl_timeout_s,
            translator_attempts=args.lean_translator_attempts,
            prover_attempts=args.lean_prover_attempts,
            memory_top_k=args.memory_top_k,
            inner_memory_top_k=args.lean_inner_memory_top_k,
            outer_memory_top_k=args.lean_outer_memory_top_k,
        ),
    }


def _require_args(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(args, name.replace("-", "_")) is None]
    if missing:
        rendered = ", ".join(f"--{name}" for name in missing)
        raise RuntimeError(f"Missing required backend arguments: {rendered}")


def load_rows(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.problem:
        yield {"index": "0", "problem": args.problem, "answer": args.answer}
        return
    if not args.input_jsonl:
        raise RuntimeError("Supply either --problem or --input-jsonl.")
    with Path(args.input_jsonl).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if args.limit is not None and idx >= args.limit:
                break
            item = json.loads(line)
            yield {
                "index": item.get(args.index_key, idx),
                "problem": str(item[args.problem_key]),
                "answer": item.get(args.answer_key) if args.answer_key else None,
            }


def judge_answer(args: argparse.Namespace, prompt_dir: str, problem: str, prediction: str, ground_truth: str) -> str:
    _require_args(args, "judge-base-url", "judge-model")
    endpoint = ModelEndpoint(args.judge_base_url, args.judge_model, args.judge_api_key_env, args.judge_api_key)
    prompt = render_prompt(
        "answer_judge.txt",
        prompt_dir,
        answer=prediction,
        ground_truth=ground_truth,
    )
    response = chat_completion(
        endpoint,
        [{"role": "user", "content": prompt}],
        GenerationConfig(
            temperature=args.judge_temperature,
            top_p=args.judge_top_p,
            seed=args.seed,
            extra_body=_json_arg(args.judge_extra_body),
        ),
    )
    return (response.choices[0].message.content or "").strip()


def _json_arg(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("JSON argument must decode to an object.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the math tool router on one problem or a JSONL dataset.")
    parser.add_argument("--problem")
    parser.add_argument("--answer")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--problem-key", default="problem")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--index-key", default="index")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-dir", default=str(DEFAULT_PROMPT_DIR))

    parser.add_argument("--outer-base-url", required=True)
    parser.add_argument("--outer-api-key-env")
    parser.add_argument("--outer-api-key")
    parser.add_argument("--outer-model", required=True)
    parser.add_argument("--outer-temperature", type=float)
    parser.add_argument("--outer-top-p", type=float)
    parser.add_argument("--outer-presence-penalty", type=float)
    parser.add_argument("--outer-frequency-penalty", type=float)
    parser.add_argument("--outer-extra-body")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int, required=True)

    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--embedding-api-key-env")
    parser.add_argument("--embedding-api-key")
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--router-model-path", required=True)
    parser.add_argument("--weight-performance", type=float, required=True)
    parser.add_argument("--router-difficulty-epsilon", type=float, required=True)
    parser.add_argument("--router-cost-hidden-dim", type=int)
    parser.add_argument("--deepseek-base-url")
    parser.add_argument("--deepseek-api-key-env")
    parser.add_argument("--deepseek-api-key")
    parser.add_argument("--deepseek-model")
    parser.add_argument("--deepseek-verifier-attempts", type=int)

    parser.add_argument("--lean-formalizer-base-url")
    parser.add_argument("--lean-formalizer-api-key-env")
    parser.add_argument("--lean-formalizer-api-key")
    parser.add_argument("--lean-formalizer-model")
    parser.add_argument("--lean-backtranslate-base-url")
    parser.add_argument("--lean-backtranslate-api-key-env")
    parser.add_argument("--lean-backtranslate-api-key")
    parser.add_argument("--lean-backtranslate-model")
    parser.add_argument("--lean-prover-base-url")
    parser.add_argument("--lean-prover-api-key-env")
    parser.add_argument("--lean-prover-api-key")
    parser.add_argument("--lean-prover-model")
    parser.add_argument("--lean-repl")
    parser.add_argument("--lean-lake-path")
    parser.add_argument("--lean-project-root")
    parser.add_argument("--lean-repl-timeout-s", type=int)
    parser.add_argument("--lean-translator-attempts", type=int)
    parser.add_argument("--lean-prover-attempts", type=int)
    parser.add_argument("--lean-inner-memory-top-k", type=int)
    parser.add_argument("--lean-outer-memory-top-k", type=int)

    parser.add_argument("--memory-embedding-base-url")
    parser.add_argument("--memory-embedding-api-key-env")
    parser.add_argument("--memory-embedding-api-key")
    parser.add_argument("--memory-embedding-model")
    parser.add_argument("--memory-top-k", type=int, required=True)
    parser.add_argument("--verifier-temperature", type=float)
    parser.add_argument("--verifier-top-p", type=float)
    parser.add_argument("--verifier-extra-body")

    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key-env")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-temperature", type=float)
    parser.add_argument("--judge-top-p", type=float)
    parser.add_argument("--judge-extra-body")
    return parser


if __name__ == "__main__":
    main()
