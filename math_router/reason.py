from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .llm import GenerationConfig, ModelEndpoint, chat_completion
from .prompts import render_prompt
from .router import ToolRouter
from .verifiers import BaseVerifier


@dataclass
class WorkflowConfig:
    outer_endpoint: ModelEndpoint
    outer_generation: GenerationConfig
    prompt_dir: str
    max_steps: int
    memory_top_k: int


def run_workflow(
    problem: str,
    config: WorkflowConfig,
    router: ToolRouter,
    verifiers: Dict[str, BaseVerifier],
) -> Tuple[Optional[str], Dict[str, Any]]:
    state: Dict[str, Any] = {
        "problem": problem,
        "final_answer": None,
        "scratchpad": [],
        "candidate_step_nl": None,
        "tool_context": None,
        "memory": [],
        "tool_trace": [],
        "total_tokens": 0,
        "reasoning_model_tokens": 0,
        "formal_tokens": 0,
    }
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": render_prompt("reasoner_system.txt", config.prompt_dir)},
        {"role": "user", "content": render_prompt("reasoner_user.txt", config.prompt_dir, problem=problem)},
    ]

    for _ in range(config.max_steps):
        response = chat_completion(
            config.outer_endpoint,
            messages,
            config.outer_generation,
            tools=[_verification_tool_schema()],
            tool_choice="auto",
        )
        state["total_tokens"] += _usage_tokens(response)
        state["reasoning_model_tokens"] += _usage_tokens(response)
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        messages.append(_message_to_dict(message))

        if finish_reason == "length":
            content = message.content or ""
            if content.strip():
                state["scratchpad"].append(content.strip())
                state["final_answer"] = content.strip()
            else:
                state["final_answer"] = "Final Answer: stopped early due to generation limit."
            break

        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            tool_call = tool_calls[0]
            args = _parse_tool_args(tool_call.function.arguments)
            proof_step = args.get("proof_step", "").strip()
            state["candidate_step_nl"] = proof_step
            selected = router.route(state, allowed_tools=["lean4", "deepseek"])
            verifier = verifiers.get(selected)
            if verifier is None:
                raise RuntimeError(f"Router selected {selected}, but that verifier is not enabled.")
            state["tool_trace"].append(selected)
            try:
                result = verifier.verify(problem, proof_step, state["memory"])
            except Exception:
                fallback_feedback = (
                    "VERIFICATION FAILURE: self-verify the step"
                    if selected == "lean4"
                    else "UNVERIFIED: unable to determine correctness."
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": fallback_feedback,
                    }
                )
                continue
            state["formal_tokens"] += result.tokens
            state["total_tokens"] += result.tokens
            if result.status == "CORRECT":
                extra = result.extra or {}
                updated_memory = extra.get("updated_memory")
                if isinstance(updated_memory, list):
                    state["memory"] = updated_memory
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result.feedback,
                        }
                    )
                    continue
                memory_item = extra.get("memory_item")
                if not isinstance(memory_item, dict):
                    raise RuntimeError(f"Verifier {selected} returned CORRECT without a memory_item.")
                state["memory"].append(memory_item)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.feedback,
                }
            )
            continue

        content = message.content or ""
        if content.strip():
            state["scratchpad"].append(content.strip())
        if _looks_like_final_answer(content):
            state["final_answer"] = content.strip()
            break

    if (
        (state["final_answer"] is None or str(state["final_answer"]).strip() == "")
        and len(state.get("scratchpad", [])) == 0
        and int(state.get("total_tokens", 0) or 0) == 0
    ):
        raise RuntimeError("EMPTY_WORKFLOW_ABORT: no final answer, no scratchpad, total_tokens=0")

    return state["final_answer"], state


def _verification_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "verify_one_mathematical_step",
            "description": "Formally validates a single mathematical reasoning step. Use this for critical proof steps that should be checked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "proof_step": {
                        "type": "string",
                        "description": "One natural-language mathematical proof step in English, including all necessary context.",
                    }
                },
                "required": ["proof_step"],
            },
        },
    }


def _parse_tool_args(raw: str) -> Dict[str, str]:
    import json

    try:
        data = json.loads(raw or "{}")
        if isinstance(data, dict):
            return {str(key): str(value) for key, value in data.items()}
    except Exception:
        pass
    return {}


def _message_to_dict(message: Any) -> Dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return message
    raise TypeError(f"Unsupported message type: {type(message)}")


def _usage_tokens(response: object) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    total = getattr(usage, "total_tokens", None)
    return int(total or 0)


def _strip_think(text: str) -> str:
    if not text:
        return ""
    if re.search(r"<think>", text, re.IGNORECASE) and not re.search(r"</think>", text, re.IGNORECASE):
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _looks_like_final_answer(text: str) -> bool:
    text = _strip_think(text)
    if not text:
        return False
    patterns = [
        r"\bfinal\s*answer\b",
        r"\bthe\s+answer\s+is\b",
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)
