from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from .llm import ModelEndpoint, embed
from .router_nnmodel import MathRouter, select_tool


@dataclass
class RouterConfig:
    model_path: Path
    embedding_endpoint: ModelEndpoint
    weight_performance: float
    difficulty_epsilon: float
    cost_hidden_dim: Optional[int] = None


class ToolRouter:
    def __init__(self, config: RouterConfig):
        self.config = config
        self._model: Optional[MathRouter] = None
        self._embedding_dim: Optional[int] = None

    def route(self, state: Dict[str, Any], allowed_tools: Optional[Iterable[str]] = None) -> str:
        model = self._load_model()

        text = self._routing_text(state)
        embedding = embed(self.config.embedding_endpoint, text)
        embedding = self._fit_embedding_dim(embedding, self._embedding_dim)
        tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

        _, details = model.route(
            tensor,
            weight_performance=self.config.weight_performance,
        )
        allowed = list(allowed_tools) if allowed_tools is not None else list(model.tool_names)
        allowed_model_names = [name for name in model.tool_names if name in allowed]
        if not allowed_model_names:
            raise RuntimeError(
                f"Router checkpoint tools {model.tool_names} do not match enabled verifier tools {allowed}."
            )

        best_idx = select_tool(
            allowed_model_names,
            {name: details[0][name] for name in allowed_model_names},
        )
        return self._map_tool_name(allowed_model_names[best_idx])

    def _load_model(self) -> MathRouter:
        if self._model is not None:
            return self._model
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Router checkpoint not found: {self.config.model_path}")

        checkpoint = torch.load(self.config.model_path, weights_only=False, map_location="cpu")
        embedding_dim = int(checkpoint["embedding_dim"])
        if "tool_names" not in checkpoint:
            raise KeyError("Router checkpoint missing required field: tool_names.")
        tool_names = list(checkpoint["tool_names"])
        unsupported_tools = sorted(set(tool_names) - {"lean4", "deepseek"})
        if unsupported_tools:
            raise ValueError(f"Router checkpoint contains unsupported tools: {unsupported_tools}")
        specialty_dim = int(checkpoint.get("specialty_dim"))
        cost_meta = checkpoint.get("cost_model", {}) or {}
        cost_hidden = self.config.cost_hidden_dim or checkpoint.get("cost_hidden_dim") or cost_meta.get("hidden_dim")
        if cost_hidden is None:
            raise KeyError("Router checkpoint does not contain cost_hidden_dim or cost_model.hidden_dim.")
        cost_hidden_dim = int(cost_hidden)

        model = MathRouter(
            tool_names=tool_names,
            embedding_dim=embedding_dim,
            specialty_dim=specialty_dim,
            cost_hidden_dim=cost_hidden_dim,
            difficulty_epsilon=self.config.difficulty_epsilon,
        )
        state_dict = checkpoint["model_state_dict"]
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            missing = ", ".join(incompatible.missing_keys)
            unexpected = ", ".join(incompatible.unexpected_keys)
            raise RuntimeError(f"Router checkpoint mismatch. Missing: {missing}; unexpected: {unexpected}")
        model.eval()
        self._model = model
        self._embedding_dim = embedding_dim
        return model

    @staticmethod
    def _routing_text(state: Dict[str, Any]) -> str:
        step = str(state.get("candidate_step_nl") or "")
        context = str(state.get("tool_context") or "")
        if not step.strip():
            raise RuntimeError("Router requires candidate_step_nl for step-level routing.")
        if context.strip():
            return f"{step.strip()}\n\n{context.strip()}"
        return step.strip()

    @staticmethod
    def _fit_embedding_dim(vector: List[float], dim: Optional[int]) -> List[float]:
        if dim is None or len(vector) == dim:
            return vector
        if len(vector) > dim:
            return vector[:dim]
        return vector + [0.0] * (dim - len(vector))

    @staticmethod
    def _map_tool_name(model_tool_name: str) -> str:
        return model_tool_name
