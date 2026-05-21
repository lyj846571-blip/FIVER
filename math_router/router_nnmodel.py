from __future__ import annotations

from typing import Dict, List, Tuple

import torch


class Improved2PLModel(torch.nn.Module):
    def __init__(self, embedding_dim: int, n_tools: int, specialty_dim: int, difficulty_epsilon: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_tools = n_tools
        self.specialty_dim = specialty_dim
        self.difficulty_epsilon = float(difficulty_epsilon)
        self.W_d = torch.nn.Linear(embedding_dim, 1, bias=False)
        self.W_r = torch.nn.Linear(embedding_dim, specialty_dim, bias=False)
        self.worker_strength = torch.nn.Parameter(torch.zeros(n_tools))
        self.worker_specialty = torch.nn.Parameter(torch.randn(n_tools, specialty_dim) * 0.1)

    def forward(self, embeddings: torch.Tensor, tool_idx: int) -> torch.Tensor:
        difficulty = torch.abs(self.W_d(embeddings)) + self.difficulty_epsilon
        specialty = self.W_r(embeddings)
        strength = self.worker_strength[tool_idx]
        match = torch.matmul(specialty, self.worker_specialty[tool_idx])
        return torch.sigmoid(strength - difficulty.squeeze(-1) + match)


class CostPredictor(torch.nn.Module):
    def __init__(self, embedding_dim: int, n_tools: int, hidden_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_tools = n_tools
        self.hidden_dim = hidden_dim
        self.base_logits = torch.nn.Parameter(torch.zeros(n_tools))
        self.heads = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(embedding_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
                for _ in range(n_tools)
            ]
        )

    def set_costs(self, costs: List[float]) -> None:
        values = torch.tensor(costs, dtype=torch.float32)
        if values.numel() != self.base_logits.numel():
            raise ValueError("Checkpoint cost vector length does not match tool count.")
        values = torch.nan_to_num(values, nan=1.0, posinf=1.0, neginf=0.0)
        max_cost = values.max()
        if max_cost > 1:
            values = values / max_cost
        values = values.clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            self.base_logits.copy_(torch.log(values / (1 - values)))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        head_logits = torch.cat([head(embeddings) for head in self.heads], dim=1)
        return torch.sigmoid(head_logits + self.base_logits.unsqueeze(0))


class MathRouter(torch.nn.Module):
    def __init__(
        self,
        tool_names: List[str],
        embedding_dim: int,
        specialty_dim: int,
        cost_hidden_dim: int,
        difficulty_epsilon: float,
    ):
        super().__init__()
        self.tool_names = tool_names
        self.embedding_dim = embedding_dim
        self.irt_model = Improved2PLModel(embedding_dim, len(tool_names), specialty_dim, difficulty_epsilon)
        self.cost_predictor = CostPredictor(embedding_dim, len(tool_names), cost_hidden_dim)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        costs = self.cost_predictor(embeddings)
        return {
            tool_name: {
                "p_correct": self.irt_model(embeddings, idx),
                "cost": costs[:, idx],
            }
            for idx, tool_name in enumerate(self.tool_names)
        }

    def route(
        self,
        embeddings: torch.Tensor,
        weight_performance: float,
    ) -> Tuple[List[int], Dict[int, Dict[str, Dict[str, float]]]]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(embeddings)

        selected: List[int] = []
        details: Dict[int, Dict[str, Dict[str, float]]] = {}
        for row in range(embeddings.shape[0]):
            row_details: Dict[str, Dict[str, float]] = {}
            for name in self.tool_names:
                p_correct = outputs[name]["p_correct"][row].item()
                cost = outputs[name]["cost"][row].item()
                score = weight_performance * p_correct - (1 - weight_performance) * cost
                row_details[name] = {"p_correct": p_correct, "cost": cost, "score": score}
            selected.append(select_tool(self.tool_names, row_details))
            details[row] = row_details
        return selected, details


def select_tool(
    tool_names: List[str],
    tool_details: Dict[str, Dict[str, float]],
) -> int:
    best = max(
        tool_names,
        key=lambda name: (
            tool_details[name]["score"],
            tool_details[name]["p_correct"],
            -tool_details[name]["cost"],
        ),
    )
    return tool_names.index(best)
