from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..router_nnmodel import MathRouter
from .data import RouterDataset, RouterSample, latency_to_cost_target


@dataclass
class TrainConfig:
    tool_names: List[str]
    specialty_dim: int
    cost_hidden_dim: int
    difficulty_epsilon: float
    epochs: int
    success_lr: float
    cost_lr: float
    batch_size: int
    seed: Optional[int]
    device: str


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_val(
    samples: List[RouterSample],
    val_ratio: float,
    seed: Optional[int],
) -> Tuple[List[RouterSample], List[RouterSample]]:
    if val_ratio <= 0:
        return samples, []
    indices = np.arange(len(samples))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = int(round(len(samples) * val_ratio))
    val_ids = set(indices[:val_size].tolist())
    train = [sample for idx, sample in enumerate(samples) if idx not in val_ids]
    val = [sample for idx, sample in enumerate(samples) if idx in val_ids]
    return train, val


def train_router(
    samples: List[RouterSample],
    embeddings: np.ndarray,
    latency_norm_stats: Dict[str, Dict[str, float]],
    config: TrainConfig,
) -> MathRouter:
    set_seed(config.seed)
    dataset = RouterDataset(samples, config.tool_names, embeddings, latency_norm_stats)
    generator = torch.Generator()
    if config.seed is not None:
        generator.manual_seed(config.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator if config.seed is not None else None,
    )
    router = MathRouter(
        tool_names=config.tool_names,
        embedding_dim=embeddings.shape[1],
        specialty_dim=config.specialty_dim,
        cost_hidden_dim=config.cost_hidden_dim,
        difficulty_epsilon=config.difficulty_epsilon,
    ).to(config.device)
    _initialize_costs(router, samples, latency_norm_stats)

    success_optimizer = torch.optim.Adam(router.irt_model.parameters(), lr=config.success_lr)
    cost_optimizer = torch.optim.Adam(router.cost_predictor.parameters(), lr=config.cost_lr)
    criterion = torch.nn.BCELoss()

    router.train()
    for epoch in range(config.epochs):
        total_success_loss = 0.0
        total_cost_loss = 0.0
        batches = 0
        for batch in dataloader:
            embedding = batch["embedding"].to(config.device)
            success = batch["success"].to(config.device)
            cost_target = batch["cost_target"].to(config.device)
            latency_mask = batch["latency_mask"].to(config.device)

            success_optimizer.zero_grad()
            success_loss = 0.0
            for tool_idx, _tool_name in enumerate(config.tool_names):
                p_correct = router.irt_model(embedding, tool_idx)
                success_loss = success_loss + criterion(p_correct, success[:, tool_idx])
            success_loss.backward()
            success_optimizer.step()

            cost_optimizer.zero_grad()
            predicted_cost = router.cost_predictor(embedding)
            cost_loss_matrix = F.smooth_l1_loss(predicted_cost, cost_target, reduction="none")
            cost_loss = (cost_loss_matrix * latency_mask).sum() / latency_mask.sum().clamp(min=1.0)
            cost_loss.backward()
            cost_optimizer.step()

            total_success_loss += float(success_loss.item())
            total_cost_loss += float(cost_loss.item())
            batches += 1
        print(
            f"epoch={epoch + 1} "
            f"success_loss={total_success_loss / max(batches, 1):.6f} "
            f"cost_loss={total_cost_loss / max(batches, 1):.6f}"
        )
    return router


def evaluate_router(
    router: MathRouter,
    samples: List[RouterSample],
    embeddings: np.ndarray,
    device: str,
    weight_performance: float,
) -> Dict[str, Any]:
    if not samples:
        return {
            "oracle_cheapest_hit": 0.0,
            "selected_success": 0.0,
            "avg_latency": 0.0,
            "avg_pred_cost": 0.0,
            **{f"{name}_count": 0 for name in router.tool_names},
        }
    router.eval().to(device)
    oracle_hit = 0
    selected_success = 0
    total_latency = 0.0
    latency_count = 0
    total_pred_cost = 0.0
    counts = {name: 0 for name in router.tool_names}
    with torch.no_grad():
        for index, sample in enumerate(samples):
            embedding = torch.tensor(embeddings[index], dtype=torch.float32).unsqueeze(0).to(device)
            selected, details = router.route(
                embedding,
                weight_performance=weight_performance,
            )
            chosen = router.tool_names[selected[0]]
            counts[chosen] += 1
            result = sample.tool_results.get(chosen)
            if result is not None:
                selected_success += int(result.get("success", 0))
                total_latency += float(result.get("latency_s", 0.0) or 0.0)
                latency_count += 1
            total_pred_cost += float(details[0][chosen]["cost"])

            successful_tools = [
                name
                for name in router.tool_names
                if int(sample.tool_results.get(name, {}).get("success", 0)) == 1
            ]
            if successful_tools:
                oracle_tool = min(
                    successful_tools,
                    key=lambda name: float(sample.tool_results[name].get("latency_s", float("inf"))),
                )
                oracle_hit += int(chosen == oracle_tool)
    total = len(samples)
    return {
        "oracle_cheapest_hit": oracle_hit / total,
        "selected_success": selected_success / total,
        "avg_latency": total_latency / max(latency_count, 1),
        "avg_pred_cost": total_pred_cost / total,
        **{f"{name}_count": counts[name] for name in router.tool_names},
    }


def _initialize_costs(
    router: MathRouter,
    samples: List[RouterSample],
    latency_norm_stats: Dict[str, Dict[str, float]],
) -> None:
    by_tool: Dict[str, List[float]] = {name: [] for name in router.tool_names}
    for sample in samples:
        for tool_name, result in sample.tool_results.items():
            if tool_name not in by_tool:
                continue
            latency = float(result.get("latency_s", 0.0) or 0.0)
            if latency > 0:
                by_tool[tool_name].append(latency_to_cost_target(latency, latency_norm_stats))
    router.cost_predictor.set_costs(
        [float(np.mean(by_tool[name])) if by_tool[name] else 1.0 for name in router.tool_names]
    )
