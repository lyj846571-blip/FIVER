from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from math_router.llm import ModelEndpoint
from math_router.training.data import (
    build_latency_norm_stats,
    load_router_jsonl,
)
from math_router.training.embeddings import embed_texts
from math_router.training.trainer import (
    TrainConfig,
    evaluate_router,
    split_train_val,
    train_router,
)


def main() -> None:
    args = build_parser().parse_args()
    tool_names = [tool.strip() for tool in args.tool_names.split(",") if tool.strip()]
    if not tool_names:
        raise RuntimeError("--tool-names must contain at least one tool.")

    all_samples = load_router_jsonl(args.train_jsonl, tool_names)
    if args.max_samples is not None:
        all_samples = all_samples[: args.max_samples]
    train_samples, val_samples = split_train_val(all_samples, args.val_ratio, args.seed)
    if not train_samples:
        raise RuntimeError("No training samples after split.")

    endpoint = ModelEndpoint(
        base_url=args.embedding_base_url,
        model=args.embedding_model,
        api_key_env=args.embedding_api_key_env,
        api_key=args.embedding_api_key,
    )
    train_embeddings = embed_texts([sample.route_text for sample in train_samples], endpoint)
    val_embeddings = (
        embed_texts([sample.route_text for sample in val_samples], endpoint)
        if val_samples
        else train_embeddings[:0]
    )
    latency_norm_stats = build_latency_norm_stats(train_samples, tool_names)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TrainConfig(
        tool_names=tool_names,
        specialty_dim=args.specialty_dim,
        cost_hidden_dim=args.cost_hidden_dim,
        difficulty_epsilon=args.router_difficulty_epsilon,
        epochs=args.epochs,
        success_lr=args.success_lr,
        cost_lr=args.cost_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    router = train_router(train_samples, train_embeddings, latency_norm_stats, config)

    train_metrics = evaluate_router(
        router,
        train_samples,
        train_embeddings,
        device=device,
        weight_performance=args.weight_performance,
    )
    val_metrics = (
        evaluate_router(
            router,
            val_samples,
            val_embeddings,
            device=device,
            weight_performance=args.weight_performance,
        )
        if val_samples
        else {}
    )

    checkpoint_path = Path(args.output_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "tool_names": router.tool_names,
        "model_state_dict": router.state_dict(),
        "embedding_dim": router.embedding_dim,
        "specialty_dim": args.specialty_dim,
        "cost_model": {
            "type": "per_tool_query_dependent_sigmoid_mlp",
            "hidden_dim": args.cost_hidden_dim,
        },
        "costs": torch.sigmoid(router.cost_predictor.base_logits).detach().cpu().tolist(),
        "latency_norm_stats": latency_norm_stats,
        "cost_normalization": "global_log_latency_minmax",
    }
    torch.save(checkpoint, checkpoint_path)

    meta_path = Path(args.output_meta) if args.output_meta else checkpoint_path.with_suffix(".meta.json")
    meta = dict(checkpoint)
    meta.pop("model_state_dict", None)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"saved_checkpoint={checkpoint_path}")
    print(f"saved_metadata={meta_path}")
    print(
        json.dumps(
            {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "weight_performance": args.weight_performance,
            },
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the math tool router from router evaluation JSONL.")
    parser.add_argument("--train-jsonl", nargs="+", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-meta")
    parser.add_argument("--tool-names", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--val-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], required=True)

    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--embedding-api-key-env")
    parser.add_argument("--embedding-api-key")
    parser.add_argument("--embedding-model", required=True)

    parser.add_argument("--specialty-dim", type=int, required=True)
    parser.add_argument("--cost-hidden-dim", type=int, required=True)
    parser.add_argument("--router-difficulty-epsilon", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--success-lr", type=float, required=True)
    parser.add_argument("--cost-lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)

    parser.add_argument("--weight-performance", type=float, required=True)
    return parser


if __name__ == "__main__":
    main()
