# FIVER: An Adaptive Agent for Integrating Formal and Informal Verification in LLM Mathematical Reasoning

## Install

```bash
pip install -r requirements.txt
```

## Train the Router

The training set is:

```text
data/router_training/step_verdict_grouped_final_complete__regraded.jsonl
```

```bash
python train_router.py \
  --train-jsonl data/router_training/step_verdict_grouped_final_complete__regraded.jsonl \
  --output-checkpoint router_models/router_model_retrained.pt \
  --tool-names lean4,deepseek \
  --val-ratio 0.15 \
  --device auto \
  --embedding-base-url "$ROUTER_EMBEDDING_BASE_URL" \
  --embedding-api-key-env ROUTER_EMBEDDING_API_KEY \
  --embedding-model "$ROUTER_EMBEDDING_MODEL" \
  --specialty-dim 8 \
  --cost-hidden-dim 64 \
  --router-difficulty-epsilon "$ROUTER_DIFFICULTY_EPSILON" \
  --epochs 40 \
  --success-lr 0.01 \
  --cost-lr 0.01 \
  --batch-size 16 \
  --weight-performance 0.8
```

This creates:

```text
router_models/router_model_retrained.pt
```

## Run Problem

```bash
python run_problem.py \
  --problem "Convert the point $(0,3)$ to polar coordinates." \
  --outer-base-url "$OUTER_BASE_URL" \
  --outer-api-key-env OUTER_API_KEY \
  --outer-model "$OUTER_MODEL" \
  --embedding-base-url "$ROUTER_EMBEDDING_BASE_URL" \
  --embedding-api-key-env ROUTER_EMBEDDING_API_KEY \
  --embedding-model "$ROUTER_EMBEDDING_MODEL" \
  --memory-embedding-base-url "$MEMORY_EMBEDDING_BASE_URL" \
  --memory-embedding-api-key-env MEMORY_EMBEDDING_API_KEY \
  --memory-embedding-model "$MEMORY_EMBEDDING_MODEL" \
  --router-model-path router_models/router_model_retrained.pt \
  --weight-performance 0.8 \
  --router-difficulty-epsilon "$ROUTER_DIFFICULTY_EPSILON" \
  --max-steps 50 \
  --memory-top-k 3 \
  --deepseek-verifier-attempts 1 \
  --deepseek-base-url "$DEEPSEEK_BASE_URL" \
  --deepseek-api-key-env DEEPSEEK_API_KEY \
  --deepseek-model "$DEEPSEEK_MODEL" \
  --lean-formalizer-base-url "$LEAN_FORMALIZER_BASE_URL" \
  --lean-formalizer-api-key-env LEAN_FORMALIZER_API_KEY \
  --lean-formalizer-model "$LEAN_FORMALIZER_MODEL" \
  --lean-backtranslate-base-url "$LEAN_BACKTRANSLATE_BASE_URL" \
  --lean-backtranslate-api-key-env LEAN_BACKTRANSLATE_API_KEY \
  --lean-backtranslate-model "$LEAN_BACKTRANSLATE_MODEL" \
  --lean-prover-base-url "$LEAN_PROVER_BASE_URL" \
  --lean-prover-api-key-env LEAN_PROVER_API_KEY \
  --lean-prover-model "$LEAN_PROVER_MODEL" \
  --lean-repl "$LEAN_REPL" \
  --lean-lake-path "$LEAN_LAKE_PATH" \
  --lean-project-root "$LEAN_PROJECT_ROOT" \
  --lean-repl-timeout-s 300 \
  --lean-translator-attempts 4 \
  --lean-prover-attempts 4 \
  --lean-inner-memory-top-k 2 \
  --lean-outer-memory-top-k 1
```

For a dataset, use JSON, JSONL, or Arrow input:

```bash
python run_problem.py \
  --input-arrow data/datasets/aime/default/0.0.0/563bb8404243c5f09de6ec262f2db674fe5bce9b/aime25-test.arrow \
  --problem-key problem \
  --answer-key answer \
  --index-key id \
  --output-jsonl outputs/aime25_results.jsonl \
  [the runtime arguments above]
```
