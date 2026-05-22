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
  --output-checkpoint <router_checkpoint_path> \
  --tool-names lean4,deepseek \
  --val-ratio <validation_ratio> \
  --device <cpu_or_cuda_or_auto> \
  --embedding-base-url <router_embedding_base_url> \
  --embedding-api-key <router_embedding_api_key> \
  --embedding-model <router_embedding_model> \
  --specialty-dim <specialty_dim> \
  --cost-hidden-dim <cost_hidden_dim> \
  --router-difficulty-epsilon <difficulty_epsilon> \
  --epochs <num_epochs> \
  --success-lr <success_learning_rate> \
  --cost-lr <cost_learning_rate> \
  --batch-size <batch_size> \
  --weight-performance <router_weight>
```

This creates:

```text
<router_checkpoint_path>
```

## Run Problem

```bash
python run_problem.py \
  --problem "Convert the point $(0,3)$ to polar coordinates." \
  --outer-base-url <outer_base_url> \
  --outer-api-key <outer_api_key> \
  --outer-model <outer_model> \
  --embedding-base-url <router_embedding_base_url> \
  --embedding-api-key <router_embedding_api_key> \
  --embedding-model <router_embedding_model> \
  --memory-embedding-base-url <memory_embedding_base_url> \
  --memory-embedding-api-key <memory_embedding_api_key> \
  --memory-embedding-model <memory_embedding_model> \
  --router-model-path <router_checkpoint_path> \
  --weight-performance <router_weight> \
  --router-difficulty-epsilon <difficulty_epsilon> \
  --max-steps <max_steps> \
  --memory-top-k <memory_top_k> \
  --deepseek-verifier-attempts <deepseek_verifier_attempts> \
  --deepseek-base-url <deepseek_base_url> \
  --deepseek-api-key <deepseek_api_key> \
  --deepseek-model <deepseek_model> \
  --lean-formalizer-base-url <lean_formalizer_base_url> \
  --lean-formalizer-api-key <lean_formalizer_api_key> \
  --lean-formalizer-model <lean_formalizer_model> \
  --lean-backtranslate-base-url <lean_backtranslate_base_url> \
  --lean-backtranslate-api-key <lean_backtranslate_api_key> \
  --lean-backtranslate-model <lean_backtranslate_model> \
  --lean-prover-base-url <lean_prover_base_url> \
  --lean-prover-api-key <lean_prover_api_key> \
  --lean-prover-model <lean_prover_model> \
  --lean-repl <lean_repl_path> \
  --lean-lake-path <lean_lake_path> \
  --lean-project-root <lean_project_root> \
  --lean-repl-timeout-s <lean_repl_timeout_seconds> \
  --lean-translator-attempts <lean_translator_attempts> \
  --lean-prover-attempts <lean_prover_attempts> \
  --lean-inner-memory-top-k <lean_inner_memory_top_k> \
  --lean-outer-memory-top-k <lean_outer_memory_top_k>
```

For a dataset, use JSON, JSONL, or Arrow input:

```bash
python run_problem.py \
  --input-arrow <dataset_arrow_path> \
  --problem-key <problem_field> \
  --answer-key <answer_field> \
  --index-key <index_field> \
  --output-jsonl <output_jsonl_path> \
  [the runtime arguments above]
```
