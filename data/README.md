# Data

This directory contains the datasets used by the public release.

## Evaluation Datasets

- `datasets/mine`: MinervaMath-style dataset cache.
- `datasets/hm2`: HARDMath2 dataset cache.
- `datasets/math398`: MATH-398 dataset cache.
- `datasets/aime`: AIME dataset cache.

These directories are stored in Hugging Face `datasets` disk/cache format where applicable.

## Router Training Data

`router_training/` contains the step-level tool-evaluation JSONL files used to train the router:

- `step_verdict_grouped_final_complete__regraded.jsonl`
- `HuggingFaceH4_MATH-500.jsonl`
- `JVRoggeveen_HARDMath2.jsonl`

Each row is one verifier result for one reasoning step and one tool. Rows sharing the same `sid` are grouped into one router training sample by `train_router.py`.
