from __future__ import annotations

import time
from typing import List

import numpy as np

from ..llm import ModelEndpoint, embed


def embed_texts(texts: List[str], endpoint: ModelEndpoint, normalize: bool = True) -> np.ndarray:
    vectors = []
    start = time.perf_counter()
    for index, text in enumerate(texts, start=1):
        vector = np.array(embed(endpoint, text), dtype=np.float32)
        if normalize:
            vector = vector / (np.linalg.norm(vector) + 1e-9)
        vectors.append(vector)
        if index == 1:
            print(f"embedding_dim={vector.shape[0]}")
        if index % 50 == 0:
            print(f"embedded {index}/{len(texts)}")
    print(f"embedded {len(texts)} texts in {time.perf_counter() - start:.2f}s")
    return np.array(vectors, dtype=np.float32)
