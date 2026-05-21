from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str, prompt_dir: str | Path | None = None) -> str:
    base = Path(prompt_dir) if prompt_dir is not None else DEFAULT_PROMPT_DIR
    path = base / name
    return path.read_text(encoding="utf-8").strip()


def render_prompt(name: str, prompt_dir: str | Path | None = None, **kwargs: Any) -> str:
    return load_prompt(name, prompt_dir).format(**kwargs)
