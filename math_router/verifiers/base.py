from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VerificationResult:
    status: str
    feedback: str
    tokens: int = 0
    extra: Optional[Dict[str, Any]] = None


class BaseVerifier:
    def verify(self, problem: str, proof_step: str, memory: List[Dict[str, Any]]) -> VerificationResult:
        raise NotImplementedError
