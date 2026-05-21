from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class GenerationConfig:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def kwargs(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "seed"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.extra_body:
            data["extra_body"] = self.extra_body
        return data


@dataclass
class ModelEndpoint:
    base_url: str
    model: str
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("ModelEndpoint.base_url is required and cannot be empty.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("ModelEndpoint.model is required and cannot be empty.")
        if not self.api_key and not self.api_key_env:
            raise ValueError("ModelEndpoint requires either api_key or api_key_env.")
        self.base_url = self.base_url.strip()
        self.model = self.model.strip()
        if self.api_key_env is not None:
            self.api_key_env = self.api_key_env.strip()
            if not self.api_key_env:
                raise ValueError("ModelEndpoint.api_key_env cannot be empty.")
        if self.api_key is not None:
            self.api_key = self.api_key.strip()
            if not self.api_key:
                raise ValueError("ModelEndpoint.api_key cannot be empty.")

    def resolved_api_key(self) -> str:
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env:
            value = os.getenv(self.api_key_env)
            if value:
                return value
            raise RuntimeError(f"Environment variable {self.api_key_env} is not set.")
        raise RuntimeError("No API key was supplied. Use an API key environment variable or pass one explicitly.")

    def client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key=self.resolved_api_key())


def chat_completion(
    endpoint: ModelEndpoint,
    messages: List[Dict[str, Any]],
    generation: GenerationConfig,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> Any:
    kwargs: Dict[str, Any] = {
        "model": endpoint.model,
        "messages": messages,
        **generation.kwargs(),
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return endpoint.client().chat.completions.create(**kwargs)


def embed(endpoint: ModelEndpoint, text: str) -> List[float]:
    response = endpoint.client().embeddings.create(input=text, model=endpoint.model)
    return response.data[0].embedding
