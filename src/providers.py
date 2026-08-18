from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] | None = None


@dataclass
class ModelSpec:
    id: str
    label: str
    class_name: str
    parameter_b: float
    active_parameter_b: float
    deployment: str
    provider: str
    model: str
    base_url: str
    api_key: str
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    enabled: bool = True

    @classmethod
    def from_toml(cls, data: dict[str, Any]) -> "ModelSpec":
        model = data.get("model", "")
        if data.get("model_env"):
            model = os.getenv(data["model_env"], model)
        base_url = data.get("base_url", "")
        if data.get("base_url_env"):
            base_url = os.getenv(data["base_url_env"], base_url)
        api_key = data.get("api_key", "")
        if data.get("api_key_env"):
            api_key = os.getenv(data["api_key_env"], api_key)
        return cls(
            id=data["id"],
            label=data.get("label", data["id"]),
            class_name=data.get("class", "unknown"),
            parameter_b=float(data.get("parameter_b", 0.0)),
            active_parameter_b=float(data.get("active_parameter_b", data.get("parameter_b", 0.0))),
            deployment=data.get("deployment", "local"),
            provider=data.get("provider", "openai_compatible"),
            model=model,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            input_cost_per_million=float(data.get("input_cost_per_million", 0.0)),
            output_cost_per_million=float(data.get("output_cost_per_million", 0.0)),
            enabled=bool(data.get("enabled", True)),
        )

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_million / 1_000_000
            + output_tokens * self.output_cost_per_million / 1_000_000
        )


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    def __init__(self, spec: ModelSpec, timeout_s: int = 180):
        self.spec = spec
        self.timeout_s = timeout_s

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        if not self.spec.model:
            raise ProviderError(f"model is not configured for {self.spec.id}")
        if not self.spec.base_url:
            raise ProviderError(f"base URL is not configured for {self.spec.id}")

        url = self.spec.base_url + "/chat/completions"
        payload = {
            "model": self.spec.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.spec.api_key:
            headers["Authorization"] = f"Bearer {self.spec.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw_bytes = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:4000]
            raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise ProviderError(str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        data = json.loads(raw_bytes)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {str(data)[:2000]}") from exc
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=latency_ms,
            raw=data,
        )
