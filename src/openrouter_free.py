from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from providers import ModelSpec, OpenAICompatibleProvider
from rlm import correct, run_condition
from tasks import make_task
from util import ROOT, append_jsonl, load_dotenv, now_iso, stable_id, write_json


@dataclass(frozen=True)
class FreeModel:
    model_id: str
    name: str
    context_length: int
    architecture: str


@dataclass(frozen=True)
class Scenario:
    family: str
    difficulty: str
    condition: str
    recursion: int
    seed: int

    @property
    def max_calls(self) -> int:
        return 1 if self.condition == "base" else max(1, self.recursion)


def _zero(value: Any) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def is_zero_pricing(pricing: Any) -> bool:
    """Strict free check. Missing pricing dimensions are not assumed to be free."""
    if not isinstance(pricing, dict):
        return False
    return all(key in pricing and _zero(pricing[key]) for key in ("prompt", "completion", "request"))


def discover_free_models(payload: dict[str, Any]) -> list[FreeModel]:
    rows: list[FreeModel] = []
    for row in payload.get("data", []):
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "")
        if not model_id or model_id == "openrouter/free":
            continue
        # Specific free variants are stable enough to compare model-by-model;
        # the random free router is measured separately.
        if not model_id.endswith(":free") or not is_zero_pricing(row.get("pricing")):
            continue
        architecture = row.get("architecture") or {}
        if isinstance(architecture, dict):
            arch = str(architecture.get("tokenizer") or architecture.get("instruct_type") or "")
        else:
            arch = ""
        rows.append(
            FreeModel(
                model_id=model_id,
                name=str(row.get("name") or model_id),
                context_length=int(row.get("context_length") or 0),
                architecture=arch,
            )
        )
    rows.sort(key=lambda row: (-row.context_length, row.model_id))
    return rows


def fetch_model_catalog(base_url: str, api_key: str, timeout_s: int = 30) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(base_url.rstrip("/") + "/models", headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


def _spec(model_id: str, base_url: str, api_key: str) -> ModelSpec:
    return ModelSpec.from_toml(
        {
            "id": "free-" + stable_id(model_id)[:12],
            "label": model_id,
            "class": "openrouter-free",
            "parameter_b": 0.0,
            "active_parameter_b": 0.0,
            "deployment": "remote",
            "provider": "openai_compatible",
            "model": model_id,
            "base_url": base_url,
            "api_key": api_key,
            "input_cost_per_million": 0.0,
            "output_cost_per_million": 0.0,
        }
    )


def scenarios() -> list[Scenario]:
    # Broad enough to exercise exact answers, iterative reasoning, Prolog calls,
    # and StarIntel-shaped symbolic state while staying inside free-tier request limits.
    return [
        Scenario("reachability", "easy", "base", 1, 545),
        Scenario("authorization", "medium", "base", 1, 546),
        Scenario("state", "medium", "agent_plain", 2, 547),
        Scenario("reachability", "medium", "rlm_verified", 2, 548),
        Scenario("starintel_actor_loop", "medium", "rlm_verified", 2, 550),
    ]


def run_deep(
    models: list[FreeModel],
    *,
    base_url: str,
    api_key: str,
    max_models: int,
    max_requests: int,
    include_router: bool,
) -> int:
    chosen = models[:max_models]
    model_ids = [row.model_id for row in chosen]
    if include_router:
        model_ids.append("openrouter/free")

    if not model_ids:
        print("no strictly zero-price :free models discovered")
        return 2

    session_id = stable_id("openrouter-free", now_iso(), max_models, max_requests)
    out_dir = ROOT / "results" / "free"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.jsonl"
    manifest_path = out_dir / f"{session_id}.manifest.json"
    manifest = {
        "session_id": session_id,
        "started_at": now_iso(),
        "models": model_ids,
        "max_models": max_models,
        "max_requests": max_requests,
        "strict_zero_cost": True,
        "scenarios": [scenario.__dict__ for scenario in scenarios()],
    }
    write_json(manifest_path, manifest)

    request_count = 0
    completed = 0
    successes = 0
    failures = 0
    stopped_reason = ""

    for model_id in model_ids:
        provider = OpenAICompatibleProvider(_spec(model_id, base_url, api_key), timeout_s=180)
        for scenario in scenarios():
            if scenario.condition.startswith("rlm") and not shutil.which("swipl"):
                continue
            if request_count + scenario.max_calls > max_requests:
                stopped_reason = f"request fuse reached: {request_count}/{max_requests}"
                break

            task = make_task(scenario.family, scenario.difficulty, scenario.seed)
            trace = run_condition(
                provider,
                task,
                scenario.condition,
                scenario.recursion,
                max_tokens=256,
                temperature=0.0,
            )
            request_count += trace.model_calls
            completed += 1
            ok = not trace.error and correct(trace.answer, task.expected)
            successes += int(ok)
            failures += int(not ok)

            # OpenRouter can return actual usage cost even when the registry says
            # zero. Treat any nonzero amount as a safety failure and stop.
            if trace.cost_usd > 0:
                stopped_reason = (
                    f"NONZERO_FREE_COST: {model_id} reported ${trace.cost_usd:.8f}; "
                    "aborting free benchmark"
                )

            routed_model = ""
            for event in trace.events:
                if event.get("type") == "model":
                    # Current generic event does not retain raw response model;
                    # the requested exact :free model remains the comparison key.
                    routed_model = model_id
                    break

            append_jsonl(
                out_path,
                {
                    "session_id": session_id,
                    "timestamp": now_iso(),
                    "requested_model": model_id,
                    "routed_model": routed_model,
                    "task_family": task.family,
                    "difficulty": task.difficulty,
                    "condition": scenario.condition,
                    "recursion_level": scenario.recursion,
                    "success": ok,
                    "expected": task.expected,
                    "answer": trace.answer,
                    "model_calls": trace.model_calls,
                    "prolog_calls": trace.prolog_calls,
                    "input_tokens": trace.input_tokens,
                    "output_tokens": trace.output_tokens,
                    "latency_ms": round(trace.model_latency_ms, 3),
                    "cost_usd": round(trace.cost_usd, 10),
                    "error": trace.error,
                },
            )
            print(
                f"{model_id:<48} {scenario.family:<28} {scenario.condition:<13} "
                f"{'PASS' if ok else 'FAIL'} calls={trace.model_calls} cost=${trace.cost_usd:.8f}"
            )
            if stopped_reason:
                break
        if stopped_reason:
            break

    manifest.update(
        {
            "finished_at": now_iso(),
            "requests_used": request_count,
            "completed_scenarios": completed,
            "successes": successes,
            "failures": failures,
            "stopped_reason": stopped_reason or None,
            "data_file": str(out_path.relative_to(ROOT)),
        }
    )
    write_json(manifest_path, manifest)
    print(f"free-suite scenarios={completed} requests={request_count}/{max_requests} pass={successes} fail={failures}")
    print(f"raw={out_path.relative_to(ROOT)}")
    if stopped_reason:
        print(stopped_reason)
        return 4 if stopped_reason.startswith("NONZERO_FREE_COST") else 0
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover and stress-test strictly free OpenRouter models")
    parser.add_argument("command", choices=["list", "deep"])
    parser.add_argument("--max-models", type=int, default=int(os.getenv("OPENROUTER_FREE_MAX_MODELS", "5")))
    parser.add_argument("--max-requests", type=int, default=int(os.getenv("OPENROUTER_FREE_MAX_REQUESTS", "40")))
    parser.add_argument("--include-router", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("OPENROUTER_API_KEY is required")
        return 2
    catalog = fetch_model_catalog(args.base_url, api_key)
    free_models = discover_free_models(catalog)
    if args.command == "list":
        print(f"strictly free specific models: {len(free_models)}")
        for row in free_models:
            print(f"{row.model_id:<56} context={row.context_length:<9} {row.name}")
        print("router: openrouter/free (random free model; benchmark separately)")
        return 0
    return run_deep(
        free_models,
        base_url=args.base_url,
        api_key=api_key,
        max_models=max(1, args.max_models),
        max_requests=max(1, args.max_requests),
        include_router=args.include_router,
    )


if __name__ == "__main__":
    raise SystemExit(main())
