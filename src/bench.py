from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

from providers import ModelSpec, OpenAICompatibleProvider
from rlm import correct, run_condition
from tasks import make_task
from util import ROOT, append_jsonl, load_dotenv, now_iso, stable_id, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prolog-RLM capability substitution benchmark")
    parser.add_argument("--preset", default="smoke")
    parser.add_argument("--deployment", choices=["local", "remote", "all"], default="local")
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="hard cap on run count")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config() -> tuple[dict[str, Any], list[ModelSpec]]:
    load_dotenv()
    experiment = tomllib.loads((ROOT / "config/experiment.toml").read_text())
    model_data = tomllib.loads((ROOT / "config/models.toml").read_text())
    models = [ModelSpec.from_toml(row) for row in model_data["models"]]
    return experiment, models


def resolve_matrix(config: dict[str, Any], preset_name: str) -> dict[str, Any]:
    base = config["experiment"]
    task = config["task"]
    recursion = config["recursion"]
    preset = config.get("presets", {}).get(preset_name, {})
    return {
        "trials": int(preset.get("trials", base["trials"])),
        "instances": int(preset.get("instances_per_difficulty", task["instances_per_difficulty"])),
        "families": list(preset.get("families", task["families"])),
        "difficulties": list(preset.get("difficulties", task["difficulties"])),
        "recursion_levels": list(preset.get("recursion_levels", recursion["levels"])),
        "conditions": list(preset.get("conditions", recursion["conditions"])),
        "max_tokens": int(base["max_output_tokens"]),
        "temperature": float(base["temperature"]),
        "timeout_s": int(base["request_timeout_s"]),
        "seed": int(base["seed"]),
    }


def select_models(models: list[ModelSpec], args: argparse.Namespace) -> list[ModelSpec]:
    selected = []
    requested = set(args.model_id)
    for model in models:
        if requested and model.id not in requested:
            continue
        if not requested and not model.enabled:
            continue
        if args.deployment != "all" and model.deployment != args.deployment:
            continue
        if model.deployment == "remote" and not args.allow_paid:
            continue
        selected.append(model)
    return selected


def main() -> int:
    args = parse_args()
    config, models = load_config()
    matrix = resolve_matrix(config, args.preset)
    selected = select_models(models, args)
    if not selected:
        print("No models selected. Use --model-id ID, enable a model in config/models.toml, or add --allow-paid for remote models.")
        return 2

    combinations = list(
        itertools.product(
            selected,
            matrix["families"],
            matrix["difficulties"],
            range(matrix["instances"]),
            matrix["conditions"],
            matrix["recursion_levels"],
            range(matrix["trials"]),
        )
    )
    if args.limit:
        combinations = combinations[: args.limit]
    print(f"planned runs: {len(combinations)}")
    if args.dry_run:
        for model, family, difficulty, instance, condition, recursion, trial in combinations[:50]:
            print(model.id, family, difficulty, instance, condition, recursion, trial)
        return 0

    session_id = stable_id(now_iso(), args.preset, len(combinations))
    out_path = ROOT / "results/raw" / f"{session_id}.jsonl"
    manifest_path = ROOT / "results/raw" / f"{session_id}.manifest.json"
    manifest = {
        "session_id": session_id,
        "started_at": now_iso(),
        "preset": args.preset,
        "deployment": args.deployment,
        "models": [model.id for model in selected],
        "matrix": matrix,
        "planned_runs": len(combinations),
    }
    write_json(manifest_path, manifest)

    successes = 0
    failures = 0
    started = time.perf_counter()
    for index, (model, family, difficulty, instance, condition, recursion, trial) in enumerate(combinations, 1):
        seed = matrix["seed"] + instance * 1009 + trial * 104729 + sum(map(ord, family + difficulty))
        task = make_task(family, difficulty, seed)
        run_id = stable_id(session_id, model.id, task.id, condition, recursion, trial)
        provider = OpenAICompatibleProvider(model, timeout_s=matrix["timeout_s"])
        trace = run_condition(
            provider,
            task,
            condition,
            recursion,
            max_tokens=matrix["max_tokens"],
            temperature=matrix["temperature"],
        )
        ok = not trace.error and correct(trace.answer, task.expected)
        successes += int(ok)
        failures += int(not ok)
        cost = model.cost(trace.input_tokens, trace.output_tokens)
        row = {
            "run_id": run_id,
            "session_id": session_id,
            "timestamp": now_iso(),
            "model_id": model.id,
            "model_label": model.label,
            "model_class": model.class_name,
            "parameter_b": model.parameter_b,
            "active_parameter_b": model.active_parameter_b,
            "provider": model.provider,
            "deployment": model.deployment,
            "task_id": task.id,
            "task_family": task.family,
            "difficulty": task.difficulty,
            "task_seed": task.seed,
            "task_metadata": task.metadata,
            "condition": condition,
            "recursion_level": recursion,
            "trial": trial,
            "success": ok,
            "expected": task.expected,
            "answer": trace.answer,
            "model_calls": trace.model_calls,
            "prolog_calls": trace.prolog_calls,
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "total_tokens": trace.input_tokens + trace.output_tokens,
            "model_latency_ms": round(trace.model_latency_ms, 3),
            "prolog_latency_ms": round(trace.prolog_latency_ms, 3),
            "total_latency_ms": round(trace.model_latency_ms + trace.prolog_latency_ms, 3),
            "cost_usd": round(cost, 8),
            "error": trace.error,
            "trace": trace.events,
        }
        append_jsonl(out_path, row)
        if not args.quiet:
            status = "PASS" if ok else "FAIL"
            print(
                f"[{index:>5}/{len(combinations)}] {status} {model.id:<18} {family:<14} "
                f"{difficulty:<6} {condition:<14} r={recursion:<2} tok={row['total_tokens']}"
            )
        elif index % max(1, min(25, len(combinations))) == 0 or index == len(combinations):
            print(f"progress {index}/{len(combinations)} pass={successes} fail={failures}")

    elapsed = time.perf_counter() - started
    manifest.update({
        "finished_at": now_iso(),
        "successes": successes,
        "failures": failures,
        "elapsed_s": round(elapsed, 3),
        "data_file": str(out_path.relative_to(ROOT)),
    })
    write_json(manifest_path, manifest)
    print(f"session={session_id} runs={len(combinations)} pass={successes} fail={failures} elapsed={elapsed:.1f}s")
    print(f"raw={out_path.relative_to(ROOT)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
