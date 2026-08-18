from __future__ import annotations

import argparse
import itertools
import time
import tomllib
from typing import Any

from providers import CostBudget, ModelSpec, OpenAICompatibleProvider
from rlm import correct, run_condition
from tasks import make_task
from util import ROOT, append_jsonl, load_dotenv, now_iso, stable_id, write_json


SINGLE_CALL_CONDITIONS = {"base", "vanilla"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prolog-RLM capability substitution benchmark")
    parser.add_argument("--preset", default="smoke")
    parser.add_argument("--deployment", choices=["local", "remote", "all"], default="local")
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="hard cap on run count")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.0,
        help="application-side paid-request fuse; refuses the next call if it could cross this spend",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip /models endpoint validation (not recommended for real runs)",
    )
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
        "max_tokens": int(preset.get("max_output_tokens", base["max_output_tokens"])),
        "temperature": float(preset.get("temperature", base["temperature"])),
        "timeout_s": int(preset.get("request_timeout_s", base["request_timeout_s"])),
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


def build_combinations(
    selected: list[ModelSpec], matrix: dict[str, Any], *, interleave_models: bool
) -> list[tuple[ModelSpec, str, str, int, str, int, int]]:
    recursion_levels = matrix["recursion_levels"]
    if not recursion_levels:
        return []
    first_recursion = recursion_levels[0]

    rows: list[tuple[ModelSpec, str, str, int, str, int, int]] = []
    if not interleave_models:
        rows = list(
            itertools.product(
                selected,
                matrix["families"],
                matrix["difficulties"],
                range(matrix["instances"]),
                matrix["conditions"],
                recursion_levels,
                range(matrix["trials"]),
            )
        )
    else:
        for family, difficulty, instance, condition, recursion, trial, model in itertools.product(
            matrix["families"],
            matrix["difficulties"],
            range(matrix["instances"]),
            matrix["conditions"],
            recursion_levels,
            range(matrix["trials"]),
            selected,
        ):
            rows.append((model, family, difficulty, instance, condition, recursion, trial))

    # A one-shot base model has no recursion axis. Repeating it at r=2/4/8/etc.
    # only duplicates identical work, biases aggregates, and wastes paid calls.
    return [
        row
        for row in rows
        if row[4] not in SINGLE_CALL_CONDITIONS or row[5] == first_recursion
    ]


def preflight_models(models: list[ModelSpec], timeout_s: int) -> bool:
    print("provider preflight:")
    all_ok = True
    for model in models:
        result = OpenAICompatibleProvider(model, timeout_s=timeout_s).probe()
        status = "OK" if result.ok else "FAIL"
        print(f"  [{status}] {model.id}: {result.detail}")
        all_ok = all_ok and result.ok
    return all_ok


def main() -> int:
    args = parse_args()
    if args.max_cost_usd < 0:
        print("--max-cost-usd must be non-negative")
        return 2

    config, models = load_config()
    matrix = resolve_matrix(config, args.preset)
    selected = select_models(models, args)
    if not selected:
        print("No models selected. Use --model-id ID, enable a model in config/models.toml, or add --allow-paid for remote models.")
        return 2

    if args.max_cost_usd > 0:
        unpriced = [
            model.id
            for model in selected
            if model.input_cost_per_million <= 0 or model.output_cost_per_million <= 0
        ]
        if unpriced:
            print(
                "hard cost budget requires nonzero input/output pricing metadata; missing: "
                + ", ".join(unpriced)
            )
            return 2

    combinations = build_combinations(
        selected,
        matrix,
        interleave_models=args.max_cost_usd > 0 and len(selected) > 1,
    )
    if args.limit:
        combinations = combinations[: args.limit]
    print(f"planned runs: {len(combinations)}")
    if args.max_cost_usd > 0:
        print(f"cost fuse: ${args.max_cost_usd:.2f} (models interleaved)")
    if args.dry_run:
        for model, family, difficulty, instance, condition, recursion, trial in combinations[:50]:
            print(model.id, family, difficulty, instance, condition, recursion, trial)
        return 0

    if not args.skip_preflight and not preflight_models(selected, matrix["timeout_s"]):
        print("preflight failed; no benchmark rows were written")
        return 3

    budget = CostBudget(args.max_cost_usd) if args.max_cost_usd > 0 else None
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
        "preflight": "skipped" if args.skip_preflight else "passed",
        "cost_budget_usd": args.max_cost_usd or None,
    }
    write_json(manifest_path, manifest)

    successes = 0
    failures = 0
    runtime_errors = 0
    completed_runs = 0
    stopped_reason = ""
    started = time.perf_counter()
    for index, (model, family, difficulty, instance, condition, recursion, trial) in enumerate(combinations, 1):
        seed = matrix["seed"] + instance * 1009 + trial * 104729 + sum(map(ord, family + difficulty))
        task = make_task(family, difficulty, seed)
        run_id = stable_id(session_id, model.id, task.id, condition, recursion, trial)
        provider = OpenAICompatibleProvider(
            model,
            timeout_s=matrix["timeout_s"],
            budget=budget,
        )
        trace = run_condition(
            provider,
            task,
            condition,
            recursion,
            max_tokens=matrix["max_tokens"],
            temperature=matrix["temperature"],
        )

        if trace.error.startswith("COST_BUDGET_EXCEEDED:"):
            stopped_reason = trace.error
            print(stopped_reason)
            print("cost fuse stopped the matrix before another paid request")
            break

        runtime_errors += int(bool(trace.error))
        ok = not trace.error and correct(trace.answer, task.expected)
        successes += int(ok)
        failures += int(not ok)
        completed_runs += 1
        cost = trace.cost_usd if trace.cost_usd > 0 else model.cost(trace.input_tokens, trace.output_tokens)
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
            "provider_model": model.model,
            "provider_base_url": model.base_url,
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
            "session_cost_usd": round(budget.spent_usd, 8) if budget else 0.0,
            "error": trace.error,
            "trace": trace.events,
        }
        append_jsonl(out_path, row)
        if not args.quiet:
            status = "PASS" if ok else "FAIL"
            spend = f" cost=${budget.spent_usd:.4f}" if budget else ""
            print(
                f"[{index:>5}/{len(combinations)}] {status} {model.id:<18} {family:<14} "
                f"{difficulty:<6} {condition:<14} r={recursion:<2} tok={row['total_tokens']}{spend}"
            )
        elif index % max(1, min(25, len(combinations))) == 0 or index == len(combinations):
            spend = f" cost=${budget.spent_usd:.4f}" if budget else ""
            print(f"progress {index}/{len(combinations)} pass={successes} fail={failures}{spend}")

    elapsed = time.perf_counter() - started
    manifest.update({
        "finished_at": now_iso(),
        "completed_runs": completed_runs,
        "successes": successes,
        "failures": failures,
        "runtime_errors": runtime_errors,
        "elapsed_s": round(elapsed, 3),
        "cost_spent_usd": round(budget.spent_usd, 8) if budget else 0.0,
        "stopped_reason": stopped_reason or None,
        "data_file": str(out_path.relative_to(ROOT)),
    })
    write_json(manifest_path, manifest)
    print(
        f"session={session_id} runs={completed_runs}/{len(combinations)} "
        f"pass={successes} fail={failures} errors={runtime_errors} elapsed={elapsed:.1f}s"
    )
    if budget:
        print(f"spend=${budget.spent_usd:.4f}/${budget.limit_usd:.2f}")
    print(f"raw={out_path.relative_to(ROOT)}")
    if stopped_reason:
        return 4
    if runtime_errors:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
