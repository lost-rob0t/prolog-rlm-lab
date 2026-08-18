from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import matplotlib.pyplot as plt

from util import ROOT

SUMMARY = ROOT / "results/summary/summary.csv"
PLOTS = ROOT / "results/plots"


def load_rows() -> list[dict[str, Any]]:
    with SUMMARY.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "parameter_b", "active_parameter_b", "recursion_level", "runs", "success_rate",
        "avg_total_tokens", "avg_total_latency_ms", "avg_model_calls", "avg_prolog_calls",
        "symbolic_work_share", "token_offload_ratio", "cost_per_success_usd", "failure_rate",
    }
    for row in rows:
        for key in numeric:
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def save(name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS / name, format="svg", bbox_inches="tight")
    plt.close()


def grouped_mean(rows: list[dict[str, Any]], keys: tuple[str, ...], value: str) -> dict[tuple[Any, ...], float]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(float(row[value]))
    return {k: mean(v) for k, v in groups.items()}


def line_by_condition(rows: list[dict[str, Any]], metric: str, ylabel: str, filename: str) -> None:
    grouped = grouped_mean(rows, ("condition", "recursion_level"), metric)
    conditions = sorted({k[0] for k in grouped})
    plt.figure(figsize=(10, 5.6))
    for condition in conditions:
        points = sorted((float(k[1]), v) for k, v in grouped.items() if k[0] == condition)
        if points:
            xs, ys = zip(*points)
            plt.plot(xs, ys, marker="o", label=condition)
    plt.xlabel("Recursion level")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by recursion level")
    plt.legend()
    plt.grid(alpha=0.18)
    save(filename)


def success_by_model(rows: list[dict[str, Any]]) -> None:
    grouped = grouped_mean(rows, ("model_label", "condition"), "success_rate")
    labels = sorted({k[0] for k in grouped})
    conditions = sorted({k[1] for k in grouped})
    width = 0.8 / max(1, len(conditions))
    xs = list(range(len(labels)))
    plt.figure(figsize=(max(10, len(labels) * 1.2), 6))
    for ci, condition in enumerate(conditions):
        vals = [grouped.get((label, condition), 0.0) for label in labels]
        offsets = [x - 0.4 + width / 2 + ci * width for x in xs]
        plt.bar(offsets, vals, width=width, label=condition)
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.ylabel("Success rate")
    plt.ylim(0, 1.05)
    plt.title("Success by model and condition")
    plt.legend()
    plt.grid(axis="y", alpha=0.18)
    save("success-by-model.svg")


def success_by_family(rows: list[dict[str, Any]]) -> None:
    grouped = grouped_mean(rows, ("task_family", "condition"), "success_rate")
    families = sorted({k[0] for k in grouped})
    conditions = sorted({k[1] for k in grouped})
    width = 0.8 / max(1, len(conditions))
    xs = list(range(len(families)))
    plt.figure(figsize=(10, 5.8))
    for ci, condition in enumerate(conditions):
        vals = [grouped.get((family, condition), 0.0) for family in families]
        offsets = [x - 0.4 + width / 2 + ci * width for x in xs]
        plt.bar(offsets, vals, width=width, label=condition)
    plt.xticks(xs, families, rotation=20)
    plt.ylabel("Success rate")
    plt.ylim(0, 1.05)
    plt.title("Success by task family")
    plt.legend()
    plt.grid(axis="y", alpha=0.18)
    save("success-by-family.svg")


def scatter_tokens(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 6))
    conditions = sorted({r["condition"] for r in rows})
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        plt.scatter([r["avg_total_tokens"] for r in subset], [r["success_rate"] for r in subset], alpha=0.65, label=condition)
    plt.xlabel("Average model tokens")
    plt.ylabel("Success rate")
    plt.ylim(-0.03, 1.03)
    plt.title("Accuracy vs model-token consumption")
    plt.legend()
    plt.grid(alpha=0.18)
    save("accuracy-vs-tokens.svg")


def scatter_params(rows: list[dict[str, Any]]) -> None:
    subset = [r for r in rows if float(r.get("parameter_b", 0)) > 0]
    if not subset:
        return
    plt.figure(figsize=(9, 6))
    for condition in sorted({r["condition"] for r in subset}):
        rs = [r for r in subset if r["condition"] == condition]
        plt.scatter([r["parameter_b"] for r in rs], [r["success_rate"] for r in rs], alpha=0.65, label=condition)
    plt.xscale("log")
    plt.xlabel("Declared total parameters (billions, log scale)")
    plt.ylabel("Success rate")
    plt.ylim(-0.03, 1.03)
    plt.title("Accuracy vs open-weight parameter scale")
    plt.legend()
    plt.grid(alpha=0.18)
    save("accuracy-vs-parameters.svg")


def heatmap(rows: list[dict[str, Any]]) -> None:
    grouped = grouped_mean(rows, ("model_label", "task_family"), "success_rate")
    models = sorted({k[0] for k in grouped})
    families = sorted({k[1] for k in grouped})
    if not models or not families:
        return
    matrix = [[grouped.get((model, family), math.nan) for family in families] for model in models]
    plt.figure(figsize=(max(8, len(families) * 1.8), max(4.8, len(models) * 0.7)))
    image = plt.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(image, label="Success rate")
    plt.xticks(range(len(families)), families, rotation=25, ha="right")
    plt.yticks(range(len(models)), models)
    plt.title("Task-family performance heatmap")
    save("task-family-heatmap.svg")


def pareto(rows: list[dict[str, Any]]) -> None:
    grouped = grouped_mean(rows, ("model_label", "condition"), "success_rate")
    token_group = grouped_mean(rows, ("model_label", "condition"), "avg_total_tokens")
    points = []
    for key, success in grouped.items():
        points.append((token_group.get(key, 0.0), success, key))
    plt.figure(figsize=(10, 6))
    for tokens, success, (model, condition) in points:
        plt.scatter(tokens, success)
        plt.annotate(f"{model}\n{condition}", (tokens, success), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Average model tokens")
    plt.ylabel("Success rate")
    plt.ylim(-0.03, 1.03)
    plt.title("Capability / token Pareto field")
    plt.grid(alpha=0.18)
    save("pareto.svg")


def main() -> int:
    if not SUMMARY.exists() or SUMMARY.stat().st_size == 0:
        print("summary.csv missing; run analyze first")
        return 2
    rows = load_rows()
    if not rows:
        print("summary.csv is empty")
        return 2
    success_by_model(rows)
    success_by_family(rows)
    line_by_condition(rows, "success_rate", "Success rate", "success-by-recursion.svg")
    line_by_condition(rows, "avg_total_tokens", "Average model tokens", "tokens-by-recursion.svg")
    line_by_condition(rows, "avg_total_latency_ms", "Average latency (ms)", "latency-by-recursion.svg")
    line_by_condition(rows, "avg_model_calls", "Average model calls", "model-calls-by-recursion.svg")
    line_by_condition(rows, "avg_prolog_calls", "Average Prolog calls", "prolog-calls-by-recursion.svg")
    line_by_condition(rows, "symbolic_work_share", "Symbolic work share", "symbolic-work-share.svg")
    line_by_condition(rows, "token_offload_ratio", "Token offload ratio", "token-offload-ratio.svg")
    line_by_condition(rows, "failure_rate", "Failure rate", "failure-rate.svg")
    line_by_condition(rows, "cost_per_success_usd", "Cost per success (USD)", "cost-per-success.svg")
    scatter_tokens(rows)
    scatter_params(rows)
    heatmap(rows)
    pareto(rows)

    generated = sorted(PLOTS.glob("*.svg"))
    org = ["* Graphs\n\n", "The report generates one SVG per core metric so the plots remain sharp when zoomed or printed.\n\n"]
    for path in generated:
        title = path.stem.replace("-", " ").title()
        org += [f"** {title}\n\n", f"[[file:results/plots/{path.name}]]\n\n"]
    (ROOT / "results/summary/plots.org").write_text("".join(org), encoding="utf-8")
    print(f"plots generated: {len(generated)}")
    for path in generated:
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
