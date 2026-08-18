from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from util import ROOT, read_jsonl, write_csv, write_json


def avg(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0) or 0) for r in rows]
    return mean(vals) if vals else 0.0


def success_rate(rows: list[dict[str, Any]]) -> float:
    return mean([1.0 if r.get("success") else 0.0 for r in rows]) if rows else 0.0


def group(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(k) for k in keys)].append(row)
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("model_id", "model_label", "model_class", "parameter_b", "active_parameter_b", "deployment", "condition", "recursion_level", "task_family", "difficulty")
    result = []
    for values, subset in sorted(group(rows, keys).items(), key=lambda kv: tuple(map(str, kv[0]))):
        item = dict(zip(keys, values))
        calls = avg(subset, "model_calls") + avg(subset, "prolog_calls")
        item.update({
            "runs": len(subset),
            "success_rate": round(success_rate(subset), 6),
            "avg_total_tokens": round(avg(subset, "total_tokens"), 3),
            "avg_input_tokens": round(avg(subset, "input_tokens"), 3),
            "avg_output_tokens": round(avg(subset, "output_tokens"), 3),
            "avg_total_latency_ms": round(avg(subset, "total_latency_ms"), 3),
            "avg_model_latency_ms": round(avg(subset, "model_latency_ms"), 3),
            "avg_prolog_latency_ms": round(avg(subset, "prolog_latency_ms"), 3),
            "avg_model_calls": round(avg(subset, "model_calls"), 3),
            "avg_prolog_calls": round(avg(subset, "prolog_calls"), 3),
            "symbolic_work_share": round(avg(subset, "prolog_calls") / calls, 6) if calls else 0.0,
            "total_cost_usd": round(sum(float(r.get("cost_usd", 0) or 0) for r in subset), 8),
            "failure_rate": round(1.0 - success_rate(subset), 6),
        })
        successes = sum(1 for r in subset if r.get("success"))
        item["cost_per_success_usd"] = round(item["total_cost_usd"] / successes, 8) if successes else 0.0
        result.append(item)
    return result


def add_token_offload(summary: list[dict[str, Any]]) -> None:
    baseline: dict[tuple[Any, ...], float] = {}
    for row in summary:
        if row["condition"] == "vanilla":
            key = (row["model_id"], row["recursion_level"], row["task_family"], row["difficulty"])
            baseline[key] = float(row["avg_total_tokens"])
    for row in summary:
        key = (row["model_id"], row["recursion_level"], row["task_family"], row["difficulty"])
        base = baseline.get(key, 0.0)
        current = float(row["avg_total_tokens"])
        row["token_offload_ratio"] = round(1.0 - current / base, 6) if base > 0 else 0.0


def capability_replacement(summary: list[dict[str, Any]], threshold: float = 0.80) -> dict[str, Any]:
    collapsed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        if float(row.get("parameter_b", 0) or 0) > 0:
            collapsed[(row["model_id"], row["condition"])].append(row)
    scores = []
    for (model_id, condition), rows in collapsed.items():
        scores.append({
            "model_id": model_id,
            "condition": condition,
            "parameter_b": float(rows[0]["parameter_b"]),
            "success_rate": mean(float(r["success_rate"]) for r in rows),
        })
    vanilla = sorted((r for r in scores if r["condition"] == "vanilla" and r["success_rate"] >= threshold), key=lambda r: r["parameter_b"])
    rlm = sorted((r for r in scores if r["condition"] == "rlm_verified" and r["success_rate"] >= threshold), key=lambda r: r["parameter_b"])
    if not vanilla or not rlm or rlm[0]["parameter_b"] <= 0:
        return {"threshold": threshold, "crr": None, "vanilla": vanilla[:1], "rlm_verified": rlm[:1]}
    return {
        "threshold": threshold,
        "crr": round(vanilla[0]["parameter_b"] / rlm[0]["parameter_b"], 4),
        "vanilla": vanilla[0],
        "rlm_verified": rlm[0],
    }


def org_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int = 30) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |\n"
    sep = "|" + "+".join("-" * (len(label) + 2) for _, label in columns) + "|\n"
    body = []
    for row in rows[:limit]:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(str(value).replace("|", "¦"))
        body.append("| " + " | ".join(cells) + " |\n")
    return header + sep + "".join(body)


def write_org(summary: list[dict[str, Any]], crr: dict[str, Any], raw_count: int) -> None:
    overall_rows = []
    grouped = group(summary, ("model_id", "model_label", "model_class", "condition", "recursion_level"))
    for values, subset in grouped.items():
        model_id, label, cls, condition, recursion = values
        weights = [max(1, int(r["runs"])) for r in subset]
        total_weight = sum(weights)
        weighted = lambda key: sum(float(r[key]) * w for r, w in zip(subset, weights)) / total_weight
        overall_rows.append({
            "model_id": model_id,
            "model_label": label,
            "model_class": cls,
            "condition": condition,
            "recursion_level": recursion,
            "success_rate": weighted("success_rate"),
            "avg_total_tokens": weighted("avg_total_tokens"),
            "avg_total_latency_ms": weighted("avg_total_latency_ms"),
            "avg_model_calls": weighted("avg_model_calls"),
            "avg_prolog_calls": weighted("avg_prolog_calls"),
            "symbolic_work_share": weighted("symbolic_work_share"),
            "failure_rate": weighted("failure_rate"),
        })
    overall_rows.sort(key=lambda r: (str(r["model_id"]), str(r["condition"]), int(r["recursion_level"])))

    crr_text = "not established"
    if crr.get("crr") is not None:
        crr_text = f"{crr['crr']:.2f}x at success threshold {crr['threshold']:.0%}"
    text = f"""* Results dashboard

#+begin_export html
<div class="metric-strip">
  <div class="metric"><span>Raw runs</span><strong>{raw_count:,}</strong></div>
  <div class="metric"><span>CRR</span><strong>{crr_text}</strong></div>
  <div class="metric"><span>Primary RLM mode</span><strong>rlm_verified</strong></div>
  <div class="metric"><span>Evidence</span><strong>machine-scored</strong></div>
</div>
#+end_export

** Aggregate metrics

"""
    text += org_table(
        overall_rows,
        [
            ("model_label", "Model"),
            ("model_class", "Class"),
            ("condition", "Condition"),
            ("recursion_level", "R"),
            ("success_rate", "Success"),
            ("avg_total_tokens", "Tokens"),
            ("avg_model_calls", "Model calls"),
            ("avg_prolog_calls", "Prolog calls"),
            ("symbolic_work_share", "SWS"),
            ("avg_total_latency_ms", "Latency ms"),
        ],
        limit=120,
    )
    text += "\n** Capability Replacement Ratio\n\n"
    text += f"Threshold: {crr['threshold']:.0%}. CRR: *{crr_text}*.\n\n"
    if crr.get("vanilla"):
        text += f"Smallest vanilla model above threshold: ={crr['vanilla']}=.\n\n"
    if crr.get("rlm_verified"):
        text += f"Smallest RLM-verified model above threshold: ={crr['rlm_verified']}=.\n"
    (ROOT / "results/summary/metrics.org").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()
    paths = sorted((ROOT / "results/raw").glob("*.jsonl"))
    rows = read_jsonl(paths)
    if not rows:
        print("No raw result rows found. Run ./scripts/demo or a benchmark first.")
        return 2
    summary = summarize(rows)
    add_token_offload(summary)
    crr = capability_replacement(summary, args.threshold)
    write_csv(ROOT / "results/summary/summary.csv", summary)
    write_json(ROOT / "results/summary/capability.json", crr)
    write_org(summary, crr, len(rows))
    print(f"raw rows: {len(rows)}")
    print(f"summary rows: {len(summary)}")
    print(f"CRR: {crr.get('crr')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
