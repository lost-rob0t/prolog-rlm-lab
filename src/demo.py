from __future__ import annotations

import math
import random
from pathlib import Path

from util import ROOT, append_jsonl, now_iso


def main() -> int:
    path = ROOT / "results/raw/demo-synthetic.jsonl"
    path.unlink(missing_ok=True)
    rng = random.Random(545)
    models = [
        ("demo-2b", "Demo 2B", "1to3b", 2.0),
        ("demo-9b", "Demo 9B", "7to12b", 9.0),
        ("demo-27b", "Demo 27B", "20to30b-dense", 27.0),
    ]
    conditions = ["vanilla", "self_recursive", "rlm", "rlm_verified"]
    recursions = [1, 2, 4, 8, 16]
    families = ["reachability", "constraints", "state", "authorization"]
    run = 0
    for model_id, label, cls, params in models:
        capability = math.log2(params + 1) / math.log2(28)
        for condition in conditions:
            bonus = {"vanilla": 0.0, "self_recursive": 0.05, "rlm": 0.18, "rlm_verified": 0.25}[condition]
            for recursion in recursions:
                recursion_gain = min(0.18, math.log2(recursion) * (0.025 if condition == "self_recursive" else 0.045))
                for family in families:
                    for trial in range(8):
                        run += 1
                        family_penalty = {"reachability": 0.02, "constraints": 0.08, "state": 0.06, "authorization": 0.03}[family]
                        p = min(0.98, max(0.05, 0.28 + capability * 0.50 + bonus + recursion_gain - family_penalty))
                        success = rng.random() < p
                        model_calls = 1 if condition == "vanilla" else min(recursion, 1 + int(rng.random() * max(1, recursion)))
                        prolog_calls = 0 if condition in {"vanilla", "self_recursive"} else max(1, model_calls - 1)
                        tokens = int((380 + params * 10) * model_calls * (1.0 if condition != "rlm_verified" else 0.86))
                        row = {
                            "run_id": f"demo-{run:06d}",
                            "session_id": "demo-synthetic",
                            "timestamp": now_iso(),
                            "model_id": model_id,
                            "model_label": label,
                            "model_class": cls,
                            "parameter_b": params,
                            "active_parameter_b": params,
                            "provider": "synthetic",
                            "deployment": "demo",
                            "task_id": f"demo-{family}-{trial}",
                            "task_family": family,
                            "difficulty": "demo",
                            "task_seed": 545 + run,
                            "condition": condition,
                            "recursion_level": recursion,
                            "trial": trial,
                            "success": success,
                            "expected": "DEMO",
                            "answer": "DEMO",
                            "model_calls": model_calls,
                            "prolog_calls": prolog_calls,
                            "input_tokens": int(tokens * 0.72),
                            "output_tokens": int(tokens * 0.28),
                            "total_tokens": tokens,
                            "model_latency_ms": tokens * 0.8,
                            "prolog_latency_ms": prolog_calls * 2.5,
                            "total_latency_ms": tokens * 0.8 + prolog_calls * 2.5,
                            "cost_usd": 0.0,
                            "error": "",
                            "trace": [],
                        }
                        append_jsonl(path, row)
    print(f"synthetic demo rows: {run}")
    print(f"raw={path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
