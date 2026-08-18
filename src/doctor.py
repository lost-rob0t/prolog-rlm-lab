from __future__ import annotations

import argparse
import tomllib

from providers import ModelSpec, OpenAICompatibleProvider
from util import ROOT, load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark model endpoints before running")
    parser.add_argument("--deployment", choices=["local", "remote", "all"], default="local")
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--timeout", type=int, default=15)
    return parser.parse_args()


def load_models() -> list[ModelSpec]:
    load_dotenv()
    model_data = tomllib.loads((ROOT / "config/models.toml").read_text())
    return [ModelSpec.from_toml(row) for row in model_data["models"]]


def select_models(models: list[ModelSpec], args: argparse.Namespace) -> list[ModelSpec]:
    requested = set(args.model_id)
    selected: list[ModelSpec] = []
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
    selected = select_models(load_models(), args)
    if not selected:
        print("No models selected.")
        return 2

    failures = 0
    for model in selected:
        result = OpenAICompatibleProvider(model, timeout_s=args.timeout).probe()
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {model.id}")
        print(f"  endpoint: {model.base_url or '<unset>'}")
        print(f"  model:    {model.model or '<unset>'}")
        print(f"  detail:   {result.detail}")
        failures += int(not result.ok)

    if failures:
        print(f"doctor: {len(selected) - failures} ready, {failures} failed")
        return 1

    print(f"doctor: {len(selected)} model(s) ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
