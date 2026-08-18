from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bench import resolve_matrix
from providers import BudgetExceeded, CostBudget, ModelSpec, OpenAICompatibleProvider
from rlm import PrologToolbox, correct, normalize_answer
from tasks import make_task


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload


class LabTests(unittest.TestCase):
    def test_answer_normalization(self):
        self.assertEqual(normalize_answer('{"answer":"YES"}'), "YES")
        self.assertEqual(normalize_answer("The answer is no."), "NO")
        self.assertEqual(normalize_answer('{"answer":"-17"}'), "-17")
        self.assertTrue(correct("YES", "YES"))

    def test_task_determinism(self):
        a = make_task("reachability", "medium", 545)
        b = make_task("reachability", "medium", 545)
        self.assertEqual(a.id, b.id)
        self.assertEqual(a.prompt, b.prompt)
        self.assertEqual(a.expected, b.expected)

    def test_blank_env_override_preserves_registry_default(self):
        row = {
            "id": "test",
            "model": "registry-model",
            "model_env": "RLM_TEST_MODEL",
            "base_url": "http://127.0.0.1:8000/v1",
        }
        with mock.patch.dict(os.environ, {"RLM_TEST_MODEL": ""}, clear=False):
            spec = ModelSpec.from_toml(row)
        self.assertEqual(spec.model, "registry-model")

    @mock.patch("urllib.request.urlopen")
    def test_provider_probe_accepts_advertised_model(self, urlopen):
        urlopen.return_value = FakeResponse({"data": [{"id": "Qwen3-8B-Q4_K_M"}]})
        spec = ModelSpec.from_toml({
            "id": "qwen3-8b-q4-k-m",
            "model": "Qwen3-8B-Q4_K_M",
            "base_url": "http://127.0.0.1:8000/v1",
        })
        result = OpenAICompatibleProvider(spec).probe()
        self.assertTrue(result.ok)
        self.assertEqual(result.advertised_models, ("Qwen3-8B-Q4_K_M",))

    @mock.patch("urllib.request.urlopen")
    def test_provider_probe_reports_alias_mismatch(self, urlopen):
        urlopen.return_value = FakeResponse({"data": [{"id": "wrong-model"}]})
        spec = ModelSpec.from_toml({
            "id": "qwen35-4b",
            "model": "Qwen/Qwen3.5-4B",
            "model_env": "QWEN35_4B_LOCAL_MODEL",
            "base_url": "http://127.0.0.1:8000/v1",
        })
        result = OpenAICompatibleProvider(spec).probe()
        self.assertFalse(result.ok)
        self.assertIn("QWEN35_4B_LOCAL_MODEL", result.detail)
        self.assertIn("wrong-model", result.detail)

    def test_cost_budget_rejects_call_that_could_cross_limit(self):
        spec = ModelSpec.from_toml({
            "id": "paid",
            "model": "paid/model",
            "base_url": "https://example.invalid/v1",
            "input_cost_per_million": 5.0,
            "output_cost_per_million": 30.0,
        })
        budget = CostBudget(limit_usd=0.01)
        with self.assertRaises(BudgetExceeded):
            budget.ensure_can_call(
                spec,
                [{"role": "user", "content": "x" * 4000}],
                max_tokens=1024,
            )

    @mock.patch("urllib.request.urlopen")
    def test_provider_records_actual_cost_against_shared_budget(self, urlopen):
        urlopen.return_value = FakeResponse({
            "choices": [{"message": {"content": '{"answer":"YES"}'}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "cost": 0.0125,
            },
        })
        spec = ModelSpec.from_toml({
            "id": "paid",
            "model": "paid/model",
            "base_url": "https://example.invalid/v1",
            "input_cost_per_million": 5.0,
            "output_cost_per_million": 30.0,
        })
        budget = CostBudget(limit_usd=1.0)
        completion = OpenAICompatibleProvider(spec, budget=budget).complete(
            [{"role": "user", "content": "test"}],
            max_tokens=128,
            temperature=0.0,
        )
        self.assertAlmostEqual(completion.cost_usd, 0.0125)
        self.assertAlmostEqual(budget.spent_usd, 0.0125)

    def test_frontier5_preset_is_small_and_output_capped(self):
        config = {
            "experiment": {
                "trials": 5,
                "max_output_tokens": 1024,
                "temperature": 0.0,
                "request_timeout_s": 180,
                "seed": 545,
            },
            "task": {
                "families": ["reachability", "constraints", "state", "authorization"],
                "difficulties": ["easy", "medium", "hard"],
                "instances_per_difficulty": 10,
            },
            "recursion": {
                "levels": [1, 2, 4, 8, 16],
                "conditions": ["vanilla", "self_recursive", "rlm", "rlm_verified"],
            },
            "presets": {
                "frontier5": {
                    "trials": 1,
                    "instances_per_difficulty": 2,
                    "difficulties": ["medium", "hard"],
                    "recursion_levels": [1, 2],
                    "conditions": ["vanilla", "rlm_verified"],
                    "max_output_tokens": 256,
                }
            },
        }
        matrix = resolve_matrix(config, "frontier5")
        self.assertEqual(matrix["max_tokens"], 256)
        self.assertEqual(matrix["recursion_levels"], [1, 2])
        self.assertEqual(matrix["conditions"], ["vanilla", "rlm_verified"])
        runs_per_model = (
            len(matrix["families"])
            * len(matrix["difficulties"])
            * matrix["instances"]
            * len(matrix["conditions"])
            * len(matrix["recursion_levels"])
            * matrix["trials"]
        )
        self.assertEqual(runs_per_model, 64)

    @unittest.skipUnless(shutil.which("swipl"), "SWI-Prolog not installed")
    def test_reachability_tool(self):
        task = make_task("reachability", "easy", 545)
        result, _ = PrologToolbox(task).call("reachable", ["n0", "n3"])
        self.assertEqual(result, "[[]]")

    @unittest.skipUnless(shutil.which("swipl"), "SWI-Prolog not installed")
    def test_state_tool(self):
        task = make_task("state", "easy", 545)
        result, _ = PrologToolbox(task).call("final_value", ["?X"])
        self.assertTrue(result.startswith("[["))


if __name__ == "__main__":
    unittest.main()
