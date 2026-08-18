from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from providers import Completion
from rlm import run_condition
from tasks import STARINTEL_FAMILIES, make_task


class FakeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, max_tokens: int, temperature: float) -> Completion:
        self.calls += 1
        return Completion(
            text='{"answer":"YES"}',
            input_tokens=100,
            output_tokens=10,
            latency_ms=1.0,
        )


class BenchmarkLayerTests(unittest.TestCase):
    def test_base_is_one_call(self):
        provider = FakeProvider()
        task = make_task("reachability", "easy", 545)
        trace = run_condition(
            provider,
            task,
            "base",
            8,
            max_tokens=128,
            temperature=0.0,
        )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(trace.model_calls, 1)
        self.assertEqual(trace.prolog_calls, 0)

    def test_plain_agent_uses_turn_budget_without_prolog(self):
        provider = FakeProvider()
        task = make_task("reachability", "easy", 545)
        trace = run_condition(
            provider,
            task,
            "agent_plain",
            4,
            max_tokens=128,
            temperature=0.0,
        )
        self.assertEqual(provider.calls, 4)
        self.assertEqual(trace.model_calls, 4)
        self.assertEqual(trace.prolog_calls, 0)

    def test_starintel_families_are_deterministic(self):
        self.assertGreaterEqual(len(STARINTEL_FAMILIES), 4)
        for family in STARINTEL_FAMILIES:
            with self.subTest(family=family):
                a = make_task(family, "medium", 545)
                b = make_task(family, "medium", 545)
                self.assertEqual(a.id, b.id)
                self.assertEqual(a.prompt, b.prompt)
                self.assertEqual(a.expected, b.expected)
                self.assertTrue(a.tools)
                self.assertTrue(a.metadata.get("starintel"))


if __name__ == "__main__":
    unittest.main()
