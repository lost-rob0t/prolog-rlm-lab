from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openrouter_free import discover_free_models, is_zero_pricing


class OpenRouterFreeTests(unittest.TestCase):
    def test_zero_pricing_requires_prompt_completion_and_request_to_be_zero(self):
        self.assertTrue(is_zero_pricing({"prompt": "0", "completion": "0", "request": "0"}))
        self.assertFalse(is_zero_pricing({"prompt": "0", "completion": "0.000001", "request": "0"}))
        self.assertFalse(is_zero_pricing({"prompt": "0", "completion": "0"}))
        self.assertFalse(is_zero_pricing(None))

    def test_discovery_keeps_only_strictly_free_specific_models(self):
        payload = {
            "data": [
                {
                    "id": "vendor/a:free",
                    "name": "A (free)",
                    "context_length": 32000,
                    "pricing": {"prompt": "0", "completion": "0", "request": "0"},
                },
                {
                    "id": "vendor/b",
                    "name": "B",
                    "context_length": 128000,
                    "pricing": {"prompt": "0.000001", "completion": "0", "request": "0"},
                },
                {
                    "id": "openrouter/free",
                    "name": "Free Models Router",
                    "context_length": 200000,
                    "pricing": {"prompt": "0", "completion": "0", "request": "0"},
                },
            ]
        }
        rows = discover_free_models(payload)
        self.assertEqual([row.model_id for row in rows], ["vendor/a:free"])

    def test_discovery_sorts_by_context_then_id(self):
        payload = {
            "data": [
                {"id": "z:free", "context_length": 1000, "pricing": {"prompt": "0", "completion": "0", "request": "0"}},
                {"id": "a:free", "context_length": 2000, "pricing": {"prompt": "0", "completion": "0", "request": "0"}},
            ]
        }
        rows = discover_free_models(payload)
        self.assertEqual([row.model_id for row in rows], ["a:free", "z:free"])


if __name__ == "__main__":
    unittest.main()
