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

from providers import ModelSpec, OpenAICompatibleProvider
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
            "base_url": "http://127.0.0.1:8080/v1",
        }
        with mock.patch.dict(os.environ, {"RLM_TEST_MODEL": ""}, clear=False):
            spec = ModelSpec.from_toml(row)
        self.assertEqual(spec.model, "registry-model")

    @mock.patch("urllib.request.urlopen")
    def test_provider_probe_accepts_advertised_model(self, urlopen):
        urlopen.return_value = FakeResponse({"data": [{"id": "Qwen/Qwen3.5-4B"}]})
        spec = ModelSpec.from_toml({
            "id": "qwen35-4b",
            "model": "Qwen/Qwen3.5-4B",
            "base_url": "http://127.0.0.1:8080/v1",
        })
        result = OpenAICompatibleProvider(spec).probe()
        self.assertTrue(result.ok)
        self.assertEqual(result.advertised_models, ("Qwen/Qwen3.5-4B",))

    @mock.patch("urllib.request.urlopen")
    def test_provider_probe_reports_alias_mismatch(self, urlopen):
        urlopen.return_value = FakeResponse({"data": [{"id": "local-qwen.gguf"}]})
        spec = ModelSpec.from_toml({
            "id": "qwen35-4b",
            "model": "Qwen/Qwen3.5-4B",
            "model_env": "QWEN35_4B_LOCAL_MODEL",
            "base_url": "http://127.0.0.1:8080/v1",
        })
        result = OpenAICompatibleProvider(spec).probe()
        self.assertFalse(result.ok)
        self.assertIn("QWEN35_4B_LOCAL_MODEL", result.detail)
        self.assertIn("local-qwen.gguf", result.detail)

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
