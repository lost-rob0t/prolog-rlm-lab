from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlm import PrologToolbox, correct, normalize_answer
from tasks import make_task


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
