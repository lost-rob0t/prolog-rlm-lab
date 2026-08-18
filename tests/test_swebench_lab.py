from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swebench_lab import CONDITIONS, extract_patch, prediction_record, safe_repo_path


class SwebenchLabTests(unittest.TestCase):
    def test_conditions_form_causal_ladder(self):
        self.assertEqual(CONDITIONS, ("base", "agent_plain", "rlm_agent"))

    def test_extract_patch_strips_fences_and_preamble(self):
        text = "Here is the fix:\n```diff\ndiff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n```"
        patch = extract_patch(text)
        self.assertTrue(patch.startswith("diff --git a/a.py b/a.py"))
        self.assertIn("+new", patch)
        self.assertNotIn("```", patch)

    def test_prediction_record_matches_official_shape(self):
        row = prediction_record("sympy__sympy-1", "model#base", "diff --git a/a b/a\n")
        self.assertEqual(
            set(row),
            {"instance_id", "model_name_or_path", "model_patch"},
        )

    def test_safe_repo_path_rejects_escape(self):
        root = Path("/tmp/workspace").resolve()
        with self.assertRaises(ValueError):
            safe_repo_path(root, "../etc/passwd")
        self.assertEqual(safe_repo_path(root, "src/a.py"), root / "src/a.py")


if __name__ == "__main__":
    unittest.main()
