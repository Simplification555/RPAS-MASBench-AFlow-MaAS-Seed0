"""Regression checks for recovered RPAS MaAS MATH prompt helpers."""

from __future__ import annotations

import hashlib
import shutil
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from maas_context_guard import patch_template_directory


_BASELINE = (
    Path(__file__).resolve().parents[1]
    / "external_baselines/MaAS/maas/ext/maas/scripts/optimized/MATH"
)
_PINNED_SHA256 = {
    "prompt.py": "a0734f6491d7f9cce6f4cfb814e2c91bb8a13904856ecac1a556f90bb116f8a2",
    "op_prompt.py": "07a78706b8c28800e1e4773497a4d0d5f1358d53a9e6794e13798946951b7c94",
}
_RECOVERED_HELPER_SHA256 = {
    "maas_prompt_repair.py": "dbfad9d7c66e1b80df18a2550a9c1c6a4766943d900d909810ed7bbc9fa2f63c",
    "maas_concise_prompts_v2.py": "a4af1fd1f791b6b9fdb3a98c74c92418222ac3b80cc692b15ebe216b787f583f",
}


def _load_prompts(directory: Path) -> tuple[dict, dict]:
    prompt, operator = {}, {}
    exec((directory / "prompt.py").read_text(encoding="utf-8"), prompt)
    exec((directory / "op_prompt.py").read_text(encoding="utf-8"), operator)
    return prompt, operator


class MaASPromptRepairTests(unittest.TestCase):
    def test_recovered_helpers_match_source_hashes(self) -> None:
        for name, expected in _RECOVERED_HELPER_SHA256.items():
            digest = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, f"recovered RPAS helper differs: {name}")

    def _check_repair(self, directory: Path, concise: bool) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            result = patch_template_directory(directory, concise=concise)
        prompt, operator = _load_prompts(directory)

        self.assertEqual(result["concise_prompt_variant"], "maas_concise_v2" if concise else "native_repaired")
        self.assertEqual(result["prompt_adapter_origin"], "recovered_rpas_working_copy")
        self.assertEqual(result["helper_source_commit"], "ba1ca711f148642f6e61bb38c52a0e8994e54c98")
        self.assertIn(r"\boxed", prompt["GENERATE_SOLUTION_PROMPT"])
        self.assertNotIn("\x08", prompt["GENERATE_SOLUTION_PROMPT"])
        self.assertNotIn("\x0c", operator["GENERATE_COT_PROMPT"])
        self.assertNotIn("\t", operator["GENERATE_COT_PROMPT"])
        rendered = operator["GENERATE_COT_PROMPT"].format(instruction="check", input="2 + 2")
        self.assertIn("2 + 2", rendered)
        if not concise:
            self.assertIn(r"\boxed{-2}", rendered)
            self.assertIn(r"\frac{1}{3}", rendered)
        code_prompt = operator["PYTHON_CODE_VERIFIER_PROMPT"].format(
            problem="2 + 2", analysis="4", feedback="none"
        )
        self.assertIn("2 + 2", code_prompt)
        self.assertIn("4", code_prompt)
        if concise:
            self.assertIn("Return one fenced Python code block", code_prompt)
            before = {name: (directory / name).read_bytes() for name in _PINNED_SHA256}
            patch_template_directory(directory, concise=True)
            self.assertEqual(before, {name: (directory / name).read_bytes() for name in _PINNED_SHA256})

    def test_pinned_template_shape_offline(self) -> None:
        # A compact fixture of the problematic literal forms and required
        # active prompt names, so this test runs before baseline setup too.
        prompt_source = r'''GENERATE_SOLUTION_PROMPT = """Keep \boxed{4}."""
DETAILED_SOLUTION_PROMPT = """Explain \boxed{4}."""
REFINE_ANSWER_PROMPT = """Verify \boxed{4}."""
'''
        operator_source = r"""GENERATE_COT_PROMPT = '''Instruction: {instruction}
Problem: {input}
Example: \frac{1}{3}; \boxed{-2}; 6\times 3.
'''
PYTHON_CODE_VERIFIER_PROMPT = "Problem: {problem}; Analysis: {analysis}; Feedback: {feedback}."
"""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "prompt.py").write_text(prompt_source, encoding="utf-8")
            (directory / "op_prompt.py").write_text(operator_source, encoding="utf-8")
            self._check_repair(directory, concise=False)

    def test_exact_pinned_native_templates_when_checked_out(self) -> None:
        for split in ("train", "test"):
            source = _BASELINE / split / "template"
            if not source.is_dir():
                self.skipTest("pinned MaAS checkout is not installed")
            for name, expected in _PINNED_SHA256.items():
                digest = hashlib.sha256((source / name).read_bytes()).hexdigest()
                self.assertEqual(digest, expected, f"MaAS {split}/{name} differs from pinned commit")
            for concise in (False, True):
                with self.subTest(split=split, concise=concise), TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    for name in _PINNED_SHA256:
                        shutil.copy2(source / name, directory / name)
                    self._check_repair(directory, concise)


if __name__ == "__main__":
    unittest.main()
