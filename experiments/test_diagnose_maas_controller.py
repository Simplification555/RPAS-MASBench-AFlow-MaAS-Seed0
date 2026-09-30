"""Offline checks for the synthetic MaAS controller diagnostic boundary."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings
from unittest.mock import patch

from scripts.diagnose_maas_controller import (
    MAAS_COMMIT,
    WORKSPACE,
    prepare_isolated_repo,
    validate_endpoint,
    validate_output_root,
    verify_source,
    write_synthetic_jsonl,
)


class MaASControllerDiagnosticTests(unittest.TestCase):
    def test_only_loopback_model_endpoint_and_separate_output(self) -> None:
        self.assertEqual(validate_endpoint("http://127.0.0.1:8000/v1/"), "http://127.0.0.1:8000/v1")
        for endpoint in ("https://127.0.0.1:8000/v1", "http://example.com/v1", "http://localhost:8000/v2"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_endpoint(endpoint)
        source = WORKSPACE / "external_baselines/MaAS"
        for forbidden in (WORKSPACE / "outputs/diagnostic", WORKSPACE / "data/masbench/diagnostic"):
            with self.subTest(path=forbidden), self.assertRaises(ValueError):
                validate_output_root(forbidden, source)
        with TemporaryDirectory() as temporary, patch.dict("os.environ", {"RPAS_OUTPUT_ROOT": temporary}):
            with self.assertRaises(ValueError):
                validate_output_root(Path(temporary) / "maas-diagnostic", source)

    def test_pinned_repo_copy_has_only_two_synthetic_training_rows(self) -> None:
        source = WORKSPACE / "external_baselines/MaAS"
        if not source.is_dir():
            self.skipTest("pinned MaAS checkout is not installed")
        source = verify_source(source)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            embedding = root / "embedding"
            embedding.mkdir()
            repo = root / "repo"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                warnings.simplefilter("ignore", SyntaxWarning)
                staging = prepare_isolated_repo(
                    source, repo, "http://127.0.0.1:8000/v1", embedding, 1536,
                )
            rows = [json.loads(line) for line in
                    (repo / "maas/ext/maas/data/math_train.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(staging["source_commit"], MAAS_COMMIT)
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["diagnostic_id"] for row in rows},
                             {"synthetic_addition", "synthetic_subtraction"})
            self.assertTrue(all("\\boxed" in row["solution"] for row in rows))
            self.assertEqual((repo / "maas/ext/maas/data/math_test.jsonl").read_bytes(), b"")
            self.assertFalse((repo / ".git").exists())
            prompt = (repo / "workspace/rpas_synthetic_maas/MATH/train/template/prompt.py").read_text()
            operator = (repo / "workspace/rpas_synthetic_maas/MATH/train/template/op_prompt.py").read_text()
            self.assertTrue(all(repair["concise_prompt_variant"] == "native_repaired"
                                for repair in staging["template_repairs"]))
            self.assertIn("Please solve the given mathematical problem step by step.", prompt)
            self.assertNotIn("Keep every calculation needed to obtain the answer.", prompt)
            self.assertIn("GENERATE_COT_PROMPT", operator)

    def test_no_empty_or_large_synthetic_batch(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            for rows in ((), ({}, {}, {})):
                with self.subTest(size=len(rows)), self.assertRaises(ValueError):
                    write_synthetic_jsonl(path, rows)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
