import hashlib
import json
import os
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from build_search_select_manifest import AXES, build_manifest
from native_masbench_external import (
    controller_update_evidence, freeze_selection, load_test_after_selection,
    maas_training_repeat_evidence, model_asset_provenance, run_session_provenance,
    validate_search_select,
)
from masbench_data_protocol import canonical_gold, expected_horizon_width, validate_frozen_splits


def _rows(axis, split, count, prefix):
    official = "test" if split == "test" else "train"
    return [{"id": f"{prefix}{i}", "dataset": "masbench", "axis": axis,
             "axis_value": "1", "official_split": official, "answer": str(i % 23),
             "input": f"problem {i}"} for i in range(count)]


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeTensor:
    dtype = "float32"

    def __init__(self, values):
        self.values = tuple(values)
        self.shape = (len(self.values),)

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, _shape):
        return self

    def tolist(self):
        return list(self.values)

    def equal(self, other):
        return self.values == other.values


class NativeMasBenchProtocolTests(unittest.TestCase):
    def test_model_and_session_provenance_require_matching_verified_assets(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            deployment = root / "deployment"
            deployment.mkdir()
            specifications = {
                "qwen": {"revision": "qwen-revision", "files": {"qwen.bin": [3, "q" * 64]}},
                "embedding": {"revision": "embedding-revision", "files": {"embed.bin": [3, "e" * 64]}},
            }
            revisions_path = deployment / "model_revisions.json"
            revisions_path.write_text(json.dumps(specifications), encoding="utf-8")
            qwen_path, embedding_path = root / "qwen", root / "embedding"
            qwen_path.mkdir()
            embedding_path.mkdir()
            report = {
                "manifest_sha256": _hash(revisions_path),
                "models": {
                    "qwen": {"revision": "qwen-revision", "path": str(qwen_path),
                             "sha256_verified": {"qwen.bin": "q" * 64}},
                    "embedding": {"revision": "embedding-revision", "path": str(embedding_path),
                                  "sha256_verified": {"embed.bin": "e" * 64}},
                },
            }
            report_path = root / "verified.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            session_path = root / "session.json"
            session_path.write_text(json.dumps({
                "project_root": str(root), "model_revision": "qwen-revision",
                "model_path": str(qwen_path), "embedding_revision": "embedding-revision",
                "embedding_path": str(embedding_path),
                "model_verification_report_sha256": _hash(report_path),
            }), encoding="utf-8")
            variables = {
                "RPAS_MODEL_REVISION": "qwen-revision",
                "RPAS_MAAS_EMBEDDING_REVISION": "embedding-revision",
                "RPAS_MODEL_PATH": str(qwen_path),
                "RPAS_MAAS_EMBEDDING_PATH": str(embedding_path),
                "RPAS_MODEL_VERIFICATION_REPORT": str(report_path),
                "RPAS_RUN_SESSION_RECORD": str(session_path),
            }
            with patch.dict(os.environ, variables):
                assets = model_asset_provenance(root, "maas")
                session = run_session_provenance(root, assets)
                self.assertEqual(session["sha256"], _hash(session_path))
                self.assertEqual(assets["verification_report"]["sha256"], _hash(report_path))
                report["models"]["qwen"]["sha256_verified"]["qwen.bin"] = "wrong"
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "does not verify configured qwen"):
                    model_asset_provenance(root, "maas")

    def test_controller_training_requires_step_and_changed_trainable_parameter(self):
        before = {"weight": _FakeTensor([1.0, 2.0]), "buffer": _FakeTensor([1.0])}
        updated = {"weight": _FakeTensor([1.1, 2.0]), "buffer": _FakeTensor([1.0])}
        evidence = controller_update_evidence(before, updated, {"weight"}, 1)
        self.assertEqual(evidence["optimizer_steps"], 1)
        self.assertEqual(evidence["changed_parameter_tensors"], 1)
        self.assertNotEqual(evidence["initial_state_sha256"], evidence["final_state_sha256"])
        with self.assertRaisesRegex(RuntimeError, "was not updated"):
            controller_update_evidence(before, updated, {"weight"}, 0)
        with self.assertRaisesRegex(RuntimeError, "was not updated"):
            controller_update_evidence(before, before, {"weight"}, 1)

    def test_maas_training_repetitions_require_observed_full_result_rows(self):
        audit = {
            "native_sample_argument": 1, "effective_repetitions": 4,
            "examples_per_repetition": 24, "observed_result_rows": 96,
            "completed_repetitions": 4,
        }
        self.assertEqual(maas_training_repeat_evidence([audit], 1, 24), audit)
        with self.assertRaisesRegex(RuntimeError, "repetitions differ"):
            maas_training_repeat_evidence([{**audit, "observed_result_rows": 24}], 1, 24)

    def test_gold_is_canonical_and_restricted_to_z23(self):
        self.assertEqual(canonical_gold("02<<horizon>>004"), "2<<horizon>>4")
        with self.assertRaises(ValueError):
            canonical_gold("23")
        with self.assertRaises(ValueError):
            canonical_gold("not a number")

    def test_horizon_width_comes_from_explicit_output_template_not_axis_values(self):
        prompt = "axis_value=2,4,6\n### Final Answers\nProblem 1: \\boxed{[answer1]}\nProblem 2: \\boxed{[answer2]}"
        self.assertEqual(expected_horizon_width(prompt), 2)

    def test_robustness_preserves_raw_injected_values(self):
        self.assertEqual(canonical_gold("08966231<<horizon>>11", allow_raw=True), "8966231<<horizon>>11")
        with self.assertRaisesRegex(ValueError, "Z_23"):
            canonical_gold("8966231<<horizon>>11", allow_raw=False)

    def test_search_select_manifest_builder_never_needs_d_test(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for axis in AXES:
                axis_dir = root / axis
                axis_dir.mkdir()
                _write_jsonl(axis_dir / "search_24.jsonl", _rows(axis, "search", 24, f"{axis}-s"))
                _write_jsonl(axis_dir / "select_24.jsonl", _rows(axis, "select", 24, f"{axis}-v"))
            manifest = build_manifest(root)
        self.assertEqual(manifest["split_sizes"], {"search": 24, "select": 24})
        self.assertNotIn("test", json.dumps(manifest).lower())

    def test_preselection_validator_uses_exact_frozen_rows(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            axis = "breadth"
            search, select = _rows(axis, "search", 24, "s"), _rows(axis, "select", 24, "v")
            files = {}
            for name, rows in (("search", search), ("select", select)):
                path = root / f"{name}_24.jsonl"
                _write_jsonl(path, rows)
                files[name] = _hash(path)
            manifest = {"schema_version": "rpas_masbench_search_select_blind_v1", "data_seed": 2026,
                        "split_sizes": {"search": 24, "select": 24},
                        "axes": {axis: {"row_ids": {"search": [r["id"] for r in search],
                                                       "select": [r["id"] for r in select]},
                                        "file_sha256": files}}}
            audit = validate_search_select({"search": search, "select": select}, manifest, axis, 2026, files)
        self.assertTrue(audit["search_select_disjoint"])
        self.assertEqual({k: len(v) for k, v in audit["split_ids"].items()}, {"search": 24, "select": 24})

    def test_full_frozen_split_audit_checks_official_file_hashes(self):
        axis = "breadth"
        rows = {"search": _rows(axis, "search", 24, "s"),
                "select": _rows(axis, "select", 24, "v"),
                "test": _rows(axis, "test", 60, "t")}
        hashes = {name: f"file-hash-{name}" for name in rows}
        for split in rows.values():
            for row in split:
                row["content_sha256"] = hashlib.sha256(row["input"].encode()).hexdigest()
        content_hashes = {name: hashlib.sha256("\n".join(r["content_sha256"] for r in split).encode()).hexdigest()
                          for name, split in rows.items()}
        id_hashes = {name: hashlib.sha256("\n".join(r["id"] for r in split).encode()).hexdigest()
                     for name, split in rows.items()}
        manifest = {"schema": "rpas_frozen_splits_v1", "group": axis, "seed": 2026,
                    "source": {"dataset": "Salesforce/MASBench", "axis": axis},
                    "counts": {"search": 24, "select": 24, "test": 60},
                    "files": {"search": f"{axis}/search_24.jsonl", "select": f"{axis}/select_24.jsonl",
                              "test": f"{axis}/test_60.jsonl"},
                    "content_sha256": content_hashes, "id_sha256": id_hashes}
        audit = validate_frozen_splits(rows, manifest, axis, 2026, hashes)
        self.assertTrue(audit["pairwise_disjoint"])
        self.assertEqual(len(audit["split_ids"]["test"]), 60)
        manifest["content_sha256"]["test"] = "wrong"
        with self.assertRaisesRegex(ValueError, "official frozen content"):
            validate_frozen_splits(rows, manifest, axis, 2026, hashes)

    def test_d_test_load_requires_lock_before_any_test_read(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            data_dir = root / "data"
            (data_dir / "breadth").mkdir(parents=True)
            args = Namespace(method="aflow", axis="breadth", seed=0, data_seed=2026,
                             data_dir=data_dir, run_dir=run_dir)
            selection_path = run_dir / "selection_frozen.json"
            selection_path.write_text('{"status":"tampered"}', encoding="utf-8")
            with patch("native_masbench_external.read_jsonl", side_effect=AssertionError("D_test read before lock")):
                with self.assertRaisesRegex(RuntimeError, "selection lock"):
                    load_test_after_selection(args, {"search": [], "select": []}, selection_path, "wrong-hash", {})

    def test_d_test_load_succeeds_after_selection_lock(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, axis_dir, run_dir = root / "data", root / "data" / "breadth", root / "run"
            axis_dir.mkdir(parents=True)
            run_dir.mkdir()
            rows = {"search": _rows("breadth", "search", 24, "s"),
                    "select": _rows("breadth", "select", 24, "v"),
                    "test": _rows("breadth", "test", 60, "t")}
            for split in rows.values():
                for row in split:
                    row["content_sha256"] = hashlib.sha256(row["input"].encode()).hexdigest()
            for split, filename in (("search", "search_24.jsonl"), ("select", "select_24.jsonl"),
                                    ("test", "test_60.jsonl")):
                _write_jsonl(axis_dir / filename, rows[split])
            file_hashes = {name: _hash(axis_dir / filename) for name, filename in
                      (("search", "search_24.jsonl"), ("select", "select_24.jsonl"), ("test", "test_60.jsonl"))}
            id_hashes = {name: hashlib.sha256("\n".join(r["id"] for r in split).encode()).hexdigest()
                         for name, split in rows.items()}
            content_hashes = {name: hashlib.sha256("\n".join(r["content_sha256"] for r in split).encode()).hexdigest()
                              for name, split in rows.items()}
            full_manifest = {"schema": "rpas_frozen_splits_v1", "group": "breadth", "seed": 2026,
                             "source": {"dataset": "Salesforce/MASBench", "axis": "breadth"},
                             "counts": {"search": 24, "select": 24, "test": 60},
                             "files": {"search": "breadth/search_24.jsonl", "select": "breadth/select_24.jsonl",
                                       "test": "breadth/test_60.jsonl"}, "content_sha256": content_hashes,
                             "id_sha256": id_hashes}
            (axis_dir / "manifest.json").write_text(json.dumps(full_manifest), encoding="utf-8")
            args = Namespace(method="aflow", axis="breadth", seed=0, data_seed=2026,
                             data_dir=data_dir, run_dir=run_dir)
            pre_audit = {name: [r["id"] for r in rows[name]] for name in ("search", "select")}
            run_manifest = {"pre_test_split_audit": {"split_ids": pre_audit}, "data_sha256": {}}
            selection_path, selection_hash = freeze_selection(args, {"Q": {"round": 1}}, {"score": 1.0})
            audit = load_test_after_selection(args, {"search": rows["search"], "select": rows["select"]},
                                              selection_path, selection_hash, run_manifest)
        self.assertTrue(audit["pairwise_disjoint"])
        self.assertTrue(run_manifest["d_test_opened"])
        self.assertEqual(run_manifest["d_test_access"]["examples"], 60)


if __name__ == "__main__":
    unittest.main()
