import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from masbench_quality_gate import (
    BASELINE_COMMITS, BASELINE_REQUIRED, RUNTIME_ADAPTER_FILES, sha256_file, validate_run,
)


class MasBenchQualityGateTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        repo = root / "repo"
        exp = repo / "experiments"
        exp.mkdir(parents=True)
        (exp / "native_masbench_external.py").write_text("runner", encoding="utf-8")
        (exp / "native_aime_formal.py").write_text("base runner", encoding="utf-8")
        (exp / "masbench_answer_protocol.py").write_text("parser", encoding="utf-8")
        for name in RUNTIME_ADAPTER_FILES["maas"]:
            if name in ("maas_prompt_repair.py", "maas_concise_prompts_v2.py"):
                (exp / name).write_bytes(Path(__file__).with_name(name).read_bytes())
            else:
                (exp / name).write_text(name, encoding="utf-8")
        deployment = repo / "deployment"
        deployment.mkdir()
        model_specs = {
            "qwen": {"revision": "1" * 40, "files": {"model.safetensors": [3, "2" * 64]}},
            "embedding": {"revision": "3" * 40, "files": {"model.safetensors": [3, "4" * 64]}},
        }
        (deployment / "model_revisions.json").write_text(json.dumps(model_specs), encoding="utf-8")
        model_paths = {}
        for name in model_specs:
            path = repo / "models" / name
            path.mkdir(parents=True)
            model_paths[name] = str(path)
        baseline_root = repo / "baseline"
        baseline_hashes = {}
        for relative in BASELINE_REQUIRED["aflow"]:
            path = baseline_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative, encoding="utf-8")
            baseline_hashes[relative] = sha256_file(path)
        data_root = repo / "data" / "masbench"
        data_root.mkdir(parents=True)
        safe_manifest = data_root / "manifest_search_select.json"
        safe_manifest.write_text('{"schema_version":"rpas_ec1_masbench_aime_v1"}', encoding="utf-8")
        full_manifest = data_root / "breadth" / "manifest.json"
        full_manifest.parent.mkdir(parents=True)
        full_manifest.write_text('{"test_metadata":"opened only after lock"}', encoding="utf-8")
        test_file = data_root / "breadth" / "test_60.jsonl"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text('{"id":"test-row"}\n', encoding="utf-8")
        run_dir = root / "outputs" / "aflow" / "breadth" / "seed_0"
        run_dir.mkdir(parents=True)
        session_record = run_dir / "run_session_record.json"
        session_record.write_text(json.dumps({
            "project_root": str(repo),
            "model_revision": model_specs["qwen"]["revision"],
            "model_path": model_paths["qwen"],
            "embedding_revision": model_specs["embedding"]["revision"],
            "embedding_path": model_paths["embedding"],
            "model_verification_report_sha256": None,
        }), encoding="utf-8")
        point = {
            "num_examples": 60,
            "correct": 30,
            "accuracy": 0.5,
            "avg_cost": 100.0,
            "total_tokens": 6000,
            "calls": 60,
            "resource_usage": {"calls": 60, "total_tokens": 6000},
            "output_audit": {"csv_rows": 60, "failure_fraction": 0.0},
            "valid_answer_rate": 1.0,
            "component_accuracy": 0.5,
            "mean_score_0_1_2": 1.0,
        }
        controls = {
            "model": "Qwen/Qwen3.5-9B", "temperature": 0.0, "top_p": 1.0,
            "thinking": False, "max_tokens": 6144, "max_model_len": 8192,
            "max_num_seqs": 24, "tensor_parallel_size": 1, "concurrency": 8,
            "data_seed": 2026, "search_seed": 0, "search_size": 24,
            "selection_size": 24, "test_size": 60,
            "selection_split": "D_select", "test_split": "D_test",
        }
        result = {
            "run_id": "test-run", "method": "aflow", "controls": controls,
            "native_method_config": {
                "dataset": "MATH", "question_type": "math",
                "operators": ["Custom", "ScEnsemble", "Programmer"],
                "sample": 4, "check_convergence": False, "initial_round": 1,
                "max_rounds": 8, "validation_rounds": 1,
                "mutation_generation_retry_cap": 8,
            },
            "telemetry": {"context_guard": {"context_limit": 8192, "calls_seen": 100}},
            "selection_rows": [{"status": "evaluated", "round": 1,
                                "output_audit": {"failure_fraction": 0.0}}],
            "selected_round": {"round": 1}, "selected_efficiency_round": {"round": 1},
            "test": {"Q": point, "E": dict(point)},
        }
        result["test"]["Q"]["selected_round"] = 1
        result["test"]["E"]["selected_round"] = 1
        selection_lock = {
            "status": "frozen_before_test_access", "method": "aflow", "axis": "breadth", "seed": 0,
            "d_test_opened": False, "frozen_at_utc": "2026-01-01T00:00:00+00:00",
        }
        selection_path = run_dir / "selection_frozen.json"
        selection_path.write_text(json.dumps(selection_lock), encoding="utf-8")
        selection_hash = sha256_file(selection_path)
        result["selection_lock"] = {"path": selection_path.name, "sha256": selection_hash}
        result_path = run_dir / "native_result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        manifest = {
            "protocol_version": "masbench_external_protocol_v1", "status": "complete",
            "method": "aflow", "axis": "breadth", "seed": 0, "run_id": "test-run",
            "pairwise_disjoint": True, "split_counts": {"search": 24, "select": 24, "test": 60},
            "native_result_sha256": sha256_file(result_path), "rpas_root": str(repo),
            "data_sha256": {
                "manifest_search_select.json": sha256_file(safe_manifest),
                "breadth/manifest.json": sha256_file(full_manifest),
                "breadth/test_60.jsonl": sha256_file(test_file),
            },
            "d_test_opened": True,
            "selection_lock_sha256": selection_hash,
            "d_test_access": {
                "opened_at_utc": "2026-01-01T00:00:01+00:00",
                "selection_lock_sha256": selection_hash,
                "test_file_sha256": sha256_file(test_file),
                "examples": 60,
            },
            "official_baseline_source": str(baseline_root),
            "baseline_source_sha256": baseline_hashes,
            "runner_sha256": sha256_file(exp / "native_masbench_external.py"),
            "base_runner_sha256": sha256_file(exp / "native_aime_formal.py"),
            "runtime_adapter_sha256": {
                name: sha256_file(exp / name) for name in RUNTIME_ADAPTER_FILES["aflow"]
            },
            "model_asset_provenance": {
                "revisions_manifest_sha256": sha256_file(deployment / "model_revisions.json"),
                "revisions": {"qwen": model_specs["qwen"]["revision"]},
                "paths": {"qwen": model_paths["qwen"]},
            },
            "run_session_record": {
                "path": str(session_record), "sha256": sha256_file(session_record),
            },
            "answer_parser_sha256": sha256_file(exp / "masbench_answer_protocol.py"),
            "baseline_git_commit": BASELINE_COMMITS["aflow"],
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return run_dir

    def make_maas_fixture(self, root: Path):
        run_dir = self.make_fixture(root)
        result_path = run_dir / "native_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["method"] = "maas"
        result["native_method_config"] = {
            "dataset": "MATH", "question_type": "math", "operators": ["native-controller"],
            "controller_sample": 1, "round": 1, "batch_size": 8,
            "learning_rate": 0.01, "is_textgrad": False, "training_repetitions": 4,
        }
        result["test"] = {"breadth": result["test"]}
        result["selection"] = {
            "num_examples": 24, "output_audit": {"csv_rows": 24, "failure_fraction": 0.0},
        }
        checkpoint = run_dir / "controller.ckpt"
        checkpoint.write_bytes(b"test controller")
        result["controller_checkpoint"] = str(checkpoint)
        result["controller_checkpoint_sha256"] = sha256_file(checkpoint)
        repeat_audit = {
            "native_sample_argument": 1, "effective_repetitions": 4,
            "examples_per_repetition": 24, "observed_result_rows": 96,
            "completed_repetitions": 4,
        }
        result["telemetry"]["training_repeat_audit"] = [repeat_audit]
        result["controller_training"] = {
            "optimizer_steps": 1, "parameter_tensors": 3,
            "changed_parameter_tensors": 1,
            "initial_state_sha256": "a" * 64,
            "final_state_sha256": "b" * 64,
            "checkpoint_state_sha256": "b" * 64,
            "checkpoint_file_sha256": sha256_file(checkpoint),
            "training_repeat_audit": repeat_audit,
        }
        repairs = []
        for split in ("train", "test"):
            template = (run_dir / "maas_repo" / "maas" / "ext" / "maas" / "scripts" /
                        "optimized" / "MATH" / split / "template")
            template.mkdir(parents=True)
            prompt = template / "prompt.py"
            operator = template / "op_prompt.py"
            prompt.write_text(f"prompt for {split}", encoding="utf-8")
            operator.write_text(f"operator for {split}", encoding="utf-8")
            repairs.append({
                "version": "maas_context_guard_v1", "path": str(template),
                "concise_prompt_variant": "native_repaired",
                "prompt_adapter_origin": "recovered_rpas_working_copy",
                "helper_source_repository": "Simplification555/RPAS-EC2-A100",
                "helper_source_commit": "ba1ca711f148642f6e61bb38c52a0e8994e54c98",
                "helper_source_path": "experiments/external_baselines/masbench_aflow_maas_seed0/experiments",
                "original_helper_git_commit": None,
                "prompt_sha256": sha256_file(prompt),
                "operator_sha256": sha256_file(operator),
            })
        result["prompt_repairs"] = repairs
        lock_path = run_dir / "selection_frozen.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["method"] = "maas"
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        lock_hash = sha256_file(lock_path)
        result["selection_lock"]["sha256"] = lock_hash
        result_path.write_text(json.dumps(result), encoding="utf-8")

        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        baseline_root = Path(manifest["official_baseline_source"])
        maas_hashes = {}
        for relative in BASELINE_REQUIRED["maas"]:
            path = baseline_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative, encoding="utf-8")
            maas_hashes[relative] = sha256_file(path)
        exp = Path(manifest["rpas_root"]) / "experiments"
        manifest.update({
            "method": "maas", "selection_lock_sha256": lock_hash,
            "native_result_sha256": sha256_file(result_path),
            "baseline_source_sha256": maas_hashes,
            "baseline_git_commit": BASELINE_COMMITS["maas"],
            "runtime_adapter_sha256": {
                name: sha256_file(exp / name) for name in RUNTIME_ADAPTER_FILES["maas"]
            },
        })
        model_specs = json.loads((Path(manifest["rpas_root"]) / "deployment/model_revisions.json").read_text())
        manifest["model_asset_provenance"]["revisions"]["embedding"] = model_specs["embedding"]["revision"]
        manifest["model_asset_provenance"]["paths"]["embedding"] = str(
            Path(manifest["rpas_root"]) / "models/embedding"
        )
        manifest["d_test_access"]["selection_lock_sha256"] = lock_hash
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return run_dir

    def validate_fixture(self, run_dir: Path, method: str):
        with patch("masbench_quality_gate.subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout=BASELINE_COMMITS[method] + "\n"
        )):
            return validate_run(run_dir)

    def replace_result(self, run_dir: Path, result: dict):
        result_path = run_dir / "native_result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["native_result_sha256"] = sha256_file(result_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_complete_consistent_run_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("masbench_quality_gate.subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout=BASELINE_COMMITS["aflow"] + "\n"
            )):
                report = validate_run(self.make_fixture(Path(directory)))
        self.assertTrue(report["passed"], report["errors"])

    def test_failure_fraction_above_five_percent_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.make_fixture(Path(directory))
            result_path = run_dir / "native_result.json"
            result = json.loads(result_path.read_text())
            result["test"]["Q"]["output_audit"]["failure_fraction"] = 0.1
            result_path.write_text(json.dumps(result), encoding="utf-8")
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["native_result_sha256"] = sha256_file(result_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("masbench_quality_gate.subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout=BASELINE_COMMITS["aflow"] + "\n"
            )):
                report = validate_run(run_dir)
        self.assertFalse(report["passed"])
        self.assertTrue(any("failure fraction" in item for item in report["errors"]))

    def test_maas_native_controller_and_24_24_60_protocol_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.validate_fixture(self.make_maas_fixture(Path(directory)), "maas")
        self.assertTrue(report["passed"], report["errors"])

    def test_maas_requires_optimizer_step_and_parameter_change(self):
        for key, value, expected_error in (
            ("optimizer_steps", 0, "optimizer did not record"),
            ("changed_parameter_tensors", 0, "no verified parameter update"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                run_dir = self.make_maas_fixture(Path(directory))
                result = json.loads((run_dir / "native_result.json").read_text(encoding="utf-8"))
                result["controller_training"][key] = value
                self.replace_result(run_dir, result)
                report = self.validate_fixture(run_dir, "maas")
                self.assertFalse(report["passed"])
                self.assertTrue(any(expected_error in error for error in report["errors"]), report["errors"])

    def test_maas_requires_observed_four_training_repetitions(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.make_maas_fixture(Path(directory))
            result = json.loads((run_dir / "native_result.json").read_text(encoding="utf-8"))
            result["telemetry"]["training_repeat_audit"][0]["observed_result_rows"] = 24
            self.replace_result(run_dir, result)
            report = self.validate_fixture(run_dir, "maas")
        self.assertFalse(report["passed"])
        self.assertTrue(any("did not complete four" in error for error in report["errors"]), report["errors"])

    def test_base_runner_and_runtime_adapter_hashes_are_checked(self):
        for name, expected_error in (
            ("native_aime_formal.py", "base runner hash mismatch"),
            ("maas_context_guard.py", "runtime adapter hash mismatch"),
            ("maas_prompt_repair.py", "runtime adapter hash mismatch"),
            ("maas_concise_prompts_v2.py", "runtime adapter hash mismatch"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                run_dir = self.make_maas_fixture(Path(directory))
                manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
                target = Path(manifest["rpas_root"]) / "experiments" / name
                target.write_text("tampered", encoding="utf-8")
                report = self.validate_fixture(run_dir, "maas")
                self.assertFalse(report["passed"])
                self.assertTrue(any(expected_error in error for error in report["errors"]), report["errors"])

    def test_maas_repaired_prompt_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.make_maas_fixture(Path(directory))
            result = json.loads((run_dir / "native_result.json").read_text(encoding="utf-8"))
            prompt_path = Path(result["prompt_repairs"][0]["path"]) / "prompt.py"
            prompt_path.write_text("tampered", encoding="utf-8")
            report = self.validate_fixture(run_dir, "maas")
        self.assertFalse(report["passed"])
        self.assertTrue(any("repaired prompt.py hash mismatch" in error for error in report["errors"]),
                        report["errors"])

    def test_maas_helper_hash_must_match_recovered_source(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.make_maas_fixture(Path(directory))
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            helper = Path(manifest["rpas_root"]) / "experiments/maas_prompt_repair.py"
            helper.write_text("changed helper", encoding="utf-8")
            manifest["runtime_adapter_sha256"][helper.name] = sha256_file(helper)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = self.validate_fixture(run_dir, "maas")
        self.assertFalse(report["passed"])
        self.assertTrue(any("does not match pinned source" in error for error in report["errors"]),
                        report["errors"])

    def test_maas_rejects_concise_prompt_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.make_maas_fixture(Path(directory))
            result = json.loads((run_dir / "native_result.json").read_text(encoding="utf-8"))
            result["prompt_repairs"][0]["concise_prompt_variant"] = "maas_concise_v2"
            self.replace_result(run_dir, result)
            report = self.validate_fixture(run_dir, "maas")
        self.assertFalse(report["passed"])
        self.assertTrue(any("not the frozen native repair" in error for error in report["errors"]),
                        report["errors"])

    def test_run_session_record_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.make_maas_fixture(Path(directory))
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            session_path = Path(manifest["run_session_record"]["path"])
            session_path.write_text("tampered", encoding="utf-8")
            report = self.validate_fixture(run_dir, "maas")
        self.assertFalse(report["passed"])
        self.assertTrue(any("run session record hash mismatch" in error for error in report["errors"]),
                        report["errors"])

    def test_verified_model_report_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.make_maas_fixture(Path(directory))
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            spec_path = Path(manifest["rpas_root"]) / "deployment/model_revisions.json"
            specifications = json.loads(spec_path.read_text(encoding="utf-8"))
            report_path = run_dir / "verified_models.json"
            report_path.write_text(json.dumps({
                "manifest_sha256": sha256_file(spec_path),
                "models": {
                    name: {
                        "revision": specifications[name]["revision"],
                        "path": manifest["model_asset_provenance"]["paths"][name],
                        "sha256_verified": {
                            filename: digest for filename, (_, digest) in specifications[name]["files"].items()
                        },
                    } for name in ("qwen", "embedding")
                },
            }), encoding="utf-8")
            manifest["model_asset_provenance"]["verification_report"] = {
                "path": str(report_path), "sha256": sha256_file(report_path),
            }
            session_path = Path(manifest["run_session_record"]["path"])
            session = json.loads(session_path.read_text(encoding="utf-8"))
            session["model_verification_report_sha256"] = sha256_file(report_path)
            session_path.write_text(json.dumps(session), encoding="utf-8")
            manifest["run_session_record"]["sha256"] = sha256_file(session_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(self.validate_fixture(run_dir, "maas")["passed"])
            report_path.write_text("tampered", encoding="utf-8")
            report = self.validate_fixture(run_dir, "maas")
        self.assertFalse(report["passed"])
        self.assertTrue(any("model verification report hash mismatch" in error for error in report["errors"]),
                        report["errors"])


if __name__ == "__main__":
    unittest.main()
