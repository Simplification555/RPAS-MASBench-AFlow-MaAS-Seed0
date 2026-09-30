#!/usr/bin/env python3
"""Fail-closed quality gate for one native MASBench search/select/test run."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONTEXT_LIMIT = 8192
OUTPUT_LIMIT = 6144
MAX_SAMPLE_FAILURE_RATE = 0.05
PROTOCOL_VERSION = "masbench_external_protocol_v1"
AXES = ("breadth", "depth", "horizon", "parallel", "robustness")
BASELINE_REQUIRED = {
    "aflow": (
        "scripts/optimizer.py", "scripts/evaluator.py",
        "scripts/optimizer_utils/evaluation_utils.py", "scripts/optimizer_utils/graph_utils.py",
        "benchmarks/math.py",
    ),
    "maas": (
        "maas/ext/maas/scripts/optimizer.py", "maas/ext/maas/scripts/evaluator.py",
        "maas/ext/maas/scripts/optimizer_utils/evaluation_utils.py",
        "maas/ext/maas/scripts/optimizer_utils/graph_utils.py",
        "maas/ext/maas/benchmark/benchmark.py", "maas/ext/maas/benchmark/math.py",
    ),
}
BASELINE_COMMITS = {
    "aflow": "3f457218fc716093fe53f6df8a5d5e6379d66346",
    "maas": "987f3c1bc9a96e844fe090db3791446e3ef0f5c7",
}
RUNTIME_ADAPTER_FILES = {
    "aflow": ("maas_context_guard.py",),
    "maas": (
        "maas_context_guard.py",
        "maas_prompt_repair.py",
        "maas_concise_prompts_v2.py",
    ),
}
RECOVERED_MAAS_HELPER_SHA256 = {
    "maas_prompt_repair.py": "dbfad9d7c66e1b80df18a2550a9c1c6a4766943d900d909810ed7bbc9fa2f63c",
    "maas_concise_prompts_v2.py": "a4af1fd1f791b6b9fdb3a98c74c92418222ac3b80cc692b15ebe216b787f583f",
}
MAAS_PROMPT_ADAPTER_PROVENANCE = {
    "prompt_adapter_origin": "recovered_rpas_working_copy",
    "helper_source_repository": "Simplification555/RPAS-EC2-A100",
    "helper_source_commit": "ba1ca711f148642f6e61bb38c52a0e8994e54c98",
    "helper_source_path": "experiments/external_baselines/masbench_aflow_maas_seed0/experiments",
    "original_helper_git_commit": None,
}


def sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail_if(condition: bool, reason: str, errors: list[str]) -> None:
    if condition:
        errors.append(reason)


def _points(result: dict[str, Any], axis: str) -> dict[str, dict[str, Any]]:
    if result.get("method") == "aflow":
        return result.get("test", {})
    if result.get("method") == "maas":
        axis_result = result.get("test", {}).get(axis, {})
        return {key: axis_result.get(key, {}) for key in ("Q", "E")}
    return {}


def validate_run(run_dir: Path, rpas_root: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = run_dir / "run_manifest.json"
    result_path = run_dir / "native_result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        return {"passed": False, "errors": ["run manifest or native result is missing"]}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    method, axis, seed = manifest.get("method"), manifest.get("axis"), manifest.get("seed")
    _fail_if(manifest.get("protocol_version") != PROTOCOL_VERSION, "unexpected protocol version", errors)
    _fail_if(manifest.get("status") != "complete", f"run status is {manifest.get('status')!r}, not complete", errors)
    _fail_if(method not in ("aflow", "maas"), "unknown method", errors)
    _fail_if(axis not in AXES, "unknown MASBench axis", errors)
    _fail_if(seed != 0, "seed is not the requested single frozen seed 0", errors)
    _fail_if(result.get("run_id") != manifest.get("run_id"), "run ID mismatch", errors)
    _fail_if(result.get("method") != method, "method mismatch between result and manifest", errors)
    _fail_if(manifest.get("pairwise_disjoint") is not True, "frozen splits are not verified pairwise-disjoint", errors)
    _fail_if(manifest.get("d_test_opened") is not True, "D_test was not evaluated after the selection lock", errors)
    d_test_access = manifest.get("d_test_access", {})
    selection_path = run_dir / "selection_frozen.json"
    selection_hash = manifest.get("selection_lock_sha256")
    _fail_if(not selection_path.is_file(), "frozen D_select selection lock is missing", errors)
    if selection_path.is_file():
        lock = json.loads(selection_path.read_text(encoding="utf-8"))
        _fail_if(lock.get("status") != "frozen_before_test_access", "selection lock status is invalid", errors)
        _fail_if(lock.get("d_test_opened") is not False, "selection lock was not frozen before test access", errors)
        _fail_if(lock.get("method") != method or lock.get("axis") != axis or lock.get("seed") != seed,
                 "selection lock does not match run method/axis/seed", errors)
        _fail_if(not selection_hash or sha256_file(selection_path) != selection_hash,
                 "selection lock hash mismatch", errors)
        _fail_if(d_test_access.get("selection_lock_sha256") != selection_hash,
                 "D_test access audit does not reference the frozen selection lock", errors)
        _fail_if(d_test_access.get("examples") != 60, "D_test access audit does not record all 60 examples", errors)
        _fail_if(not d_test_access.get("test_file_sha256"), "D_test file hash missing from access audit", errors)
        result_lock = result.get("selection_lock", {})
        _fail_if(result_lock.get("path") != selection_path.name or result_lock.get("sha256") != selection_hash,
                 "native result does not reference the frozen selection lock", errors)
        try:
            locked_at = datetime.fromisoformat(str(lock.get("frozen_at_utc")))
            opened_at = datetime.fromisoformat(str(d_test_access.get("opened_at_utc")))
            _fail_if(opened_at <= locked_at, "D_test access timestamp precedes candidate freeze", errors)
        except (TypeError, ValueError):
            errors.append("selection-lock or D_test-access timestamp is missing/invalid")
    if axis == "robustness":
        _fail_if(not manifest.get("task_protocol_exception"), "robustness raw-fact/magic-number rule is not recorded", errors)
    _fail_if(manifest.get("native_result_sha256") != sha256_file(result_path), "native result hash mismatch", errors)

    controls = result.get("controls", {})
    expected_controls = {
        "model": "Qwen/Qwen3.5-9B",
        "temperature": 0.0,
        "top_p": 1.0,
        "thinking": False,
        "max_tokens": OUTPUT_LIMIT,
        "max_model_len": CONTEXT_LIMIT,
        "max_num_seqs": 24,
        "tensor_parallel_size": 1,
        "concurrency": 8,
        "data_seed": 2026,
        "search_seed": seed,
        "selection_split": "D_select",
        "test_split": "D_test",
        "search_size": 24,
        "selection_size": 24,
        "test_size": 60,
    }
    # MaAS stores nested Q/E points and omits explicit split-size keys in its
    # result controls only if a legacy runner produced the artifact.
    for key, expected in expected_controls.items():
        actual = controls.get(key)
        if key == "test_size" and method == "maas":
            actual = controls.get("test_size", manifest.get("split_counts", {}).get("test"))
        _fail_if(actual != expected, f"control {key}={actual!r}, expected {expected!r}", errors)

    method_config = result.get("native_method_config", {})
    if method == "aflow":
        expected_method_config = {
            "dataset": "MATH", "question_type": "math",
            "operators": ["Custom", "ScEnsemble", "Programmer"],
            "sample": 4, "check_convergence": False, "initial_round": 1,
            "max_rounds": 8, "validation_rounds": 1,
            "mutation_generation_retry_cap": 8,
        }
        for key, expected in expected_method_config.items():
            _fail_if(method_config.get(key) != expected,
                     f"AFlow native configuration {key}={method_config.get(key)!r}, expected {expected!r}", errors)
    elif method == "maas":
        _fail_if(method_config.get("dataset") != "MATH" or method_config.get("question_type") != "math",
                 "MaAS native task configuration is not MATH/math", errors)
        _fail_if(not isinstance(method_config.get("operators"), list) or not method_config.get("operators"),
                 "MaAS native operator/controller configuration is missing", errors)
        for key, expected in {"controller_sample": 1, "round": 1, "batch_size": 8,
                              "learning_rate": 0.01, "is_textgrad": False,
                              "training_repetitions": 4}.items():
            _fail_if(method_config.get(key) != expected,
                     f"MaAS native configuration {key}={method_config.get(key)!r}, expected {expected!r}", errors)

    guard = result.get("telemetry", {}).get("context_guard")
    _fail_if(not isinstance(guard, dict), "context guard summary missing", errors)
    if isinstance(guard, dict):
        _fail_if(guard.get("context_limit") != CONTEXT_LIMIT, "context guard did not enforce 8192 tokens", errors)
        _fail_if(int(guard.get("calls_seen", 0)) <= 0, "context guard saw no model requests", errors)

    effective_rpas_root = rpas_root or (Path(manifest["rpas_root"]) if manifest.get("rpas_root") else None)
    if effective_rpas_root is None or not effective_rpas_root.is_dir():
        errors.append("RPAS source root is missing; runtime hashes cannot be verified")
    else:
        data_root = effective_rpas_root / "data" / "masbench"
        for relative, expected_hash in manifest.get("data_sha256", {}).items():
            path = data_root / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                errors.append(f"frozen dataset hash mismatch: {relative}")
        parser_path = effective_rpas_root / "experiments" / "masbench_answer_protocol.py"
        if not parser_path.is_file() or sha256_file(parser_path) != manifest.get("answer_parser_sha256"):
            errors.append("answer parser hash mismatch")
        runner_path = effective_rpas_root / "experiments" / "native_masbench_external.py"
        if not runner_path.is_file() or sha256_file(runner_path) != manifest.get("runner_sha256"):
            errors.append("runner hash mismatch")
        base_runner_path = effective_rpas_root / "experiments" / "native_aime_formal.py"
        if not base_runner_path.is_file() or sha256_file(base_runner_path) != manifest.get("base_runner_sha256"):
            errors.append("base runner hash mismatch")
        runtime_hashes = manifest.get("runtime_adapter_sha256", {})
        required_runtime_files = RUNTIME_ADAPTER_FILES.get(str(method), ())
        if not isinstance(runtime_hashes, dict) or set(runtime_hashes) != set(required_runtime_files):
            errors.append("runtime adapter hash inventory mismatch")
        for name in required_runtime_files:
            path = effective_rpas_root / "experiments" / name
            if not path.is_file() or sha256_file(path) != (
                runtime_hashes.get(name) if isinstance(runtime_hashes, dict) else None
            ):
                errors.append(f"runtime adapter hash mismatch: {name}")
        if method == "maas" and isinstance(runtime_hashes, dict):
            for name, expected_hash in RECOVERED_MAAS_HELPER_SHA256.items():
                _fail_if(runtime_hashes.get(name) != expected_hash,
                         f"MaAS recovered helper does not match pinned source: {name}", errors)
        revisions_path = effective_rpas_root / "deployment/model_revisions.json"
        model_assets = manifest.get("model_asset_provenance")
        if not revisions_path.is_file() or not isinstance(model_assets, dict):
            errors.append("model asset provenance is missing")
        else:
            specifications = json.loads(revisions_path.read_text(encoding="utf-8"))
            _fail_if(model_assets.get("revisions_manifest_sha256") != sha256_file(revisions_path),
                     "model revision manifest hash mismatch", errors)
            required_models = ("qwen", "embedding") if method == "maas" else ("qwen",)
            revisions = model_assets.get("revisions")
            expected_revisions = {name: specifications[name]["revision"] for name in required_models}
            _fail_if(revisions != expected_revisions, "model asset revisions differ from frozen pins", errors)
            paths = model_assets.get("paths")
            if not isinstance(paths, dict) or set(paths) != set(required_models) or any(
                not isinstance(paths[name], str) or not Path(paths[name]).is_dir() for name in required_models
            ):
                errors.append("local model asset paths are missing")
            verification = model_assets.get("verification_report")
            if verification is not None:
                if not isinstance(verification, dict):
                    errors.append("model verification report reference is invalid")
                else:
                    report_path = Path(str(verification.get("path", "")))
                    if not report_path.is_file() or sha256_file(report_path) != verification.get("sha256"):
                        errors.append("model verification report hash mismatch")
                    else:
                        report = json.loads(report_path.read_text(encoding="utf-8"))
                        _fail_if(report.get("manifest_sha256") != sha256_file(revisions_path),
                                 "model verification report uses a different revision manifest", errors)
                        for name in required_models:
                            entry = report.get("models", {}).get(name, {})
                            expected_files = {
                                filename: digest for filename, (_, digest) in specifications[name]["files"].items()
                            }
                            _fail_if(entry.get("revision") != expected_revisions[name] or
                                     entry.get("sha256_verified") != expected_files or
                                     not isinstance(paths, dict) or entry.get("path") != paths.get(name),
                                     f"model verification report does not match {name} provenance", errors)
        session_reference = manifest.get("run_session_record")
        if not isinstance(session_reference, dict):
            errors.append("run session record is missing")
        else:
            session_path = Path(str(session_reference.get("path", "")))
            if not session_path.is_file() or sha256_file(session_path) != session_reference.get("sha256"):
                errors.append("run session record hash mismatch")
            else:
                session = json.loads(session_path.read_text(encoding="utf-8"))
                _fail_if(session.get("project_root") != str(effective_rpas_root.resolve()),
                         "run session record project root mismatch", errors)
                if isinstance(model_assets, dict):
                    revisions = model_assets.get("revisions", {})
                    paths = model_assets.get("paths", {})
                    verification = model_assets.get("verification_report", {})
                    if not isinstance(revisions, dict):
                        revisions = {}
                    if not isinstance(paths, dict):
                        paths = {}
                    if not isinstance(verification, dict):
                        verification = {}
                    _fail_if(session.get("model_revision") != revisions.get("qwen") or
                             not isinstance(session.get("model_path"), str) or
                             not isinstance(paths.get("qwen"), str) or
                             Path(session["model_path"]).resolve() != Path(paths["qwen"]).resolve() or
                             session.get("model_verification_report_sha256") != verification.get("sha256"),
                             "run session model provenance mismatch", errors)
                    if method == "maas":
                        _fail_if(session.get("embedding_revision") != revisions.get("embedding") or
                                 not isinstance(session.get("embedding_path"), str) or
                                 not isinstance(paths.get("embedding"), str) or
                                 Path(session["embedding_path"]).resolve() != Path(paths["embedding"]).resolve(),
                                 "run session embedding provenance mismatch", errors)
        selection_manifest_path = data_root / "manifest_search_select.json"
        expected_safe_manifest = manifest.get("data_sha256", {}).get("manifest_search_select.json")
        if not selection_manifest_path.is_file() or sha256_file(selection_manifest_path) != expected_safe_manifest:
            errors.append("pre-selection search/select manifest hash mismatch")

    baseline_root = Path(manifest.get("official_baseline_source", ""))
    baseline_hashes = manifest.get("baseline_source_sha256", {})
    if not baseline_root.is_dir():
        errors.append("native baseline source directory is missing")
    for relative in BASELINE_REQUIRED.get(str(method), ()):
        expected_hash = baseline_hashes.get(relative)
        source_file = baseline_root / relative
        if not expected_hash or not source_file.is_file() or sha256_file(source_file) != expected_hash:
            errors.append(f"native baseline source hash mismatch: {relative}")
    pinned_commit = manifest.get("baseline_git_commit")
    if not pinned_commit or pinned_commit != BASELINE_COMMITS.get(str(method)):
        errors.append("native baseline Git commit is not the frozen upstream revision")
    elif baseline_root.is_dir():
        completed = subprocess.run(
            ["git", "-C", str(baseline_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0 or completed.stdout.strip() != pinned_commit:
            errors.append("native baseline checkout does not match its frozen Git commit")

    points = _points(result, str(axis))
    test_n = int(manifest.get("split_counts", {}).get("test", 0))
    _fail_if(set(points) != {"Q", "E"}, "both frozen Q and E test operating points are required", errors)
    point_summary: dict[str, Any] = {}
    for point_name in ("Q", "E"):
        point = points.get(point_name, {})
        _fail_if(not isinstance(point, dict) or not point, f"{point_name} test result missing", errors)
        if not isinstance(point, dict) or not point:
            continue
        _fail_if(point.get("num_examples") != test_n or test_n != 60,
                 f"{point_name} test count {point.get('num_examples')} does not match frozen D_test size {test_n}", errors)
        correct, accuracy = point.get("correct"), point.get("accuracy")
        _fail_if(not isinstance(correct, int) or not isinstance(accuracy, (int, float)),
                 f"{point_name} exact-match metrics are missing", errors)
        if isinstance(correct, int) and isinstance(accuracy, (int, float)) and test_n:
            _fail_if(not math.isclose(float(accuracy), correct / test_n, rel_tol=0.0, abs_tol=1e-8),
                     f"{point_name} accuracy does not equal exact correct / N", errors)
        audit = point.get("output_audit", {})
        _fail_if(audit.get("csv_rows") != test_n, f"{point_name} failure audit row count mismatch", errors)
        failure_fraction = audit.get("failure_fraction")
        _fail_if(not isinstance(failure_fraction, (int, float)) or failure_fraction > MAX_SAMPLE_FAILURE_RATE,
                 f"{point_name} sample failure fraction exceeds 5% or is missing", errors)
        usage = point.get("resource_usage", {})
        calls, tokens = point.get("calls"), point.get("total_tokens")
        _fail_if(not isinstance(calls, int) or calls <= 0, f"{point_name} has no recorded model calls", errors)
        _fail_if(not isinstance(tokens, int) or tokens <= 0, f"{point_name} has no recorded token usage", errors)
        if isinstance(usage, dict):
            _fail_if(usage.get("calls") != calls or usage.get("total_tokens") != tokens,
                     f"{point_name} resource telemetry disagrees with point totals", errors)
        avg_cost = point.get("avg_cost")
        if isinstance(tokens, int) and test_n and isinstance(avg_cost, (int, float)):
            _fail_if(not math.isclose(float(avg_cost), tokens / test_n, rel_tol=0.0, abs_tol=1e-8),
                     f"{point_name} token cost per example disagrees with total tokens", errors)
        point_summary[point_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "tokens_per_example": point.get("avg_cost"),
            "calls": calls,
            "failure_fraction": failure_fraction,
            "parser_valid_rate": point.get("valid_answer_rate"),
            "component_accuracy": point.get("component_accuracy"),
            "mean_score_0_1_2": point.get("mean_score_0_1_2"),
        }

    selection_audit = result.get("selection", {}).get("output_audit") if method == "maas" else None
    if method == "maas":
        selection = result.get("selection", {})
        _fail_if(selection.get("num_examples") != 24, "MaAS D_select evaluation must contain 24 rows", errors)
        selection_fraction = (selection_audit or {}).get("failure_fraction")
        _fail_if(not isinstance(selection_fraction, (int, float)) or selection_fraction > MAX_SAMPLE_FAILURE_RATE,
                 "MaAS D_select failure fraction exceeds 5% or is missing", errors)
        if selection_audit:
            _fail_if(selection_audit.get("csv_rows") != 24, "MaAS D_select failure audit row count mismatch", errors)
        checkpoint = Path(str(result.get("controller_checkpoint", "")))
        _fail_if(not checkpoint.is_file(), "MaAS selected controller checkpoint is missing", errors)
        if checkpoint.is_file():
            _fail_if(sha256_file(checkpoint) != result.get("controller_checkpoint_sha256"),
                     "MaAS controller checkpoint hash mismatch", errors)
        training = result.get("controller_training", {})
        if not isinstance(training, dict):
            training = {}
        steps = training.get("optimizer_steps")
        tensors = training.get("parameter_tensors")
        changed = training.get("changed_parameter_tensors")
        _fail_if(type(steps) is not int or steps < 1, "MaAS optimizer did not record a training step", errors)
        _fail_if(type(tensors) is not int or tensors < 1, "MaAS controller parameter count is invalid", errors)
        _fail_if(type(changed) is not int or changed < 1 or (
            type(tensors) is int and changed > tensors
        ), "MaAS controller has no verified parameter update", errors)
        state_hashes = {
            name: training.get(name) for name in (
                "initial_state_sha256", "final_state_sha256", "checkpoint_state_sha256",
            )
        }
        for name, digest in state_hashes.items():
            _fail_if(not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None,
                     f"MaAS controller {name} is missing or invalid", errors)
        _fail_if(state_hashes["initial_state_sha256"] == state_hashes["final_state_sha256"],
                 "MaAS controller state did not change during training", errors)
        _fail_if(state_hashes["final_state_sha256"] != state_hashes["checkpoint_state_sha256"],
                 "MaAS checkpoint state differs from trained controller", errors)
        _fail_if(training.get("checkpoint_file_sha256") != result.get("controller_checkpoint_sha256"),
                 "MaAS training audit checkpoint file hash mismatch", errors)
        repeat_audits = result.get("telemetry", {}).get("training_repeat_audit")
        expected_repeat = {
            "native_sample_argument": 1,
            "effective_repetitions": 4,
            "examples_per_repetition": 24,
            "observed_result_rows": 96,
            "completed_repetitions": 4,
        }
        if not isinstance(repeat_audits, list) or len(repeat_audits) != 1 or not isinstance(repeat_audits[0], dict):
            errors.append("MaAS native training repetition audit is missing")
        else:
            observed_repeat = repeat_audits[0]
            if any(type(observed_repeat.get(name)) is not int or observed_repeat[name] != value
                   for name, value in expected_repeat.items()):
                errors.append("MaAS native training did not complete four 24-row repetitions")
            _fail_if(training.get("training_repeat_audit") != observed_repeat,
                     "MaAS controller training audit disagrees with native repetition telemetry", errors)
        prompt_repairs = result.get("prompt_repairs")
        if not isinstance(prompt_repairs, list) or len(prompt_repairs) != 2:
            errors.append("MaAS train/test prompt repair audits are missing")
        else:
            repaired_splits: set[str] = set()
            for repair in prompt_repairs:
                if not isinstance(repair, dict):
                    errors.append("MaAS prompt repair audit is malformed")
                    continue
                path = Path(str(repair.get("path", "")))
                if not path.is_dir() or not path.resolve().is_relative_to(run_dir.resolve()):
                    errors.append("MaAS prompt repair path is outside the run directory or missing")
                    continue
                repaired_splits.add(path.parent.name)
                expected_template = (
                    run_dir / "maas_repo/maas/ext/maas/scripts/optimized/MATH" /
                    path.parent.name / "template"
                )
                _fail_if(path.resolve() != expected_template.resolve(),
                         "MaAS prompt repair path is not the native MATH template", errors)
                _fail_if(repair.get("concise_prompt_variant") != "native_repaired",
                         "MaAS prompt repair variant is not the frozen native repair", errors)
                _fail_if(repair.get("version") != "maas_context_guard_v1" or
                         any(repair.get(key) != value for key, value in MAAS_PROMPT_ADAPTER_PROVENANCE.items()) or
                         "original_helper_git_commit" not in repair,
                         "MaAS prompt repair provenance is incorrect", errors)
                for filename, field in (("prompt.py", "prompt_sha256"), ("op_prompt.py", "operator_sha256")):
                    source = path / filename
                    if not source.is_file() or sha256_file(source) != repair.get(field):
                        errors.append(f"MaAS repaired {filename} hash mismatch")
            _fail_if(repaired_splits != {"train", "test"},
                     "MaAS prompt repairs do not cover both native train and test templates", errors)
    else:
        valid = [row for row in result.get("selection_rows", []) if row.get("status") == "evaluated"]
        _fail_if(not valid, "AFlow has no valid D_select candidate", errors)
        for row in valid:
            if row.get("output_audit", {}).get("failure_fraction", 1.0) > MAX_SAMPLE_FAILURE_RATE:
                errors.append(f"AFlow selected from an invalid D_select round: {row.get('round')}")
        selected_rounds = {
            "Q": result.get("selected_round", {}).get("round"),
            "E": result.get("selected_efficiency_round", {}).get("round"),
        }
        valid_rounds = {row.get("round") for row in valid}
        for point_name, selected_round in selected_rounds.items():
            _fail_if(selected_round not in valid_rounds,
                     f"AFlow {point_name} round was not frozen from a valid D_select candidate", errors)
            point = points.get(point_name, {})
            _fail_if(point.get("selected_round") != selected_round,
                     f"AFlow {point_name} D_test result does not match selected D_select round", errors)

    return {
        "passed": not errors,
        "errors": errors,
        "run_id": manifest.get("run_id"),
        "method": method,
        "axis": axis,
        "seed": seed,
        "point_summary": point_summary,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rpas-root", type=Path, default=None)
    args = parser.parse_args()
    report = validate_run(args.run_dir.resolve(), args.rpas_root.resolve() if args.rpas_root else None)
    (args.run_dir / "quality_gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
