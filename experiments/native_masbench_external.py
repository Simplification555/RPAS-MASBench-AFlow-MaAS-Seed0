#!/usr/bin/env python3
"""Native AFlow/MaAS search-select-test runs on frozen MASBench axes."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from external_comparison.common.pareto import efficiency_operating_point, quality_operating_point
from masbench_answer_protocol import (
    SEPARATOR,
    VERSION as ANSWER_PARSER,
    extract_masbench_answer,
    native_objective_score,
    score_masbench,
)
from masbench_data_protocol import canonical_gold, expected_horizon_width, validate_frozen_splits
from native_aime_formal import (
    CONTEXT_LIMIT,
    MAX_SAMPLE_FAILURE_RATE,
    MODEL,
    OUTPUT_LIMIT,
    audit_aflow_outputs,
    copy_tree,
    dump_json,
    empty_phase_usage,
    ensure_package_tree,
    evaluate_aflow_graph,
    make_aflow_config,
    make_maas_config,
    new_phase_totals,
    patch_aflow_code_formatter,
    patch_aflow_graph_namespace,
    patch_aflow_runtime,
    patch_maas_embedding_path,
    patch_maas_optional_backoff,
    patch_maas_optional_provider_import,
    patch_maas_optional_tools,
    patch_maas_provider_surface,
    patch_maas_runtime,
    phase_scope,
    phase_snapshot,
    seed_everything,
    sha256_file,
    usage_delta,
    validate_aflow_workflow_round,
)


PROTOCOL_VERSION = "masbench_external_protocol_v1"
AXES = ("breadth", "depth", "horizon", "parallel", "robustness")
EXPECTED_BASELINE_COMMITS = {
    "aflow": "3f457218fc716093fe53f6df8a5d5e6379d66346",
    "maas": "987f3c1bc9a96e844fe090db3791446e3ef0f5c7",
}
BASELINE_FILES = {
    "aflow": (
        "scripts/optimizer.py",
        "scripts/evaluator.py",
        "scripts/optimizer_utils/evaluation_utils.py",
        "scripts/optimizer_utils/graph_utils.py",
        "benchmarks/math.py",
    ),
    "maas": (
        "maas/ext/maas/scripts/optimizer.py",
        "maas/ext/maas/scripts/evaluator.py",
        "maas/ext/maas/scripts/optimizer_utils/evaluation_utils.py",
        "maas/ext/maas/scripts/optimizer_utils/graph_utils.py",
        "maas/ext/maas/benchmark/benchmark.py",
        "maas/ext/maas/benchmark/math.py",
    ),
}
RUNTIME_ADAPTER_FILES = {
    "aflow": ("maas_context_guard.py",),
    "maas": (
        "maas_context_guard.py",
        "maas_prompt_repair.py",
        "maas_concise_prompts_v2.py",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("aflow", "maas"), required=True)
    parser.add_argument("--axis", choices=AXES, required=True)
    parser.add_argument("--seed", type=int, choices=(0,), required=True)
    parser.add_argument("--rpas-root", type=Path, default=Path.home() / "RPAS")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--maas-samples", type=int, default=1)
    parser.add_argument("--maas-batch-size", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty MASBench split: {path}")
    return rows


def baseline_source_hashes(source_root: Path, method: str) -> dict[str, str]:
    hashes = {}
    for relative in BASELINE_FILES[method]:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required native {method} baseline file missing: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def baseline_commit(source_root: Path, method: str) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    commit = completed.stdout.strip()
    if commit != EXPECTED_BASELINE_COMMITS[method]:
        raise RuntimeError(
            f"native {method} baseline commit {commit} differs from frozen commit "
            f"{EXPECTED_BASELINE_COMMITS[method]}"
        )
    return commit


def controller_state_sha256(state: dict[str, Any]) -> str:
    """Hash tensor names, shapes, dtypes, and values independent of torch.save metadata."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu()
        try:
            encoded = json.dumps(
                [name, str(tensor.dtype), list(tensor.shape), tensor.reshape(-1).tolist()],
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
        except ValueError as exc:
            raise RuntimeError("MaAS controller state contains nonfinite values") from exc
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def controller_state_snapshot(controller: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in controller.state_dict().items()}


def controller_update_evidence(
    before: dict[str, Any], after: dict[str, Any], parameter_names: set[str], optimizer_steps: int,
) -> dict[str, Any]:
    """Require a native optimizer step and a changed trainable controller tensor."""
    if before.keys() != after.keys() or not parameter_names or not parameter_names <= before.keys():
        raise RuntimeError("MaAS controller parameter layout changed during training")
    changed = sum(not before[name].equal(after[name]) for name in parameter_names)
    initial_hash = controller_state_sha256(before)
    final_hash = controller_state_sha256(after)
    if optimizer_steps < 1 or changed < 1 or initial_hash == final_hash:
        raise RuntimeError(
            f"MaAS controller was not updated: optimizer_steps={optimizer_steps}, "
            f"changed_parameter_tensors={changed}"
        )
    return {
        "optimizer_steps": optimizer_steps,
        "parameter_tensors": len(parameter_names),
        "changed_parameter_tensors": changed,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
    }


def maas_training_repeat_evidence(audits: Any, sample: int, examples: int) -> dict[str, int]:
    """Require observed native training rows from four full repetitions."""
    if not isinstance(audits, list) or len(audits) != 1 or not isinstance(audits[0], dict):
        raise RuntimeError("MaAS did not record exactly one native controller training audit")
    audit = audits[0]
    expected = {
        "native_sample_argument": sample,
        "effective_repetitions": 4,
        "examples_per_repetition": examples,
        "observed_result_rows": examples * 4,
        "completed_repetitions": 4,
    }
    if any(type(audit.get(name)) is not int or audit[name] != value for name, value in expected.items()):
        raise RuntimeError(f"MaAS native controller training repetitions differ from {expected}: {audit}")
    return dict(expected)


def model_asset_provenance(rpas_root: Path, method: str) -> dict[str, Any]:
    """Record pinned revisions and a matching completed local asset audit, if present."""
    revisions_path = rpas_root / "deployment/model_revisions.json"
    specifications = json.loads(revisions_path.read_text(encoding="utf-8"))
    revisions = {"qwen": os.environ.get("RPAS_MODEL_REVISION")}
    if method == "maas":
        revisions["embedding"] = os.environ.get("RPAS_MAAS_EMBEDDING_REVISION")
    for name, revision in revisions.items():
        expected = specifications[name]["revision"]
        if revision != expected:
            raise ValueError(f"{name} model revision {revision!r} does not match pinned {expected!r}")
    model_paths = {
        name: os.environ.get("RPAS_MODEL_PATH" if name == "qwen" else "RPAS_MAAS_EMBEDDING_PATH")
        for name in revisions
    }
    if any(not path or not Path(path).is_dir() for path in model_paths.values()):
        raise RuntimeError("configured local model or embedding path is missing")
    provenance: dict[str, Any] = {
        "revisions_manifest_sha256": sha256_file(revisions_path),
        "revisions": revisions,
        "paths": {name: str(Path(path).resolve()) for name, path in model_paths.items()},
    }
    report_name = os.environ.get("RPAS_MODEL_VERIFICATION_REPORT")
    if not report_name:
        return provenance
    report_path = Path(report_name).expanduser().resolve()
    if not report_path.is_file():
        return provenance
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("manifest_sha256") != provenance["revisions_manifest_sha256"]:
        raise RuntimeError("local model verification report used a different revision manifest")
    for name, revision in revisions.items():
        entry = report.get("models", {}).get(name, {})
        expected_files = {filename: digest for filename, (_, digest) in specifications[name]["files"].items()}
        if (entry.get("revision") != revision or entry.get("sha256_verified") != expected_files or
                Path(entry.get("path", "")).resolve() != Path(model_paths[name]).resolve()):
            raise RuntimeError(f"local model verification report does not verify configured {name} model")
    provenance["verification_report"] = {
        "path": str(report_path), "sha256": sha256_file(report_path),
    }
    return provenance


def run_session_provenance(rpas_root: Path, model_assets: dict[str, Any]) -> dict[str, str]:
    record_name = os.environ.get("RPAS_RUN_SESSION_RECORD")
    if not record_name:
        raise RuntimeError("RPAS_RUN_SESSION_RECORD is required for a formal run")
    record_path = Path(record_name).expanduser().resolve()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    model_report = model_assets.get("verification_report", {})
    if (record.get("project_root") != str(rpas_root.resolve()) or
            record.get("model_revision") != model_assets["revisions"]["qwen"] or
            Path(record.get("model_path", "")).resolve() != Path(model_assets["paths"]["qwen"]) or
            record.get("model_verification_report_sha256") != model_report.get("sha256")):
        raise RuntimeError("run session record disagrees with local model provenance")
    if "embedding" in model_assets["revisions"] and (
        record.get("embedding_revision") != model_assets["revisions"]["embedding"] or
        Path(record.get("embedding_path", "")).resolve() != Path(model_assets["paths"]["embedding"])
    ):
        raise RuntimeError("run session record disagrees with MaAS embedding provenance")
    return {"path": str(record_path), "sha256": sha256_file(record_path)}


def atomic_dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_search_select(
    rows: dict[str, list[dict[str, Any]]], manifest: dict[str, Any], axis: str, data_seed: int,
    split_file_hashes: dict[str, str],
) -> dict[str, Any]:
    """Validate only the pre-selection splits; this manifest intentionally contains no D_test metadata."""
    if manifest.get("schema_version") != "rpas_masbench_search_select_blind_v1":
        raise ValueError("unexpected search/select manifest schema")
    if int(manifest.get("data_seed", -1)) != data_seed:
        raise ValueError("MASBench dataset seed does not match requested frozen seed")
    axis_manifest = manifest.get("axes", {}).get(axis)
    if not isinstance(axis_manifest, dict):
        raise ValueError(f"MASBench dataset has no frozen metadata for axis {axis!r}")

    split_ids: dict[str, list[str]] = {}
    value_counts: dict[str, dict[str, int]] = {}
    expected_counts = {"search": 24, "select": 24}
    if manifest.get("split_sizes") != expected_counts:
        raise ValueError("search/select manifest split sizes differ from required 24/24")
    for split in ("search", "select"):
        current = rows.get(split, [])
        ids = [str(row.get("id", "")) for row in current]
        if len(current) != expected_counts[split] or any(not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{axis}/{split} must contain {expected_counts[split]} unique rows with IDs")
        expected_ids = axis_manifest.get("row_ids", {}).get(split)
        if not expected_ids or ids != [str(item) for item in expected_ids]:
            raise ValueError(f"{axis}/{split} row IDs/order differ from frozen search/select metadata")
        if split_file_hashes.get(split) != axis_manifest.get("file_sha256", {}).get(split):
            raise ValueError(f"{axis}/{split} bytes differ from D_test-blind manifest hash")
        counts: dict[str, int] = {}
        for row in current:
            if row.get("axis") != axis or row.get("dataset") != "masbench":
                raise ValueError(f"row {row.get('id')} has mismatched dataset/axis metadata")
            if row.get("official_split") != "train":
                raise ValueError(f"row {row.get('id')} is not from the official training split")
            canonical = canonical_gold(row.get("answer"), allow_raw=(axis == "robustness"))
            if axis == "horizon":
                expected_width = expected_horizon_width(row.get("input"))
                if len(canonical.split(SEPARATOR)) != expected_width:
                    raise ValueError(f"horizon gold width mismatch for row {row.get('id')}")
            key = str(row.get("axis_value"))
            counts[key] = counts.get(key, 0) + 1
        expected_values = axis_manifest.get(f"{split}_values")
        if expected_values is not None and counts != {str(k): int(v) for k, v in expected_values.items()}:
            raise ValueError(f"{axis}/{split} axis-value distribution differs from frozen metadata")
        split_ids[split] = ids
        value_counts[split] = dict(sorted(counts.items()))

    if set(split_ids["search"]) & set(split_ids["select"]):
        raise ValueError("D_search and D_select must be disjoint")
    return {"split_ids": split_ids, "axis_value_counts": value_counts, "search_select_disjoint": True}


def freeze_selection(
    args: argparse.Namespace, candidates: dict[str, Any], details: dict[str, Any]
) -> tuple[Path, str]:
    """Durably record the D_select decision before any code reads a D_test row."""
    path = args.run_dir / "selection_frozen.json"
    payload = {
        "status": "frozen_before_test_access",
        "protocol_version": PROTOCOL_VERSION,
        "method": args.method,
        "axis": args.axis,
        "seed": args.seed,
        "selection_split": "D_select",
        "test_split": "D_test",
        "d_test_opened": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected": candidates,
        "selection_details": details,
    }
    atomic_dump_json(path, payload)
    return path, sha256_file(path)


def load_test_after_selection(
    args: argparse.Namespace,
    rows: dict[str, list[dict[str, Any]]],
    selection_path: Path,
    selection_hash: str,
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    lock = json.loads(selection_path.read_text(encoding="utf-8"))
    if (lock.get("status") != "frozen_before_test_access" or lock.get("d_test_opened") is not False
            or lock.get("method") != args.method or lock.get("axis") != args.axis or lock.get("seed") != args.seed
            or sha256_file(selection_path) != selection_hash):
        raise RuntimeError("D_test access blocked: D_select selection lock is missing, mismatched, or modified")

    # This is the first point in a run where either D_test examples or its full
    # manifest (which contains test IDs) are read.
    test_path = args.data_dir / args.axis / "test_60.jsonl"
    test_rows = read_jsonl(test_path)
    full_manifest_path = args.data_dir / args.axis / "manifest.json"
    full_manifest = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    rows["test"] = test_rows
    file_hashes = {
        "search": sha256_file(args.data_dir / args.axis / "search_24.jsonl"),
        "select": sha256_file(args.data_dir / args.axis / "select_24.jsonl"),
        "test": sha256_file(test_path),
    }
    audit = validate_frozen_splits(rows, full_manifest, args.axis, args.data_seed, file_hashes)
    pre_test_ids = run_manifest["pre_test_split_audit"]["split_ids"]
    for split in ("search", "select"):
        if audit["split_ids"][split] != pre_test_ids[split]:
            raise RuntimeError(f"{split} rows changed between pre-selection validation and D_test access")

    opened_at = datetime.now(timezone.utc).isoformat()
    run_manifest["split_counts"] = {split: len(values) for split, values in rows.items()}
    run_manifest["pairwise_disjoint"] = audit["pairwise_disjoint"]
    run_manifest["selection_lock_sha256"] = selection_hash
    run_manifest["axis_value_counts"] = audit["axis_value_counts"]
    run_manifest["split_ids"] = audit["split_ids"]
    run_manifest["d_test_access"] = {
        "opened_at_utc": opened_at,
        "selection_lock": str(selection_path.name),
        "selection_lock_sha256": selection_hash,
        "test_file_sha256": sha256_file(test_path),
        "full_manifest_sha256": sha256_file(full_manifest_path),
        "examples": len(test_rows),
    }
    run_manifest["d_test_opened"] = True
    run_manifest["data_sha256"][f"{args.axis}/test_60.jsonl"] = run_manifest["d_test_access"]["test_file_sha256"]
    run_manifest["data_sha256"][f"{args.axis}/manifest.json"] = run_manifest["d_test_access"]["full_manifest_sha256"]
    atomic_dump_json(args.run_dir / "run_manifest.json", run_manifest)
    atomic_dump_json(args.run_dir / "split_manifest.json", run_manifest)
    return audit


def math_rows(rows: list[dict[str, Any]], instruction: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        answer = canonical_gold(row["answer"], allow_raw=(row.get("axis") == "robustness"))
        result.append({
            "problem": f"{instruction.rstrip()}\n\n{row['input']}",
            "solution": f"FINAL ANSWER: {answer}",
            "masbench_id": str(row["id"]),
            "answer": answer,
        })
    return result


def shared_masbench_metrics(log_dir: Path, expected_rows: list[dict[str, Any]], instruction: str) -> dict[str, Any]:
    csv_paths = sorted(log_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime)
    if not csv_paths:
        raise RuntimeError(f"native evaluator produced no CSV output in {log_dir}")
    with csv_paths[-1].open(newline="", encoding="utf-8-sig") as stream:
        outputs = list(csv.DictReader(stream))
    if len(outputs) != len(expected_rows):
        raise RuntimeError(f"MASBench output count mismatch in {log_dir}: {len(outputs)} != {len(expected_rows)}")

    expected_native = math_rows(expected_rows, instruction)
    scores = []
    for index, (output, expected) in enumerate(zip(outputs, expected_native)):
        question = re.sub(r"\s+", " ", str(output.get("question", "") or "")).strip()
        expected_question = re.sub(r"\s+", " ", expected["problem"]).strip()
        if question and question != expected_question:
            raise RuntimeError(f"MASBench evaluator row/order mismatch at {log_dir} row {index}")
        csv_gold = extract_masbench_answer(output.get("expected_output", ""), len(expected["answer"].split(SEPARATOR)))
        if csv_gold and csv_gold != score_masbench("FINAL ANSWER: " + expected["answer"], expected["answer"])["gold"]:
            raise RuntimeError(f"MASBench CSV gold mismatch at {log_dir} row {index}")
        if output.get("expected_output") and not csv_gold:
            raise RuntimeError(f"MASBench CSV gold is unparseable at {log_dir} row {index}")
        scores.append(score_masbench(output.get("prediction", ""), expected["answer"]))

    n = len(scores)
    correct = sum(bool(row["correct"]) for row in scores)
    component_total = sum(int(row["component_total"]) for row in scores)
    component_correct = sum(int(row["component_correct"]) for row in scores)
    return {
        "score": correct / n if n else 0.0,
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "num_examples": n,
        "valid_answer_rate": sum(bool(row["parser_valid"]) for row in scores) / n if n else 0.0,
        "component_accuracy": component_correct / component_total if component_total else 0.0,
        "component_correct": component_correct,
        "component_total": component_total,
        "mean_score_0_1_2": sum(int(row["score_0_1_2"]) for row in scores) / n if n else 0.0,
        "answer_parser": ANSWER_PARSER,
    }


def patch_native_math_scorer(module_name: str) -> None:
    module = __import__(module_name, fromlist=["MATHBenchmark"])
    benchmark_class = module.MATHBenchmark

    def calculate_score(self: Any, expected_output: str, prediction: str):
        gold = extract_masbench_answer(expected_output)
        if not gold:
            raise ValueError("MASBench expected answer could not be parsed")
        # Shared ordinal objective: wrong=0, partially correct vector=1, exact=2.
        return native_objective_score(prediction, gold)

    benchmark_class.calculate_score = calculate_score


def patch_masbench_graph_namespace(graph_utils: Any) -> None:
    """Use an isolated MASBench package name in all generated AFlow rounds."""
    from native_aime_formal import patch_aflow_graph_namespace

    patch_aflow_graph_namespace(graph_utils)
    original_write = graph_utils.write_graph_files

    def write_graph_files(self: Any, directory: str, response: dict[str, Any], round_number: int, dataset: str) -> None:
        original_write(self, directory, response, round_number, dataset)
        for filename in ("graph.py", "prompt.py"):
            path = Path(directory) / filename
            if path.exists():
                text = path.read_text(encoding="utf-8").replace(
                    "workspace.rpas_aime_native.MATH", "workspace.rpas_masbench_native.MATH"
                )
                path.write_text(text, encoding="utf-8")

    graph_utils.write_graph_files = write_graph_files


def model_math_rows(rows: list[dict[str, Any]], instruction: str) -> list[dict[str, Any]]:
    return math_rows(rows, instruction)


def run_aflow(
    args: argparse.Namespace, source: Path, data_dir: Path, run_id: str,
    rows: dict[str, list[dict[str, Any]]], instruction: str,
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    run_started = time.time()
    repo = args.run_dir / "aflow_repo"
    copy_tree(source, repo)
    patch_aflow_code_formatter(repo)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    validate_file = repo / "data/datasets/math_validate.jsonl"
    test_file = repo / "data/datasets/math_test.jsonl"
    from native_aime_formal import write_jsonl
    write_jsonl(validate_file, model_math_rows(rows["search"], instruction))
    # AFlow ships with a native test split. Keep it empty through search and
    # D_select so an upstream helper cannot accidentally score on D_test.
    write_jsonl(test_file, [])

    telemetry = patch_aflow_runtime(args.max_tokens, args.concurrency)
    patch_native_math_scorer("benchmarks.math")
    from scripts.optimizer import Optimizer
    from scripts.evaluator import Evaluator
    from scripts.optimizer_utils.graph_utils import GraphUtils

    relative_root = Path("workspace") / "rpas_masbench_native" / "MATH"
    source_seed = repo / "workspace/MATH/workflows/round_1"
    target_seed = repo / relative_root / "workflows/round_1"
    target_seed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_seed, target_seed)
    shutil.copytree(repo / "workspace/MATH/workflows/template", repo / relative_root / "workflows/template", dirs_exist_ok=True)
    ensure_package_tree(repo / relative_root / "workflows", repo)
    for filename in ("graph.py", "prompt.py"):
        path = target_seed / filename
        if path.exists():
            text = path.read_text(encoding="utf-8").replace(
                "workspace.MATH", "workspace.rpas_masbench_native.MATH"
            )
            if filename == "prompt.py" and re.search(
                r"(?m)^#\s*[A-Z][A-Z0-9_]*_PROMPT\s*=\s*['\"]{1,3}", text
            ):
                text = "\n".join(
                    line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line
                    for line in text.splitlines()
                ) + "\n"
            path.write_text(text, encoding="utf-8")
    patch_masbench_graph_namespace(GraphUtils)

    config = make_aflow_config(args.endpoint)
    seed_everything(args.seed)
    optimizer = Optimizer(
        dataset="MATH",
        question_type="math",
        opt_llm_config=config,
        exec_llm_config=config,
        operators=["Custom", "ScEnsemble", "Programmer"],
        sample=4,
        check_convergence=False,
        optimized_path=str(relative_root.parent),
        initial_round=1,
        max_rounds=args.max_rounds,
        validation_rounds=1,
    )
    # The released MCTS retry loop is unbounded when the model repeats a
    # previously rejected modification. Bound only this failure path; valid
    # native modifications and evaluation logic remain unchanged.
    original_optimize_graph = optimizer._optimize_graph
    original_check_modification = optimizer.experience_utils.check_modification
    mutation_retry_cap = 8

    async def bounded_optimize_graph():
        attempts = 0

        def bounded_check(processed_experience: Any, modification: str, sample_round: int) -> bool:
            nonlocal attempts
            attempts += 1
            if attempts > mutation_retry_cap:
                raise RuntimeError(f"AFlow repeated/rejected {attempts} graph mutations; capped at {mutation_retry_cap}")
            return original_check_modification(processed_experience, modification, sample_round)

        optimizer.experience_utils.check_modification = bounded_check
        try:
            return await original_optimize_graph()
        finally:
            optimizer.experience_utils.check_modification = original_check_modification

    optimizer._optimize_graph = bounded_optimize_graph
    phase_wall: dict[str, float] = {}
    with phase_scope("search", phase_wall):
        optimizer.optimize("Graph")
    workflow_root = repo / relative_root / "workflows"
    rounds = sorted(
        int(match.group(1))
        for path in workflow_root.glob("round_*/graph.py")
        if (match := re.fullmatch(r"round_(\d+)", path.parent.name)) and int(match.group(1)) <= args.max_rounds + 1
    )
    if not rounds:
        raise RuntimeError("AFlow native search produced no workflow rounds")

    write_jsonl(validate_file, model_math_rows(rows["select"], instruction))
    evaluator = Evaluator(eval_path=str(args.run_dir / "aflow_selection"))
    graph_utils = GraphUtils(str(relative_root))
    selection_rows = []
    with phase_scope("select", phase_wall):
        for round_number in rounds:
            log_dir = args.run_dir / "aflow_selection" / f"round_{round_number}"
            issues = validate_aflow_workflow_round(workflow_root, round_number)
            if issues:
                selection_rows.append({"round": round_number, "status": "invalid", "errors": issues})
                continue
            try:
                graph = graph_utils.load_graph(round_number, str(relative_root / "workflows"))
                before = phase_snapshot(telemetry, "select")
                native_score, native_avg_cost, native_total_cost = evaluate_aflow_graph(
                    evaluator=evaluator, graph=graph, config=config, dataset_path=validate_file,
                    log_dir=log_dir, is_test=False,
                )
                shared = shared_masbench_metrics(log_dir, rows["select"], instruction)
                usage = usage_delta(phase_snapshot(telemetry, "select"), before)
                audit = audit_aflow_outputs(log_dir)
                total_tokens = int(usage["total_tokens"])
                valid = audit["failure_fraction"] <= MAX_SAMPLE_FAILURE_RATE
                candidate = {
                    "round": round_number,
                    "status": "evaluated" if valid else "invalid",
                    "candidate_id": f"round_{round_number}",
                    "native_score": native_score,
                    "score": shared["score"],
                    "shared_metrics": shared,
                    "valid": valid,
                    "total_calls": usage["calls"],
                    "total_tokens": total_tokens,
                    "avg_cost": total_tokens / max(1, len(rows["select"])),
                    "total_cost": total_tokens,
                    "cost_unit": "model_tokens",
                    "native_avg_cost": native_avg_cost,
                    "native_total_cost": native_total_cost,
                    "output_audit": audit,
                }
                if not valid:
                    candidate["errors"] = [f"failure rate {audit['failure_fraction']:.1%} exceeds 5%"]
                selection_rows.append(candidate)
            except Exception as exc:
                selection_rows.append({"round": round_number, "status": "invalid", "errors": [f"{type(exc).__name__}: {exc}"]})

    valid_rows = [row for row in selection_rows if row.get("status") == "evaluated"]
    if not valid_rows:
        raise RuntimeError("AFlow produced no valid MASBench D_select candidates")
    q_pick = quality_operating_point(valid_rows)
    e_pick = efficiency_operating_point(valid_rows, delta=0.05)
    if q_pick is None or e_pick is None:
        raise RuntimeError("AFlow failed to select MASBench Q/E points on D_select")
    selected = {"Q": next(row for row in valid_rows if row["round"] == q_pick["round"]),
                "E": next(row for row in valid_rows if row["round"] == e_pick["round"])}

    selection_path, selection_hash = freeze_selection(
        args,
        {name: {"round": int(candidate["round"]), "candidate_id": candidate["candidate_id"]}
         for name, candidate in selected.items()},
        {"selection_rows": selection_rows, "quality_point": q_pick, "efficiency_point": e_pick},
    )
    load_test_after_selection(args, rows, selection_path, selection_hash, run_manifest)
    write_jsonl(test_file, model_math_rows(rows["test"], instruction))

    test_results = {}
    for point, candidate in selected.items():
        if point == "E" and selected["E"]["round"] == selected["Q"]["round"]:
            test_results["E"] = {**test_results["Q"], "same_candidate_as": "Q", "operating_point": "E"}
            continue
        candidate_round = int(candidate["round"])
        phase = f"test:{args.axis}:{point}"
        test_log = args.run_dir / "aflow_test" / point
        test_graph = graph_utils.load_graph(candidate_round, str(relative_root / "workflows"))
        with phase_scope(phase, phase_wall):
            native_score, native_avg_cost, native_total_cost = evaluate_aflow_graph(
                evaluator=evaluator, graph=test_graph, config=config, dataset_path=test_file,
                log_dir=test_log, is_test=True,
            )
        shared = shared_masbench_metrics(test_log, rows["test"], instruction)
        audit = audit_aflow_outputs(test_log)
        usage = phase_snapshot(telemetry, phase)
        test_results[point] = {
            **shared,
            "native_score": native_score,
            "avg_cost": usage["total_tokens"] / max(1, len(rows["test"])),
            "total_cost": usage["total_tokens"],
            "cost_unit": "model_tokens",
            "native_avg_cost": native_avg_cost,
            "native_total_cost": native_total_cost,
            "resource_usage": usage,
            "calls": usage["calls"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "failed_calls": usage["failed_calls"],
            "output_audit": audit,
            "selected_round": candidate_round,
            "operating_point": point,
        }
    if selected["Q"]["round"] == selected["E"]["round"]:
        test_results["E"] = {**test_results["Q"], "same_candidate_as": "Q", "operating_point": "E"}

    guard = telemetry.pop("context_guard", None)
    result = {
        "method": "aflow",
        "search_entrypoint": "AFlow native MCTS Optimizer.optimize(Graph)",
        "native_method_config": {
            "dataset": "MATH",
            "question_type": "math",
            "operators": ["Custom", "ScEnsemble", "Programmer"],
            "sample": 4,
            "check_convergence": False,
            "initial_round": 1,
            "max_rounds": args.max_rounds,
            "validation_rounds": 1,
            "mutation_generation_retry_cap": mutation_retry_cap,
        },
        "search_rounds": rounds,
        "selection_rows": selection_rows,
        "selected_round": selected["Q"],
        "selected_efficiency_round": selected["E"],
        "selection_lock": {"path": selection_path.name, "sha256": selection_hash},
        "test": test_results,
        "telemetry": {**telemetry, "context_guard": guard.summary() if guard is not None else None,
                      "phase_wall_seconds": phase_wall, "wall_time_seconds": time.time() - run_started},
        "controls": controls(args),
    }
    dump_json(args.run_dir / "native_result.json", result)
    return result


def run_maas(
    args: argparse.Namespace, source: Path, data_dir: Path, run_id: str,
    rows: dict[str, list[dict[str, Any]]], instruction: str,
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    run_started = time.time()
    repo = args.run_dir / "maas_repo"
    copy_tree(source, repo)
    patch_maas_optional_provider_import(repo)
    patch_maas_provider_surface(repo)
    patch_maas_optional_backoff(repo)
    patch_maas_optional_tools(repo)
    patch_maas_embedding_path(repo)
    make_maas_config(repo, args.endpoint, args.max_tokens)
    data_path = repo / "maas/ext/maas/data"
    from native_aime_formal import write_jsonl
    write_jsonl(data_path / "math_train.jsonl", model_math_rows(rows["search"], instruction))
    write_jsonl(data_path / "math_test.jsonl", model_math_rows(rows["select"], instruction))

    relative_root = Path("workspace") / "rpas_masbench_native" / args.axis / f"seed_{args.seed}" / "MATH"
    train_source = repo / "maas/ext/maas/scripts/optimized/MATH/train"
    test_source = repo / "maas/ext/maas/scripts/optimized/MATH/test"
    from maas_context_guard import patch_template_directory
    prompt_repairs = [patch_template_directory(train_source / "template", concise=False),
                      patch_template_directory(test_source / "template", concise=False)]
    shutil.copytree(train_source, repo / relative_root / "train", dirs_exist_ok=True)
    shutil.copytree(test_source, repo / relative_root / "test", dirs_exist_ok=True)
    ensure_package_tree(repo / relative_root, repo)
    os.environ["METAGPT_PROJECT_ROOT"] = str(repo)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    telemetry = patch_maas_runtime(args.concurrency, args.max_tokens)
    patch_native_math_scorer("maas.ext.maas.benchmark.math")

    from maas.configs.models_config import ModelsConfig
    from maas.ext.maas.benchmark.experiment_configs import EXPERIMENT_CONFIGS
    from maas.ext.maas.scripts.optimizer import Optimizer
    config = EXPERIMENT_CONFIGS["MATH"]
    models = ModelsConfig.default()
    llm_config = models.get("qwen35_9b")
    if llm_config is None:
        raise RuntimeError("MaAS qwen35_9b model config is unavailable")
    seed_everything(args.seed)
    optimizer = Optimizer(
        dataset=config.dataset, question_type=config.question_type,
        opt_llm_config=llm_config, exec_llm_config=llm_config,
        operators=config.operators, optimized_path=str(relative_root.parent),
        sample=args.maas_samples, round=1, batch_size=args.maas_batch_size,
        lr=0.01, is_textgrad=False,
    )
    import torch

    before_controller = controller_state_snapshot(optimizer.controller)
    parameter_names = {name for name, _ in optimizer.controller.named_parameters()}
    original_step = optimizer.optimizer.step
    optimizer_steps = 0

    def observed_step(*step_args: Any, **step_kwargs: Any) -> Any:
        nonlocal optimizer_steps
        outcome = original_step(*step_args, **step_kwargs)
        optimizer_steps += 1
        return outcome

    optimizer.optimizer.step = observed_step
    phase_wall: dict[str, float] = {}
    try:
        with phase_scope("search", phase_wall):
            optimizer.optimize("Graph")
    finally:
        optimizer.optimizer.step = original_step
    checkpoint = repo / relative_root / "train" / "round_1" / f"MATH_controller_sample{args.maas_samples}.pth"
    if not checkpoint.exists():
        raise RuntimeError(f"MaAS native search did not save controller: {checkpoint}")
    controller_training = controller_update_evidence(
        before_controller, controller_state_snapshot(optimizer.controller),
        parameter_names, optimizer_steps,
    )
    controller_training["training_repeat_audit"] = maas_training_repeat_evidence(
        telemetry.get("training_repeat_audit"), args.maas_samples, len(rows["search"]),
    )
    checkpoint_state_hash = controller_state_sha256(torch.load(checkpoint, map_location="cpu"))
    if checkpoint_state_hash != controller_training["final_state_sha256"]:
        raise RuntimeError("MaAS saved controller differs from trained controller state")
    controller_training["checkpoint_state_sha256"] = checkpoint_state_hash
    controller_training["checkpoint_file_sha256"] = sha256_file(checkpoint)

    loop = asyncio.new_event_loop()
    try:
        with phase_scope("select", phase_wall):
            native_selection_score = float(loop.run_until_complete(optimizer.test()))
        eval_dir = repo / relative_root / "test" / "round_1"
        selection = shared_masbench_metrics(eval_dir, rows["select"], instruction)
        selection_audit = audit_aflow_outputs(eval_dir)
        if selection_audit["failure_fraction"] > MAX_SAMPLE_FAILURE_RATE:
            raise RuntimeError(
                f"MaAS D_select failures {selection_audit['failure_fraction']:.1%} exceed "
                f"the {MAX_SAMPLE_FAILURE_RATE:.1%} quality threshold; D_test will not be opened"
            )
        selection_usage = phase_snapshot(telemetry, "select")
        selection_path, selection_hash = freeze_selection(
            args,
            {"Q": {"controller_checkpoint": str(checkpoint),
                   "controller_checkpoint_sha256": sha256_file(checkpoint)}},
            {"selection": selection, "native_selection_score": native_selection_score,
             "output_audit": selection_audit},
        )
        load_test_after_selection(args, rows, selection_path, selection_hash, run_manifest)
        test_results = {}
        for split_name in ("test",):
            write_jsonl(data_path / "math_test.jsonl", model_math_rows(rows[split_name], instruction))
            phase = f"test:{args.axis}"
            with phase_scope(phase, phase_wall):
                native_score = float(loop.run_until_complete(optimizer.test()))
            shared = shared_masbench_metrics(eval_dir, rows[split_name], instruction)
            audit = audit_aflow_outputs(eval_dir)
            usage = phase_snapshot(telemetry, phase)
            point = {
                **shared,
                "native_score": native_score,
                "avg_cost": usage["total_tokens"] / max(1, len(rows[split_name])),
                "total_cost": usage["total_tokens"],
                "cost_unit": "model_tokens",
                "resource_usage": usage,
                "calls": usage["calls"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "failed_calls": usage["failed_calls"],
                "output_audit": audit,
                "selected_controller": str(checkpoint),
                "operating_point": "Q",
            }
            test_results["Q"] = point
            test_results["E"] = {**point, "same_candidate_as": "Q", "operating_point": "E"}
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    context_guard = telemetry.pop("_context_guard", None)
    result = {
        "method": "maas",
        "search_entrypoint": "MaAS native Optimizer.optimize(Graph)",
        "native_method_config": {
            "dataset": config.dataset,
            "question_type": config.question_type,
            "operators": list(config.operators),
            "controller_sample": args.maas_samples,
            "round": 1,
            "batch_size": args.maas_batch_size,
            "learning_rate": float(optimizer.lr),
            "is_textgrad": False,
            "training_repetitions": 4,
        },
        "selection": {"score": selection["score"], "native_score": native_selection_score,
                      **selection, "split": "D_select", "num_examples": len(rows["select"]),
                      "resource_usage": selection_usage, "output_audit": selection_audit},
        "controller_checkpoint": str(checkpoint),
        "controller_checkpoint_sha256": sha256_file(checkpoint),
        "controller_training": controller_training,
        "selection_lock": {"path": selection_path.name, "sha256": selection_hash},
        "test": {args.axis: {**test_results["Q"], "Q": test_results["Q"], "E": test_results["E"]}},
        "telemetry": {**telemetry, "context_guard": context_guard.summary() if context_guard is not None else None,
                      "phase_wall_seconds": phase_wall, "wall_time_seconds": time.time() - run_started},
        "prompt_repairs": prompt_repairs,
        "controls": controls(args),
    }
    dump_json(args.run_dir / "native_result.json", result)
    return result


def controls(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": MODEL,
        "endpoint": args.endpoint,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "thinking": False,
        "concurrency": args.concurrency,
        "max_model_len": CONTEXT_LIMIT,
        "max_num_seqs": int(os.environ.get("RPAS_MAX_NUM_SEQS", "24")),
        "tensor_parallel_size": int(os.environ.get("RPAS_TP_SIZE", "1")),
        "data_seed": args.data_seed,
        "search_seed": args.seed,
        "search_size": 24,
        "selection_size": 24,
        "test_size": 60,
        "native_search_round_limit": args.max_rounds if args.method == "aflow" else 1,
        "native_search_mutation_retry_cap": 8 if args.method == "aflow" else None,
        "maas_samples": args.maas_samples if args.method == "maas" else None,
        "maas_batch_size": args.maas_batch_size if args.method == "maas" else None,
        "gold_protocol": f"{ANSWER_PARSER}; exact match + component accuracy + 0/1/2 ordinal score",
        "selection_split": "D_select",
        "test_split": "D_test",
    }


def main() -> None:
    args = parse_args()
    fixed = {
        "seed": (args.seed, 0),
        "data_seed": (args.data_seed, 2026),
        "max_tokens": (args.max_tokens, OUTPUT_LIMIT),
        "concurrency": (args.concurrency, 8),
        "max_rounds": (args.max_rounds, 8),
        "maas_samples": (args.maas_samples, 1),
        "maas_batch_size": (args.maas_batch_size, 8),
        "max_model_len": (int(os.environ.get("RPAS_MAX_MODEL_LEN", "8192")), CONTEXT_LIMIT),
        "max_num_seqs": (int(os.environ.get("RPAS_MAX_NUM_SEQS", "24")), 24),
        "tensor_parallel_size": (int(os.environ.get("RPAS_TP_SIZE", "1")), 1),
    }
    mismatched = [f"{name}={actual} (required {expected})" for name, (actual, expected) in fixed.items()
                  if actual != expected]
    if mismatched:
        raise ValueError("refusing non-protocol MASBench settings: " + ", ".join(mismatched))
    if os.environ.get("RPAS_GPU_MEMORY_UTILIZATION", "0.92") != "0.92":
        raise ValueError("refusing non-protocol GPU memory utilization; expected 0.92")
    args.rpas_root = args.rpas_root.resolve()
    args.data_dir = (args.data_dir or args.rpas_root / "data/masbench").resolve()
    args.run_dir = args.run_dir.resolve()
    if args.run_dir.exists():
        raise FileExistsError(f"refusing to overwrite prior/partial result: {args.run_dir}")
    from phase2_wan_agent_search import MASBENCH_INSTRUCTION
    task_instruction = MASBENCH_INSTRUCTION
    if args.axis == "robustness":
        task_instruction += (
            "Robustness-axis rule: compute values defined by algebraic relations in Z_23, but copy every "
            "explicitly requested injected fact or magic number exactly as written; these raw retrieval values "
            "are not reduced modulo 23. Preserve the complete requested answer order.\n"
        )

    axis_dir = args.data_dir / args.axis
    rows = {name: read_jsonl(axis_dir / f"{name}_24.jsonl") for name in ("search", "select")}
    # The pre-selection manifest is deliberately D_test-blind: it contains no
    # test IDs, labels, values, or distribution statistics.
    search_select_manifest_path = args.data_dir / "manifest_search_select.json"
    search_select_manifest = json.loads(search_select_manifest_path.read_text(encoding="utf-8"))
    frozen_split_audit = validate_search_select(
        rows, search_select_manifest, args.axis, args.data_seed,
        {name: sha256_file(axis_dir / f"{name}_24.jsonl") for name in ("search", "select")},
    )
    data_files = {
        f"{args.axis}/search_24.jsonl": sha256_file(axis_dir / "search_24.jsonl"),
        f"{args.axis}/select_24.jsonl": sha256_file(axis_dir / "select_24.jsonl"),
        "manifest_search_select.json": sha256_file(search_select_manifest_path),
    }
    contract_path = args.rpas_root / "experiments/MASBENCH_TASK_CONTRACT_v1.txt"
    task_contract_sha = sha256_file(contract_path)
    instruction_sha = hashlib.sha256(task_instruction.encode("utf-8")).hexdigest()
    source_root = Path(os.environ.get("RPAS_AFLOW_SOURCE" if args.method == "aflow" else "RPAS_MAAS_SOURCE",
                                     str(Path.home() / "challenge_cup_ec1/external_baselines" / ("AFlow" if args.method == "aflow" else "MaAS")))).resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    frozen_baseline_commit = baseline_commit(source_root, args.method)
    baseline_hashes = baseline_source_hashes(source_root, args.method)
    endpoint = args.endpoint.rstrip("/") + "/models"
    from urllib.request import urlopen
    with urlopen(endpoint, timeout=10) as response:
        model_data = json.loads(response.read().decode("utf-8")).get("data", [])
    if MODEL not in {str(item.get("id", "")) for item in model_data}:
        raise RuntimeError(f"endpoint {endpoint} does not serve {MODEL}")

    run_id = f"masbench:{args.method}:{args.axis}:seed_{args.seed}:{PROTOCOL_VERSION}"
    os.environ["MASBENCH_RUN_ID"] = run_id
    os.environ["RPAS_MAX_MODEL_LEN"] = str(CONTEXT_LIMIT)
    os.environ["RPAS_MAX_NUM_SEQS"] = os.environ.get("RPAS_MAX_NUM_SEQS", "24")
    os.environ["RPAS_TP_SIZE"] = "1"
    model_assets = model_asset_provenance(args.rpas_root, args.method)
    session_record = run_session_provenance(args.rpas_root, model_assets)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "method": args.method,
        "dataset": "Salesforce/MASBench",
        "axis": args.axis,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "split_counts": {name: len(values) for name, values in rows.items()},
        "pre_test_split_audit": frozen_split_audit,
        "search_select_disjoint": frozen_split_audit["search_select_disjoint"],
        "data_sha256": data_files,
        "task_contract_sha256": task_contract_sha,
        "task_instruction_sha256": instruction_sha,
        "task_protocol_exception": (
            "robustness preserves explicitly requested injected fact/magic-number values verbatim; algebraic values remain Z_23"
            if args.axis == "robustness" else None
        ),
        "answer_parser": ANSWER_PARSER,
        "answer_parser_sha256": sha256_file(Path(__file__).with_name("masbench_answer_protocol.py")),
        "runner_sha256": sha256_file(Path(__file__)),
        "base_runner_sha256": sha256_file(Path(__file__).with_name("native_aime_formal.py")),
        "runtime_adapter_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in RUNTIME_ADAPTER_FILES[args.method]
        },
        "model_asset_provenance": model_assets,
        "run_session_record": session_record,
        "model": MODEL,
        "context_limit": CONTEXT_LIMIT,
        "output_limit": OUTPUT_LIMIT,
        "selection_split": "D_select",
        "test_split": "D_test",
        "d_test_opened": False,
        "selection_lock_required_before_test": True,
        "baseline_source_sha256": baseline_hashes,
        "baseline_git_commit": frozen_baseline_commit,
        "official_baseline_source": str(source_root),
        "rpas_root": str(args.rpas_root),
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir()
    dump_json(args.run_dir / "run_manifest.json", manifest)
    dump_json(args.run_dir / "split_manifest.json", manifest)
    try:
        if args.method == "aflow":
            result = run_aflow(args, source_root, args.data_dir, run_id, rows, task_instruction, manifest)
        else:
            result = run_maas(args, source_root, args.data_dir, run_id, rows, task_instruction, manifest)
        result["run_id"] = run_id
        dump_json(args.run_dir / "native_result.json", result)
        if not manifest.get("d_test_access", {}).get("selection_lock_sha256"):
            raise RuntimeError("run cannot complete without a pre-test D_select selection lock")
        manifest["d_test_opened"] = True
        manifest["status"] = "complete"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["native_result_sha256"] = sha256_file(args.run_dir / "native_result.json")
        dump_json(args.run_dir / "run_manifest.json", manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        dump_json(args.run_dir / "run_manifest.json", manifest)
        raise


def manifest_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("id", "")) for row in rows]


if __name__ == "__main__":
    main()
