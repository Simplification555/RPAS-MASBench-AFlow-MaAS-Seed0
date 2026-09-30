#!/usr/bin/env python3
"""Exercise one native AFlow graph mutation on two synthetic math problems.

This diagnostic does not open MASBench data or formal run output. It copies the
pinned AFlow checkout into a new diagnostic directory, runs the released MATH
optimizer once, and scores both the seed and generated graphs separately.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT))

EXPECTED_COMMIT = "3f457218fc716093fe53f6df8a5d5e6379d66346"
MODEL = "Qwen/Qwen3.5-9B"
TASKS = (
    {"id": "synthetic_add", "problem": "Compute 3 + 4. Give the final integer as FINAL ANSWER: <integer>.", "answer": "7"},
    {"id": "synthetic_multiply", "problem": "Compute 3 times 4. Give the final integer as FINAL ANSWER: <integer>.", "answer": "12"},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aflow-source", type=Path, default=ROOT / "external_baselines" / "AFlow")
    parser.add_argument("--endpoint", default=os.environ.get("RPAS_ENDPOINT", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model-path", type=Path, default=os.environ.get("RPAS_MODEL_PATH"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--check-only", action="store_true", help="verify pinned checkout and template without a model service")
    args = parser.parse_args()
    if not 128 <= args.max_tokens <= 6144:
        parser.error("--max-tokens must be between 128 and 6144")
    if args.timeout_seconds < 30:
        parser.error("--timeout-seconds must be at least 30")
    return args


def validate_source(source: Path) -> dict[str, str]:
    if not source.is_dir():
        raise FileNotFoundError(f"AFlow checkout is missing: {source}")
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"AFlow checkout {commit} differs from pinned {EXPECTED_COMMIT}")
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("AFlow checkout has local changes; use a clean pinned checkout for diagnosis")
    required = (
        "scripts/optimizer.py",
        "scripts/evaluator.py",
        "scripts/formatter.py",
        "workspace/MATH/workflows/round_1/graph.py",
        "workspace/MATH/workflows/round_1/prompt.py",
        "workspace/MATH/workflows/template/operator.json",
    )
    for relative in required:
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"AFlow checkout lacks {relative}")
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {"commit": commit, "optimizer_sha256": sha256_file(source / "scripts/optimizer.py")}


def local_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("AFlow diagnostic requires a local HTTP endpoint")
    if parsed.path.rstrip("/") != "/v1":
        raise ValueError("--endpoint must end in /v1")
    return endpoint.rstrip("/")


def probe_model(endpoint: str) -> dict:
    from masbench_answer_protocol import score_masbench

    opener = build_opener(ProxyHandler({}))
    with opener.open(endpoint + "/models", timeout=15) as response:
        models = json.load(response)
    served = {str(item.get("id", "")) for item in models.get("data", [])}
    if MODEL not in served:
        raise RuntimeError(f"local endpoint serves {sorted(served)!r}, expected {MODEL}")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": TASKS[0]["problem"]}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = Request(
        endpoint + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with opener.open(request, timeout=180) as response:
        completion = json.load(response)
    if completion.get("model") != MODEL:
        raise RuntimeError(f"completion used {completion.get('model')!r}, expected {MODEL!r}")
    answer = str(completion["choices"][0]["message"].get("content") or "")
    if not answer.strip():
        raise RuntimeError("local model returned empty content for the synthetic probe")
    return {
        "model": completion.get("model"),
        "response": answer,
        "score": score_masbench(answer, TASKS[0]["answer"]),
        "usage": completion.get("usage"),
    }


def safe_run_dir(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        requested = ROOT.parent / "tmp" / "rpas" / "diagnostics" / "aflow" / f"{stamp}_{os.getpid()}"
    run_dir = requested.expanduser().resolve()
    forbidden = [ROOT / "outputs", ROOT / "data" / "masbench", ROOT / "external_baselines"]
    if os.environ.get("RPAS_OUTPUT_ROOT"):
        forbidden.append(Path(os.environ["RPAS_OUTPUT_ROOT"]).expanduser().resolve())
    if any(run_dir == path.resolve() or path.resolve() in run_dir.parents for path in forbidden):
        raise ValueError("diagnostic output must be separate from formal output, benchmark data, and baseline checkouts")
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic run: {run_dir}")
    return run_dir


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_workflow(repo: Path) -> tuple[Path, Path]:
    from native_aime_formal import ensure_package_tree, patch_aflow_code_formatter, write_jsonl

    patch_aflow_code_formatter(repo)
    data_path = repo / "data" / "datasets" / "math_validate.jsonl"
    write_jsonl(
        data_path,
        [
            {"problem": task["problem"], "solution": f"FINAL ANSWER: {task['answer']}", "synthetic_id": task["id"]}
            for task in TASKS
        ],
    )
    write_jsonl(repo / "data" / "datasets" / "math_test.jsonl", [])
    relative_root = Path("workspace") / "rpas_aime_native" / "MATH"
    workflow_root = repo / relative_root / "workflows"
    workflow_root.mkdir(parents=True)
    shutil.copytree(repo / "workspace/MATH/workflows/round_1", workflow_root / "round_1")
    shutil.copytree(repo / "workspace/MATH/workflows/template", workflow_root / "template")
    # The released seed contains an old MATH mismatch log. Do not feed those
    # unrelated examples into this two-problem diagnostic's mutation prompt.
    (workflow_root / "round_1" / "log.json").unlink(missing_ok=True)
    ensure_package_tree(workflow_root, repo)
    for filename in ("graph.py", "prompt.py"):
        path = workflow_root / "round_1" / filename
        path.write_text(
            path.read_text(encoding="utf-8").replace("workspace.MATH", "workspace.rpas_aime_native.MATH"),
            encoding="utf-8",
        )
    return relative_root, workflow_root


def score_round(round_dir: Path) -> dict:
    from masbench_answer_protocol import score_masbench

    csv_paths = sorted(round_dir.glob("*.csv"))
    if len(csv_paths) != 1:
        raise RuntimeError(f"expected one native AFlow CSV in {round_dir}, found {len(csv_paths)}")
    with csv_paths[0].open(newline="", encoding="utf-8") as stream:
        outputs = list(csv.DictReader(stream))
    if len(outputs) != len(TASKS):
        raise RuntimeError(f"native AFlow scored {len(outputs)} samples; expected {len(TASKS)}")
    scored = []
    for task, output in zip(TASKS, outputs):
        if re.sub(r"\s+", " ", output["question"]).strip() != re.sub(r"\s+", " ", task["problem"]).strip():
            raise RuntimeError(f"native AFlow output order differs at {task['id']}")
        if output["expected_output"].strip() != f"FINAL ANSWER: {task['answer']}":
            raise RuntimeError(f"native AFlow gold differs at {task['id']}")
        prediction = output.get("prediction", "")
        scored.append({
            "synthetic_id": task["id"],
            "prediction": prediction,
            "native_score": output.get("score"),
            "shared_score": score_masbench(prediction, task["answer"]),
        })
    return {
        "csv": str(csv_paths[0]),
        "samples": scored,
        "correct": sum(bool(item["shared_score"]["correct"]) for item in scored),
        "parser_valid": sum(bool(item["shared_score"]["parser_valid"]) for item in scored),
    }


def run_native(source: Path, run_dir: Path, endpoint: str, max_tokens: int, timeout_seconds: int) -> dict:
    from native_aime_formal import (
        copy_tree, make_aflow_config, patch_aflow_graph_namespace,
        patch_aflow_runtime, phase_scope, seed_everything, validate_aflow_workflow_round,
    )
    from native_masbench_external import patch_native_math_scorer

    repo = run_dir / "aflow_repo"
    copy_tree(source, repo)
    relative_root, workflow_root = prepare_workflow(repo)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    telemetry = patch_aflow_runtime(max_tokens, concurrency=len(TASKS))
    patch_native_math_scorer("benchmarks.math")
    from scripts.optimizer import Optimizer
    from scripts.optimizer_utils.graph_utils import GraphUtils

    patch_aflow_graph_namespace(GraphUtils)
    seed_everything(0)
    config = make_aflow_config(endpoint)
    optimizer = Optimizer(
        dataset="MATH", question_type="math", opt_llm_config=config,
        exec_llm_config=config, operators=["Custom", "ScEnsemble", "Programmer"],
        sample=1, check_convergence=False, optimized_path=str(relative_root.parent),
        initial_round=1, max_rounds=1, validation_rounds=1,
    )
    original_check = optimizer.experience_utils.check_modification
    attempts = 0

    def bounded_check(processed_experience, modification, sample_round):
        nonlocal attempts
        attempts += 1
        if attempts > 4:
            raise RuntimeError("AFlow rejected more than four synthetic graph mutations")
        return original_check(processed_experience, modification, sample_round)

    optimizer.experience_utils.check_modification = bounded_check
    try:
        with phase_scope("search", telemetry["phase_wall_seconds"]):
            search_score = asyncio.run(asyncio.wait_for(optimizer._optimize_graph(), timeout=timeout_seconds))
    finally:
        optimizer.experience_utils.check_modification = original_check

    issues = validate_aflow_workflow_round(workflow_root, 2)
    if issues:
        raise RuntimeError(f"generated AFlow graph is invalid: {issues}")
    generated_graph = workflow_root / "round_2" / "graph.py"
    if not generated_graph.is_file():
        raise RuntimeError("native AFlow did not generate round_2/graph.py")
    rounds = {f"round_{number}": score_round(workflow_root / f"round_{number}") for number in (1, 2)}
    if rounds["round_2"]["parser_valid"] < 1:
        raise RuntimeError("generated graph produced no parseable synthetic answer")
    if telemetry["calls"] < 3 or telemetry["failed_calls"]:
        raise RuntimeError(f"AFlow model call telemetry is unhealthy: {telemetry['calls']} calls, {telemetry['failed_calls']} failed")
    guard = telemetry.pop("context_guard", None)
    return {
        "native_search_score": search_score,
        "graph_mutation_attempts": attempts,
        "generated_graph_sha256": sha256_file(generated_graph),
        "rounds": rounds,
        "telemetry": {**telemetry, "context_guard": guard.summary() if guard else None},
    }


def main() -> None:
    args = parse_args()
    source = args.aflow_source.expanduser().resolve()
    source_info = validate_source(source)
    if args.check_only:
        print(json.dumps({"status": "offline_structure_pass", "source": str(source), **source_info}, indent=2))
        return
    endpoint = local_endpoint(args.endpoint)
    if args.model_path is not None:
        model_path = args.model_path.expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"local model directory is missing: {model_path}")
        os.environ["RPAS_TOKENIZER_PATH"] = str(model_path)
    elif not os.environ.get("RPAS_TOKENIZER_PATH"):
        raise ValueError("set --model-path or RPAS_TOKENIZER_PATH for the local Qwen tokenizer")
    run_dir = safe_run_dir(args.run_dir)
    probe = probe_model(endpoint)
    run_dir.mkdir(parents=True)
    result = {
        "status": "running",
        "kind": "synthetic_aflow_diagnostic",
        "source": str(source),
        "source_info": source_info,
        "endpoint": endpoint,
        "max_tokens": args.max_tokens,
        "seed": 0,
        "sample_count": len(TASKS),
        "direct_probe": probe,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = run_dir / "diagnostic.json"
    write_json(report_path, result)
    try:
        result["native_aflow"] = run_native(source, run_dir, endpoint, args.max_tokens, args.timeout_seconds)
        result["status"] = "passed"
        print(f"AFLOW_DIAGNOSTIC_PASS {report_path}")
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"AFLOW_DIAGNOSTIC_FAIL {report_path}: {result['error']}", file=sys.stderr)
        raise
    finally:
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, result)


if __name__ == "__main__":
    main()
