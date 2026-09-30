#!/usr/bin/env python3
"""Run a disposable native MaAS controller update on synthetic MATH rows.

This diagnostic never imports MASBench rows or formal outputs. It requires a
local Qwen endpoint, local tokenizer and MiniLM weights, and the exact pinned
MaAS checkout. All native execution occurs in a temporary copy; only its
logs, training artifacts and a controller-update audit are retained.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback
from urllib.parse import urlparse
from urllib.request import build_opener, ProxyHandler


WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENTS = WORKSPACE / "experiments"
MODEL = "Qwen/Qwen3.5-9B"
MAAS_COMMIT = "987f3c1bc9a96e844fe090db3791446e3ef0f5c7"
RELATIVE_ROOT = Path("workspace/rpas_synthetic_maas/MATH")
SYNTHETIC_ROWS = (
    {"problem": "Compute 2 + 3. Give the final answer in \\boxed{...}.",
     "solution": "\\boxed{5}", "diagnostic_id": "synthetic_addition"},
    {"problem": "Compute 7 - 4. Give the final answer in \\boxed{...}.",
     "solution": "\\boxed{3}", "diagnostic_id": "synthetic_subtraction"},
)


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def validate_output_root(path: Path, source: Path) -> Path:
    """Keep disposable diagnostics clear of benchmark data and formal output."""
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {resolved}")
    forbidden = [WORKSPACE / "outputs", WORKSPACE / "data", source.resolve()]
    for environment_name in ("RPAS_OUTPUT_ROOT", "RPAS_DATA_DIR"):
        configured = os.environ.get(environment_name)
        if configured:
            forbidden.append(Path(configured).expanduser().resolve())
    if any(_inside(resolved, directory.resolve()) for directory in forbidden):
        raise ValueError("diagnostic output must be outside formal outputs, data and MaAS source")
    if _inside(WORKSPACE, resolved):
        raise ValueError("diagnostic output cannot be an ancestor of the workspace")
    return resolved


def validate_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint.rstrip("/"))
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path != "/v1"):
        raise ValueError("endpoint must be a loopback HTTP URL ending in /v1")
    return endpoint.rstrip("/")


def verify_model(endpoint: str) -> None:
    # Bypass a workstation proxy when checking the loopback service.
    opener = build_opener(ProxyHandler({}))
    with opener.open(endpoint + "/models", timeout=10) as response:
        payload = json.load(response)
    served = {str(item.get("id", "")) for item in payload.get("data", [])}
    if MODEL not in served:
        raise RuntimeError(f"local endpoint serves {sorted(served)!r}; expected {MODEL}")


def verify_source(source: Path) -> Path:
    source = source.expanduser().resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(source)
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    top_level = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if Path(top_level).resolve() != source:
        raise RuntimeError(f"MaAS source must be the checkout root: {source}")
    if head != MAAS_COMMIT:
        raise RuntimeError(f"MaAS checkout is {head}; required {MAAS_COMMIT}")
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=normal"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("MaAS checkout has local changes; use the clean pinned source")
    return source


def write_synthetic_jsonl(path: Path, rows: tuple[dict, ...] = SYNTHETIC_ROWS) -> None:
    if not (1 <= len(rows) <= 2):
        raise ValueError("MaAS diagnostic permits one or two synthetic rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare_isolated_repo(
    source: Path, repo: Path, endpoint: str, embedding_path: Path, max_tokens: int,
) -> dict:
    """Stage only fixed MaAS code and embedded synthetic rows in a fresh copy."""
    source = verify_source(source)
    if repo.exists():
        raise FileExistsError(repo)
    # Copy tracked files only. This excludes any ignored/untracked benchmark
    # material that might happen to be present beside the pinned checkout.
    tracked = subprocess.run(
        ["git", "-C", str(source), "ls-files", "-z"],
        check=True, capture_output=True,
    ).stdout.split(b"\0")
    repo.mkdir(parents=True)
    for encoded in tracked:
        if not encoded:
            continue
        relative = Path(os.fsdecode(encoded))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe path in MaAS checkout: {relative}")
        original = source / relative
        if original.is_symlink():
            raise ValueError(f"refusing symlink in MaAS checkout: {relative}")
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    sys.path.insert(0, str(EXPERIMENTS))
    sys.path.insert(0, str(WORKSPACE))
    from native_aime_formal import (
        ensure_package_tree,
        make_maas_config,
        patch_maas_embedding_path,
        patch_maas_optional_backoff,
        patch_maas_optional_provider_import,
        patch_maas_optional_tools,
        patch_maas_provider_surface,
    )
    from maas_context_guard import patch_template_directory

    patch_maas_optional_provider_import(repo)
    patch_maas_provider_surface(repo)
    patch_maas_optional_backoff(repo)
    patch_maas_optional_tools(repo)
    previous_embedding = os.environ.get("RPAS_MAAS_EMBEDDING_PATH")
    os.environ["RPAS_MAAS_EMBEDDING_PATH"] = str(embedding_path)
    try:
        patch_maas_embedding_path(repo)
    finally:
        if previous_embedding is None:
            os.environ.pop("RPAS_MAAS_EMBEDDING_PATH", None)
        else:
            os.environ["RPAS_MAAS_EMBEDDING_PATH"] = previous_embedding
    make_maas_config(repo, endpoint, max_tokens)

    native_root = repo / "maas/ext/maas/scripts/optimized/MATH"
    repairs = [
        patch_template_directory(native_root / split / "template", concise=False)
        for split in ("train", "test")
    ]
    destination = repo / RELATIVE_ROOT
    for split in ("train", "test"):
        shutil.copytree(native_root / split, destination / split, dirs_exist_ok=True)
    ensure_package_tree(destination, repo)

    data_dir = repo / "maas/ext/maas/data"
    write_synthetic_jsonl(data_dir / "math_train.jsonl")
    (data_dir / "math_test.jsonl").write_text("", encoding="utf-8")
    return {
        "source_commit": MAAS_COMMIT,
        "native_root": str(RELATIVE_ROOT),
        "train_row_count": len(SYNTHETIC_ROWS),
        "test_row_count": 0,
        "train_data_sha256": _sha256(data_dir / "math_train.jsonl"),
        "template_repairs": repairs,
    }


def _worker(spec_path: Path) -> None:
    """Run the real native MaAS optimizer and audit its Adam updates."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    repo = Path(spec["repo"])
    diagnostic_root = Path(spec["diagnostic_root"])
    os.chdir(repo)
    sys.path.insert(0, str(WORKSPACE))
    sys.path.insert(0, str(EXPERIMENTS))
    sys.path.insert(0, str(repo))
    os.environ["METAGPT_PROJECT_ROOT"] = str(repo)

    events: list[dict] = []
    native_repetitions: list[dict] = []
    telemetry: dict = {}
    try:
        import torch
        from native_aime_formal import patch_maas_runtime, seed_everything
        from maas.ext.maas.benchmark.benchmark import BaseBenchmark

        # The formal compatibility layer captures this method and requests
        # four repetitions. Observe the actual native method it invokes.
        native_train = BaseBenchmark.evaluate_all_problems

        async def observed_native_train(
            self, data, graph, max_concurrent_tasks=30, repetitions=4, is_textgrad=False,
        ):
            native_repetitions.append({"rows": len(data), "repetitions": repetitions})
            return await native_train(
                self, data, graph, max_concurrent_tasks=max_concurrent_tasks,
                repetitions=repetitions, is_textgrad=is_textgrad,
            )

        BaseBenchmark.evaluate_all_problems = observed_native_train
        telemetry = patch_maas_runtime(concurrency=2, max_tokens=int(spec["max_tokens"]))

        from maas.configs.models_config import ModelsConfig
        from maas.ext.maas.benchmark.experiment_configs import EXPERIMENT_CONFIGS
        from maas.ext.maas.scripts.optimizer import Optimizer

        config = EXPERIMENT_CONFIGS["MATH"]
        llm_config = ModelsConfig.default().get("qwen35_9b")
        if llm_config is None:
            raise RuntimeError("MaAS qwen35_9b model config is unavailable")
        seed_everything(0)
        optimizer = Optimizer(
            dataset=config.dataset, question_type=config.question_type,
            opt_llm_config=llm_config, exec_llm_config=llm_config,
            operators=config.operators, optimized_path=str(RELATIVE_ROOT.parent),
            sample=1, round=1, batch_size=2, lr=0.01, is_textgrad=False,
        )
        starting_params = {
            name: parameter.detach().clone()
            for name, parameter in optimizer.controller.named_parameters()
        }
        original_step = optimizer.optimizer.step
        step_path = diagnostic_root / "optimizer_steps.jsonl"

        def observed_step(*args, **kwargs):
            parameters = [
                parameter for group in optimizer.optimizer.param_groups
                for parameter in group["params"]
            ]
            before = [parameter.detach().clone() for parameter in parameters]
            with torch.no_grad():
                gradient_max = max(
                    (float(parameter.grad.detach().abs().max().item())
                     for parameter in parameters if parameter.grad is not None),
                    default=0.0,
                )
            result = original_step(*args, **kwargs)
            with torch.no_grad():
                deltas = [float((parameter.detach() - earlier).abs().max().item())
                          for parameter, earlier in zip(parameters, before)]
            event = {
                "step": len(events) + 1,
                "gradient_max_abs": gradient_max,
                "changed_parameter_tensors": sum(delta > 0 for delta in deltas),
                "parameter_max_abs_delta": max(deltas, default=0.0),
            }
            events.append(event)
            with step_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            return result

        optimizer.optimizer.step = observed_step
        # This must invoke the released Graph path, including its controller
        # forward pass, scoring, backward pass, and native checkpoint write.
        optimizer.optimize("Graph")

        checkpoint = repo / RELATIVE_ROOT / "train/round_1/MATH_controller_sample1.pth"
        if not checkpoint.is_file():
            raise RuntimeError("native Optimizer.optimize('Graph') did not save its checkpoint")
        if native_repetitions != [{"rows": 2, "repetitions": 4}]:
            raise RuntimeError(f"expected four native training repetitions, observed {native_repetitions!r}")
        repeat_audit = telemetry.get("training_repeat_audit", [])
        if (len(repeat_audit) != 1 or repeat_audit[0].get("completed_repetitions") != 4
                or repeat_audit[0].get("observed_result_rows") != 8):
            raise RuntimeError(f"MaAS four-repetition completion audit failed: {repeat_audit!r}")
        if not events or not any(event["parameter_max_abs_delta"] > 0 for event in events):
            raise RuntimeError("native Adam step did not change any controller parameter")
        if telemetry.get("calls", 0) < 1:
            raise RuntimeError("native graph made no local model requests")

        with torch.no_grad():
            final_deltas = {
                name: float((parameter.detach() - starting_params[name]).abs().max().item())
                for name, parameter in optimizer.controller.named_parameters()
            }
        if max(final_deltas.values(), default=0.0) <= 0:
            raise RuntimeError("final controller parameters equal their initial values")
        saved_state = torch.load(checkpoint, map_location="cpu")
        checkpoint_deltas = {
            name: float((saved_state[name] - starting_params[name].cpu()).abs().max().item())
            for name in starting_params
        }
        if max(checkpoint_deltas.values(), default=0.0) <= 0:
            raise RuntimeError("saved controller checkpoint equals its initial parameters")

        results_path = repo / RELATIVE_ROOT / "train/results.json"
        if not results_path.is_file():
            raise RuntimeError("native Optimizer.optimize('Graph') did not save train/results.json")
        native_results = json.loads(results_path.read_text(encoding="utf-8"))
        if len(native_results) != 1 or native_results[0].get("round") != 1:
            raise RuntimeError(f"unexpected native MaAS train results: {native_results!r}")
        guard = telemetry.pop("_context_guard", None)
        _json_dump(diagnostic_root / "worker_result.json", {
            "status": "passed",
            "entrypoint": "MaAS native Optimizer.optimize('Graph')",
            "controller_sample": 1,
            "native_training_repetitions": native_repetitions,
            "training_repeat_audit": repeat_audit,
            "optimizer_steps": events,
            "controller_changed_parameter_tensors": sum(delta > 0 for delta in final_deltas.values()),
            "controller_max_abs_delta": max(final_deltas.values(), default=0.0),
            "checkpoint_max_abs_delta_from_initial": max(checkpoint_deltas.values(), default=0.0),
            "checkpoint_relative_path": str(checkpoint.relative_to(repo)),
            "checkpoint_sha256": _sha256(checkpoint),
            "native_train_results": native_results,
            "local_model_calls": telemetry.get("calls", 0),
            "failed_model_calls": telemetry.get("failed_calls", 0),
            "prompt_tokens": telemetry.get("prompt_tokens", 0),
            "completion_tokens": telemetry.get("completion_tokens", 0),
            "context_guard": guard.summary() if guard is not None else None,
        })
    except BaseException as exc:
        _json_dump(diagnostic_root / "worker_result.json", {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "native_training_repetitions": native_repetitions,
            "optimizer_steps": events,
            "local_model_calls": telemetry.get("calls", 0),
        })
        raise


def _run_worker(spec_path: Path, environment: dict[str, str], log, timeout_seconds: int) -> int:
    """Bound the worker and its native code-execution child processes."""
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)],
        env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout_seconds)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise


def run_diagnostic(args: argparse.Namespace) -> dict:
    source = verify_source(args.source)
    diagnostic_root = validate_output_root(args.output_root, source)
    endpoint = validate_endpoint(args.endpoint)
    tokenizer_path = args.tokenizer_path.expanduser().resolve(strict=True)
    embedding_path = args.embedding_path.expanduser().resolve(strict=True)
    if not tokenizer_path.is_dir() or not embedding_path.is_dir():
        raise ValueError("tokenizer and embedding paths must be local directories")
    if not (64 <= args.max_tokens <= 2048):
        raise ValueError("diagnostic max_tokens must be within 64..2048")
    if not (60 <= args.timeout_seconds <= 7200):
        raise ValueError("diagnostic timeout must be within 60..7200 seconds")
    verify_model(endpoint)

    diagnostic_root.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_root.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    status = "failed"
    error: str | None = None
    staging: dict | None = None
    exit_code: int | None = None
    with TemporaryDirectory(prefix="maas_source_", dir=diagnostic_root) as temporary:
        repo = Path(temporary) / "repo"
        try:
            staging = prepare_isolated_repo(source, repo, endpoint, embedding_path, args.max_tokens)
            shutil.copy2(repo / "maas/ext/maas/data/math_train.jsonl", diagnostic_root / "synthetic_train.jsonl")
            spec_path = diagnostic_root / "worker_spec.json"
            _json_dump(spec_path, {
                "repo": str(repo), "diagnostic_root": str(diagnostic_root),
                "max_tokens": args.max_tokens,
            })
            worker_env = os.environ.copy()
            worker_env.update({
                "RPAS_MAAS_EMBEDDING_PATH": str(embedding_path),
                "RPAS_TOKENIZER_PATH": str(tokenizer_path),
                "RPAS_MAAS_SAMPLE_TIMEOUT_S": "120",
                "RPAS_MAAS_SAMPLE_RETRIES": "1",
                "RPAS_MAAS_CODE_TIMEOUT_S": "30",
                "RPAS_MAAS_REQUEST_TIMEOUT_S": "120",
                "METAGPT_PROJECT_ROOT": str(repo),
                "NO_PROXY": "127.0.0.1,localhost,::1",
                "no_proxy": "127.0.0.1,localhost,::1",
            })
            with (diagnostic_root / "run.log").open("w", encoding="utf-8") as log:
                try:
                    exit_code = _run_worker(spec_path, worker_env, log, args.timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    error = f"worker exceeded {args.timeout_seconds} seconds: {exc}"
            worker_result_path = diagnostic_root / "worker_result.json"
            if worker_result_path.exists():
                worker_result = json.loads(worker_result_path.read_text(encoding="utf-8"))
                if worker_result.get("status") == "passed" and exit_code == 0:
                    status = "passed"
                elif error is None:
                    error = worker_result.get("error") or f"worker exit code {exit_code}"
            elif error is None:
                error = f"worker exit code {exit_code}; inspect run.log"
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            native_outputs = repo / RELATIVE_ROOT
            if native_outputs.exists():
                shutil.copytree(native_outputs, diagnostic_root / "native_artifacts", dirs_exist_ok=True)

    report = {
        "status": status,
        "synthetic_only": True,
        "diagnostic_root": str(diagnostic_root),
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "endpoint": endpoint,
        "maas_source": str(source),
        "maas_commit": MAAS_COMMIT,
        "local_tokenizer_path": str(tokenizer_path),
        "local_embedding_path": str(embedding_path),
        "max_tokens": args.max_tokens,
        "concurrency": 2,
        "controller_sample": 1,
        "expected_native_training_repetitions": 4,
        "synthetic_row_count": len(SYNTHETIC_ROWS),
        "worker_exit_code": exit_code,
        "staging": staging,
        "error": error,
    }
    _json_dump(diagnostic_root / "report.json", report)
    return report


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-spec":
        _worker(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=WORKSPACE / "external_baselines/MaAS")
    parser.add_argument("--endpoint", required=True, help="local Qwen OpenAI-compatible /v1 endpoint")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="local Qwen3.5-9B model directory")
    parser.add_argument("--embedding-path", type=Path, required=True, help="local all-MiniLM-L6-v2 directory")
    parser.add_argument("--output-root", type=Path, required=True, help="new diagnostic directory outside outputs/")
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    report = run_diagnostic(args)
    print(json.dumps({"status": report["status"], "report": str(Path(report["diagnostic_root"]) / "report.json"),
                      "error": report["error"]}, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
