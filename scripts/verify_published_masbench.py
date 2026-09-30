#!/usr/bin/env python3
"""Verify published MASBench evidence offline without recreating the host gate.

Archive members are read in memory, never extracted. Prediction scores are
recomputed with the shared parser against the repository's frozen data. Token
counts are checked for internal consistency, not retokenized or rerun.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
import tarfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from masbench_answer_protocol import VERSION, SEPARATOR, extract_masbench_answer, score_masbench
from masbench_data_protocol import canonical_gold

AXES = ("breadth", "depth", "horizon", "parallel", "robustness")
METHODS = ("aflow", "maas")
METADATA = ("native_result.json", "run_manifest.json", "selection_frozen.json",
            "split_manifest.json", "quality_gate.json")
USAGE_KEYS = ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "failed_calls")
METRICS = ("accuracy", "tokens_per_example", "component_accuracy", "mean_score_0_1_2")
ROBUSTNESS_RULE = (
    "Robustness-axis rule: compute values defined by algebraic relations in Z_23, but copy every "
    "explicitly requested injected fact or magic number exactly as written; these raw retrieval values "
    "are not reduced modulo 23. Preserve the complete requested answer order.\n"
)


class VerificationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(value: str) -> str:
    require(isinstance(value, str) and bool(value), "empty or non-string evidence path")
    path = PurePosixPath(value)
    require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts and "\\" not in value
            and str(path) == value and ":" not in path.parts[0],
            f"unsafe or noncanonical evidence path: {value!r}")
    return value


def equal_number(actual: Any, expected: int | float, label: str) -> None:
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8),
            f"{label} differs from recomputed/recorded value")


def verify_direct_files(bundle: Path) -> int:
    require(bundle.is_dir() and not bundle.is_symlink(), "bundle directory is missing or a symlink")
    sums = bundle / "SHA256SUMS"
    require(sums.is_file() and not sums.is_symlink(), "bundle SHA256SUMS is missing")
    indexed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        require(match is not None, "invalid SHA256SUMS line")
        expected, name = match.groups()
        safe_name(name)
        require(name != "SHA256SUMS" and name not in indexed, f"duplicate checksum path: {name}")
        indexed[name] = expected
    actual: set[str] = set()
    for path in bundle.rglob("*"):
        require(not path.is_symlink(), f"published bundle contains a symlink: {path.name}")
        require(path.is_file() or path.is_dir(), f"published bundle contains a nonregular file: {path.name}")
        if path.is_file() and path != sums:
            name = safe_name(path.relative_to(bundle).as_posix())
            actual.add(name)
            require(indexed.get(name) == digest(path.read_bytes()), f"direct file checksum mismatch: {name}")
    require(actual == set(indexed), "SHA256SUMS does not cover exactly all published direct files")
    return len(actual)


class Evidence:
    def __init__(self, bundle: Path):
        index = json.loads((bundle / "file_index.json").read_text(encoding="utf-8"))
        require(index.get("schema_version") == "masbench_evidence_index_v1", "unexpected evidence-index schema")
        self.entries: dict[str, dict[str, Any]] = {}
        self.sources: dict[str, str] = {}
        self.content: dict[str, bytes] = {}
        for entry in index["files"]:
            name = safe_name(entry["path"])
            source = entry["source_path"]
            require(isinstance(source, str) and source.startswith("/"), "evidence source path is not absolute")
            require(name not in self.entries, f"duplicate index member: {name}")
            require(type(entry.get("size")) is int and entry["size"] >= 0
                    and re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))) is not None,
                    f"invalid index size/hash: {name}")
            if source in self.sources:
                previous = self.entries[self.sources[source]]
                require((previous["size"], previous["sha256"]) == (entry["size"], entry["sha256"]),
                        "conflicting evidence copies for one source path")
            self.entries[name] = entry
            self.sources.setdefault(source, name)
        require(bool(self.entries), "empty evidence index")
        seen: set[str] = set()
        with tarfile.open(bundle / "evidence.tar.gz", mode="r|gz") as archive:
            for member in archive:
                name = safe_name(member.name)
                require(member.isfile() and not member.issym() and not member.islnk(),
                        f"archive member is not a regular file: {name}")
                require(name not in seen and name in self.entries, f"duplicate/unindexed archive member: {name}")
                seen.add(name)
                entry = self.entries[name]
                require(member.size == entry["size"], f"archive member size mismatch: {name}")
                stream = archive.extractfile(member)
                require(stream is not None, f"archive member cannot be read: {name}")
                data = stream.read()
                require(len(data) == entry["size"] and digest(data) == entry["sha256"],
                        f"archive member checksum mismatch: {name}")
                if PurePosixPath(name).suffix in (".json", ".csv"):
                    self.content[name] = data
        require(seen == set(self.entries), "archive/index file sets differ")

    def name_for_source(self, source: str) -> str:
        require(source in self.sources, f"original source path absent from evidence index: {source}")
        return self.sources[source]

    def json(self, name: str) -> Any:
        require(name in self.content, f"required JSON evidence absent: {name}")
        return json.loads(self.content[name].decode("utf-8"))


def instruction_text() -> str:
    def constant_text(node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(constant_text(value) for value in node.values)
        if (isinstance(node, ast.FormattedValue) and isinstance(node.value, ast.Name)
                and node.value.id == "MASBENCH_ANSWER_SEPARATOR" and node.format_spec is None):
            return SEPARATOR
        raise VerificationError("unsupported shared MASBench instruction expression")

    tree = ast.parse((ROOT / "experiments/phase2_wan_agent_search.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "MASBENCH_INSTRUCTION" for target in node.targets):
            return constant_text(node.value)
    raise VerificationError("shared MASBench instruction constant is missing")


def failure_reason(prediction: str, parser_valid: bool) -> str | None:
    """Recognize standalone native failures, never substring mentions in answers."""
    value = prediction.strip()
    if not value:
        return "empty_prediction"
    lower = value.lower()
    markers = ("aflow sample timeout", "maas sample failed", "no code generated",
               "maximum retries reached", "traceback (most recent call last)",
               "code execution timed out", "code execution exited with code",
               "code execution returned invalid output")
    if any(lower == marker or lower.startswith(marker + ":")
           or lower.startswith(marker + " ") for marker in markers):
        return "runtime_failure"
    if re.fullmatch(r"['\"][^\n]+['\"] object has no attribute ['\"][^\n]+['\"]", value):
        return "attribute_error"
    if not parser_valid and (re.match(r"^(?:[A-Za-z]+Error|[A-Za-z]+Exception):[^\n]*$", value)
                            or re.match(r"^(?:execution error|unknown error|error):[^\n]*$", lower)):
        return "runtime_failure"
    return None


def score_csv(evidence: Evidence, source: str, axis: str, split: str,
              data: list[dict[str, Any]], instruction: str) -> tuple[dict[str, Any], dict[str, Any]]:
    name = evidence.name_for_source(source)
    require(name in evidence.content, f"CSV evidence is missing: {name}")
    reader = csv.DictReader(io.StringIO(evidence.content[name].decode("utf-8-sig"), newline=""))
    require({"question", "prediction", "expected_output", "score"}.issubset(reader.fieldnames or []),
            f"CSV schema is incomplete: {name}")
    rows = list(reader)
    expected_n = 24 if split == "select" else 60
    require(len(rows) == len(data) == expected_n, f"CSV row count mismatch: {name}")
    scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (row, original) in enumerate(zip(rows, data), 1):
        require(None not in row and all(isinstance(row.get(key), str)
                for key in ("question", "prediction", "expected_output", "score")),
                f"malformed CSV row {index}: {name}")
        gold = canonical_gold(original["answer"], allow_raw=(axis == "robustness"))
        question = re.sub(r"\s+", " ", row["question"] or "").strip()
        wanted_question = re.sub(r"\s+", " ", f"{instruction.rstrip()}\n\n{original['input']}").strip()
        require(question == wanted_question, f"CSV frozen-data row/order mismatch at {name} row {index}")
        require(extract_masbench_answer(row["expected_output"], len(gold.split(SEPARATOR))) == gold,
                f"CSV gold differs from frozen data at {name} row {index}")
        scored = score_masbench(row["prediction"], gold)
        equal_number(float(row["score"]), scored["score_0_1_2"], f"CSV ordinal score {name} row {index}")
        scores.append(scored)
        reason = failure_reason(row["prediction"], scored["parser_valid"])
        if reason:
            failures.append({"row": index, "kind": reason})
    correct = sum(item["correct"] for item in scores)
    component_correct = sum(item["component_correct"] for item in scores)
    component_total = sum(item["component_total"] for item in scores)
    metrics = {
        "score": correct / expected_n, "accuracy": correct / expected_n,
        "correct": correct, "num_examples": expected_n,
        "valid_answer_rate": sum(item["parser_valid"] for item in scores) / expected_n,
        "component_accuracy": component_correct / component_total,
        "component_correct": component_correct, "component_total": component_total,
        "mean_score_0_1_2": sum(item["score_0_1_2"] for item in scores) / expected_n,
        "answer_parser": VERSION,
    }
    audit = {"csv": name, "rows": expected_n, "failures": failures,
             "failure_fraction": len(failures) / expected_n}
    return metrics, audit


def compare_metrics(recorded: dict[str, Any], computed: dict[str, Any], label: str) -> None:
    for key, value in computed.items():
        if isinstance(value, str):
            require(recorded.get(key) == value, f"{label} {key} mismatch")
        else:
            equal_number(recorded.get(key), value, f"{label} {key}")


def check_usage(usage: dict[str, Any], label: str) -> None:
    require(isinstance(usage, dict), f"{label} usage is missing")
    for key in USAGE_KEYS:
        require(type(usage.get(key)) is int and usage[key] >= 0, f"{label} invalid {key}")
    require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
            f"{label} prompt + completion token sum mismatch")
    require(usage["failed_calls"] <= usage["calls"], f"{label} failed calls exceed all calls")


def verify_run(bundle: Path, evidence: Evidence, method: str, axis: str,
               instruction: str) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    prefix = f"{method}/{axis}/seed_0"
    label = f"{method}/{axis}"
    metadata: dict[str, Any] = {}
    for filename in METADATA:
        name = f"{prefix}/{filename}"
        direct = bundle / "runs" / prefix / filename
        require(name in evidence.content and direct.read_bytes() == evidence.content[name],
                f"direct/archive metadata differs: {name}")
        metadata[filename] = evidence.json(name)
    result, manifest, lock, gate = (metadata[name] for name in (
        "native_result.json", "run_manifest.json", "selection_frozen.json", "quality_gate.json"))
    for record in (manifest, lock, gate):
        require((record.get("method"), record.get("axis"), record.get("seed")) == (method, axis, 0),
                f"{label} metadata identity mismatch")
    require(manifest.get("status") == "complete" and gate.get("passed") is True,
            f"{label} does not have a completed, stored passing host gate")
    require(result.get("method") == method and result.get("run_id") == manifest.get("run_id") == gate.get("run_id"),
            f"{label} run identity mismatch")
    result_hash = evidence.entries[f"{prefix}/native_result.json"]["sha256"]
    lock_hash = evidence.entries[f"{prefix}/selection_frozen.json"]["sha256"]
    require(manifest.get("native_result_sha256") == result_hash, f"{label} manifest result hash mismatch")
    require(manifest.get("selection_lock_sha256") == lock_hash
            and result.get("selection_lock") == {"path": "selection_frozen.json", "sha256": lock_hash}
            and manifest.get("d_test_access", {}).get("selection_lock_sha256") == lock_hash,
            f"{label} selection lock linkage mismatch")
    require(lock.get("status") == "frozen_before_test_access" and lock.get("d_test_opened") is False
            and manifest.get("d_test_opened") is True,
            f"{label} recorded selection/test access order is invalid")
    require(datetime.fromisoformat(lock["frozen_at_utc"]) <
            datetime.fromisoformat(manifest["d_test_access"]["opened_at_utc"]),
            f"{label} recorded freeze occurs after test access")
    parser_hash = digest((ROOT / "experiments/masbench_answer_protocol.py").read_bytes())
    require(manifest.get("answer_parser_sha256") == parser_hash, f"{label} shared parser hash mismatch")
    require(manifest.get("task_instruction_sha256") == digest(instruction.encode("utf-8")),
            f"{label} shared task instruction hash mismatch")
    data: dict[str, list[dict[str, Any]]] = {}
    for relative, wanted_hash in manifest["data_sha256"].items():
        path = ROOT / "data/masbench" / safe_name(relative)
        require(path.is_file() and digest(path.read_bytes()) == wanted_hash,
                f"{label} frozen repository dataset hash mismatch: {relative}")
    for split, size in (("select", 24), ("test", 60)):
        path = ROOT / "data/masbench" / axis / f"{split}_{size}.jsonl"
        data[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        require(len(data[split]) == size and [row["id"] for row in data[split]] == manifest["split_ids"][split],
                f"{label} frozen {split} IDs/count mismatch")
    telemetry = result["telemetry"]
    check_usage(telemetry, label)
    phases = telemetry["phase_totals"]
    for phase, usage in phases.items():
        check_usage(usage, f"{label} phase {phase}")
    for key in USAGE_KEYS:
        require(telemetry[key] == sum(phase[key] for phase in phases.values()),
                f"{label} all-phase {key} sum mismatch")
    selection_audits: list[dict[str, Any]] = []
    selection_count = 0
    if method == "aflow":
        selection = result["selection_rows"]
        require(lock["selection_details"]["selection_rows"] == selection, f"{label} frozen selection rows differ")
        selected = {"Q": result["selected_round"], "E": result["selected_efficiency_round"]}
        selected_rounds = {row["round"] for row in selected.values()}
        valid_rounds = {row["round"] for row in selection if row.get("status") == "evaluated"}
        require(selected_rounds <= valid_rounds, f"{label} selected round was not recorded as evaluated")
        for point, row in selected.items():
            require(lock["selected"][point] == {"round": row["round"], "candidate_id": row["candidate_id"]},
                    f"{label} {point} selected round disagrees with freeze")
        for row in selection:
            source = row.get("output_audit", {}).get("csv")
            if not source:
                continue
            computed, audit = score_csv(evidence, source, axis, "select", data["select"], instruction)
            compare_metrics(row["shared_metrics"], computed, f"{label} D_select round {row['round']}")
            equal_number(row["score"], computed["score"], f"{label} selection objective")
            equal_number(row["native_score"], computed["mean_score_0_1_2"], f"{label} native selection objective")
            equal_number(row["avg_cost"], row["total_tokens"] / 24, f"{label} selection tokens/example")
            require(row["total_cost"] == row["total_tokens"], f"{label} selection token total mismatch")
            selection_count += 1
            audit.update({"round": row["round"], "selected": row["round"] in selected_rounds})
            selection_audits.append(audit)
            if audit["selected"]:
                require(audit["failure_fraction"] <= 0.05, f"{label} selected D_select runtime failures exceed 5%")
        for target, key in (("calls", "total_calls"), ("total_tokens", "total_tokens")):
            require(phases["select"][target] == sum(row.get(key, 0) for row in selection),
                    f"{label} selection {target} sum mismatch")
        points = result["test"]
    else:
        selection = result["selection"]
        computed, audit = score_csv(evidence, selection["output_audit"]["csv"], axis, "select", data["select"], instruction)
        compare_metrics(selection, computed, f"{label} D_select")
        equal_number(selection["native_score"], computed["mean_score_0_1_2"], f"{label} native selection objective")
        compare_metrics(lock["selection_details"]["selection"], computed, f"{label} locked D_select")
        require(audit["failure_fraction"] <= 0.05, f"{label} selected D_select runtime failures exceed 5%")
        audit["selected"] = True
        selection_audits.append(audit)
        selection_count = 1
        check_usage(selection["resource_usage"], f"{label} D_select")
        require(selection["resource_usage"] == phases["select"], f"{label} D_select usage differs from telemetry")
        checkpoint = result["controller_checkpoint"]
        name = evidence.name_for_source(checkpoint)
        checkpoint_hash = evidence.entries[name]["sha256"]
        require(checkpoint_hash == result["controller_checkpoint_sha256"]
                == lock["selected"]["Q"]["controller_checkpoint_sha256"]
                == result["controller_training"]["checkpoint_file_sha256"]
                and lock["selected"]["Q"]["controller_checkpoint"] == checkpoint,
                f"{label} controller checkpoint hash/linkage mismatch")
        points = {point: result["test"][axis][point] for point in ("Q", "E")}
    computed_points: dict[str, dict[str, float]] = {}
    final_audits: dict[str, dict[str, Any]] = {}
    for point_name in ("Q", "E"):
        point = points[point_name]
        computed, audit = score_csv(evidence, point["output_audit"]["csv"], axis, "test", data["test"], instruction)
        compare_metrics(point, computed, f"{label} D_test {point_name}")
        equal_number(point["native_score"], computed["mean_score_0_1_2"], f"{label} native D_test objective")
        require(audit["failure_fraction"] <= 0.05, f"{label} {point_name} D_test runtime failures exceed 5%")
        usage = point["resource_usage"]
        check_usage(usage, f"{label} D_test {point_name}")
        for key in USAGE_KEYS:
            require(point.get(key) == usage[key], f"{label} {point_name} point/usage {key} mismatch")
        equal_number(point["avg_cost"], usage["total_tokens"] / 60, f"{label} D_test tokens/example")
        require(point["cost_unit"] == "model_tokens" and point["total_cost"] == usage["total_tokens"],
                f"{label} D_test token cost mismatch")
        if method == "aflow":
            require(point["selected_round"] == selected[point_name]["round"], f"{label} D_test round mismatch")
            actual_point = "Q" if point.get("same_candidate_as") == "Q" else point_name
            phase_key = f"test:{axis}:{actual_point}"
        else:
            require(point["selected_controller"] == checkpoint, f"{label} D_test controller mismatch")
            phase_key = f"test:{axis}"
        require(usage == phases[phase_key], f"{label} {point_name} usage differs from phase telemetry")
        computed_points[point_name] = {
            "accuracy": computed["accuracy"], "tokens_per_example": usage["total_tokens"] / 60,
            "component_accuracy": computed["component_accuracy"], "mean_score_0_1_2": computed["mean_score_0_1_2"],
        }
        for metric in METRICS:
            equal_number(gate["point_summary"][point_name][metric], computed_points[point_name][metric],
                         f"{label} stored gate {point_name}/{metric}")
        final_audits[point_name] = audit
    return {"run_id": manifest["run_id"], "selection_csvs_recomputed": selection_count,
            "final_prediction_audit": final_audits, "selection_prediction_audit": selection_audits,
            "recorded_total_tokens": telemetry["total_tokens"]}, computed_points


def verify_summary(bundle: Path, runs: dict[str, dict[str, dict[str, dict[str, float]]]],
                   run_ids: list[str]) -> None:
    summary = json.loads((bundle / "seed0_summary/aggregate.json").read_text(encoding="utf-8"))
    require(summary.get("schema_version") == "masbench_single_seed_summary_v1" and summary.get("seed") == 0
            and summary.get("axes") == list(AXES) and summary.get("run_ids") == run_ids,
            "aggregate JSON identity/axis/run-ID mismatch")
    expected_rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for method in METHODS:
        for axis in AXES:
            for point in ("Q", "E"):
                expected = runs[method][axis][point]
                for metric in METRICS:
                    equal_number(summary["methods"][method]["axes"][axis][point][metric], expected[metric],
                                 f"aggregate {method}/{axis}/{point}/{metric}")
                expected_rows[(method, axis, point, "")] = expected
        for point in ("Q", "E"):
            for metric in METRICS:
                expected = statistics.mean(runs[method][axis][point][metric] for axis in AXES)
                record = summary["methods"][method]["macro_avg"][point][metric]
                equal_number(record["mean_across_five_axes"], expected, f"aggregate macro {method}/{point}/{metric}")
                require(record["n_axes"] == 5 and record["seed"] == 0, "aggregate macro axis/seed count mismatch")
                expected_rows[(method, "macro_avg", point, metric)] = {"value": expected, "n_axes": 5}
    with (bundle / "seed0_summary/aggregate.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(expected_rows), "aggregate CSV row count mismatch")
    seen = set()
    for row in rows:
        key = (row["method"], row["axis"], row["point"], row["metric"])
        require(key in expected_rows and key not in seen and row["seed"] == "0", "aggregate CSV identity/duplicate mismatch")
        seen.add(key)
        for metric, value in expected_rows[key].items():
            equal_number(float(row[metric]), value, f"aggregate CSV {'/'.join(key)} {metric}")
    require(seen == set(expected_rows), "aggregate CSV rows are incomplete")


def verify_bundle(bundle: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "masbench_published_verification_v1", "passed": False,
        "host_quality_gate_reexecuted": False,
        "token_verification": "Recorded prompt/completion counts, phase sums, point totals and summary arithmetic; no retokenization.",
        "errors": [], "runs": {},
    }
    try:
        report["direct_files_verified"] = verify_direct_files(bundle)
        evidence = Evidence(bundle)
        report["archive_files_verified"] = len(evidence.entries)
        instruction = instruction_text()
        computed: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        run_ids: list[str] = []
        for method in METHODS:
            computed[method] = {}
            for axis in AXES:
                try:
                    current_instruction = instruction + (ROBUSTNESS_RULE if axis == "robustness" else "")
                    details, points = verify_run(bundle, evidence, method, axis, current_instruction)
                    report["runs"][f"{method}/{axis}"] = details
                    computed[method][axis] = points
                    run_ids.append(details["run_id"])
                except (VerificationError, KeyError, TypeError, ValueError, OSError) as exc:
                    report["errors"].append(f"{method}/{axis}: {exc}")
        if not report["errors"]:
            verify_summary(bundle, computed, run_ids)
        report["passed"] = not report["errors"]
    except (VerificationError, KeyError, TypeError, ValueError, OSError, tarfile.TarError) as exc:
        report["errors"].append(str(exc))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "results/masbench_seed0_20260930")
    parser.add_argument("--report", type=Path, help="write a detailed JSON report outside the immutable bundle")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if args.report is not None:
        report_path = args.report.resolve()
        require(not report_path.is_relative_to(bundle), "--report must be outside the checksummed bundle")
        require(not report_path.exists(), "refusing to overwrite an existing verification report")
    report = verify_bundle(bundle)
    if args.report is not None:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {key: report.get(key) for key in (
        "passed", "host_quality_gate_reexecuted", "direct_files_verified", "archive_files_verified", "errors")}
    summary["runs_verified"] = len(report["runs"])
    summary["selection_csvs_recomputed"] = sum(run["selection_csvs_recomputed"] for run in report["runs"].values())
    summary["test_points_recomputed"] = 2 * len(report["runs"])
    summary["final_failure_counts"] = {run: {point: len(audit["failures"])
        for point, audit in details["final_prediction_audit"].items()} for run, details in report["runs"].items()}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
