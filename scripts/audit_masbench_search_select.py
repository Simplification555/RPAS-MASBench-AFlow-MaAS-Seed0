#!/usr/bin/env python3
"""Audit packaged D_search/D_select against pinned Salesforce train Parquet.

This audit never parses D_test or downloads the official test split. It only
streams packaged test files to compare their byte hashes with SHA256SUMS.
Requires pyarrow; downloaded source Parquet files stay outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as parquet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.build_search_select_manifest import build_manifest


AXES = ("breadth", "depth", "horizon", "parallel", "robustness")
SOURCE_REPO = "Salesforce/MASBench"
SOURCE_REVISION = "f1d57e20c304c5e42fa77c1a0412de2bc7ad52a3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksums(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative_path = line.split("  ", 1)
        result[relative_path] = digest
    return result


def source_parquet(axis: str, cache_dir: Path, expected_sha: str) -> Path:
    destination = cache_dir / f"{axis}-train-{SOURCE_REVISION}.parquet"
    if destination.exists() and sha256_file(destination) == expected_sha:
        return destination
    url = (
        f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/"
        f"{SOURCE_REVISION}/{axis}/train-00000-of-00001.parquet"
    )
    temporary = destination.with_suffix(".partial")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
                reported_size = response.headers.get("Content-Length")
            if reported_size is not None and temporary.stat().st_size != int(reported_size):
                raise ValueError(f"short download: {temporary.stat().st_size}/{reported_size} bytes")
            actual_sha = sha256_file(temporary)
            if actual_sha != expected_sha:
                raise ValueError(f"SHA mismatch: {actual_sha}")
            temporary.replace(destination)
            return destination
        except (OSError, ValueError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_row(local: dict, official: dict, index: int, axis: str) -> None:
    prompt = json.loads(official["prompt_json"])
    reward = json.loads(official["reward_model_json"])
    extra = json.loads(official["extra_info_json"])
    if not isinstance(prompt, list) or len(prompt) != 1:
        raise ValueError(f"{axis}:{index}: expected one official prompt message")
    expected_source_id = extra.get("example_id", extra.get("index", index))
    checks = {
        "source_index": int(local["source_index"]) == index,
        "official_split": local["official_split"] == "train",
        "dataset": local["dataset"] == "masbench",
        "source_example_id": str(local["source_example_id"]) == str(expected_source_id),
        "input": local["input"] == str(prompt[0]["content"]).strip(),
        "answer": str(local["answer"]) == str(reward["ground_truth"]),
        "axis": local["axis"] == axis == official["axis"],
        "axis_value": str(local["axis_value"]) == str(official["value"]),
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"{axis}:{index}: mismatched fields: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "masbench"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    package_hashes = checksums(root / "SHA256SUMS")
    report: dict = {
        "source_repository": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "scope": "D_search/D_select source rows; D_test byte hash only",
        "axes": {},
    }

    for axis in AXES:
        manifest_path = data_dir / axis / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_name = f"{axis}/train-00000-of-00001.parquet"
        source_sha = manifest["source"]["files"][source_name]
        official_path = source_parquet(axis, args.cache_dir, source_sha)
        official_rows = parquet.read_table(official_path).to_pylist()
        axis_report = {
            "official_train_parquet_sha256": source_sha,
            "official_train_rows": len(official_rows),
            "search_verified": 0,
            "select_verified": 0,
        }
        for split in ("search", "select"):
            relative = f"data/masbench/{axis}/{split}_24.jsonl"
            local_path = root / relative
            actual_sha = sha256_file(local_path)
            if actual_sha != package_hashes[relative]:
                raise ValueError(f"{relative}: package SHA mismatch")
            local_rows = read_rows(local_path)
            if len(local_rows) != 24:
                raise ValueError(f"{axis}/{split}: expected 24 rows, got {len(local_rows)}")
            for local in local_rows:
                index = int(local["source_index"])
                if not 0 <= index < len(official_rows):
                    raise ValueError(f"{axis}/{split}: source_index {index} is out of range")
                verify_row(local, official_rows[index], index, axis)
            axis_report[f"{split}_verified"] = len(local_rows)
            axis_report[f"{split}_file_sha256"] = actual_sha

        # Byte-level check only: do not parse the packaged D_test or obtain
        # any official test examples before candidate selection is frozen.
        test_relative = f"data/masbench/{axis}/test_60.jsonl"
        test_sha = sha256_file(root / test_relative)
        if test_sha != package_hashes[test_relative]:
            raise ValueError(f"{test_relative}: package SHA mismatch")
        axis_report["test_file_sha256_matches_package"] = True
        report["axes"][axis] = axis_report

    safe_manifest_path = data_dir / "manifest_search_select.json"
    safe_manifest_sha = sha256_file(safe_manifest_path)
    reconstructed = (json.dumps(build_manifest(data_dir), ensure_ascii=False, indent=2) + "\n").encode()
    if safe_manifest_path.read_bytes() != reconstructed:
        raise ValueError("D_test-blind manifest differs from deterministic builder output")
    report["search_select_manifest_sha256"] = safe_manifest_sha
    report["search_select_manifest_rebuilt_byte_identical"] = True
    report["total_search_select_rows_verified"] = sum(
        item["search_verified"] + item["select_verified"] for item in report["axes"].values()
    )
    report["d_test_content_opened"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MASBENCH_SOURCE_AUDIT_PASS {args.report}")


if __name__ == "__main__":
    main()
