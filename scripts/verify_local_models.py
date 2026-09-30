#!/usr/bin/env python3
"""Verify pinned local model assets without contacting Hugging Face."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "deployment/model_revisions.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(path: Path, specification: dict) -> dict:
    checked = {}
    for name, expected_blob in specification["metadata_git_blob_sha1"].items():
        candidate = path / name
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise FileNotFoundError(f"required model metadata missing or empty: {candidate}")
        content = candidate.read_bytes()
        actual_blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if actual_blob != expected_blob:
            raise ValueError(f"{candidate}: Git blob SHA-1 {actual_blob} != {expected_blob}")
    for name, (expected_size, expected_sha) in specification["files"].items():
        candidate = path / name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.stat().st_size != expected_size:
            raise ValueError(f"{candidate}: size {candidate.stat().st_size} != {expected_size}")
        actual_sha = sha256_file(candidate)
        if actual_sha != expected_sha:
            raise ValueError(f"{candidate}: SHA-256 {actual_sha} != {expected_sha}")
        checked[name] = actual_sha
    return {"repository": specification["repository"], "revision": specification["revision"],
            "path": str(path.resolve()), "sha256_verified": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, default=Path("/data/jxc/models/Qwen3.5-9B"))
    parser.add_argument("--embedding", type=Path, default=Path("/data/jxc/models/all-MiniLM-L6-v2"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error(f"refusing to overwrite {args.report}")
    specification = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(MANIFEST),
        "models": {
            "qwen": verify_model(args.qwen, specification["qwen"]),
            "embedding": verify_model(args.embedding, specification["embedding"]),
        },
    }
    index = json.loads((args.qwen / "model.safetensors.index.json").read_text(encoding="utf-8"))
    actual_shards = set(index["weight_map"].values())
    expected_shards = {name for name in specification["qwen"]["files"] if name.endswith(".safetensors")}
    if actual_shards != expected_shards:
        raise ValueError(f"Qwen index shard set differs: {sorted(actual_shards ^ expected_shards)}")
    report["models"]["qwen"]["index_shards_verified"] = sorted(actual_shards)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"LOCAL_MODELS_VERIFIED {args.report}")


if __name__ == "__main__":
    main()
