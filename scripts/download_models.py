#!/usr/bin/env python3
"""Download the fixed model revisions into /data before any formal run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("/data/jxc/models"))
    parser.add_argument("--cache-root", type=Path, default=Path("/data/jxc/cache/rpas"))
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    args.model_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.cache_root / "huggingface"))
    os.environ.setdefault("HF_XET_CACHE", str(args.cache_root / "xet"))
    from huggingface_hub import snapshot_download

    specification = json.loads((ROOT / "deployment/model_revisions.json").read_text(encoding="utf-8"))
    for key, local_name in (("embedding", "all-MiniLM-L6-v2"), ("qwen", "Qwen3.5-9B")):
        item = specification[key]
        destination = args.model_root / local_name
        print(f"DOWNLOAD_START {item['repository']}@{item['revision']} -> {destination}", flush=True)
        snapshot_download(
            repo_id=item["repository"], revision=item["revision"],
            local_dir=destination, max_workers=args.max_workers,
        )
        print(f"DOWNLOAD_DONE {destination}", flush=True)


if __name__ == "__main__":
    main()
