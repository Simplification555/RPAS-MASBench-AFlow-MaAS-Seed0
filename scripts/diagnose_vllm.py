#!/usr/bin/env python3
"""Disposable local vLLM probe using synthetic prompts only.

This does not read MASBench data or write into a formal output root.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


MODEL = "Qwen/Qwen3.5-9B"


def request_json(url: str, payload: dict | None = None, timeout: int = 180) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def gpu_used_mib(index: int) -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--id", str(index), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        return None
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def probe(endpoint: str, sequence: int, max_tokens: int) -> dict:
    started = time.monotonic()
    try:
        response = request_json(
            endpoint + "/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": f"Reply with only the integer {sequence + 7}."}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        choice = response["choices"][0]
        answer = str(choice.get("message", {}).get("content") or "").strip()
        expected = str(sequence + 7)
        return {
            "ok": answer == expected,
            "seconds": round(time.monotonic() - started, 3),
            "finish_reason": choice.get("finish_reason"),
            "answer": answer,
            "expected": expected,
            "usage": response.get("usage", {}),
        }
    except Exception as exc:
        return {"ok": False, "seconds": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if any(level < 1 or level > 8 for level in args.levels):
        parser.error("levels must be within 1..8")
    endpoint = args.endpoint.rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.path != "/v1":
        parser.error("--endpoint must be a loopback HTTP URL ending in /v1")
    models = request_json(endpoint + "/models")
    if not any(item.get("id") == MODEL for item in models.get("data", [])):
        parser.error(f"{endpoint} does not serve the required model ID {MODEL}")

    stop = threading.Event()
    samples: list[int] = []

    def sample_gpu() -> None:
        while not stop.is_set():
            used = gpu_used_mib(args.gpu_index)
            if used is not None:
                samples.append(used)
            stop.wait(0.2)

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    levels = []
    try:
        for level in args.levels:
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=level) as pool:
                results = list(pool.map(lambda number: probe(endpoint, number, args.max_tokens), range(level)))
            elapsed = time.monotonic() - started
            prompt_tokens = sum(int(result.get("usage", {}).get("prompt_tokens", 0)) for result in results)
            completion_tokens = sum(int(result.get("usage", {}).get("completion_tokens", 0)) for result in results)
            levels.append({
                "concurrency": level,
                "seconds": round(elapsed, 3),
                "successes": sum(bool(result["ok"]) for result in results),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "completion_tokens_per_second": round(completion_tokens / elapsed, 3) if elapsed else None,
                "requests": results,
            })
            print(json.dumps(levels[-1], ensure_ascii=False), flush=True)
            if levels[-1]["successes"] != level:
                break
    finally:
        stop.set()
        sampler.join(timeout=2)
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model": MODEL,
        "synthetic_only": True,
        "max_tokens": args.max_tokens,
        "gpu_index": args.gpu_index,
        "gpu_used_mib_min": min(samples) if samples else None,
        "gpu_used_mib_peak": max(samples) if samples else None,
        "levels": levels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(level["successes"] != level["concurrency"] for level in levels):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
