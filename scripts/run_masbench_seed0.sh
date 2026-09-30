#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEGACY_PYTHON="${RPAS_PYTHON:-python3}"
SERVE_PYTHON="${RPAS_SERVE_PYTHON:-$LEGACY_PYTHON}"
AFLOW_PYTHON="${RPAS_AFLOW_PYTHON:-$LEGACY_PYTHON}"
MAAS_PYTHON="${RPAS_MAAS_PYTHON:-$LEGACY_PYTHON}"
if [[ -n "${RPAS_VLLM:-}" ]]; then
  VLLM="$RPAS_VLLM"
elif [[ "$SERVE_PYTHON" == */* ]]; then
  VLLM="${SERVE_PYTHON%/*}/vllm"
else
  VLLM=vllm
fi
MODEL_PATH="${RPAS_MODEL_PATH:-}"
METHOD="${1:-all}"
PORT="${RPAS_PORT:-8000}"
OUT_ROOT="${RPAS_OUTPUT_ROOT:-$ROOT/outputs/masbench_aflow_maas_seed0}"
DATA_DIR="$ROOT/data/masbench"

case "$METHOD" in aflow|maas|all) ;; *) echo "usage: bash scripts/run_masbench_seed0.sh [aflow|maas|all]" >&2; exit 2 ;; esac
[[ -n "$MODEL_PATH" && -d "$MODEL_PATH" ]] || { echo "Set RPAS_MODEL_PATH to local Qwen3.5-9B weights" >&2; exit 2; }
[[ -n "${RPAS_MODEL_VERIFICATION_REPORT:-}" && -f "$RPAS_MODEL_VERIFICATION_REPORT" ]] || {
  echo "Set RPAS_MODEL_VERIFICATION_REPORT to a fresh scripts/verify_local_models.py report" >&2; exit 2;
}
[[ -d "$ROOT/external_baselines/AFlow" && -d "$ROOT/external_baselines/MaAS" ]] || {
  echo "Run bash scripts/setup_baselines.sh first" >&2; exit 2;
}
[[ -d "$DATA_DIR" ]] || { echo "Packaged MASBench data directory missing" >&2; exit 2; }

export RPAS_PYTHON="$LEGACY_PYTHON" RPAS_SERVE_PYTHON="$SERVE_PYTHON"
export RPAS_AFLOW_PYTHON="$AFLOW_PYTHON" RPAS_MAAS_PYTHON="$MAAS_PYTHON"
export RPAS_VLLM="$VLLM" RPAS_MODEL_PATH="$MODEL_PATH"
export RPAS_TOKENIZER_PATH="${RPAS_TOKENIZER_PATH:-$MODEL_PATH}"
export RPAS_MODEL_REVISION="${RPAS_MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
export RPAS_MAAS_EMBEDDING_REVISION="${RPAS_MAAS_EMBEDDING_REVISION:-1110a243fdf4706b3f48f1d95db1a4f5529b4d41}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
RPAS_CACHE_ROOT="${RPAS_CACHE_ROOT:-/data/jxc/cache/rpas}"
RPAS_TMP_ROOT="${RPAS_TMP_ROOT:-/data/jxc/tmp/rpas}"
export HF_HOME="$RPAS_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$RPAS_CACHE_ROOT/xdg"
export TORCH_HOME="$RPAS_CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$RPAS_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$RPAS_CACHE_ROOT/torchinductor"
export VLLM_CACHE_ROOT="$RPAS_CACHE_ROOT/vllm"
export CUDA_CACHE_PATH="$RPAS_CACHE_ROOT/cuda"
export TMPDIR="$RPAS_TMP_ROOT"
mkdir -p "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH" "$TMPDIR"
LOCAL_NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
export NO_PROXY="${LOCAL_NO_PROXY:+$LOCAL_NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

export RPAS_MAAS_EMBEDDING_PATH="${RPAS_MAAS_EMBEDDING_PATH:-}"
export RPAS_AFLOW_SOURCE="$ROOT/external_baselines/AFlow"
export RPAS_MAAS_SOURCE="$ROOT/external_baselines/MaAS"
export PYTHONPATH="$ROOT:$ROOT/experiments:$ROOT/external_comparison${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export RPAS_MAX_MODEL_LEN=8192 RPAS_MAX_NUM_SEQS=24 RPAS_TP_SIZE=1 RPAS_GPU_MEMORY_UTILIZATION=0.92

"$SERVE_PYTHON" - "$PORT" <<'PY'
import socket
import sys

try:
    port = int(sys.argv[1])
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
except (OSError, ValueError) as exc:
    raise SystemExit(f"RPAS_PORT unavailable: {exc}")
PY

"$SERVE_PYTHON" "$ROOT/scripts/preflight_gpu.py" --role serve
if [[ "$METHOD" == aflow || "$METHOD" == all ]]; then
  "$AFLOW_PYTHON" "$ROOT/scripts/preflight_gpu.py" --role aflow
fi
if [[ "$METHOD" == maas || "$METHOD" == all ]]; then
  "$MAAS_PYTHON" "$ROOT/scripts/preflight_gpu.py" --role maas
fi

mkdir -p "$ROOT/logs" "$OUT_ROOT"
SERVER_LOG="$(mktemp "$ROOT/logs/vllm_seed0_${PORT}_XXXXXXXX.log")"
GPU_LOG="$(mktemp "$ROOT/logs/gpu_seed0_${PORT}_XXXXXXXX.csv")"
"$VLLM" serve "$MODEL_PATH" \
  --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 24 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --language-model-only >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
nvidia-smi --id "$GPU_ID" --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv -lms 1000 >"$GPU_LOG" 2>&1 &
GPU_MONITOR_PID=$!
cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
SESSION_RECORD="${SERVER_LOG%.log}.json"
export RPAS_RUN_SESSION_RECORD="$SESSION_RECORD"
"$SERVE_PYTHON" "$ROOT/scripts/record_run_session.py" \
  --root "$ROOT" --output "$SESSION_RECORD" --server-pid "$SERVER_PID" \
  --gpu-monitor-pid "$GPU_MONITOR_PID" --server-log "$SERVER_LOG" \
  --gpu-log "$GPU_LOG" --port "$PORT"

"$SERVE_PYTHON" - "$PORT" "$SERVER_PID" "$SERVER_LOG" <<'PY'
import json, sys, time, urllib.error, urllib.request
port, pid, log = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
url = f"http://127.0.0.1:{port}/v1/models"
for attempt in range(180):
    try:
        import os
        os.kill(pid, 0)
    except OSError:
        print(f"vLLM exited early; inspect {log}", file=sys.stderr)
        raise SystemExit(1)
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            data = json.loads(response.read().decode())
        if any(item.get("id") == "Qwen/Qwen3.5-9B" for item in data.get("data", [])):
            print("VLLM_MODEL_READY", flush=True)
            break
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        pass
    try:
        import os
        os.kill(pid, 0)
    except OSError:
        print(f"vLLM exited early; inspect {log}", file=sys.stderr)
        raise SystemExit(1)
    time.sleep(5)
else:
    print(f"vLLM did not become ready within 15 minutes; inspect {log}", file=sys.stderr)
    raise SystemExit(1)
PY

RUN_UNITS=()
if [[ "$METHOD" == all ]]; then
  RUN_UNITS=(aflow:breadth maas:breadth)
  for selected_method in aflow maas; do
    for axis in depth horizon parallel robustness; do
      RUN_UNITS+=("$selected_method:$axis")
    done
  done
else
  for axis in breadth depth horizon parallel robustness; do
    RUN_UNITS+=("$METHOD:$axis")
  done
fi

for unit in "${RUN_UNITS[@]}"; do
  selected_method="${unit%%:*}"
  axis="${unit#*:}"
  if [[ "$selected_method" == aflow ]]; then
    METHOD_PYTHON="$AFLOW_PYTHON"
  else
    METHOD_PYTHON="$MAAS_PYTHON"
  fi
  if [[ "$selected_method" == maas && -z "$RPAS_MAAS_EMBEDDING_PATH" ]]; then
    echo "Set RPAS_MAAS_EMBEDDING_PATH for MaAS" >&2; exit 2
  fi
    run_dir="$OUT_ROOT/$selected_method/$axis/seed_0"
    if [[ -e "$run_dir" ]]; then
      if [[ -f "$run_dir/quality_gate.json" ]] && "$METHOD_PYTHON" - "$run_dir" "$ROOT" <<'PY'
import json, sys
from pathlib import Path
from masbench_quality_gate import validate_run
run_dir, root = Path(sys.argv[1]), Path(sys.argv[2])
try:
    stored = json.loads((run_dir / "quality_gate.json").read_text())
    current = validate_run(run_dir, root)
    if stored.get("passed") is True and current.get("passed") is True:
        print("ALREADY_GATED_PASS")
        raise SystemExit(0)
    print(json.dumps(current, indent=2), file=sys.stderr)
except Exception as exc:
    print(f"existing run cannot be verified: {type(exc).__name__}: {exc}", file=sys.stderr)
raise SystemExit(1)
PY
      then
        echo "SKIP already complete and gated: $selected_method/$axis/seed_0"
        continue
      fi
      echo "Refusing to overwrite partial or failed run: $run_dir" >&2
      exit 3
    fi

    args=(--method "$selected_method" --axis "$axis" --seed 0
      --rpas-root "$ROOT" --data-dir "$DATA_DIR" --run-dir "$run_dir"
      --endpoint "http://127.0.0.1:$PORT/v1" --data-seed 2026
      --max-tokens 6144 --concurrency 8 --max-rounds 8
      --maas-samples 1 --maas-batch-size 8)
    echo "START $selected_method/$axis seed=0"
    "$METHOD_PYTHON" "$ROOT/experiments/native_masbench_external.py" "${args[@]}"
    "$METHOD_PYTHON" "$ROOT/experiments/masbench_quality_gate.py" --run-dir "$run_dir" --rpas-root "$ROOT"
done

if [[ "$METHOD" == all ]]; then
  if [[ -e "$OUT_ROOT/seed0_summary" ]]; then
    if [[ -f "$OUT_ROOT/seed0_summary/aggregate.json" && -f "$OUT_ROOT/seed0_summary/aggregate.csv" ]] \
      && "$AFLOW_PYTHON" "$ROOT/experiments/aggregate_masbench_seed0.py" --root "$OUT_ROOT" --verify-existing; then
      echo "SKIP aggregation; existing summary revalidated at $OUT_ROOT/seed0_summary"
    else
      echo "Refusing to trust or overwrite an incomplete/stale summary: $OUT_ROOT/seed0_summary" >&2
      exit 4
    fi
  else
    "$AFLOW_PYTHON" "$ROOT/experiments/aggregate_masbench_seed0.py" --root "$OUT_ROOT"
  fi
fi
echo "MASBENCH_SEED0_RUN_COMPLETE"
