# Native AFlow + MaAS on MASBench (seed 0)

This package is a **single-seed, five-axis pilot** of AFlow and MaAS on the frozen Salesforce/MASBench data shipped here. It is not a three-seed paper table and it makes no across-seed variance/significance claim. The baseline implementations are fetched from their upstream repositories at pinned commits; run-local compatibility/data/scoring adapters are patched. The missing RPAS-project MaAS helper files were later supplied in a separate project branch and pinned here by SHA-256. MASBench uses their native-template compatibility repair without the optional concise prompt rewrite. Each method/axis gets an isolated output tree.

## Completed seed-0 results

All ten method/axis cells completed by **2026-09-30 12:23:09 Asia/Shanghai**. Each stored quality gate passed, and fresh validation of the aggregate passed on the original host. The [published result bundle](results/masbench_seed0_20260930/README.md) includes the original summaries, run/selection records, raw predictions, generated workflows, trained controllers, runtime code, configuration, and execution logs. It also provides an offline checksum and scoring verifier for a fresh clone.

| Method / operating point | Mean exact accuracy across five axes | Mean D_test tokens per example |
|---|---:|---:|
| AFlow Q | 20.67% | 19,191.55 |
| AFlow E | 21.00% | 12,584.53 |
| MaAS RPAS compatibility/context adapter variant (Q = E) | 20.33% | 54,583.57 |

These are one-seed results. A passing execution/protocol gate does not imply high accuracy; all three rows have 0/60 exact matches on robustness. MaAS was retrained in a fresh output root after an initial breadth timeout failure, before MaAS D_test access. The recovery used graph/request timeouts of 1500/600 seconds, three sample attempts, and a 60-second generated-code timeout. The [recovery record](results/masbench_seed0_20260930/RECOVERY_NOTE.md) explains the change and preserves the failed attempt separately.

## Locked protocol

- Methods: native AFlow optimizer and native MaAS optimizer/controller with recorded RPAS compatibility, data, scoring, context, and execution adapters; five axes: `breadth`, `depth`, `horizon`, `parallel`, `robustness`.
- One search seed only: `0` (pilot); frozen data seed: `2026`; each axis uses the canonical RPAS split of 24 `D_search`, 24 `D_select`, and 60 `D_test` examples.
- Model: `Qwen/Qwen3.5-9B`; context limit 8192; generation ceiling 6144; deterministic decoding; concurrency 8; vLLM TP=1, max-num-seqs=24, GPU memory utilization 0.92.
- Shared answer parser reports exact-match accuracy, component accuracy, and ordinal score 0/1/2 (wrong / partially correct / exact). Robustness explicitly preserves injected raw retrieval values while algebraic values remain in Z_23.
- AFlow uses its native MCTS/Graph optimizer with the fixed MATH/math operator configuration, sample 4, max rounds 8, and an 8-retry cap only for repeated/rejected mutations. MaAS uses its native Agentic Supernet/controller optimization, native MATH/math operators, one controller sample, one search round, batch 8, learning rate 0.01, no TextGrad, and its four native training repetitions. MaAS produces one native controller; its Q/E entries therefore reference the same selected controller rather than inventing a second method-specific policy.
- The packaged pre-selection manifest contains only `D_search`/`D_select` IDs and file hashes; the optional builder reads only those two split files, never D_test or the per-axis manifests. The runner validates only `D_search`/`D_select` before search. It writes and hashes `selection_frozen.json` after the `D_select` decision, and only then reads `test_60.jsonl` and the full per-axis manifest. If a run is interrupted or selection is invalid, it fails closed and does not open D_test. Existing/partial run directories are never overwritten.

## Single RTX 5880 Ada 48 GB deployment

The target host has an RTX 5880 Ada Generation (compute capability 8.9, approximately 48 GB) with a driver reporting CUDA 13.0 support. The [official vLLM Qwen3.5-9B recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-9B) lists BF16 deployment on one 24 GB GPU with vLLM 0.17.0 or newer. This supports attempting the model load; the frozen 8192 context, 24 maximum sequences, eight requests, and MaAS process sharing the GPU still require measured acceptance.

The upstream projects do not define a verified shared environment. Use separate Python 3.11 environments for vLLM, AFlow, and MaAS. MaAS pins an older PyTorch stack that must not replace the serving stack. `scripts/preflight_gpu.py` checks each role separately; it does not replace model loading or a real native method run. No model weights are in this repository and the formal runner never downloads them implicitly. The pinned model revisions and large-file hashes are in `deployment/model_revisions.json`.

The [environment setup record](deployment/ENVIRONMENT_SETUP.md) gives the installation commands and frozen package lists used on this host.
The [single-GPU acceptance record](deployment/ACCEPTANCE_2026-09-28.md) gives measured model, memory, concurrency, and native-method diagnostic results before formal execution.

## Run on Linux

Set paths to the installed Python environments and verified local weights:

```bash
cd /data/jxc/RPAS-MASBench-AFlow-MaAS-Seed0
export RPAS_SERVE_PYTHON=/data/jxc/envs/rpas-serve/bin/python
export RPAS_AFLOW_PYTHON=/data/jxc/envs/rpas-aflow/bin/python
export RPAS_MAAS_PYTHON=/data/jxc/envs/rpas-maas/bin/python
export RPAS_MODEL_PATH=/data/jxc/models/Qwen3.5-9B
export RPAS_MAAS_EMBEDDING_PATH=/data/jxc/models/all-MiniLM-L6-v2
export RPAS_MODEL_VERIFICATION_REPORT=/data/jxc/tmp/rpas/model_verification.json
export CUDA_VISIBLE_DEVICES=0
# Runtime limits used by the successful recovered MaAS run:
export RPAS_MAAS_SAMPLE_TIMEOUT_S=1500
export RPAS_MAAS_REQUEST_TIMEOUT_S=600
export RPAS_MAAS_SAMPLE_RETRIES=3
export RPAS_MAAS_CODE_TIMEOUT_S=60

bash scripts/setup_baselines.sh
python scripts/verify_local_models.py --report "$RPAS_MODEL_VERIFICATION_REPORT"
# Run in a persistent tmux session after the disposable service/method diagnostics pass.
bash scripts/run_masbench_seed0.sh all
```

To run a single method, use `bash scripts/run_masbench_seed0.sh aflow` or `... maas`. `RPAS_OUTPUT_ROOT` and `RPAS_PORT` may be changed to a new location/port. A method-axis-seed directory is never overwritten: the runner skips it only if its stored gate and a fresh gate verification both pass; any partial/failed unit stops the run. If a unit was interrupted, preserve that root for diagnosis and use a fresh root for a clean rerun. Fully gated earlier units can safely be skipped if the same command is re-invoked. MaAS requires local Sentence-Transformers `all-MiniLM-L6-v2` weights.

The completed host run used frozen source commits `f9ec43e` for AFlow and `56e7392` for MaAS. AFlow results remain in `outputs/masbench_aflow_then_maas_seed0/aflow`; the successful MaAS results and combined aggregate are in `outputs/masbench_aflow_maas_recovery_seed0_20260929`, whose `aflow` entry links to the completed AFlow tree. Each result manifest points to its own frozen source root for fresh host validation. The published archive materializes that link and preserves the original records. Historical absolute paths in those records are resolved by the offline verifier; the original full gate additionally requires the original source/model/service environment.

The script launches vLLM in BF16 on localhost, waits for the exact served model ID, evaluates each axis, runs the fail-closed quality gate after every run, and stops its own server on exit. `all` gates AFlow breadth and MaAS breadth first, then runs the remaining AFlow and MaAS axes serially. The runner records each native source commit/hash and exact run settings. It does not submit to a scheduler and has no global wall-time guarantee. It never resumes partial output directories: preserve and diagnose an interrupted directory, then use a fresh output root for a clean run.

The `all` runner automatically writes the aggregate after all 10 runs pass their individual gates. To revalidate an existing aggregate without overwriting it:

```bash
python experiments/aggregate_masbench_seed0.py \
  --root outputs/masbench_aflow_maas_recovery_seed0_20260929 --verify-existing
```

Aggregation refuses incomplete/failed/mismatched runs and writes `aggregate.json` and `aggregate.csv` in a new `seed0_summary` directory. The macro average is across the five axes for seed 0; it is **not** a standard deviation across seeds. Do not fill a three-seed mean±SD table from this package.

## Pre-run checks and tests

```bash
python -m py_compile experiments/*.py
PYTHONPATH="$PWD:$PWD/experiments:$PWD/external_comparison" \
  python -m unittest \
    experiments.test_native_masbench_external \
    experiments.test_masbench_answer_protocol \
    experiments.test_masbench_quality_gate \
    experiments.test_aggregate_masbench_seed0 \
    experiments.test_maas_prompt_repair \
    experiments.test_diagnose_maas_controller -v
PYTHONPATH="$PWD:$PWD/experiments:$PWD/external_comparison" \
  python experiments/masbench_runtime_smoke.py
```

`scripts/preflight_gpu.py` runs automatically for each selected role. Before the long run, start the same vLLM configuration separately and run `scripts/diagnose_vllm.py --output /data/jxc/tmp/rpas/vllm_probe.json` with synthetic prompts. It checks concurrency 1, 2, 4, and 8 and records request usage, latency, and observed GPU memory. Complete an AFlow model-call diagnostic and a MaAS controller-update diagnostic before opening a formal result root. Keep diagnostics outside the formal output tree.

## Source and data notes

- AFlow: [FoundationAgents/AFlow](https://github.com/FoundationAgents/AFlow), pinned commit `3f457218fc716093fe53f6df8a5d5e6379d66346`.
- MaAS: [bingreeky/MaAS](https://github.com/bingreeky/MaAS), pinned commit `987f3c1bc9a96e844fe090db3791446e3ef0f5c7`.
- Execution clones the third-party baselines at pinned revisions. `scripts/setup_baselines.sh` refuses to replace an existing checkout unless it is already at the exact pinned revision. The published evidence archive retains the pinned source and actual runtime copies with source attribution.
- The packaged MASBench files state provenance from `JiangyueAnn/RPAS` at `e12f588`, but that repository is currently inaccessible. A [source audit](data/masbench/CHECKSUM_AUDIT.md) verified every D_search/D_select row against a pinned [Salesforce/MASBench](https://huggingface.co/datasets/Salesforce/MASBench) official train Parquet and corrected one stale checksum entry. D_test content was not opened during that audit. The historical RPAS split-generation process remains independently unverified.
- The [MaAS helper provenance](experiments/MAAS_HELPER_PROVENANCE.md) records the two files copied from `Simplification555/RPAS-EC2-A100` commit `ba1ca711f148642f6e61bb38c52a0e8994e54c98` and their SHA-256 values. That repository reports recovery from an RPAS working copy without original Git metadata; this package independently checked the published file hashes, not the earlier working copy. MASBench applies `maas_prompt_repair.py` with `concise=False`; `maas_concise_prompts_v2.py` is retained for provenance but its wording edits are not used. The context guard remains a recorded adapter, so label MaAS results accordingly.
- This is a one-seed execution package, not evidence for the screenshot's three-seed mean±SD table. D_test is included because the RPAS canonical data branch contains it; do not use it for search, tuning, prompt changes, or candidate selection. Publicly sharing benchmark test examples may affect benchmark hygiene, even where the upstream dataset license permits redistribution.
- No separate license is asserted for this package's custom harness code. The method repositories and dataset keep their upstream licenses; see their linked source repositories/cards.
- The published evidence contains benchmark questions and model outputs, with the existing dataset provenance and license notices retained. Model weights, credentials, caches, and unrelated upstream checkpoints are excluded.
