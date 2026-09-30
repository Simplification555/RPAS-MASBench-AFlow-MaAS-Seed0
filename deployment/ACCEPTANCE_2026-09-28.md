# Single RTX 5880 Ada acceptance before formal seed-0 execution

The following checks ran on GPU 0 of `ubuntu-SYS-521GE-TNRT` on 2026-09-28
(Asia/Shanghai). Synthetic diagnostics are separate from formal MASBench
outputs and did not parse D_test.

| Check | Observed result |
|---|---|
| Python environments | Python 3.11.15; vLLM 0.17.0, Torch 2.10.0+cu130, Transformers 4.57.6; all three `uv pip check` commands passed. Frozen package lists are in this directory. |
| Pinned weights | Qwen3.5-9B four shard SHA-256 values, index, metadata and tokenizer verified; all-MiniLM-L6-v2 weights and metadata verified. Report: `/data/jxc/tmp/rpas/model_verification.json`. |
| GPU and local embedding | GPU 0 has 49,140 MiB physical memory and supports BF16. MiniLM loaded locally on CUDA and encoded one sentence to a 384-dimensional vector using about 96 MiB allocated tensor memory. |
| vLLM frozen configuration | Localhost, BF16, text-only, TP=1, 8192 context, 6144 maximum generation, 24 maximum sequences, 16384 batched tokens, 0.92 GPU memory utilization, prefix caching and Qwen3 reasoning parser. Weight load: 16.8 GiB in 5.76 s; engine initialization: 40.09 s. |
| Correctness and concurrency | 1/1, 2/2, 4/4 and 8/8 synthetic arithmetic responses correct; no request errors. Hot 8-request level: 0.181 s and 21 completion tokens. GPU peak: 43,388 MiB. Report: `/data/jxc/tmp/rpas/diagnostics/vllm_probe.json`. |
| Full output ceiling | A 1428-token synthetic input with `max_tokens=6144` generated all 6144 tokens (7572 total, below 8192), finished in 119.119 s, and peaked at 43,392 MiB; no OOM. Report: `/data/jxc/tmp/rpas/diagnostics/vllm_context_probe.json`. |
| Native AFlow | Pinned optimizer generated one new graph; both original and generated graphs answered 2/2 synthetic questions correctly. 13 local model calls, none failed. Report: `/data/jxc/tmp/rpas/diagnostics/aflow_real_20260928/diagnostic.json`. |
| Native MaAS | Pinned `Optimizer.optimize('Graph')` completed 4 repetitions on 2 synthetic rows (8 observed results), 4 Adam steps, changed 16 controller parameter tensors, and wrote a changed checkpoint. 61 local model calls, none failed. Report: `/data/jxc/tmp/rpas/diagnostics/maas_real_20260928_retry4/report.json`. |
| Shared GPU residency | While MaAS trained, GPU 0 showed about 44,278 MiB total used: vLLM 43,368 MiB and MaAS 880 MiB. No OOM was observed. |
| Search/select provenance | 240 packaged D_search/D_select rows verified against pinned official Salesforce/MASBench train Parquet; D_test content was not opened by the audit. See `../data/masbench/CHECKSUM_AUDIT.md`. |

The MaAS GPU diagnostic above used reconstructed prompt helpers before the
original RPAS-project helper files were supplied. The two supplied files were
copied from `Simplification555/RPAS-EC2-A100@ba1ca711` and match its published
SHA-256 values; the pinned native MaAS train/test templates pass offline repair
tests with them. The updated MASBench protocol uses `concise=False`, retaining
native prompt wording apart from literal/format-string compatibility repair.
Repeat the MaAS GPU diagnostic with the recovered helpers before its formal
run. The source repository reports recovery from a working copy without
original Git metadata, so formal results should state “MaAS with RPAS-project
compatibility/context adapter,” not unmodified upstream MaAS.

These checks establish readiness to attempt the ten formal method-axis cells.
Their scores and quality gates must be evaluated from the formal output tree.

## Follow-up on 2026-09-29

The required diagnostic with the recovered helpers passed at approximately
00:14 Asia/Shanghai on 2026-09-29, before formal MaAS execution. It used two
synthetic rows, concurrency 2, and a 1536-token generation ceiling. Its original
report and logs are retained under
`provenance/diagnostics/maas_recovered_20260929/` in the
[published evidence archive](../results/masbench_seed0_20260930/README.md).
This small diagnostic verifies a controller update with the recovered helpers;
it is separate from the full 6144-token formal workload. The initial formal
MaAS breadth attempt failed its 180-second graph timeout; the fresh recovery
and all completed formal results are documented in that result bundle.
