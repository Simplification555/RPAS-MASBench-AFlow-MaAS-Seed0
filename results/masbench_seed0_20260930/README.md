# Completed AFlow + MaAS MASBench pilot, seed 0

Completed **2026-09-30 12:23:09 Asia/Shanghai** on a single RTX 5880 Ada 48 GB.
All ten method/axis units passed their stored execution/protocol gates. A fresh
`aggregate_masbench_seed0.py --verify-existing` validation also passed on the
original host. This package preserves the original result bytes and records.

**MaAS label: RPAS-project compatibility/context adapter variant.** Its native
optimizer and controller are used with the recorded template compatibility,
context, scoring/data, training-repetition, and runtime adapters. The recovered
helper files are RPAS project files, with no independently verified original
Git history. This is not an unmodified upstream MaAS reproduction.

## Results

Each axis has 60 held-out D_test examples. Entries below are exact matches,
with the native Q/E operating points selected only on D_select.

| Axis | AFlow Q | AFlow E | MaAS Q = E | AFlow Q tokens/example | AFlow E tokens/example | MaAS tokens/example |
|---|---:|---:|---:|---:|---:|---:|
| breadth | 23/60 | 24/60 | 21/60 | 20,342.53 | 17,514.02 | 69,064.75 |
| depth | 19/60 | 21/60 | 18/60 | 29,081.37 | 24,403.85 | 63,977.48 |
| horizon | 8/60 | 8/60 | 11/60 | 6,482.42 | 6,482.42 | 50,544.40 |
| parallel | 12/60 | 10/60 | 11/60 | 29,194.12 | 3,665.07 | 39,981.82 |
| robustness | 0/60 | 0/60 | 0/60 | 10,857.30 | 10,857.30 | 49,349.42 |
| Mean across five axes | 20.67% | 21.00% | 20.33% | 19,191.55 | 12,584.53 | 54,583.57 |

The [original aggregate JSON](seed0_summary/aggregate.json) and
[CSV](seed0_summary/aggregate.csv) also contain component accuracy and the
0/1/2 ordinal score. Token counts include prompt and completion tokens across
the selected workflow's model calls during D_test; they exclude D_search and
D_select. Those phases have separate telemetry in each native result. MaAS Q
and E refer to the same controller and test evaluation; they are not two
independent evaluations.

Only search seed 0 was run. The macro mean is over five different axes, with no
across-seed standard deviation or significance claim. A passing quality gate
checks execution and protocol consistency, not high benchmark performance.
MaAS robustness has 13/60 parse-valid answers and 11/420 correct components,
despite 0/60 exact matches. Context compaction and the 6144-token generation
ceiling remain performance limitations. Price-based fields from the local
service do not represent a measured dollar cost.

## Contents

- `seed0_summary/`: original combined aggregate, also present in the archive.
- `runs/{aflow,maas}/{axis}/seed_0/`: readable original run manifests, selection
  locks, native results, split records, and quality gates for all ten units.
- `evidence.tar.gz`: all formal predictions and training/search outputs,
  generated workflows/prompts, actual runtime baseline source copies, five
  trained MaAS controller checkpoints, and the failed initial MaAS breadth
  attempt. It also contains frozen harness source snapshots, pinned upstream
  sources, model verification, service/session records, diagnostics, original
  launchers, and execution/GPU logs under `provenance/`.
- `file_index.json`: one SHA-256, byte size, archive member path, and historical
  source path for each archived file. The archive contains regular files only.
- `source_revisions.json` and `export_inventory.json`: source commits and export
  details. Bytecode/cache directories, Git internals, image assets, `index.html`,
  and unrelated upstream `.pth` checkpoints were omitted. Model weights are
  excluded. Actual MASBench controller checkpoints are included.
- `RECOVERY_NOTE.md`, `runtime_overrides.json`, and
  `failed_maas_breadth/run_manifest.json`: original runtime recovery evidence.
- `verification_report.json`: offline verification and prediction audit report.
- `SHA256SUMS`: checksums for the published files outside the archive.

Third-party code and data keep their source attribution. The archive retains
AFlow's MIT license and the dataset license notices. The pinned MaAS source
declares MIT in `setup.py` but has no separate root license file; no new license
is asserted here. Raw prediction files contain benchmark material.

## Offline verification on a fresh clone

From the repository root, using Python 3.11 and its standard library:

```bash
python scripts/verify_published_masbench.py \
  --bundle results/masbench_seed0_20260930
```

The verifier checks published and archived file hashes, original metadata
copies, selection/result/checkpoint hashes, and recomputes final selection/test
scores with the shared MASBench answer parser. It also checks the aggregate
and audits final predictions for empty outputs and explicit runtime failures.
It reads the archive without extracting it or loading controller checkpoints.

This is an offline evidence check. The original full quality gate also checks
the original frozen source trees, Git checkouts, installed models and service
records, so it cannot be executed unchanged on an arbitrary clone. Historical
absolute paths are preserved in the original JSON; `file_index.json` resolves
them to portable archive members. The completed AFlow link in the recovery
tree was materialized when exporting. No selection lock or result record was
rewritten to replace paths.

## Execution history and known audit limitation

AFlow used frozen harness commit `f9ec43e`; MaAS used `56e7392`. Upstream commits
were AFlow `3f457218fc716093fe53f6df8a5d5e6379d66346` and MaAS
`987f3c1bc9a96e844fe090db3791446e3ef0f5c7`.

The first MaAS breadth run stopped before D_test access with 17/24 empty
D_select outputs under the adapter's 180-second graph timeout. It is preserved
as a failed attempt and excluded from the aggregate. The successful run
restarted controller training in a fresh root with graph timeout 1500 seconds,
request timeout 600 seconds, three sample attempts, and generated-code timeout
60 seconds. Model, splits, search seed, prompt wording, context, output ceiling,
and concurrency were unchanged. The timeout overrides are in the original
launcher, launch log and `runtime_overrides.json`; they are not recorded in
`native_result.controls`. See the [recovery note](RECOVERY_NOTE.md).

AFlow parallel search produced invalid mutations referencing `self.custom` in
rounds 6–8. They scored zero and were excluded; the final Q/E evaluations use
rounds 9/1. The original failure-prefix scanner does not recognize the bare
`'Workflow' object has no attribute 'custom'` string. The frozen executed source
is retained with this limitation. Final raw predictions were independently
checked for this exception and other explicit runtime failures, with none
found. Successful MaAS breadth D_test contains one empty prediction (1/60),
already reflected in its original audit and score.
