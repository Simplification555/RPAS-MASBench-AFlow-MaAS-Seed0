# Search/select manifest checksum audit

Audit date: 2026-09-28 (Asia/Shanghai).

The packaged `manifest_search_select.json` is byte-for-byte identical to the
output of `experiments/build_search_select_manifest.py` from the packaged
`search_24.jsonl` and `select_24.jsonl` files. Its SHA-256 is
`27335b968c6b70a60e3fd5eba54221a9d45085568b6f602c9cd3fd4ed95a9a62`.
The original `SHA256SUMS` entry, `d62e744773ff81a47e804edcc023a156665971f9edf373dd6fac7ed2faea2602`,
was the only package checksum that failed. The old digest's source bytes are
not available in this repository, which has only one initial commit. The
entry was corrected to match the verified generated file; no data or manifest
bytes were changed.

## Source verification

The audit used the official [Salesforce/MASBench dataset](https://huggingface.co/datasets/Salesforce/MASBench)
at revision `f1d57e20c304c5e42fa77c1a0412de2bc7ad52a3`. For each axis,
the downloaded **train** Parquet file's SHA-256 matched the corresponding
`source.files` digest in the packaged axis manifest. Every packaged
`D_search` and `D_select` row (24 + 24 per axis, 240 total) matched the
official train row at its `source_index` for the prompt text (after the
packaged whitespace trim), ground-truth answer, axis, axis value, and
source example ID. Each packaged row also identified the official `train`
split. All packaged search/select file byte hashes matched `SHA256SUMS`.

| Axis | Official train Parquet SHA-256 | Rows checked |
|---|---|---:|
| breadth | `7c0e37e979b789301276c3771c62e5b9919c1bab608643afb89e9a1a9a350e28` | 48 |
| depth | `afe7040e92edf09e5fda6a3b759cee3ca920017f02b205a885a929cc40ecaa64` | 48 |
| horizon | `9db6ee4cfcab8fdac73cea5296ca7c340a5884c22076b898edb5d78150a8fc42` | 48 |
| parallel | `47ef6e470200382f3a1eda676a32f9465f5129eaf5a3c8a01565820c7e35a1b5` | 48 |
| robustness | `0edc2212cd2964568f7215f56c7b77a7cfc42ece329f41f7b9c6c3b46c521c2d` | 48 |

The audit streamed the five packaged `test_60.jsonl` files only to check
their byte hashes against `SHA256SUMS`. It did **not** parse D_test examples,
download official test files, or inspect test answers. Candidate selection
must still be frozen before the runtime reads D_test.

The stated `JiangyueAnn/RPAS` source repository at commit
`e12f58823be5f91a32f05f9af4d36e54838ffe59` currently returns GitHub
404, so its historical split-generation process and the old manifest digest
cannot be independently reconstructed. This audit verifies the packaged
pre-selection rows against the official dataset and the packaged manifest
against its deterministic builder; it does not establish that the historical
RPAS split-generation process was identical.

To repeat the audit with Python and `pyarrow` installed:

```bash
python scripts/audit_masbench_search_select.py \
  --cache-dir /data/jxc/tmp/rpas-data-audit \
  --report /data/jxc/tmp/rpas-data-audit/audit.json
sha256sum -c SHA256SUMS
```

The Parquet cache and JSON report are stored outside the project. The audit
script does not modify benchmark data.
