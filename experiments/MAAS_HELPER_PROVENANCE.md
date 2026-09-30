# Recovered MaAS helper provenance

This package copied the two helper files byte for byte from
`Simplification555/RPAS-EC2-A100` commit
`ba1ca711f148642f6e61bb38c52a0e8994e54c98`, under
`experiments/external_baselines/masbench_aflow_maas_seed0/experiments/`.
Their hashes below were checked after copying. The earlier SCIR working-copy
recovery is the source repository's provenance statement; that working copy
and its original Git history were not available for independent verification.

The pinned upstream MaAS commit does not contain `maas_prompt_repair.py` or
`maas_concise_prompts_v2.py`; the published harness previously imported these
files without shipping them. The exact helper files were recovered from the
SCIR working copy at `/home/jianbaizhao/RPAS/experiments/` and copied byte for
byte (SHA-256 values below). That SCIR directory had no Git metadata, so an
original commit cannot be truthfully attributed to these files. They are
RPAS-project compatibility helpers, **not upstream MaAS source**.

| File | SHA-256 |
|---|---|
| `maas_prompt_repair.py` | `dbfad9d7c66e1b80df18a2550a9c1c6a4766943d900d909810ed7bbc9fa2f63c` |
| `maas_concise_prompts_v2.py` | `a4af1fd1f791b6b9fdb3a98c74c92418222ac3b80cc692b15ebe216b787f583f` |

The seed-0 MASBench runner now invokes `patch_template_directory(...,
concise=False)`: it applies only the literal/format-string compatibility repair
needed to execute the released MaAS MATH templates, and does not apply the
separate concise-v2 wording changes. The deterministic context guard remains
an adapter and is recorded in run manifests; therefore results should be
labelled “MaAS with RPAS-project compatibility/context adapter,” not as an
unmodified upstream MaAS run. The completed single-seed results and original
runtime records are published in
[`results/masbench_seed0_20260930`](../results/masbench_seed0_20260930/README.md).
