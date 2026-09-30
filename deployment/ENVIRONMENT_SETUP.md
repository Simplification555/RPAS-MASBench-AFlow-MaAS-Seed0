# Single GPU environment setup

These commands were used on the RTX 5880 Ada host. They keep model weights,
Python environments, package caches, and build temporary files under `/data`.
The serving, AFlow, and MaAS interpreters are separate because their native
dependency sets are not mutually pinned. After installation, the observed
versions are recorded in `serve.freeze.txt`, `aflow.freeze.txt`, and
`maas.freeze.txt`. On this host the PyPI mirror transfers large CUDA wheels
faster through the existing localhost proxy (`HTTPS_PROXY`/`HTTP_PROXY`/
`ALL_PROXY` at `127.0.0.1:17897`); leave those variables set for the three
`uv pip install` commands.

```bash
cd /data/jxc/RPAS-MASBench-AFlow-MaAS-Seed0
mkdir -p /data/jxc/envs /data/jxc/cache/rpas/uv /data/jxc/tmp/rpas
uv venv --python /data/jxc/envs/pyroki311/bin/python /data/jxc/envs/rpas-serve
uv venv --python /data/jxc/envs/pyroki311/bin/python /data/jxc/envs/rpas-aflow
uv venv --python /data/jxc/envs/pyroki311/bin/python /data/jxc/envs/rpas-maas

export UV_CACHE_DIR=/data/jxc/cache/rpas/uv
export TMPDIR=/data/jxc/tmp/rpas
export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=10 UV_LOCK_TIMEOUT=7200
uv pip install --python /data/jxc/envs/rpas-serve/bin/python \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  --torch-backend=auto 'vllm==0.17.0'
uv pip install --python /data/jxc/envs/rpas-aflow/bin/python \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  --torch-backend=auto -r external_baselines/AFlow/requirements.txt \
  'torch==2.10.0' 'transformers>=4.56,<5' 'huggingface-hub>=0.30,<2'
uv pip install --python /data/jxc/envs/rpas-maas/bin/python \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  --torch-backend=auto 'torch==2.10.0' 'sentence-transformers<6' \
  'openai==1.82.0' 'numpy<2' pandas pydantic tenacity PyYAML aiofiles aiohttp \
  loguru 'transformers>=4.56,<5' 'huggingface-hub>=0.30,<2' tqdm \
  python-docx 'anthropic>=0.30,<2' tiktoken chardet pillow rich pygments \
  GitPython gitignore-parser tree-sitter-python 'tree-sitter>=0.23,<0.26'

uv pip check --python /data/jxc/envs/rpas-serve/bin/python
uv pip check --python /data/jxc/envs/rpas-aflow/bin/python
uv pip check --python /data/jxc/envs/rpas-maas/bin/python
```

The model and embedding revisions are fixed in `model_revisions.json`. Download
them before running the formal script, then verify the local files without
network access:

```bash
export HF_HOME=/data/jxc/cache/rpas/huggingface
export HF_XET_CACHE=/data/jxc/cache/rpas/xet
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY \
  -u https_proxy -u http_proxy -u all_proxy \
  /data/jxc/envs/rpas-serve/bin/python scripts/download_models.py
/data/jxc/envs/rpas-serve/bin/python scripts/verify_local_models.py \
  --report /data/jxc/tmp/rpas/model_verification.json
```

`scripts/verify_local_models.py` refuses to overwrite an existing report.
Move an old report aside before intentionally rechecking model files. The
formal runner requires the report path in `RPAS_MODEL_VERIFICATION_REPORT`.
The download mirror is a transport choice; the verifier compares local files
with pinned upstream digests.
