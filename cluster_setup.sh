#!/bin/bash --login
# One-time: the CLM environment on the cluster. Not a job; run it once on a login node:
#
#   bash cluster_setup.sh
#
# Downloads vLLM + torch (~3 GB), Qwen/Qwen3-8B (16 GB), the released CLM head (75 MB) and the
# SWE-rebench trajectories (2 GB). CLM's code is cloned into external/CLM at a pinned commit (gitignored,
# Apache 2.0): precheck/run.py uses its token recipe (train/embed_utils.py) and head loader (src/clm/heads.py)
# unchanged, so the embeddings match what the released head was trained against.
set -euo pipefail
module load cuda/13.0.1
ENV="${CLM_ENV:-/work/strebl/clm_env}"
CLM_COMMIT=bb42c6c5bf914fd449bed2f6ca65be80602cb1f7
cd /work/strebl/CLM_test
export XDG_CACHE_HOME="/work/strebl/.cache" HF_HOME="/work/strebl/.cache/huggingface" CLM_CKPT_DIR="/work/strebl/.cache/clm"

[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.12
# `bash cluster_setup.sh` ignores the --login shebang, so conda's shell function is not defined: load it
eval "$(conda shell.bash hook)"
set +u; conda activate "$ENV"; set -u      # activation scripts may read unset variables
pip install --upgrade pip
# runner="pooling" (CLM's OfflineBackend) needs a recent vLLM; its wheel brings a matching torch
pip install "vllm>=0.11" pyarrow "huggingface_hub[cli]"

[ -d external/CLM ] || git clone https://github.com/Contrastive-LM/CLM external/CLM
git -C external/CLM fetch --quiet origin
git -C external/CLM checkout --quiet "$CLM_COMMIT"
pip install -e external/CLM --no-deps

hf download Qwen/Qwen3-8B
python -c "from clm.heads import download; print(download())"
mkdir -p exps/precheck/data
[ -f exps/precheck/data/trajectories.parquet ] || \
  hf download nebius/SWE-rebench-openhands-trajectories trajectories.parquet --repo-type dataset --local-dir exps/precheck/data

python - <<'PY'
import sys, torch, vllm, transformers
sys.path[:0] = ["external/CLM/src", "external/CLM/train"]
import embed_utils
from clm.heads import HeadPair, default_checkpoint
hp = HeadPair("clm-latest", default_checkpoint(), device="cpu").ensure()
print("torch", torch.__version__, "| vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| head", default_checkpoint(), hp.n_params, "params, scale", hp.scale)
PY
python -m unittest discover tests
echo "=== setup complete: $ENV ==="
