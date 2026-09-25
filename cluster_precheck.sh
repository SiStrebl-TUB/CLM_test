#!/bin/bash --login
#SBATCH --job-name=CLMPrecheck
#SBATCH --output=./logs/precheck_%j.out
#SBATCH --error=./logs/precheck_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="ccc:80|ccc:86|ccc:89|ccc:90"
#SBATCH --mail-type=end
#SBATCH --mail-user=strebl@campus.tu-berlin.de

# Pre-check: the released CLM head on 2000 agent steps it has not been trained on (resolved SWE-rebench
# OpenHands trajectories, not among CLM's ADP post-training sources). Per step the taken action competes
# against 10 negatives of one kind: random actions of other tasks (sanity check), neighbouring steps of the
# same trajectory, near-miss mutations of the call. 2 state budgets (2048 / 8192 tokens) x 2 action formats
# (turn / call), scored by the CLM head, the raw encoder and a lexical baseline.
# Qwen3-8B in bf16 needs a GPU with >= 24 GB (no ccc:75). Run cluster_setup.sh once before.
# Output: exps/precheck/<tag>.json, <tag>_items.jsonl.gz, <tag>_examples.md; tables: python precheck/analyze.py
module load cuda/13.0.1
conda activate "${CLM_ENV:-/work/strebl/clm_env}"
export XDG_CACHE_HOME="/work/strebl/.cache" HF_HOME="/work/strebl/.cache/huggingface" CLM_CKPT_DIR="/work/strebl/.cache/clm"
cd /work/strebl/CLM_test
export PYTHONPATH=.
mkdir -p logs exps/precheck
python -c "import ast; [ast.parse(open(f).read()) for f in ('precheck/run.py', 'precheck/data.py', 'precheck/mutate.py')]" || { echo "scripts do not parse"; exit 1; }
[ -f external/CLM/train/embed_utils.py ] || { echo "external/CLM missing: run cluster_setup.sh"; exit 1; }
[ -f exps/precheck/data/trajectories.parquet ] || { echo "dataset missing: run cluster_setup.sh"; exit 1; }
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), ${MEM} MiB, on $(hostname)"
[ "$MEM" -ge 23000 ] || { echo "GPU has less than 24 GB: Qwen3-8B does not fit"; exit 1; }
python -m unittest discover tests || { echo "unit tests fail"; exit 1; }
TAG=${1:-v1}
python -u precheck/run.py --parquet exps/precheck/data/trajectories.parquet --n-states 2000 --steps-per-traj 2 --k 10 --tag "$TAG"
