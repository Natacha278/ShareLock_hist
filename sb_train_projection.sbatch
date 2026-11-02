#!/bin/bash
#SBATCH --job-name=sharelock-proj
#SBATCH --account=def-CHANGEME
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# ---------------------------------------------------------------- paths (edit these)
ROOT=/project/rrg-josedolz/natgill
PAIRS=$ROOT/data/quilt1m/quilt_pairs_t05_biolord.pt    # from build_quilt_pairs.py
TEST=$ROOT/data/patch_datasets/uni_2h_feat             # --out-dir of extract_uni2h_test_data.py
TXT_MODEL=$ROOT/weights/BioLORD-2023                   # same encoder as used for $PAIRS
RUNS=$ROOT/ShareLock_hist/runs
ENV=/home/natgill/envs/biomed                          # needs torch, sentence-transformers, scikit-learn
SCRIPT_DIR=$SLURM_SUBMIT_DIR                           # folder with train_projection.py & co.

# ---------------------------------------------------------------- experiment (edit these)
PROJECT=text        # text | image | both (both also needs --embed-dim)
DEPTH=4
WIDTH=2048
EPOCHS=30
LR=1e-3

# ---------------------------------------------------------------- environment
module load python/3.11
source "$ENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$SCRIPT_DIR"
cp "$PAIRS" "$SLURM_TMPDIR/"                          # local copy: faster to load

python train_projection.py \
    --pairs      "$SLURM_TMPDIR/$(basename "$PAIRS")" \
    --test-feats "$TEST" \
    --txt-model  "$TXT_MODEL" \
    --project    "$PROJECT" \
    --depth      "$DEPTH" \
    --width      "$WIDTH" \
    --epochs     "$EPOCHS" \
    --lr         "$LR" \
    --batch-size 4096 \
    --runs-dir   "$RUNS" \
    --run-name   "biolord_${PROJECT}_d${DEPTH}_w${WIDTH}"
