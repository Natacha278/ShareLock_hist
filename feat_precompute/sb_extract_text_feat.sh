#!/bin/bash
#SBATCH --job-name=quilt-text-feats
#SBATCH --account=def-CHANGEME
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# ---------------------------------------------------------------- paths (edit these)
CSV=/scratch/quilt1m/quilt_1M_lookup.csv
WEIGHTS=/project/rrg-josedolz/natgill/weights            # same folder as probe_text_encoders.py
MODEL=BioLORD-2023                                       # any sentence-transformers folder in $WEIGHTS
OUT=/project/rrg-josedolz/natgill/features/quilt_text
ENV=~/envs/uni2h                                         # a venv with torch + sentence-transformers
SCRIPT_DIR=$SLURM_SUBMIT_DIR

# ---------------------------------------------------------------- environment
module load python/3.11
source "$ENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---------------------------------------------------------------- extract
python "$SCRIPT_DIR/extract_text_features.py" \
    --csv         "$CSV" \
    --model-path  "$WEIGHTS/$MODEL" \
    --out-dir     "$OUT" \
    --caption-col caption \
    --batch-size  512 \
    --clean-social

echo "Done. Features in $OUT/$MODEL.pt"