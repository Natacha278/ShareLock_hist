#!/bin/bash
#SBATCH --job-name=uni2h_quilt
#SBATCH --account=rrg-josedolz          
#SBATCH --array=4-10
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/uni2h_%A_%a.out

set -euo pipefail

QUILT=/scratch/quilt1m/quilt1m_zip/
CKPT=/project/rrg-josedolz/natgill/weights/uni2-h/pytorch_model.bin  # <-- where you saved the weights
OUT=/project/rrg-josedolz/natgill/data/quilt_1m_images/uni2h
PART=${SLURM_ARRAY_TASK_ID}
ZIP=images_part_${PART}_resized.zip   # adjust if your names use leading zeros

module load StdEnv/2023 python/3.11
cd /project/rrg-josedolz/natgill/ShareLock_hist/feat_precompute
source /home/natgill/envs/biomed/bin/activate

# Copy the zip to the node-local disk for fast reads
cp "$QUILT/$ZIP" "$SLURM_TMPDIR/"

python extract_uni2h.py \
    --zip "$SLURM_TMPDIR/$ZIP" \
    --ckpt "$CKPT" \
    --out "$OUT/uni2h_part_${PART}.npz"