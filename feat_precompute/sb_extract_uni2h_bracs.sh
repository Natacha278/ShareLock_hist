#!/bin/bash
#SBATCH --job-name=uni2h-bracs
#SBATCH --account=rrg-josedolz
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# Same process as sb_extract_uni2h_test_data.sh, for the BRACS RoI set only.
# Writes $OUT/bracs/{metadata.csv,uni2h_resize.pt,uni2h_tile.pt} next to the other test sets.

# ---------------------------------------------------------------- paths (edit these)
DATA=/scratch/natgill/datasets_patch
WEIGHTS=/project/rrg-josedolz/natgill/weights/uni2-h/pytorch_model.bin
OUT=/project/rrg-josedolz/natgill/data/patch_datasets/uni_2h_feat
SCRIPT_DIR=$SLURM_SUBMIT_DIR                       # where extract_uni2h_test_data.py lives

# The RoI folder from the BRACS FTP ("Region of Interest Set" / BRACS_RoI), as a folder or a tar.
# A tar is faster to stage than thousands of PNGs on Lustre; the PNGs are already compressed, so:
#   tar -cf $DATA/bracs/BRACS_RoI.tar -C <parent of BRACS_RoI> BRACS_RoI
BRACS_SRC=$DATA/bracs/BRACS_RoI.tar

# ---------------------------------------------------------------- environment
module load python/3.11
source /home/natgill/envs/biomed/bin/activate
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---------------------------------------------------------------- stage data on local disk
LOCAL=$SLURM_TMPDIR/data
stage () {  # stage <dir-or-archive> <dest>
    local src=$1 dest=$2
    mkdir -p "$dest"
    case "$src" in
        *.zip)          unzip -q "$src" -d "$dest" ;;
        *.tar.gz|*.tgz) tar -xzf "$src" -C "$dest" ;;
        *.tar)          tar -xf  "$src" -C "$dest" ;;
        *)              cp -r "$src"/. "$dest"/ ;;
    esac
}
echo "Staging data to $LOCAL ..."
stage "$BRACS_SRC" "$LOCAL/bracs"
du -sh "$LOCAL"/*

# ---------------------------------------------------------------- extract
python "$SCRIPT_DIR/extract_uni2h_test_data.py" \
    --weights          "$WEIGHTS" \
    --out-dir          "$OUT" \
    --bracs-root       "$LOCAL/bracs" \
    --modes            resizecrop,resize,tile \
    --bracs-tile-size  448 \
    --batch-size       128 \
    --num-workers      "$SLURM_CPUS_PER_TASK"

echo "Done. Features in $OUT/bracs"
