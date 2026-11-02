#!/bin/bash
#SBATCH --job-name=uni2h-feats
#SBATCH --account=rrg-josedolz
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# ---------------------------------------------------------------- paths (edit these)
DATA=/scratch/natgill/datasets_patch
WEIGHTS=/project/rrg-josedolz/natgill/weights/uni2-h/pytorch_model.bin
OUT=/project/rrg-josedolz/natgill/data/patch_datasets/uni_2h_feat
SCRIPT_DIR=$SLURM_SUBMIT_DIR                       # where extract_uni2h_features.py lives

BACH_SRC=$DATA/bach/ICIAR2018_BACH_Challenge                                 # folder containing .../Photos (or a .zip)
CRC_SRC=$DATA/crc7k/CRC-VAL-HE-7K.zip
BREAKHIS_SRC=$DATA/breakhis/BreaKHis_v1.tar.gz
MHIST_SRC=$DATA/mhist/images.zip
MHIST_CSV=$DATA/mhist/annotations.csv

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
stage "$BACH_SRC"     "$LOCAL/bach"
stage "$CRC_SRC"      "$LOCAL/crc7k"
stage "$BREAKHIS_SRC" "$LOCAL/breakhis"
stage "$MHIST_SRC"    "$LOCAL/mhist"
cp "$MHIST_CSV" "$LOCAL/mhist/"
du -sh "$LOCAL"/*

# ---------------------------------------------------------------- extract
python "$SCRIPT_DIR/extract_uni2h_test_data.py" \
    --weights       "$WEIGHTS" \
    --out-dir       "$OUT" \
    --bach-root     "$LOCAL/bach" \
    --crc-root      "$LOCAL/crc7k" \
    --mhist-root    "$LOCAL/mhist" \
    --breakhis-root "$LOCAL/breakhis" \
    --modes         resize,tile \
    --tile-size     224 \
    --batch-size    128 \
    --num-workers   "$SLURM_CPUS_PER_TASK"

echo "Done. Features in $OUT"
