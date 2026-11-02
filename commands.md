# Env Loading
```bash
module load StdEnv/2023 python/3.11
cd /project/rrg-josedolz/natgill/ShareLock_hist
source /home/natgill/envs/biomed/bin/activate
```

# Env modules

```bash
# Create env
virtualenv --no-download /home/natgill/envs/biomed

pip install --no-index --upgrade pip
pip install --no-index torch transformers accelerate sentence-transformers huggingface_hub
pip install --no-index requests

# New install for uni feat precomputing
pip install --no-index torch torchvision timm huggingface_hub numpy pillow
```

# Uni2h precomputing

Download weights
```bash
export HF_TOKEN=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
python -c "from huggingface_hub import hf_hub_download; \\
hf_hub_download('MahmoodLab/UNI2-h', 'pytorch_model.bin', \\
local_dir='/project/rrg-josedolz/natgill/weights/uni2-h')"
```

Test feat computing
```bash
salloc --gpus=h100:1 --cpus-per-task=16 --mem=32G --time=01:00:00 --account=rrg-josedolz

  python extract_uni2h.py --zip /scratch/natgill/quilt1m/quilt1m_zip/images_part_10_resized.zip \
  --ckpt /project/rrg-josedolz/natgill/weights/uni2-h/pytorch_model.bin --out /project/rrg-josedolz/natgill/data/quilt_1m_images/uni2h_part_10.npz

# uni2h_part_2.npz
```

# Language model weights

```bash
hf download FremyCompany/BioLORD-2023 --local-dir /project/rrg-josedolz/natgill/weights/BioLORD-2023
hf download BioMistral/BioMistral-7B  --local-dir /project/rrg-josedolz/natgill/weights/BioMistral-7B

hf download Qwen/Qwen3-Embedding-8B --local-dir /project/rrg-josedolz/natgill/weights/Qwen3-Embedding-8B
```

# BioLord precomputing

```Bash
salloc --gpus=h100:1 --cpus-per-task=16 --mem=32G --time=01:00:00 --account=rrg-josedolz

python "extract_text_features.py" \
    --csv         /scratch/natgill/quilt1m/quilt_1M_lookup.csv \
    --model-path  "/project/rrg-josedolz/natgill/weights/BioLORD-2023" \
    --out-dir     "/project/rrg-josedolz/natgill/data/quilt_1m_text" \
    --caption-col caption \
    --batch-size  512 \
    --clean-social
```

# Quilt filter

```Bash
git clone https://github.com/DeepMicroscopy/QuiltCleaner.git
python feat_precompute/quilt_filter.py --qc-dir /project/rrg-josedolz/natgill/ShareLock_hist/feat_precompute/QuiltCleaner \
    --quilt-csv /scratch/natgill/quilt1m/quilt_1M_lookup.csv \
    --out feat_precompute/quilt_clean_images_trial.csv
```

```Bash
python build_quilt_pairs.py \
  --kept-csv  quilt_clean_images_t05.csv \
  --img-feats "/project/rrg-josedolz/natgill/data/quilt1m/quilt_1m_images/uni2h_part_*.npz" \
  --txt-feats /project/rrg-josedolz/natgill/data/quilt1m/quilt_1m_text/BioLORD-2023.pt \
  --out       /project/rrg-josedolz/natgill/data/quilt1m/quilt_pairs_t05_biolord.pt

# {
  #     "img_feats": img_feats, "img_names": have,
  #     "txt_feats": txt_feats, "texts": texts,
  #     "pair_img": pair_img, "pair_txt": pair_txt,
  #     "img_group": groups, "img_is_val": val, "meta": meta,
  # }
```

# Patch evaluation datasets

Download raw data
```bash
mkdir -p /scratch/natgill/datasets && cd /scratch/natgill/datasets

#bach
mkdir bach && cd bach
wget -c "https://zenodo.org/records/3632035/files/ICIAR2018_BACH_Challenge.zip?download=1" -O ICIAR2018_BACH_Challenge.zip
cd ..

# CRC7k
mkdir crc7k && cd crc7k
wget -c "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1" -O CRC-VAL-HE-7K.zip
cd ..

# BreakHis
mkdir breakhis && cd breakhis
wget -c http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz
cd ..

# MHIST (link only vaid for 4h)
```

Precompute features
```Bash
DATA=/scratch/natgill/datasets_patch
WEIGHTS=/project/rrg-josedolz/natgill/weights/uni2-h/pytorch_model.bin
OUT=/project/rrg-josedolz/natgill/data/patch_datasets/uni_2h_feat
SCRIPT_DIR=$SLURM_SUBMIT_DIR                       # where extract_uni2h_features.py lives

BACH_SRC=$DATA/bach/ICIAR2018_BACH_Challenge                                 # folder containing .../Photos (or a .zip)
CRC_SRC=$DATA/crc/CRC-VAL-HE-7K.zip
BREAKHIS_SRC=$DATA/breakhis/BreaKHis_v1.tar.gz
MHIST_SRC=$DATA/mhist/images.zip
MHIST_CSV=$DATA/mhist/annotations.csv

python "extract_uni2h_test_data.py" \
    --weights       "/project/rrg-josedolz/natgill/weights/uni2-h/pytorch_model.bin" \
    --out-dir       "/project/rrg-josedolz/natgill/data/patch_datasets/uni_2h_feat" \
    --bach-root     "/scratch/natgill/datasets_patch/bach/ICIAR2018_BACH_Challenge" \
    --crc-root      "$CRC_SRC" \
    --mhist-root    "$DATA/mhist" \
    --breakhis-root "$DATA/breakhis/BreaKHis_v1.tar.gz" \
    --modes         resize,tile \
    --tile-size     224 \
    --batch-size    128 \
    --num-workers   "$SLURM_CPUS_PER_TASK"
```

Download ARCH dataset and precompute

```Bash
ROOT=/project/rrg-josedolz/natgill
ARCH=/scratch/natgill/datasets_patch/arch/pubmed_set                 
SUBSET=pubmed
UNI2H=$ROOT/weights/uni2-h/pytorch_model.bin
TXT_MODEL=$ROOT/weights/BioLORD-2023                  
OUT=$ROOT/data/patch_datasets/uni_2h_feat                                  
SCRIPT_DIR=$SLURM_SUBMIT_DIR

cp -r "$ARCH" "$SLURM_TMPDIR/arch"

python "extract_arch_features.py" \
    --arch-root     "$SLURM_TMPDIR/arch" \
    --subset        "$SUBSET" \
    --uni2h-weights "$UNI2H" \
    --txt-model     "$TXT_MODEL" \
    --out-dir       "$OUT" \
    --num-workers   "$SLURM_CPUS_PER_TASK"
```

# Train projection

```Bash
salloc --gpus=h100:1 --cpus-per-task=16 --mem=32G --time=01:00:00 --account=rrg-josedolz

module load python/3.11
source /home/natgill/envs/biomed/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /project/rrg-josedolz/natgill/ShareLock_hist

ROOT=/project/rrg-josedolz/natgill
ARCH_FEATS=$ROOT/data/patch_datasets/uni_2h_feat/arch/arch_pubmed.pt
# local copy, loads faster
cp $ROOT/data/quilt1m/quilt_pairs_t05_biolord.pt $SLURM_TMPDIR/ 
# check the GPU is visible    
nvidia-smi                                

# Trial
python train_projection.py \
  --pairs $SLURM_TMPDIR/quilt_pairs_t05_biolord.pt \
  --test-feats $ROOT/data/patch_datasets/uni_2h_feat \
  --txt-model $ROOT/weights/BioLORD-2023 \
  --runs-dir $ROOT/ShareLock_hist/runs \
  --epochs 1 --run-name smoke

# 1 run both
python train_projection.py \
    --pairs $SLURM_TMPDIR/quilt_pairs_t05_biolord.pt \
    --test-feats $ROOT/data/patch_datasets/uni_2h_feat \
    --arch-feats "$ARCH_FEATS" \
    --txt-model $ROOT/weights/BioLORD-2023 \
    --runs-dir $ROOT/ShareLock_hist/runs \
    --project both --embed-dim 1536 \
    --depth 4 --width 4096 --epochs 10 \
    --run-name biolord_both_d4_w4096_e10

# 1 run
python train_projection.py \
    --pairs $SLURM_TMPDIR/quilt_pairs_t05_biolord.pt \
    --test-feats $ROOT/data/patch_datasets/uni_2h_feat \
    --arch-feats "$ARCH_FEATS" \
    --txt-model $ROOT/weights/BioLORD-2023 \
    --runs-dir $ROOT/ShareLock_hist/runs \
    --project text \
    --depth 4 --width 4096 --epochs 10 \
    --run-name biolord_text_d4_w4096_e10

# Multiple depth and width TEXT
for D in 2 4; do for W in 1024 2048; do
  python train_projection.py \
    --pairs $SLURM_TMPDIR/quilt_pairs_t05_biolord.pt \
    --test-feats $ROOT/data/patch_datasets/uni_2h_feat \
    --arch-feats "$ARCH_FEATS" \
    --txt-model $ROOT/weights/BioLORD-2023 \
    --runs-dir $ROOT/ShareLock_hist/runs \
    --project text --depth $D --width $W --epochs 30 \
    --run-name biolord_text_d${D}_w${W}
done; done

# Multiple depth and width BOTH
for D in 2 4; do for W in 1024 2048; do
  python train_projection.py \
    --pairs $SLURM_TMPDIR/quilt_pairs_t05_biolord.pt \
    --test-feats $ROOT/data/patch_datasets/uni_2h_feat \
    --arch-feats "$ARCH_FEATS" \
    --txt-model $ROOT/weights/BioLORD-2023 \
    --runs-dir $ROOT/ShareLock_hist/runs \
    --project both --embed-dim 1536 \
    --depth $D --width $W --epochs 30 \
    --run-name biolord_text_d${D}_w${W}
done; done

# compare runs
column -s, -t < $ROOT/runs/summary.csv | less -S     
```