# PlaceDreamer — HPC (Magnolia / Ole Miss MCSR, L40S) step-by-step

The one hard part: **you can't rsync**, and the cache is **970 MB**. Solution: ship it as ONE
tarball via Google Drive, pull it with `gdown` on the login node (which has internet).

---

## PART 0 — On your laptop (once)

### 0a. Push the code
```bash
cd ~/PlaceDreamer
git push origin main          # cache/, datasets/, venv/, *.tar.gz are all gitignored
```

### 0b. Package + upload the cache
The tarball is already built: **`placedreamer_cache.tar.gz` (970 MB)**.
(To rebuild: `tar czf placedreamer_cache.tar.gz cache/graphs cache/meta.parquet cache/norm.npz`)

**Already uploaded + verified public** (gdown pulls it with no auth prompt):

```
FILE_ID = 1eCi7g7alV9aHFy2hMXGsXip-KPIakS97
```

(If you ever re-upload: Drive → Share → **Anyone with the link** → the FILE_ID is the segment
between `/d/` and `/view` in the URL.)

---

## PART 1 — On the cluster login node (once)

```bash
# 1. clone
cd ~
git clone https://github.com/BarsatKhadka/PlaceDreamer.git
cd PlaceDreamer

# 2. env
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
python -m venv venv && source venv/bin/activate

# 3. ⚠️ CUDA-matched torch FIRST (default PyPI torch silently runs on CPU here!)
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('cuda visible:', torch.cuda.is_available())"   # MUST print True

# 4. the rest
pip install -r requirements.txt
pip install torch_geometric                    # PyG (pure python, no CUDA build needed)

# 5. get the cache (login node has internet)
pip install -U gdown
gdown 1eCi7g7alV9aHFy2hMXGsXip-KPIakS97 -O placedreamer_cache.tar.gz   # ~1.0 GB, a few min
tar xzf placedreamer_cache.tar.gz              # → cache/graphs/, cache/meta.parquet, cache/norm.npz
rm placedreamer_cache.tar.gz

# 6. VERIFY the cache landed intact
ls cache/graphs/*.npz | wc -l                  # must be 1944
python -c "import pandas as pd; m=pd.read_parquet('cache/meta.parquet'); print(len(m),'flows'); print(m.columns.tolist())"
```

**If `gdown` fails on the big file** (older versions choke on Drive's confirm token), use:
```bash
gdown --fuzzy "https://drive.google.com/file/d/1eCi7g7alV9aHFy2hMXGsXip-KPIakS97/view"
```

---

## PART 2 — Run

### 2a. Smoke test FIRST (2 min — never launch a 12h job blind)
```bash
srun --partition=gpuq --gres=gpu:1 --mem=32G --time=00:20:00 --pty bash
source venv/bin/activate
FOLD=0 EPOCHS=1 python src/train_fplace.py     # one fold, one epoch — does it run on GPU?
exit
```

### 2b. Full dev CV (the real run)
```bash
mkdir -p logs runs
sbatch slurm/train_fplace.sbatch                                    # DE-HNN, all 3 dev folds
squeue -u $USER
tail -f logs/fplace_<jobid>.out
```

### 2c. Ablations (the justification table — each is a separate job)
```bash
sbatch --export=ALL,ENCODER=sage             slurm/train_fplace.sbatch
sbatch --export=ALL,ENCODER=gat              slurm/train_fplace.sbatch
sbatch --export=ALL,ENCODER=dehnn_novn       slurm/train_fplace.sbatch
sbatch --export=ALL,ENCODER=dehnn_undirected slurm/train_fplace.sbatch
```

### 2d. Loss-weight sweep (congestion is the headline target)
```bash
sbatch --export=ALL,W_NETDEM=2,OUT=runs/dehnn_w2 slurm/train_fplace.sbatch
sbatch --export=ALL,W_NETDEM=4,OUT=runs/dehnn_w4 slurm/train_fplace.sbatch
```

---

## PART 3 — The LOCKED OOD evaluation (run ONCE, at the very end)

**Do NOT run this while tuning.** Only after you've picked the final encoder + hyperparameters
from the **dev CV** results.

```bash
python src/train_fplace.py --eval-ood     # OUT=runs/<chosen> to pick which checkpoints
```
OOD designs (**never trained on, never tuned on**): `jpeg` (largest → size extrapolation),
`aes_core`, `tv80`, `wb_dma`, `i2c`. **That number is the headline result.**

---

## Data sizes (what moves where)
| what | size | how |
|---|---|---|
| **code** | few MB | `git push` / `git clone` |
| **cache** (all f_place needs) | **970 MB** | tar → Google Drive → `gdown` |
| raw EDA-Schema sky130hd | 71 GB | **do NOT move it** — the cache already has everything |

## Gotchas (learned the hard way)
- **torch must come from the cu128 index** or it silently trains on CPU. The sbatch has a hard
  `assert torch.cuda.is_available()` so it fails loudly instead.
- `--exclude=node049` (bad ECC), same as MechRL.
- Don't `git add` the tarball (it's gitignored now — a 970MB blob permanently bloats the repo).
