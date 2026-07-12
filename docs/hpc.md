# PlaceDreamer — HPC / training port

## Environment
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# torch + torch_geometric must match the cluster CUDA — see notes in requirements.txt:
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cuXXX   # your CUDA
pip install torch_geometric
```

## Data — you almost certainly do NOT need the 63 GB dataset on HPC

**For f_place training, transfer the CACHE (~1 GB), not the raw dataset.** The cache
(`cache/graphs/*.npz` + `cache/meta.parquet`) is self-contained: graph structure + features +
per-net labels (hpwl, RUDY demand) + knobs. Built locally by `scripts/cache_graphs.py`.

```bash
# from the laptop → HPC (this is all f_place needs):
rsync -avz cache/  user@hpc:~/placedreamer/cache/
rsync -avz scripts/ docs/  user@hpc:~/placedreamer/
```

## If you DO need the raw EDA-Schema-V2 on HPC (f_route labels, other stages, new features)

Only then pull the dataset. Options, best first:

1. **rsync what you already have** (recommended — you downloaded it once, no re-download):
   ```bash
   # transfer just the tables you need (each is GB-scale; skip images):
   rsync -avz --exclude='*/images/' datasets/sky130hd/  user@hpc:~/placedreamer/datasets/sky130hd/
   ```
2. **Re-download from source on HPC:**
   - **V1 (sky130-only, scriptable, works headless):**
     ```bash
     gdown --folder "https://drive.google.com/drive/folders/1B3rBvbnviBrKw1aLRpv7e1pEXSCy_vLQ" \
           -O datasets/eda_schema_v1 --remaining-ok
     ```
   - **V2 (full, 4 PDKs) — OneDrive, browser-auth-gated:** headless download is painful.
     Either (a) download on a workstation and rsync up, or (b) `rclone` a SharePoint remote against
     the share link in `paperCodes`/the repo README. Link (V2):
     `https://drexel0-my.sharepoint.com/:f:/g/personal/ps937_drexel_edu/IgDSNDP4cARmQIbi2bweaNZKAcbC5cYWopZ6A9jAN_lDdng`
   - **Trim images after unzip** (~26 GB, not needed — we're position-free):
     `find datasets/sky130hd -type d -name images -exec rm -rf {} +`

## Adding features on HPC without re-parsing graphs
The cache stores `cell_names`/`net_names`. To add a feature (e.g. per-instance power from the
`gates` table): read the table, join by `cell_names`, append a column to `cell_x`. No graph rebuild,
no full dataset needed if the feature is already a cached label — else pull just that table (option 1).

## Sizes (so you know what you're moving)
- **cache**: ~1 GB  ← move this for f_place
- raw sky130hd (images trimmed): ~71 GB (of which ~55 GB is timing data for the deferred WNS head)
- raw sky130hd full (with images): ~96 GB
