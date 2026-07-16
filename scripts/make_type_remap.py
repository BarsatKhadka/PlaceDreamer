#!/usr/bin/env python3
"""Build cache/type_remap.npz: cached cell_type id (row index) -> dense type id (0..440).

The sky130 standard_cells parquet has 5292 rows = the SAME 441 cells repeated 12x.
build_graph.py's load_cell_lib() enumerated ROWS, so cached ids are last-occurrence row
indices (4858..5290) and the type embedding was sized 5293 — 92% of it (310,528 params)
never received a gradient, and only 174 rows are ever used.

Run ONCE locally (needs datasets/). The .npz is committed, so the cluster (which has cache/
but not the 71GB datasets/) just gets it from git. No graph re-cache needed.
"""
import pyarrow.parquet as pq, numpy as np, glob, os
ROOT = os.environ.get("PD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

names = pq.read_table(f"{ROOT}/datasets/sky130hd/standard_cells/table.parquet",
                      columns=["name"]).to_pydict()["name"]
uniq  = sorted(set(names))
canon = {n: i for i, n in enumerate(uniq)}          # 441 unique, stable (sorted) order
UNK   = len(uniq)                                    # 441 = the UNK slot

old2new = np.full(len(names) + 1, UNK, np.int64)     # +1 covers the old UNK id (5292)
for row, n in enumerate(names):
    old2new[row] = canon[n]

np.savez(f"{ROOT}/cache/type_remap.npz", old2new=old2new,
         names=np.array(uniq), n_types=np.int64(len(uniq) + 1))
print(f"parquet rows {len(names)}  ->  unique types {len(uniq)}  (+1 UNK) = N_TYPES {len(uniq)+1}")

# verify against the cache: every id present must remap into range, and the mapping must be 1:1
used = set()
for p in sorted(glob.glob(f"{ROOT}/cache/graphs/*.npz"))[::37]:
    used |= set(np.load(p, allow_pickle=True)["cell_type"].tolist())
mapped = {int(old2new[i]) for i in used if i >= 0}
print(f"cached ids sampled: {len(used)} distinct, range {min(used)}..{max(used)}")
print(f"  -> remapped to {len(mapped)} distinct dense ids, range {min(mapped)}..{max(mapped)}")
assert len(mapped) == len(used), "remap collapsed distinct types — BUG"
assert max(mapped) < UNK, "remapped id out of range"
print(f"✓ 1:1, in range. params {5293*64:,} -> {(len(uniq)+1)*64:,} "
      f"(saved {(5293-len(uniq)-1)*64:,})")
