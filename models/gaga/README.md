# GAGA (Label Information Enhanced Fraud Detection, WWW '23)

Official repo: https://github.com/Orion-wyc/GAGA
Commit: see `gaga_commit.txt`
Paper: Wang et al., "Label Information Enhanced Fraud Detection against Low
Homophily in Graphs," WWW 2023

Natively supports YelpChi and Amazon only (auto-downloaded via DGL's
`FraudDataset`). T-Finance, Elliptic, and FDCompCN will need adapters,
same as was done for DGA-GNN.

## Environment: `gaga` conda env, Python 3.7

The env file (`gaga_env.yaml`) specifies `torch==1.7.1+cu110`,
`dgl==0.7.2`. Actual working setup:

- `torch==1.7.1+cu110` — installed cleanly, works fine on this A6000 (compute
  capability 8.6) via CUDA minor-version compatibility. Confirmed with a real
  GPU matmul test, not just `torch.cuda.is_available()`.
- `dgl==0.6.1` (not 0.7.2) — **the CUDA-enabled wheel for dgl 0.7.2 does not
  exist** on data.dgl.ai's package index; `dgl-cu110` tops out at 0.6.1.
  (`dgl==0.7.2` does exist but is CPU-only, which silently fails at
  `.to('cuda')` with "Device API gpu is not enabled.") Checked GAGA's actual
  DGL API usage (`to_homogeneous`, `to_simple`, `add_self_loop`,
  `heterograph`, `save_graphs`/`load_graphs`) — all stable APIs present in
  0.6.1, so this substitution is safe.
- CUDA 11.0-era libraries (`libcublas.so.11`, `libcusparse.so.11`, etc.) had
  to be installed explicitly via pip (`nvidia-cublas-cu11==11.10.3.66`,
  `nvidia-cusparse-cu11==11.7.4.91`, etc.) since dgl-cu110's compiled binary
  expects them as loose system libraries, unlike modern torch wheels which
  bundle their own. `LD_LIBRARY_PATH` must include all of these; set via a
  conda activation hook at `/venv/gaga/etc/conda/activate.d/gaga_ld_path.sh`
  so it's automatic on `conda activate gaga`.

Full installed package list: `yelpchi/evidence/environment/environment_installed.txt`

## Known bug in upstream preprocessing: `mp.Process` deadlock

`preprocessing/graph2seq_mp.py`'s multi-worker mode (`--n_workers > 1`) uses
`mp.Process` (fork-based multiprocessing) to parallelize sequence
generation. **This deadlocks on this instance** — all worker processes sit
at 0% CPU indefinitely (confirmed via `ps`/`top`, not just a slow progress
bar). Root cause: PyTorch/DGL initialize background threads (OpenMP thread
pools) at import time; forking a process after those threads exist can
leave the child with a corrupted copy of a lock held by a thread that no
longer exists in the fork. This is a known class of bug with
fork-based multiprocessing + PyTorch, not something specific to our setup.

**Fix:** `preprocessing/graph2seq_single.py` (new file, evidence copy at
`yelpchi/evidence/preprocessing/`) reimplements the same logic
single-process, in chunks of 2000 nodes, with progress/ETA logging. No
`mp.Process` call. Verified: full YelpChi preprocessing (45,954 nodes)
completes in ~178 seconds this way — actually faster than the intended
parallel version would have been, since it never got past 0%.

The bottleneck itself is a documented TODO in the upstream code
(`_group_aggregation` — comment reads "这部分是瓶颈所在" / "this part is the
bottleneck," left by the original author) but is not pathologically slow at
YelpChi's scale (~258 nodes/sec observed).

## Results: YelpChi, native paper split (train=0.4, val=0.1, test=0.5, seed=717)

Config: `configs/yelpchi_paper.json` (paper defaults, unmodified)

Training: 500 epochs (early stopping never fully triggered — counter
reached 85/100 at epoch 499), best checkpoint selected from epoch 414 based
on validation AUC.

**Final test-set metrics:**
- AUC: 0.9428
- F1-macro: 0.8313
- F1-fraud (binary-1): 0.7110
- F1-benign (binary-0): 0.9517
- Average Precision: 0.7952
- Gmean: 0.8232
- Precision(1): 0.7098, Recall(1): 0.7122
- TN=18734, FP=957, FN=946, TP=2341

Note: this uses GAGA's own native split (0.4/0.1/0.5), not the team's
unified TR40 split (0.4/0.198/0.402). AUC here (0.943) is meaningfully
lower than PC-GNN and DGA-GNN's YelpChi results (~0.977 both) — this is
plausibly a genuine architectural/protocol difference (GAGA's paper split,
different val ratio, single-seed run vs. others' more extensive validation)
rather than a bug; not adjusted or re-run to match expectations.

## What's next

1. Amazon (native, same setup as YelpChi should apply directly)
2. T-Finance, Elliptic, FDCompCN — need adapters (no native support)
3. Frozen COMP8851 TR40/30/20/10 splits for unified comparison, once
   available
