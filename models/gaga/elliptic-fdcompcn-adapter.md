# GAGA: Elliptic + FDCompCN adapter (non-native datasets)

GAGA natively supports only YelpChi and Amazon. Elliptic and FDCompCN
required a custom adapter, built on top of DGA-GNN's already-preprocessed
`.dgldata` files rather than re-sourcing raw data separately.

## Architecture

`data_utils.py`'s `load_graph()` was extended (originally had a `tfinance`-
only branch, now covers `tfinance`/`elliptic`/`fdcompcn`) to load directly
from:
- `models/dga-gnn/dga_gnn_upstream/data/processed/elliptic_of_amnet.dgldata`
- `models/dga-gnn/dga_gnn_upstream/data/processed/fdcompcn.dgldata`

This reuses DGA-GNN's provisional TR40 splits and preprocessing rather than
building yet another independent split — one less source of inconsistency
across models.

**FDCompCN note:** relations are NOT flattened. `n_relations` reflects the
actual 3 relations preserved in the DGA-GNN `.dgldata` file
(`invest_bc2bc`, `provide_bc2bc`, `sale_bc2bc`), matching how DGA-GNN
itself was benchmarked on this dataset.

## Three real bugs found and fixed in the adapter attempt

An earlier version of this adapter (`build_gaga_sequences.py`,
`load_custom_datasets.py` — since superseded, not used for final results)
tried a different approach: saving raw node features directly instead of
running them through GAGA's actual group-aggregation preprocessing. This
would have silently bypassed GAGA's core method (group aggregation is the
paper's central contribution) and produced a shape mismatch crash. Not
used; superseded by the `load_graph()` extension approach below.

**Bug 1 — Mask key names.** DGA-GNN's `.dgldata` files use
`trn_msk`/`val_msk`/`tst_msk`. The adapter's key-matching only checked for
`train_mask`/`train_masks` (etc.), missing DGA-GNN's actual naming
entirely. This silently fell through to a **hardcoded fallback split**:
first 40% of nodes by raw index as train, contiguous, unshuffled — not a
real split at all. Confirmed by inspecting T-Finance's early split
attempts: `train_nid` was `[0, 1, 2, ..., 15741]`, i.e. a perfectly
contiguous block. **All T-Finance runs made before this fix were invalid
and discarded** (never committed to git, so no history to clean up).
Fix: added `trn_msk`/`val_msk`/`tst_msk` to the alt-name lists in
`data_utils.py`. Verified fix: post-fix `train_nid` is scattered across
the full node-ID range, matching DGA-GNN's actual provisional split sizes
exactly (T-Finance: 15742/7792/15823).

**Bug 2 — `val_size` filename mismatch.** The provisional splits for
T-Finance/Elliptic/FDCompCN follow the team's TR40 convention
(train=0.4, val≈0.198), but `elliptic_paper.json`/`fdcompcn_paper.json`
were initially set to GAGA's own paper default (`val_size: 0.1`). Since
the sequence-file naming encodes `val_size` in the filename, training
looked for a `..._0.4_0.1_717...` file that either didn't exist or (worse)
pointed at a stale file left over from Bug 1/the abandoned raw-feature
approach. Fix: aligned config `val_size` to 0.198 to match the actual
generated split files.

**Bug 3 — Config key vocabulary mismatch.** `elliptic_paper.json` and
`fdcompcn_paper.json` were originally written with a different key set
entirely (`hop` instead of `n_hops`, `epochs` instead of `max_epochs`,
`hidden_dim` instead of `emb_dim`, `norm_type` instead of separate
`norm_feat`/`grp_norm` flags) — not derived from the proven working
`yelpchi_paper.json`/`amazon_paper.json` schema. This caused
`KeyError: 'n_hops'` inside `main_transformer.py`. Fix: rebuilt both
configs from the exact working schema, changing only `dataset` and
`val_size`. Verified identical key sets via direct diff before rerunning.

## Results (native/paper-style split, train=0.4, val≈0.198, seed=717)

**Elliptic:**
- AUC: 0.8817, F1-macro: 0.787, F1-fraud: 0.6053, AP: 0.5985
- Best epoch: 4 (fast convergence — worth a follow-up sanity check, not
  yet independently re-verified)

**FDCompCN:**
- AUC: 0.6657, F1-macro: 0.601, F1-fraud: 0.309, AP: 0.228
- Best epoch: 229
- Notably weak relative to other datasets. Consistent with DGA-GNN's own
  FDCompCN TR20/TR10 collapse (near-chance performance on small training
  splits) — may indicate FDCompCN is a genuinely hard, small dataset
  across multiple model architectures, not specific to GAGA. Worth a
  cross-model note in the final writeup rather than treating as a GAGA-
  specific weakness.

## Status: T-Finance

T-Finance preprocessing was started, confirmed the mask-key fix works
correctly (real split, not the broken fallback), but is very slow at this
graph's density (~2.2 nodes/sec via GAGA's own group-aggregation
bottleneck on 42.5M edges → ~5 hour estimated total). Deprioritized in
favor of Elliptic/FDCompCN. **Still needs to be run to completion.**
