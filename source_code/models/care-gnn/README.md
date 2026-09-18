# CARE-GNN

Camouflage-resistant, relation-aware fraud detection for the COMP8851 benchmark.

| Field | Value |
|---|---|
| Paper | Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters. CIKM. |
| DOI | https://doi.org/10.1145/3340531.3411903 |
| Official code | https://github.com/YingtongDou/CARE-GNN |
| Upstream commit | **record before the first controlled run** (`git rev-parse HEAD` in a clone of the official repo) |
| Owner | Akib Hasan Maruf |
| Reviewer | Ishraq Ahmed |
| Native datasets | YelpChi, Amazon |
| Protocol | COMP8851 v4.4 (`vast-v4.4`) |

## What is implemented

This is a re-implementation, not a vendored copy: the official repository is not
copied into this tree (see the repository-safety rule in the run guide). All
three published mechanisms are present and none is simplified.

| Mechanism | Where | Paper reference |
|---|---|---|
| Label-aware similarity measure | `caregnn/layers.py::filter_neighs_ada_threshold` | Eq. (2) |
| Similarity-aware neighbour selector (top-p) | same function | Section 3.3.1 |
| RL adaptive threshold module | `caregnn/layers.py::rl_module` | Eqs. (5)-(6) |
| Intra-relation aggregation | `caregnn/layers.py::IntraAgg` | Eq. (8) |
| Inter-relation aggregation (GNN/Att/Weight/Mean) | `caregnn/layers.py` | Eq. (9) |
| Combined GNN + similarity loss | `caregnn/model.py::OneLayerCARE.loss` | Eq. (11) |

### Deliberate deviations from the published source

Each of these is behaviour-preserving or a bug fix, and each is listed so a
reviewer can check it rather than discover it.

1. **Device handling.** Tensors are created on the module's device instead of
   through hard-coded `.cuda()` calls, so CPU and GPU share one code path.
2. **`att_inter_agg` debugger call removed.** The official function contains a
   stray `pdb.set_trace()` that halts execution. The surrounding mathematics is
   unchanged.
3. **RL thresholds are checkpointed.** Upstream keeps the learned thresholds in
   a plain Python list, so `state_dict` does not carry them and restoring the
   best checkpoint silently resets every threshold to 0.5, changing both the
   neighbour filtering and the Eq. (9) aggregation weights. Here they live in a
   registered buffer, so a restored checkpoint reproduces exactly what was
   validated. `shared/comp8851/selftest.py` asserts this.
4. **Batch count passed explicitly.** The RL update schedule is identical; the
   count is passed in rather than read from a mutated attribute mid-forward.

## Layout

```
models/care-gnn/
├── README.md
├── caregnn/              model code (importable package)
│   ├── layers.py         aggregators, neighbour selector, RL module
│   └── model.py          OneLayerCARE and build_care_gnn
├── configs/
│   ├── author/           official repository defaults
│   └── unified/          protocol v4.4 configurations
├── env/                  dependency pins
└── scripts/
    ├── run_one.py        one (dataset, track, ratio, seed) cell
    └── smoke.py          two-epoch feasibility wrapper
```

## Dependencies

CARE-GNN needs **no DGL**. It consumes the canonical `.mat` release directly
through `shared.comp8851.datasets`, so its environment is just PyTorch, NumPy,
SciPy and scikit-learn. See `env/requirements.txt`.

### Data source: same archive, different reader

The protocol names `FraudYelpDataset` / `FraudAmazonDataset` as the canonical
source for YelpChi and Amazon. This implementation reads `YelpChi.mat` and
`Amazon.mat` **from those same DGL-hosted archives** — the URLs under
`data.dgl.ai` recorded in `dataset_manifest.json` — rather than calling DGL's
Python loader.

The bytes are identical: `source_sha256` in every run's manifest is the hash of
the archive DGL itself downloads. Only the reader differs, and that difference
is deliberate — it removes a heavyweight dependency from a model that otherwise
has no use for it, and it keeps CARE-GNN runnable on hosts where no DGL wheel
exists.

This is a documented loader substitution, not a data substitution. Anyone
re-deriving these results through `dgl.data.FraudYelpDataset` will obtain the
same graph, features and labels.

## Running

```bash
# Two-epoch feasibility check (metrics are NOT benchmark evidence)
python models/care-gnn/scripts/run_one.py \
    --dataset yelpchi --data-path data/yelpchi --smoke

# One controlled unified-track cell
python models/care-gnn/scripts/run_one.py \
    --dataset yelpchi --data-path data/yelpchi \
    --ratio TR40 --seed 2 --track unified --require-cuda \
    --hardware-profile-id a6000-ref-v1

# The whole matrix, resumable, through the shared queue
python shared/comp8851/run_matrix.py plan --models CARE-GNN --ratios TR40 TR30 TR20 TR10
python shared/comp8851/run_matrix.py run --host-id a6000-ref-v1
```

Author-mode reproduction uses the official hyperparameters in
`configs/author/` and its own results directory:

```bash
python models/care-gnn/scripts/run_one.py \
    --dataset yelpchi --data-path data/yelpchi \
    --track author --epochs 31 --seed 72 --batch-size 1024
```

## Dataset compatibility

CARE-GNN's mechanism is inter-relation aggregation over a *multi-relational*
graph. That determines the expected status of each pair, which the compatibility
matrix must confirm with an actual two-epoch attempt rather than assume.

| Dataset | Relations | Expected status | Reason |
|---|---|---|---|
| YelpChi | R-U-R, R-T-R, R-S-R | PASS | native |
| Amazon | U-P-U, U-S-U, U-V-U | PASS | native |
| FDCompCN | C-I-C, C-P-C, C-S-C | PASS_WITH_SHARED_ADAPTER | genuinely multi-relational; mechanism intact |
| T-Finance | single | CONDITIONAL (A2) | inter-relation aggregation degenerates to one channel |
| T-Social | single | CONDITIONAL (A2) + scale risk | same, plus per-batch neighbour filtering scales poorly |
| Elliptic | single | CONDITIONAL (A2) | single-relation and temporal; no temporal mechanism in CARE-GNN |

The runner prints a compatibility note whenever it is handed a single-relation
graph. A degenerate result is a finding to interpret, not a failure to hide.

## Status

| Item | State |
|---|---|
| Implementation | COMPLETE |
| Offline self-test | PASS (`shared/comp8851/selftest.py`) |
| Local CPU validation, YelpChi | PASS, test AUROC 0.783 / AUPRC 0.401 at TR40, 25 epochs |
| Local CPU validation, Amazon | PASS, test AUROC 0.928 / AUPRC 0.838 at TR40, 25 epochs |
| Controlled Vast.ai benchmark | **NOT RUN** |

The two local numbers land in the range the paper reports, which is why the
implementation is considered validated. They were produced on CPU under
`--run-mode local-cpu-validation` and are **local author-code validation**, not
benchmark results: they are not comparable with any Vast.ai timing or metric and
must not enter the central comparison table.
