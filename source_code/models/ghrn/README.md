# GHRN

Spectral heterophily reduction for graph anomaly detection, for the COMP8851 benchmark.

| Field | Value |
|---|---|
| Paper | Gao, Y., et al. (2023). Addressing Heterophily in Graph Anomaly Detection: A Perspective of Graph Spectrum. The Web Conference. |
| DOI | https://doi.org/10.1145/3543507.3583268 |
| Official code | https://github.com/blacksingular/GHRN |
| Upstream commit | **record before the first controlled run** |
| Backbone paper | Tang, J., Li, J., Gao, Z., & Li, J. (2022). Rethinking Graph Neural Networks for Anomaly Detection. ICML, PMLR 162, 21076-21089. |
| Backbone code | https://github.com/squareRoot3/Rethinking-Anomaly-Detection |
| Owner | Akib Hasan Maruf |
| Reviewer | Pathik Ahmed |
| Native datasets | YelpChi, Amazon, T-Finance, T-Social |
| Protocol | COMP8851 v4.4 (`vast-v4.4`) |

## What is implemented

GHRN keeps BWGNN's beta-wavelet filter bank unchanged and adds a graph
refinement stage that deletes likely heterophilic edges before message passing.
Reproducing the backbone faithfully matters here: it is what makes the GHRN
result differ from BWGNN by the refinement step alone, which is the comparison
the paper makes.

| Component | Where |
|---|---|
| Beta-wavelet coefficients | `ghrnlib/bwgnn_backbone.py::calculate_theta2` |
| Polynomial spectral filter | `ghrnlib/bwgnn_backbone.py::PolyConv` |
| Backbone network | `ghrnlib/bwgnn_backbone.py::BWGNNBackbone` |
| Edge heterophily score (post-aggregation residual) | `ghrnlib/graph_refine.py::random_walk_update` |
| Edge heterophily score (KL variant) | `ghrnlib/graph_refine.py::kl_update` |
| Pruning rule | `ghrnlib/graph_refine.py::select_edges_to_delete` |
| Two-stage training wrapper | `ghrnlib/model.py::GHRN` |

### The mechanism, in one paragraph

Heterophily is positively associated with graph frequency, so an edge joining
two different classes carries high-frequency energy of the label signal. GHRN
estimates that per edge using *predicted* posteriors, since test labels are
unavailable: it aggregates posteriors over neighbours with degree normalisation
to get the low-pass component, takes the residual `ly = pred_y - ay` as the
high-pass component at each node, and scores each edge by `ly_src · ly_dst`.
Endpoints whose residuals point in opposite directions give a negative score and
indicate an inter-class edge. The lowest-scoring `del_ratio` fraction is deleted,
self-loops are protected, and the backbone is retrained on the refined graph.

### Two-stage training

The official repository requires a `del_ratio = 0` run first, to generate the
predictions that drive pruning. That is reproduced here inside one invocation:

* **Stage 1** trains the backbone on the unrefined graph. This is a
  BWGNN-equivalent run and its posteriors feed the refinement.
* **Stage 2** refines the graph, re-initialises the backbone and retrains.

`--del-ratio 0` runs stage 1 only, giving the BWGNN baseline the paper compares
against. Stage 2's checkpoint is the one evaluated on test.

Only training-visible information reaches the refinement: the posteriors come
from a model that saw training labels alone, and the canonical dataset on disk
is never modified. The pruning diagnostic in the run record is restricted to
training-node edges for the same reason.

### Reading stage 1 against stage 2

`summary.json` records both stages, so every GHRN run carries its own ablation:
stage 1 is the BWGNN-equivalent baseline and stage 2 is GHRN proper, trained on
identical splits, seeds and evaluator.

**Stage 2 is always the reported GHRN result when `del_ratio > 0`,** even when
stage 1 scored higher on validation. Selecting whichever stage won would mean
publishing a BWGNN number under the GHRN name. If refinement does not help on a
dataset, that is the finding, and it is reportable as it stands.

The expected pattern is that refinement helps most where there is heterophily to
remove. On a strongly homophilic graph there are few inter-class edges to find,
so pruning mostly discards useful ones. Amazon is the extreme case in this
benchmark: global heterophily 0.051, against 0.227 for YelpChi. Read a flat or
negative refinement effect there as consistent with the mechanism, not as a bug,
and check `refinement.edge_heterophily_before/after` in the run record to see
whether pruning actually removed inter-class edges at all.

`del_ratio` is a legitimate tunable under the protocol's budget (up to 12 TR40
trials, selected on validation AUPRC, then frozen for the lower ratios). If 0
wins that search on some dataset, say so explicitly rather than reporting the
degenerate model as though refinement had been applied.

### On the "no pruning" protocol rule

The master plan forbids "rewiring or pruning topology to raise performance". That
rule targets *benchmark-level* data manipulation. GHRN's pruning is the model's
own published mechanism, applied inside the model, identically for every dataset
and ratio, without touching the stored dataset. It is architecture, not data
preparation. This is recorded here so a reviewer does not have to adjudicate it
mid-review.

### Deliberate deviations from the published source

1. **Filter bank in an `nn.ModuleList`.** Upstream holds the filters in a plain
   Python list, so their unused projections were never registered. Behaviour is
   identical; the reported parameter count is now truthful. The unused projection
   is only created when it is actually applied.
2. **Graph backend abstraction.** The four graph operations GHRN needs are
   behind `ghrnlib/backend.py`, with a DGL implementation (the authors' stack,
   used by default and required for controlled runs) and a `torch.sparse`
   implementation used where DGL has no wheel. See the warning below.

## The torch-sparse fallback backend

DGL publishes no wheel for some platforms, including Python 3.13 on Windows. To
keep GHRN testable there, `ghrnlib/backend.py` implements the same mathematics on
`torch.sparse`. The runner prints which backend is active and records it in
`run_config.yml`.

**Controlled benchmark runs must use DGL.** Before the first final run on the
benchmark host, run `backend.compare_backends(...)` once and store the report
with the run evidence. If the two backends ever disagree, the DGL result is
authoritative.

## Layout

```
models/ghrn/
├── README.md
├── ghrnlib/
│   ├── backend.py          graph ops: DGL and torch.sparse
│   ├── bwgnn_backbone.py   beta-wavelet filter bank
│   ├── graph_refine.py     edge scoring and pruning
│   └── model.py            GHRN two-stage wrapper
├── configs/{author,unified}/
├── env/
└── scripts/{run_one.py,smoke.py}
```

## Running

```bash
# Two-epoch feasibility check
python models/ghrn/scripts/run_one.py \
    --dataset tfinance --data-path data/tfinance/tfinance --smoke

# BWGNN-equivalent baseline (stage 1 only)
python models/ghrn/scripts/run_one.py \
    --dataset tfinance --data-path data/tfinance/tfinance \
    --ratio TR40 --seed 2 --del-ratio 0 --require-cuda

# Full GHRN with refinement
python models/ghrn/scripts/run_one.py \
    --dataset tfinance --data-path data/tfinance/tfinance \
    --ratio TR40 --seed 2 --del-ratio 0.015 --require-cuda \
    --hardware-profile-id a6000-ref-v1
```

## Dataset compatibility

GHRN is a homogeneous spectral method, so multi-relational datasets are handled
through the frozen deterministic union view built by
`shared.comp8851.datasets.CanonicalDataset.homogeneous_adjacency`. That is an A1
shared adapter, documented and identical for every model that needs it.

| Dataset | Expected status | Reason |
|---|---|---|
| T-Finance | PASS | native, homogeneous |
| T-Social | PASS, memory permitting | native; OOM after a real smoke test is a recorded feasibility outcome, never a reason to shrink the graph |
| YelpChi | PASS_WITH_SHARED_ADAPTER | multi-relational, receives the union view |
| Amazon | PASS_WITH_SHARED_ADAPTER | multi-relational, receives the union view |
| FDCompCN | PASS_WITH_SHARED_ADAPTER | multi-relational, receives the union view |
| Elliptic | PASS_WITH_SHARED_ADAPTER, temporal | 166-feature locked view, chronological split |

## Status

| Item | State |
|---|---|
| Implementation | COMPLETE |
| Offline self-test | PASS on the torch-sparse backend (`shared/comp8851/selftest.py`) |
| DGL backend exercised | **NOT RUN** locally: DGL has no Python 3.13 / Windows wheel |
| Backend equivalence check | **NOT RUN**: needs a host where DGL installs |
| Local CPU validation, Amazon | PASS, test AUROC 0.9726 / AUPRC 0.8870 at TR40, del_ratio 0.015, both stages |
| Controlled Vast.ai benchmark | **NOT RUN** |

The Amazon figure is in the range the paper reports, which is the evidence the
implementation is correct. It ran on CPU under `--run-mode local-cpu-validation`
on the torch-sparse backend, so it is **local author-code validation**, not a
benchmark result, and must not enter the central comparison table.

Before any final run, on the benchmark host: install DGL from the CUDA 12.1
wheel index, re-run `shared/comp8851/selftest.py` so the DGL path is covered,
and record `compare_backends` output.
