# COMP8851 YelpChi adapter v1

This package extends the verified Amazon benchmark adapters to canonical
YelpChi without changing either model architecture.

It preserves all 45,954 nodes, 32 canonical features and the three semantic
relations (`net_rsr`, `net_rtr`, `net_rur`). It ignores DGL-generated masks and
uses the COMP8851 frozen TR40 split for the two-epoch compatibility tests.

Safeguards retained:

- validation AUROC checkpoint selection;
- validation-only threshold selection by Macro-F1;
- AUROC, AUPRC, Macro-F1, Fraud-F1 and G-Mean reporting;
- no test evaluation during training;
- a single test evaluation after compatibility training;
- bounded LEX-GNN initialization restarts;
- progress bars, timing, GPU-memory and artifact evidence.

Deploy from the extracted package:

```bash
bash scripts/deploy_and_preflight.sh
```

Run the models sequentially, beginning with LEX-GNN:

```bash
bash scripts/run_lexgnn_yelpchi_smoke.sh
```

Do not run ConsisGAD until the LEX-GNN output has been reviewed. The second
command, when approved, is:

```bash
bash scripts/run_consisgad_yelpchi_smoke.sh
```

These are compatibility results only. They are not final benchmark results.
