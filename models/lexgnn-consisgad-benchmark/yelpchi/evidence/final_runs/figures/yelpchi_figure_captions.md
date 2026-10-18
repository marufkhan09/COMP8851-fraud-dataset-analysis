# YelpChi figure captions

**Figure 1. YelpChi performance under label scarcity.** Test AUROC,
AUPRC, Macro-F1, fraud-class F1 and G-Mean for LEX-GNN and ConsisGAD
at four training ratios. Points show means across seeds 2, 42 and 72;
error bars show sample standard deviations. All panels use the common
0.50–1.00 y-axis range. Validation and test node sets remain fixed
across training ratios.

**Figure 2. YelpChi operational efficiency.** Completed epochs,
total training time, mean time per completed epoch, test-inference
time, peak GPU memory and parameter count measured on the controlled
RTX A6000 instance. Bars show three-seed means and error bars show
sample standard deviations. All efficiency panels use zero baselines
to avoid visually exaggerating resource differences.

**Figure 3. YelpChi label-scarcity sensitivity.** Absolute
percentage-point change in each metric relative to the same model's
TR40 mean. Negative values indicate degradation relative to TR40.
This within-model normalisation separates scarcity sensitivity from
the models' different absolute performance levels.

**Figure 4. Matched model differences on YelpChi.** Mean paired
difference for the same training seeds, calculated as LEX-GNN minus
ConsisGAD. Positive values favour LEX-GNN; negative values favour
ConsisGAD. Differences are descriptive because only three seeds were
available.
