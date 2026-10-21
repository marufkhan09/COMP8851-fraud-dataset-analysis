# YelpChi final-results interpretation

## Evidence base

The analysis contains 24 verified final runs: two models, four
training ratios and three matched training seeds. Hyperparameters were
selected using validation data only. Each test partition was evaluated
once after training. Values in the figures are means across seeds 2,
42 and 72, with sample-standard-deviation error bars.

## Comparative performance

LEX-GNN produced the higher mean score for all five reported metrics at
TR40, TR30 and TR20. At TR40, its mean AUROC was
0.9615, compared with
0.9074 for ConsisGAD. The
corresponding AUPRC means were
0.8483 and
0.6451.

The difference narrowed as supervision decreased. At TR10, ConsisGAD
had slightly higher mean AUROC
(0.8992 versus
0.8903) and AUPRC
(0.6107 versus
0.6037). Macro-F1 and Fraud-F1
were very similar at TR10. This is a descriptive crossover rather than
a statistical-significance claim.

## Label-scarcity response

From TR40 to TR10, LEX-GNN's mean AUROC changed by
-0.0712 and its mean AUPRC changed by
-0.2446. Over the same reduction, ConsisGAD's mean
AUROC changed by -0.0082 and its mean AUPRC
changed by -0.0344.

These results indicate that LEX-GNN achieved the stronger
higher-supervision performance, whereas ConsisGAD retained a larger
proportion of its performance under the most restricted label
condition. This conclusion is dataset- and protocol-specific.

## Efficiency

ConsisGAD used approximately
22.9% less peak GPU memory on
average across the four ratios and had 6.9%
fewer trainable parameters. Its saved test-inference measurement was
between 8.6 and
13.7 times faster than LEX-GNN.

Total training time does not favour one model uniformly: ConsisGAD was
faster at TR40, while LEX-GNN completed sooner at TR30, TR20 and TR10.
These times are operational measurements under the frozen
implementations. They should not be interpreted as architecture-only
complexity because the models use different training procedures,
epoch limits and early-stopping behaviour.

## Reporting safeguards

- Rankings apply only within the YelpChi unified protocol.
- The fixed split seed was 2; training seeds were 2, 42 and 72.
- AUPRC is reported alongside AUROC because YelpChi is imbalanced.
- Error bars represent sample standard deviations, not standard errors.
- Three-seed confidence intervals are descriptive and no formal
  significance claim is made.
- These unified-protocol results should not be treated as exact
  reproductions of the papers' original experimental protocols.
