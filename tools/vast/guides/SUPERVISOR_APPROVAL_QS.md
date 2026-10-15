# Questions for Venus before new canonical model extensions

1. We plan to preserve one frozen split registry, training seeds 2/42/72, TR40–TR10, 12-trial TR40 validation-AUPRC tuning and test isolation. Are these strictly required for all new models, even when author training differs?
2. SAWGAD is fundamentally few-shot, using 30 fraud labels and a 100+350 epoch two-phase protocol; should it be treated as a separate weak-supervision track or adapted under the shared TR label budget? What is fair model fidelity?
3. RGTAN's native script uses 5-fold cross-validation and risk-aware neighbour features. Can we replace native folds with the fixed canonical validation while preserving the core method? What leakage audit do you require?
4. EOGFD's released runner has a class-import mismatch and computes test metrics during each training epoch. May we write a separate **validation-only** evaluator and preserve the evolutionary graph mechanism?
5. On T-Social if 48 GB GPU or host RAM cannot fit full models, is a timed failure/compatibility result acceptable? Define a common maximum wall-clock and memory feasibility gate.
6. Do you want the original ten models' evidence closure completed *before* expensive experiments on the three new models? Can a native reproduction remain a separate, clearly non-canonical evidence mode?
7. Can we use exact-equivalent batched inference, deterministic preprocessing caches and checkpoint-resume without modifying model outputs, and how should their timing be reported?
