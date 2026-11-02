"""COMP8851 shared benchmark library.

This package holds the machinery that must be *identical* across every model in
the COMP8851 fraud-GNN benchmark: dataset loading, split handling, evaluation,
timing, and the result-artefact schema required by master plan v4.4.

Model-specific code (architectures, losses, optimisers, hyperparameters) lives
under ``models/<model>/`` and must never be duplicated here.

Reference: COMP8851_VastAI_Master_Plan_FINAL_v4.4.docx, sections 4, 9 and 10.
"""

from . import artifacts, datasets, evaluator, protocol, runtime, splits

__all__ = ["artifacts", "datasets", "evaluator", "protocol", "runtime", "splits",
           "aggregate", "report", "tuning", "dgl_env"]

__version__ = "1.0.0"
