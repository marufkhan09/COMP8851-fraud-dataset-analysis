"""COMP8851 phase-2 extension: SAGAD, SpaceGNN and CRoC under protocol vast-v4.4.

This package adds model-agnostic machinery that the original ``shared.comp8851``
library does not provide (per-node prediction files, resumable checkpoints,
per-dataset author defaults for tuning, result-integrity verification). It
never modifies ``shared.comp8851``: datasets, splits, the evaluator, the
artefact schema and protocol constants are imported from it unchanged.
"""

import collections
import collections.abc

# Python 3.10+ / 3.13 compatibility shim for DGL and older scientific libraries
for _name in ("Mapping", "MutableMapping", "Iterable", "Iterator", "Sequence",
              "MutableSequence", "Callable", "Container", "Set", "MutableSet"):
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

__version__ = "1.0.0"
