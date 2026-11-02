"""GHRN implementation for the COMP8851 fraud-GNN benchmark.

Paper: Gao et al. (2023), WWW. Official code: https://github.com/blacksingular/GHRN

Submodules are imported lazily. ``ghrnlib.graph_refine.select_edges_to_delete``
is pure tensor code and stays importable on machines without DGL; everything
that performs message passing needs DGL and will raise where it is missing.
"""

__all__ = ["GHRN", "build_ghrn", "bwgnn_backbone", "graph_refine", "model"]


def __getattr__(name):
    if name in ("GHRN", "build_ghrn"):
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
