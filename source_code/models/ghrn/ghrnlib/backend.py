"""Graph-operation backend for GHRN.

GHRN needs only four graph operations. Each is implemented twice:

* **DGL backend** -- the authors' stack and the default whenever DGL imports.
  This is the path used for the controlled Vast.ai benchmark.
* **Torch backend** -- the same mathematics on ``torch.sparse`` tensors, used
  when DGL has no wheel for the platform (for example Python 3.13 on Windows,
  where DGL publishes no build).

The torch backend exists so the GHRN training path can be smoke-tested on any
machine rather than only where DGL installs. It is **not** a licence to run the
final benchmark without DGL: run :func:`compare_backends` on the benchmark host
once, confirm the two agree to numerical tolerance, and record that check in the
run evidence. If they ever disagree, the DGL result is authoritative.

The four operations
-------------------
``edge_endpoints``      source and destination node IDs, one entry per edge
``laplacian_step``      x - D^-1/2 A D^-1/2 x, the unnormalised Laplacian applied
``neighbour_average``   D^-1/2 A D^-1/2 x, the low-pass component
``prune_edges``         drop edge IDs, then restore self-loops
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def dgl_available() -> bool:
    """True when DGL imports *and* its native library loads."""
    try:
        import dgl  # noqa: F401

        return True
    except Exception:
        # A missing wheel raises ImportError; a broken install raises OSError
        # from ctypes. Both mean the same thing here.
        return False


class SimpleGraph:
    """A minimal homogeneous graph backed by ``torch.sparse``.

    Mirrors the small part of the DGL graph surface that GHRN touches:
    ``num_nodes()``, ``num_edges()``, ``edges()`` and an ``ndata`` dict.
    Edges are stored as a COO pair, matching DGL's edge ordering convention so
    that per-edge scores line up with ``edges()``.
    """

    def __init__(self, src: torch.Tensor, dst: torch.Tensor, num_nodes: int,
                 ndata: Dict[str, torch.Tensor] | None = None):
        self.src = src.to(torch.long)
        self.dst = dst.to(torch.long)
        self._num_nodes = int(num_nodes)
        self.ndata: Dict[str, torch.Tensor] = dict(ndata or {})

    # -- DGL-compatible surface -------------------------------------------
    def num_nodes(self) -> int:
        return self._num_nodes

    def num_edges(self) -> int:
        return int(self.src.numel())

    def edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.src, self.dst

    @property
    def device(self) -> torch.device:
        return self.src.device

    def to(self, device) -> "SimpleGraph":
        moved = SimpleGraph(self.src.to(device), self.dst.to(device), self._num_nodes)
        moved.ndata = {key: value.to(device) for key, value in self.ndata.items()}
        return moved

    # -- helpers ----------------------------------------------------------
    def in_degrees(self) -> torch.Tensor:
        degrees = torch.zeros(self._num_nodes, device=self.device, dtype=torch.float32)
        degrees.index_add_(0, self.dst, torch.ones_like(self.dst, dtype=torch.float32))
        return degrees

    def normalised_adjacency(self) -> torch.Tensor:
        """Sparse D^-1/2 A D^-1/2 with the same clamp DGL's PolyConv uses."""
        degrees = self.in_degrees().clamp(min=1)
        d_invsqrt = degrees.pow(-0.5)
        values = d_invsqrt[self.src] * d_invsqrt[self.dst]
        indices = torch.stack([self.dst, self.src], dim=0)
        return torch.sparse_coo_tensor(
            indices, values, (self._num_nodes, self._num_nodes)
        ).coalesce()

    @staticmethod
    def from_edges(src: torch.Tensor, dst: torch.Tensor, num_nodes: int,
                   ndata: Dict[str, torch.Tensor] | None = None) -> "SimpleGraph":
        return SimpleGraph(src, dst, num_nodes, ndata)


def _is_dgl_graph(graph: Any) -> bool:
    return graph.__class__.__module__.startswith("dgl")


# --------------------------------------------------------------------------
# The four operations
# --------------------------------------------------------------------------

def edge_endpoints(graph: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Source and destination node IDs, one entry per edge."""
    src, dst = graph.edges()
    return src, dst


def laplacian_step(graph: Any, feat: torch.Tensor) -> torch.Tensor:
    """Apply the unnormalised Laplacian: ``feat - D^-1/2 A D^-1/2 feat``."""
    if _is_dgl_graph(graph):
        import dgl.function as fn

        with graph.local_scope():
            d_invsqrt = torch.pow(
                graph.in_degrees().float().clamp(min=1), -0.5
            ).unsqueeze(-1).to(feat.device)
            graph.ndata["_h"] = feat * d_invsqrt
            graph.update_all(fn.copy_u("_h", "_m"), fn.sum("_m", "_h"))
            return feat - graph.ndata.pop("_h") * d_invsqrt

    adjacency = graph.normalised_adjacency()
    return feat - torch.sparse.mm(adjacency, feat)


def neighbour_average(graph: Any, x: torch.Tensor) -> torch.Tensor:
    """Degree-normalised neighbour aggregation: ``D^-1/2 A D^-1/2 x``.

    This is the low-pass component GHRN subtracts to obtain each node's
    high-frequency residual.
    """
    if _is_dgl_graph(graph):
        import dgl.function as fn
        from dgl.nn import EdgeWeightNorm

        with graph.local_scope():
            weight = torch.ones(graph.num_edges(), device=x.device)
            normalised = EdgeWeightNorm(norm="both")(graph, weight)
            graph.ndata["_x"] = x
            graph.edata["_w"] = normalised
            graph.update_all(fn.u_mul_e("_x", "_w", "_m"), fn.sum("_m", "_ax"))
            return graph.ndata.pop("_ax")

    adjacency = graph.normalised_adjacency()
    return torch.sparse.mm(adjacency, x)


def _reset_self_loops(graph: Any) -> Any:
    """Drop every self-loop, then add exactly one per node.

    Does the work with node IDs built in the graph's own index dtype rather than
    calling ``dgl.add_self_loop``. DGL infers a graph's ``idtype`` from the
    tensors it was built from, and on a large graph that can be ``int32``, while
    ``torch.arange`` and ``torch.argsort`` both produce ``int64``. Mixing the two
    raises::

        DGLError: Expect argument "u" to have data type torch.int32.
                  But got torch.int64.

    which is why a graph the size of T-Social fails here where the smaller
    int64 datasets do not. Generating the loop IDs from ``graph.idtype`` keeps
    both sides aligned whatever dtype the graph happens to carry.
    """
    import dgl

    graph = dgl.remove_self_loop(graph)
    loops = torch.arange(graph.num_nodes(), dtype=graph.idtype, device=graph.device)
    return dgl.add_edges(graph, loops, loops)


def prune_edges(graph: Any, edge_ids: torch.Tensor) -> Any:
    """Remove the given edge IDs, then remove and re-add self-loops."""
    if _is_dgl_graph(graph):
        import dgl

        # Edge IDs arrive from torch.argsort, which always returns int64.
        edge_ids = edge_ids.to(device=graph.device, dtype=graph.idtype)
        refined = dgl.remove_edges(graph, edge_ids)
        refined = _reset_self_loops(refined)
        for key, value in graph.ndata.items():
            if key not in refined.ndata:
                refined.ndata[key] = value
        return refined

    src, dst = graph.edges()
    keep = torch.ones(src.numel(), dtype=torch.bool, device=src.device)
    keep[edge_ids] = False
    src, dst = src[keep], dst[keep]

    # Drop existing self-loops, then add one per node, matching DGL's behaviour.
    not_loop = src != dst
    src, dst = src[not_loop], dst[not_loop]
    loops = torch.arange(graph.num_nodes(), device=src.device)
    src = torch.cat([src, loops])
    dst = torch.cat([dst, loops])

    refined = SimpleGraph(src, dst, graph.num_nodes())
    refined.ndata = dict(graph.ndata)
    return refined


def build_graph(src: torch.Tensor, dst: torch.Tensor, num_nodes: int,
                ndata: Dict[str, torch.Tensor] | None = None,
                prefer_dgl: bool = True,
                idtype: torch.dtype | None = None) -> Any:
    """Build a graph on the best available backend, with self-loops added.

    ``idtype`` pins the graph's index dtype. Leaving it ``None`` keeps DGL's own
    inference, which takes the dtype of ``src``/``dst``. Pass ``torch.int32`` for
    a graph large enough that int64 indices are a memory problem; every
    downstream operation here aligns to whatever the graph ends up carrying.
    """
    if prefer_dgl and dgl_available():
        import dgl

        if idtype is not None:
            src = src.to(idtype)
            dst = dst.to(idtype)
        graph = dgl.graph((src, dst), num_nodes=num_nodes, idtype=idtype)
        graph = _reset_self_loops(graph)
        for key, value in (ndata or {}).items():
            graph.ndata[key] = value
        return graph

    not_loop = src != dst
    src, dst = src[not_loop], dst[not_loop]
    loops = torch.arange(num_nodes, device=src.device)
    return SimpleGraph(torch.cat([src, loops]), torch.cat([dst, loops]),
                       num_nodes, ndata)


def compare_backends(src: torch.Tensor, dst: torch.Tensor, num_nodes: int,
                     features: torch.Tensor,
                     tolerance: float = 1e-4) -> Dict[str, Any]:
    """Check the DGL and torch backends agree on the same graph.

    Run this once on the benchmark host, where both backends exist, and store
    the result with the run evidence. Returns a report rather than raising, so
    a caller can record a disagreement instead of losing it.
    """
    if not dgl_available():
        return {"compared": False,
                "reason": "DGL is unavailable, so there is nothing to compare against"}

    import dgl

    dgl_graph = dgl.graph((src, dst), num_nodes=num_nodes)
    dgl_graph = dgl.add_self_loop(dgl.remove_self_loop(dgl_graph))

    not_loop = src != dst
    loops = torch.arange(num_nodes)
    simple = SimpleGraph(torch.cat([src[not_loop], loops]),
                         torch.cat([dst[not_loop], loops]), num_nodes)

    report: Dict[str, Any] = {"compared": True, "tolerance": tolerance}
    for name, function in (("laplacian_step", laplacian_step),
                           ("neighbour_average", neighbour_average)):
        left = function(dgl_graph, features)
        right = function(simple, features)
        difference = float((left - right).abs().max())
        report[name] = {
            "max_absolute_difference": difference,
            "agrees": difference <= tolerance,
        }
    report["all_agree"] = all(
        report[name]["agrees"] for name in ("laplacian_step", "neighbour_average")
    )
    return report
