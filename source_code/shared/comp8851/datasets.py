"""Canonical dataset loaders for the six COMP8851 benchmark datasets.

Design rules (master plan v4.4, section 4):

* Acquisition, representation conversion and splitting are three separate
  stages. This module performs conversion only; it never invents a split and
  never applies performance-motivated cleaning.
* The canonical asset preserves the richest available representation. Multi-
  relational datasets keep their relations; a homogeneous union view is derived
  deterministically for models that genuinely require one.
* Every loader returns the same :class:`CanonicalDataset` structure so that
  CARE-GNN, GHRN and any future model consume identical data.

Views
-----
``relation`` : per-relation adjacency, required by relation-aware methods such
               as CARE-GNN.
``homogeneous`` : union of all relations with self-loops, required by
               genuinely homogeneous spectral methods such as BWGNN/GHRN.

Both views contain identical nodes, features and labels. Only the edge
representation differs, which is what makes an A1 adapter defensible.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .protocol import sha256_file

# Expected statistics, used as integrity assertions rather than as
# documentation. A mismatch means the wrong file was downloaded.
EXPECTED_STATS: Dict[str, Dict[str, Any]] = {
    "yelpchi": {"nodes": 45954, "features": 32, "relations": ["rur", "rtr", "rsr"]},
    "amazon": {"nodes": 11944, "features": 25, "relations": ["upu", "usu", "uvu"],
               "ineligible_prefix": 3305},
    "tfinance": {"features": 10},
    "tsocial": {"features": 10},
    "elliptic": {"features": 166},
    "fdcompcn": {"nodes": 5317, "features": 57, "fraud_nodes": 559,
                 "relations": ["cic", "cpc", "csc"]},
}

RELATION_NAMES: Dict[str, List[str]] = {
    "yelpchi": ["rur", "rtr", "rsr"],
    "amazon": ["upu", "usu", "uvu"],
    "fdcompcn": ["cic", "cpc", "csc"],
}


@dataclass
class CanonicalDataset:
    """A dataset in the frozen COMP8851 canonical representation."""

    name: str
    features: np.ndarray                     # (N, F) float32
    labels: np.ndarray                       # (N,) int64 in {0, 1}
    eligible_idx: np.ndarray                 # node IDs allowed in supervised splits
    relations: Dict[str, sp.spmatrix]        # relation name -> symmetric adjacency
    time_steps: Optional[np.ndarray] = None  # Elliptic chronology
    source_files: Dict[str, str] = field(default_factory=dict)
    source_sha256: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    # -- basic properties -------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def is_multi_relational(self) -> bool:
        return len(self.relations) > 1

    # -- views ------------------------------------------------------------
    def homogeneous_adjacency(self, add_self_loops: bool = True) -> sp.csr_matrix:
        """Deterministic union of every relation, symmetric and binarised."""
        if not self.relations:
            raise ValueError(f"{self.name}: no relations to build a union from.")
        union = None
        for name in sorted(self.relations):
            matrix = self.relations[name].tocsr()
            union = matrix if union is None else (union + matrix)
        assert union is not None
        union = union.maximum(union.T)          # symmetric
        union.data = np.ones_like(union.data)   # binary, no multi-edges
        if add_self_loops:
            union = union + sp.eye(self.num_nodes, format="csr")
        union = union.tocsr()
        union.data = np.ones_like(union.data)
        union.eliminate_zeros()
        return union

    def adjacency_lists(self, add_self_loops: bool = True) -> List[Dict[int, set]]:
        """Per-relation adjacency lists, the input format CARE-GNN expects.

        Returns one ``{node_id: {neighbour ids}}`` mapping per relation, in the
        canonical relation order for the dataset.
        """
        order = RELATION_NAMES.get(self.name, sorted(self.relations))
        order = [name for name in order if name in self.relations] or sorted(self.relations)
        lists = []
        for name in order:
            matrix = self.relations[name].tocsr()
            matrix = matrix.maximum(matrix.T)
            if add_self_loops:
                matrix = matrix + sp.eye(self.num_nodes, format="csr")
            matrix = matrix.tocsr()
            adj: Dict[int, set] = {}
            indptr, indices = matrix.indptr, matrix.indices
            for node in range(self.num_nodes):
                neighbours = set(indices[indptr[node]:indptr[node + 1]].tolist())
                if not neighbours:
                    # CARE-GNN indexes neighbours unconditionally; an isolated
                    # node must still have itself so the batch never breaks.
                    neighbours = {node}
                adj[node] = neighbours
            lists.append(adj)
        return lists

    def homogeneous_edges(self, add_self_loops: bool = False):
        """Edge index pair for the union view, as torch tensors.

        Library-agnostic: the caller decides whether to build a DGL graph or a
        plain sparse one, so the dataset layer stays free of model-stack
        dependencies.
        """
        import torch

        union = self.homogeneous_adjacency(add_self_loops=add_self_loops).tocoo()
        return (torch.as_tensor(union.row, dtype=torch.int64),
                torch.as_tensor(union.col, dtype=torch.int64))

    def dgl_homogeneous(self, add_self_loops: bool = True):
        """Build the homogeneous DGL graph used by spectral models."""
        import dgl
        import torch

        union = self.homogeneous_adjacency(add_self_loops=False).tocoo()
        graph = dgl.graph((torch.as_tensor(union.row, dtype=torch.int64),
                           torch.as_tensor(union.col, dtype=torch.int64)),
                          num_nodes=self.num_nodes)
        graph = dgl.to_bidirected(graph)
        graph = dgl.remove_self_loop(graph)
        if add_self_loops:
            graph = dgl.add_self_loop(graph)
        graph.ndata["feature"] = torch.as_tensor(self.features, dtype=torch.float32)
        graph.ndata["label"] = torch.as_tensor(self.labels, dtype=torch.int64)
        return graph

    # -- reporting --------------------------------------------------------
    def statistics(self) -> Dict[str, Any]:
        """Node/edge/feature/label statistics for the dataset manifest."""
        eligible_labels = self.labels[self.eligible_idx]
        fraud = int(eligible_labels.sum())
        total_eligible = int(self.eligible_idx.size)
        union = self.homogeneous_adjacency(add_self_loops=False)
        stats: Dict[str, Any] = {
            "dataset": self.name,
            "nodes": self.num_nodes,
            "feature_dimension": self.num_features,
            "eligible_nodes": total_eligible,
            "ineligible_nodes": self.num_nodes - total_eligible,
            "fraud_nodes": fraud,
            "normal_nodes": total_eligible - fraud,
            "fraud_percentage": (100.0 * fraud / total_eligible) if total_eligible else 0.0,
            "imbalance_ratio_normal_to_fraud": (
                (total_eligible - fraud) / fraud if fraud else None
            ),
            "union_stored_directed_edges": int(union.nnz),
            "union_undirected_edges": int(union.nnz // 2),
            "relations": {},
        }
        for name in sorted(self.relations):
            matrix = self.relations[name].tocsr()
            matrix = matrix.maximum(matrix.T)
            stats["relations"][name] = {
                "stored_directed_edges": int(matrix.nnz),
                "undirected_edges": int(matrix.nnz // 2),
            }
        if self.time_steps is not None:
            stats["time_steps"] = {
                "min": int(np.min(self.time_steps)),
                "max": int(np.max(self.time_steps)),
                "distinct": int(np.unique(self.time_steps).size),
            }
        return stats

    def heterophily(self) -> Dict[str, Any]:
        """Global and fraud-node local heterophily over labelled edges.

        Global H: labelled edges joining different labels divided by all
        labelled edges. Local H for a fraud node: benign labelled neighbours
        divided by all labelled neighbours. Self-loops are excluded from both.
        """
        eligible_mask = np.zeros(self.num_nodes, dtype=bool)
        eligible_mask[self.eligible_idx] = True
        union = self.homogeneous_adjacency(add_self_loops=False).tocoo()

        keep = (union.row != union.col) & eligible_mask[union.row] & eligible_mask[union.col]
        src, dst = union.row[keep], union.col[keep]
        if src.size == 0:
            return {"global_heterophily": None, "note": "no labelled edges"}
        different = float(np.mean(self.labels[src] != self.labels[dst]))

        csr = self.homogeneous_adjacency(add_self_loops=False).tocsr()
        fraud_nodes = self.eligible_idx[self.labels[self.eligible_idx] == 1]
        local: List[float] = []
        for node in fraud_nodes:
            neighbours = csr.indices[csr.indptr[node]:csr.indptr[node + 1]]
            neighbours = neighbours[(neighbours != node) & eligible_mask[neighbours]]
            if neighbours.size:
                local.append(float(np.mean(self.labels[neighbours] == 0)))
        local_array = np.asarray(local, dtype=np.float64)
        return {
            "global_heterophily": different,
            "labelled_edges_considered": int(src.size),
            "fraud_local_heterophily_mean": float(local_array.mean()) if local_array.size else None,
            "fraud_local_heterophily_median": float(np.median(local_array)) if local_array.size else None,
            "fraud_local_heterophily_std": float(local_array.std(ddof=1)) if local_array.size > 1 else None,
            "fraud_nodes_with_labelled_neighbours": int(local_array.size),
        }

    def manifest(self) -> Dict[str, Any]:
        """The ``dataset_manifest.json`` payload."""
        return {
            "dataset_id": self.name,
            "source_files": self.source_files,
            "source_sha256": self.source_sha256,
            "statistics": self.statistics(),
            "notes": self.notes,
        }

    def view_manifest(self, view: str, add_self_loops: bool = True) -> Dict[str, Any]:
        """The ``dataset_view_manifest.json`` payload for the chosen view."""
        if view not in ("relation", "homogeneous"):
            raise ValueError(f"Unknown view {view!r}; expected 'relation' or 'homogeneous'.")
        payload: Dict[str, Any] = {
            "dataset_view_id": f"{self.name}_{view}_v1",
            "view": view,
            "builder": "shared/comp8851/datasets.py",
            "builder_version": "1.0.0",
            "self_loop_policy": "added" if add_self_loops else "not added",
            "edge_convention": "symmetric, binarised, duplicates collapsed",
            "nodes": self.num_nodes,
            "feature_dimension": self.num_features,
        }
        if view == "relation":
            payload["relations"] = RELATION_NAMES.get(self.name, sorted(self.relations))
            payload["relation_edges"] = {
                name: int(self.relations[name].tocsr().maximum(self.relations[name].tocsr().T).nnz)
                for name in sorted(self.relations)
            }
        else:
            union = self.homogeneous_adjacency(add_self_loops=add_self_loops)
            payload["union_stored_directed_edges"] = int(union.nnz)
        payload["view_sha256"] = self._view_hash(view, add_self_loops)
        return payload

    def _view_hash(self, view: str, add_self_loops: bool) -> str:
        """Deterministic hash of the exact edge structure handed to the model."""
        import hashlib

        digest = hashlib.sha256()
        digest.update(self.name.encode())
        digest.update(view.encode())
        digest.update(str(add_self_loops).encode())
        digest.update(np.ascontiguousarray(self.labels).tobytes())
        digest.update(np.ascontiguousarray(self.features).tobytes())
        if view == "relation":
            for name in RELATION_NAMES.get(self.name, sorted(self.relations)):
                if name not in self.relations:
                    continue
                coo = self.relations[name].tocoo()
                order = np.lexsort((coo.col, coo.row))
                digest.update(name.encode())
                digest.update(np.ascontiguousarray(coo.row[order]).tobytes())
                digest.update(np.ascontiguousarray(coo.col[order]).tobytes())
        else:
            coo = self.homogeneous_adjacency(add_self_loops=add_self_loops).tocoo()
            order = np.lexsort((coo.col, coo.row))
            digest.update(np.ascontiguousarray(coo.row[order]).tobytes())
            digest.update(np.ascontiguousarray(coo.col[order]).tobytes())
        return digest.hexdigest()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _symmetrise(matrix: sp.spmatrix, num_nodes: int) -> sp.csr_matrix:
    """Return a symmetric, binary, self-loop-free CSR matrix."""
    matrix = sp.csr_matrix(matrix, shape=(num_nodes, num_nodes))
    matrix = matrix.maximum(matrix.T)
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    matrix.data = np.ones_like(matrix.data)
    return matrix.tocsr()


def _edges_to_matrix(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    data = np.ones(len(src), dtype=np.float32)
    matrix = sp.coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    return _symmetrise(matrix, num_nodes)


def _check(condition: bool, message: str, strict: bool = True) -> None:
    if condition:
        return
    if strict:
        raise ValueError(message)
    print(f"[dataset] WARNING {message}")


# --------------------------------------------------------------------------
# Per-dataset loaders
# --------------------------------------------------------------------------

#: Official DGL distribution points for the YelpChi and Amazon archives. These
#: are the source lineage GADBench also identifies, so they are the canonical
#: team source named in the master plan.
FRAUD_ARCHIVES = {
    "yelpchi": ("https://data.dgl.ai/dataset/FraudYelp.zip", "YelpChi.mat"),
    "amazon": ("https://data.dgl.ai/dataset/FraudAmazon.zip", "Amazon.mat"),
}

#: Relation key names as stored inside the .mat files.
MAT_RELATION_KEYS = {
    "yelpchi": {"net_rur": "rur", "net_rtr": "rtr", "net_rsr": "rsr"},
    "amazon": {"net_upu": "upu", "net_usu": "usu", "net_uvu": "uvu"},
}


def _locate_or_fetch_mat(name: str, raw_dir: Path, download: bool = True) -> Path:
    """Find ``YelpChi.mat``/``Amazon.mat``, extracting or downloading if needed."""
    url, filename = FRAUD_ARCHIVES[name]
    raw_dir = Path(raw_dir)

    if raw_dir.is_file() and raw_dir.suffix.lower() == ".mat":
        return raw_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    for found in raw_dir.rglob(filename):
        return found

    archive = raw_dir / Path(url).name
    if not archive.exists():
        if not download:
            raise FileNotFoundError(
                f"{name}: {filename} not found under {raw_dir} and downloading is disabled. "
                f"Fetch {url} manually."
            )
        import urllib.request

        print(f"[dataset] downloading {url}")
        urllib.request.urlretrieve(url, archive)

    with zipfile.ZipFile(archive) as handle:
        handle.extractall(raw_dir)
    for found in raw_dir.rglob(filename):
        return found
    raise FileNotFoundError(f"{name}: {filename} missing from {archive}")


def load_fraud_mat(name: str, raw_dir: Path, strict: bool = True,
                   download: bool = True) -> CanonicalDataset:
    """Load YelpChi or Amazon from the official ``.mat`` release.

    The archive published at data.dgl.ai is exactly what ``dgl.data``'s Fraud
    datasets wrap, so reading it directly gives byte-identical graph content
    without requiring DGL. That matters because CARE-GNN's environment does not
    otherwise need DGL at all.

    DGL's generated 70/10/20 masks are not present in the ``.mat`` file and are
    not reconstructed: the unified track uses the frozen COMP8851 split IDs.
    """
    from scipy.io import loadmat

    mat_path = _locate_or_fetch_mat(name, raw_dir, download=download)
    payload = loadmat(str(mat_path))

    labels = np.asarray(payload["label"]).reshape(-1).astype(np.int64)
    feature_block = payload["features"]
    features = (feature_block.todense().A if sp.issparse(feature_block)
                else np.asarray(feature_block)).astype(np.float32)
    num_nodes = features.shape[0]

    relations: Dict[str, sp.spmatrix] = {}
    for mat_key, relation in MAT_RELATION_KEYS[name].items():
        if mat_key not in payload:
            raise KeyError(
                f"{name}: relation {mat_key} missing from {mat_path.name}; "
                f"found {sorted(k for k in payload if not k.startswith('__'))}"
            )
        relations[relation] = _symmetrise(payload[mat_key], num_nodes)

    if name == "amazon":
        # The first 3,305 Amazon users carry no usable label. They stay in the
        # graph for message passing but are excluded from supervised splits.
        prefix = EXPECTED_STATS["amazon"]["ineligible_prefix"]
        eligible = np.arange(prefix, num_nodes, dtype=np.int64)
    else:
        eligible = np.arange(num_nodes, dtype=np.int64)

    expected = EXPECTED_STATS[name]
    _check(num_nodes == expected["nodes"],
           f"{name}: expected {expected['nodes']} nodes, found {num_nodes}", strict)
    _check(features.shape[1] == expected["features"],
           f"{name}: expected {expected['features']} features, found {features.shape[1]}", strict)

    return CanonicalDataset(
        name=name,
        features=features,
        labels=labels,
        eligible_idx=eligible,
        relations=relations,
        source_files={"mat": str(mat_path)},
        source_sha256={"mat": sha256_file(mat_path)},
        notes={
            "provenance": f"official release {FRAUD_ARCHIVES[name][0]} -> {mat_path.name}",
            "dgl_masks": "not used; the unified track applies COMP8851 split IDs",
            "relations": list(MAT_RELATION_KEYS[name].values()),
        },
    )


def _resolve_bwgnn_path(name: str, data_path: Path) -> Path:
    """Return the DGL graph file itself, given either it or its directory.

    Callers are inconsistent: a run passes the file, while the freeze step
    passes the dataset directory. Handing a directory to ``dgl.load_graphs``
    does not raise a useful error — it reads uninitialised memory and reports a
    bogus magic number, which looks exactly like a corrupt download. Resolving
    the path here removes that whole class of false alarm.
    """
    data_path = Path(data_path)
    if data_path.is_file():
        return data_path
    if not data_path.is_dir():
        raise FileNotFoundError(
            f"{name}: nothing at {data_path}. Expected the BWGNN authors' "
            f"'{name}' DGL graph file, or a directory containing it.")

    # The authors' file is named after the dataset; fall back to the largest
    # plain file, which is the graph in every layout seen so far.
    preferred = data_path / name
    if preferred.is_file():
        return preferred

    candidates = [p for p in data_path.iterdir()
                  if p.is_file() and p.suffix not in {".npz", ".json", ".zip"}]
    if not candidates:
        raise FileNotFoundError(
            f"{name}: no graph file inside {data_path}. Expected a file named "
            f"'{name}' from the BWGNN authors' release.")
    return max(candidates, key=lambda p: p.stat().st_size)


def load_bwgnn_graph(name: str, data_path: Path, strict: bool = True) -> CanonicalDataset:
    """Load T-Finance or T-Social from the BWGNN authors' released DGL graph.

    The stored label tensor is two-column; decoding it with argmax to a single
    binary class ID is representation decoding and matches the official loader.
    It is not relabelling.
    """
    import dgl

    data_path = _resolve_bwgnn_path(name, data_path)
    graphs, _ = dgl.load_graphs(str(data_path))
    graph = graphs[0]

    labels_tensor = graph.ndata["label"]
    if labels_tensor.dim() == 2:
        labels = labels_tensor.argmax(dim=1).numpy().reshape(-1).astype(np.int64)
    else:
        labels = labels_tensor.numpy().reshape(-1).astype(np.int64)
    features = graph.ndata["feature"].float().numpy().astype(np.float32)
    num_nodes = features.shape[0]

    src, dst = graph.edges()
    relations = {"homo": _edges_to_matrix(src.numpy(), dst.numpy(), num_nodes)}
    eligible = np.arange(num_nodes, dtype=np.int64)

    _check(set(np.unique(labels).tolist()) == {0, 1},
           f"{name}: labels must be binary after argmax decoding", strict)

    return CanonicalDataset(
        name=name,
        features=features,
        labels=labels,
        eligible_idx=eligible,
        relations=relations,
        source_files={"graph": str(data_path)},
        source_sha256={"graph": sha256_file(data_path)},
        notes={
            "provenance": "BWGNN authors, Rethinking-Anomaly-Detection release",
            "label_decoding": "two-column label tensor decoded with argmax",
            "single_relation": "natively homogeneous; no relation split exists",
        },
    )


def load_elliptic(raw_dir: Path, strict: bool = True) -> CanonicalDataset:
    """Build the locked GADBench-compatible 166-feature Elliptic view.

    Source: the three EllipticCo CSVs. The time-step field stays among the 166
    model features and is also retained separately for chronological splitting.
    Unknown-class transactions remain structurally present but are excluded from
    the supervised split.

    Label convention: class 1 in the raw file is illicit, which becomes fraud=1
    here; class 2 is licit, which becomes 0.
    """
    import pandas as pd

    raw_dir = Path(raw_dir)
    candidates = {
        "features": ["elliptic_txs_features.csv"],
        "classes": ["elliptic_txs_classes.csv"],
        "edges": ["elliptic_txs_edgelist.csv"],
    }
    paths: Dict[str, Path] = {}
    for key, names in candidates.items():
        for name in names:
            for found in raw_dir.rglob(name):
                paths[key] = found
                break
            if key in paths:
                break
        if key not in paths:
            raise FileNotFoundError(
                f"Elliptic: could not find {names[0]} under {raw_dir}. "
                "Download the EllipticCo dataset from Kaggle and extract it there."
            )

    features_df = pd.read_csv(paths["features"], header=None)
    classes_df = pd.read_csv(paths["classes"])
    edges_df = pd.read_csv(paths["edges"])

    tx_ids = features_df.iloc[:, 0].to_numpy()
    time_steps = features_df.iloc[:, 1].to_numpy().astype(np.int64)
    # Column 0 is the transaction ID and is dropped. Columns 1.. are kept,
    # which retains the time step inside the 166-feature model input exactly as
    # the GADBench-compatible convention requires.
    features = features_df.iloc[:, 1:].to_numpy().astype(np.float32)

    id_to_index = {tx: index for index, tx in enumerate(tx_ids)}
    num_nodes = len(tx_ids)

    class_map = dict(zip(classes_df.iloc[:, 0], classes_df.iloc[:, 1].astype(str)))
    labels = np.zeros(num_nodes, dtype=np.int64)
    eligible: List[int] = []
    for tx, index in id_to_index.items():
        raw_class = class_map.get(tx, "unknown")
        if raw_class == "1":          # illicit
            labels[index] = 1
            eligible.append(index)
        elif raw_class == "2":        # licit
            labels[index] = 0
            eligible.append(index)
        # "unknown" stays in the graph but out of the supervised split.

    src = edges_df.iloc[:, 0].map(id_to_index).to_numpy()
    dst = edges_df.iloc[:, 1].map(id_to_index).to_numpy()
    valid = ~(np.isnan(src.astype(np.float64)) | np.isnan(dst.astype(np.float64)))
    relations = {"txn": _edges_to_matrix(src[valid].astype(np.int64),
                                         dst[valid].astype(np.int64), num_nodes)}

    _check(features.shape[1] == EXPECTED_STATS["elliptic"]["features"],
           f"elliptic: expected 166 features, found {features.shape[1]}", strict)

    return CanonicalDataset(
        name="elliptic",
        features=features,
        labels=labels,
        eligible_idx=np.asarray(sorted(eligible), dtype=np.int64),
        relations=relations,
        time_steps=time_steps,
        source_files={key: str(value) for key, value in paths.items()},
        source_sha256={key: sha256_file(value) for key, value in paths.items()},
        notes={
            "provenance": "EllipticCo Kaggle release, GADBench-compatible 166-feature view",
            "feature_convention": "transaction ID dropped; time step retained among the 166 features",
            "label_convention": "raw class 1 (illicit) -> 1, class 2 (licit) -> 0, unknown excluded",
            "split_convention": "chronological; see splits.build_chronological_splits",
        },
    )


def load_fdcompcn(raw_path: Path, strict: bool = True) -> CanonicalDataset:
    """Load FDCompCN from the SplitGNN authors' released processed dataset.

    The release is ``data/FDCompCN.zip``, which contains a single DGL binary
    graph ``comp.dgl`` (not a ``.mat`` file). It is a heterograph carrying
    ``feature`` and ``label`` node data and the three semantic relations
    C-I-C, C-P-C and C-S-C as edge types.

    Accepts the ``.zip``, the extracted ``comp.dgl``, a directory containing
    either, or a ``.npz`` canonical cache written by :func:`save_canonical`.
    The cache path needs no DGL, which is how a DGL-free environment such as
    CARE-GNN's consumes this dataset: convert once on a DGL-capable machine,
    then everyone reads the frozen view.

    SplitGNN's own train/valid/test masks are present in the file and are
    deliberately not read: the unified track uses the COMP8851 split registry.
    """
    raw_path = Path(raw_path)

    # A frozen canonical cache short-circuits everything and needs no DGL.
    if raw_path.suffix.lower() == ".npz":
        return load_canonical(raw_path)
    if raw_path.is_dir():
        for cached in raw_path.rglob("fdcompcn_canonical.npz"):
            return load_canonical(cached)

    graph_path: Optional[Path] = None
    if raw_path.is_dir():
        for candidate in list(raw_path.rglob("*.dgl")) + list(raw_path.rglob("FDCompCN.zip")):
            raw_path = candidate
            break

    if raw_path.suffix.lower() == ".zip":
        extract_dir = raw_path.parent / (raw_path.stem + "_extracted")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw_path) as archive:
            archive.extractall(extract_dir)
        for candidate in extract_dir.rglob("*.dgl"):
            graph_path = candidate
            break
        if graph_path is None:
            raise FileNotFoundError(f"No .dgl graph inside {raw_path}")
    elif raw_path.suffix.lower() == ".dgl":
        graph_path = raw_path
    else:
        raise FileNotFoundError(
            f"FDCompCN: expected FDCompCN.zip, comp.dgl, or a canonical .npz, got {raw_path}"
        )

    import dgl

    graphs, _ = dgl.load_graphs(str(graph_path))
    graph = graphs[0]

    features = graph.ndata["feature"].float().numpy().astype(np.float32)
    label_tensor = graph.ndata["label"]
    labels = (label_tensor.argmax(dim=1) if label_tensor.dim() == 2 else label_tensor)
    labels = labels.numpy().reshape(-1).astype(np.int64)
    num_nodes = features.shape[0]

    relations: Dict[str, sp.spmatrix] = {}
    for etype in graph.canonical_etypes:
        relation_name = etype[1]
        normalised = relation_name.lower().replace("-", "").replace("_", "")
        target = next((r for r in RELATION_NAMES["fdcompcn"] if r in normalised), relation_name)
        src, dst = graph.edges(etype=etype)
        relations[target] = _edges_to_matrix(src.numpy(), dst.numpy(), num_nodes)

    if not relations:
        raise KeyError(f"FDCompCN: no edge types found in {graph_path.name}")

    expected = EXPECTED_STATS["fdcompcn"]
    _check(num_nodes == expected["nodes"],
           f"fdcompcn: expected {expected['nodes']} nodes, found {num_nodes}", strict)
    _check(int(labels.sum()) == expected["fraud_nodes"],
           f"fdcompcn: expected {expected['fraud_nodes']} fraud nodes, found {int(labels.sum())}",
           strict)

    return CanonicalDataset(
        name="fdcompcn",
        features=features,
        labels=labels,
        eligible_idx=np.arange(num_nodes, dtype=np.int64),
        relations=relations,
        source_files={"graph": str(graph_path)},
        source_sha256={"graph": sha256_file(graph_path)},
        notes={
            "provenance": "SplitGNN repository, data/FDCompCN.zip -> comp.dgl",
            "relations": "C-I-C, C-P-C, C-S-C preserved as released",
            "split_convention": "SplitGNN's own masks are not inherited; COMP8851 IDs are used",
        },
    )


# --------------------------------------------------------------------------
# Canonical cache
# --------------------------------------------------------------------------

def save_canonical(dataset: CanonicalDataset, path: Path | str) -> Path:
    """Freeze a canonical dataset to a single ``.npz``.

    Two purposes. First, it caches expensive preprocessing on persistent
    storage, as the plan requires. Second, it decouples a dataset from the
    library that originally read it: T-Finance, T-Social and FDCompCN ship as
    DGL binaries, but once frozen here they load in a DGL-free environment such
    as CARE-GNN's.

    The written file is a deterministic derivative of the canonical source and
    is identified by its own SHA256, so it can be listed in the view manifest.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "features": dataset.features,
        "labels": dataset.labels,
        "eligible_idx": dataset.eligible_idx,
        "relation_names": np.asarray(sorted(dataset.relations), dtype=object),
        "meta": np.asarray([json.dumps({
            "name": dataset.name,
            "source_files": dataset.source_files,
            "source_sha256": dataset.source_sha256,
            "notes": dataset.notes,
        })], dtype=object),
    }
    if dataset.time_steps is not None:
        payload["time_steps"] = dataset.time_steps

    for name in sorted(dataset.relations):
        matrix = dataset.relations[name].tocoo()
        payload[f"rel_{name}_row"] = matrix.row.astype(np.int64)
        payload[f"rel_{name}_col"] = matrix.col.astype(np.int64)

    np.savez_compressed(path, **payload)
    return path


def load_canonical(path: Path | str) -> CanonicalDataset:
    """Load a dataset frozen by :func:`save_canonical`. Requires no graph library."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        features = data["features"]
        labels = data["labels"]
        eligible = data["eligible_idx"]
        relation_names = [str(name) for name in data["relation_names"]]
        meta = json.loads(str(data["meta"][0]))
        time_steps = data["time_steps"] if "time_steps" in data.files else None

        num_nodes = int(features.shape[0])
        relations: Dict[str, sp.spmatrix] = {}
        for name in relation_names:
            row = data[f"rel_{name}_row"]
            col = data[f"rel_{name}_col"]
            relations[name] = _edges_to_matrix(row, col, num_nodes)

    return CanonicalDataset(
        name=meta["name"],
        features=features,
        labels=labels,
        eligible_idx=eligible,
        relations=relations,
        time_steps=time_steps,
        source_files=meta.get("source_files", {}),
        source_sha256=meta.get("source_sha256", {}),
        notes={**meta.get("notes", {}), "loaded_from_cache": str(path)},
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

DATA_PATH_HELP = {
    "yelpchi": "directory DGL may download FraudYelp.zip into",
    "amazon": "directory DGL may download FraudAmazon.zip into",
    "tfinance": "path to the BWGNN authors' 'tfinance' DGL graph file",
    "tsocial": "path to the BWGNN authors' 'tsocial' DGL graph file",
    "elliptic": "directory containing the three elliptic_txs_*.csv files",
    "fdcompcn": "path to FDCompCN.zip or the extracted .mat file",
}


def load_dataset(name: str, data_path: Path | str, strict: bool = True,
                 cache_dir: Optional[Path | str] = None) -> CanonicalDataset:
    """Load any of the six benchmark datasets into the canonical representation.

    ``data_path`` meaning per dataset is listed in :data:`DATA_PATH_HELP`.
    ``strict`` turns published-statistic mismatches into errors; set it False
    only to inspect an unexpected file, never for a benchmark run.
    """
    name = name.lower()
    path = Path(data_path)

    # A frozen canonical cache wins for every dataset: it is the deterministic
    # derivative of the canonical source and loads without a graph library.
    if path.suffix.lower() == ".npz":
        dataset = load_canonical(path)
        validate_dataset(dataset)
        return dataset
    if path.is_dir():
        for cached in sorted(path.glob(f"{name}_canonical.npz")):
            dataset = load_canonical(cached)
            validate_dataset(dataset)
            return dataset

    if name in ("yelpchi", "yelp"):
        dataset = load_fraud_mat("yelpchi", path, strict=strict)
    elif name == "amazon":
        dataset = load_fraud_mat("amazon", path, strict=strict)
    elif name in ("tfinance", "tsocial"):
        dataset = load_bwgnn_graph(name, path, strict=strict)
    elif name == "elliptic":
        dataset = load_elliptic(path, strict=strict)
    elif name == "fdcompcn":
        dataset = load_fdcompcn(path, strict=strict)
    else:
        raise ValueError(
            f"Unknown dataset {name!r}. Expected one of: "
            "yelpchi, amazon, tfinance, tsocial, elliptic, fdcompcn."
        )

    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: CanonicalDataset) -> Dict[str, Any]:
    """Structural validation required before any dataset enters a run.

    Checks shapes, label domain, eligibility, edge index ranges and symmetry.
    Raises on anything that would silently corrupt an experiment.
    """
    problems: List[str] = []

    if dataset.features.ndim != 2:
        problems.append(f"features must be 2-D, got shape {dataset.features.shape}")
    if dataset.labels.shape[0] != dataset.features.shape[0]:
        problems.append(
            f"label count {dataset.labels.shape[0]} != node count {dataset.features.shape[0]}"
        )
    if not np.isfinite(dataset.features).all():
        problems.append("features contain NaN or infinity")

    label_values = set(np.unique(dataset.labels).tolist())
    if not label_values <= {0, 1}:
        problems.append(f"labels outside {{0,1}}: {sorted(label_values)}")

    if dataset.eligible_idx.size == 0:
        problems.append("no eligible nodes")
    elif dataset.eligible_idx.max() >= dataset.num_nodes or dataset.eligible_idx.min() < 0:
        problems.append("eligible_idx contains out-of-range node IDs")
    else:
        eligible_labels = set(np.unique(dataset.labels[dataset.eligible_idx]).tolist())
        if eligible_labels != {0, 1}:
            problems.append(f"eligible nodes must contain both classes, found {sorted(eligible_labels)}")

    for name, matrix in dataset.relations.items():
        if matrix.shape != (dataset.num_nodes, dataset.num_nodes):
            problems.append(f"relation {name} has shape {matrix.shape}, expected square N x N")
        difference = abs(matrix - matrix.T)
        if difference.nnz != 0:
            problems.append(f"relation {name} is not symmetric")

    if dataset.time_steps is not None and dataset.time_steps.shape[0] != dataset.num_nodes:
        problems.append("time_steps length does not match node count")

    if problems:
        raise ValueError(f"{dataset.name}: dataset validation failed:\n  " + "\n  ".join(problems))

    return {"validated": True, "statistics": dataset.statistics()}


def write_dataset_report(dataset: CanonicalDataset, output_path: Path | str) -> Path:
    """Write a combined statistics + heterophily report for a dataset."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": dataset.manifest(),
        "heterophily": dataset.heterophily(),
        "views": {
            "relation": dataset.view_manifest("relation"),
            "homogeneous": dataset.view_manifest("homogeneous"),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                           encoding="utf-8")
    return output_path
