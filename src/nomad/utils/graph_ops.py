"""Pure graph operations for the NOMAD protein-peptide-precursor graph.

All functions are side-effect-free: they accept and return graphs without
mutating caller state. No DataFrame or torch objects are accepted here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import numpy as np

from nomad.utils.constants import GraphRelation, MIN_PRECURSOR_DETECTIONS, NodeType

logger = logging.getLogger(__name__)


def prune_graph(graph: nx.Graph) -> nx.Graph:
    """Removes unobserved nodes from the graph, returning a pruned copy."""
    nodes_before = graph.number_of_nodes()
    observed_precursors = {
        n for n, d in graph.nodes(data=True)
        if d.get("type") == NodeType.PRECURSOR
        and sum(1 for nb in graph[n] if graph[n][nb].get("relation") == GraphRelation.DETECTED_IN) >= MIN_PRECURSOR_DETECTIONS
    }
    observed_peptides = {
        nb for prec in observed_precursors for nb, d in graph[prec].items()
        if d.get("relation") == GraphRelation.HAS_PRECURSOR
    }
    observed_proteins = {
        nb for pep in observed_peptides for nb, d in graph[pep].items()
        if d.get("relation") == GraphRelation.PRODUCES
    }
    sample_nodes = {n for n, d in graph.nodes(data=True) if d.get("type") == NodeType.SAMPLE}

    keep = observed_precursors | observed_peptides | observed_proteins | sample_nodes
    pruned = graph.copy()
    pruned.remove_nodes_from([n for n in pruned.nodes if n not in keep])
    logger.info("Graph pruned: %d -> %d nodes.", nodes_before, pruned.number_of_nodes())
    return pruned


def get_protein_nodes(graph: nx.Graph) -> List[str]:
    return sorted(n for n, d in graph.nodes(data=True) if d.get("type") == NodeType.PROTEIN)


def get_precursor_nodes(graph: nx.Graph) -> List[str]:
    return sorted(n for n, d in graph.nodes(data=True) if d.get("type") == NodeType.PRECURSOR)


def identify_components(graph: nx.Graph) -> List[Set[Any]]:
    """Identifies connected sets of interacting biological entities."""
    nodes = [n for n, d in graph.nodes(data=True) if d.get("type") in ("Protein", "Precursor", "Peptide")]
    sub = graph.subgraph(nodes)
    return list(nx.weakly_connected_components(sub) if sub.is_directed() else nx.connected_components(sub))


def build_v_matrix(graph: nx.Graph, precursors: List[str], sample_to_idx: Dict[str, int]) -> np.ndarray:
    """Constructs the observed quantification matrix V from graph edges."""
    V = np.zeros((len(sample_to_idx), len(precursors)))
    for i, prec in enumerate(precursors):
        for sample, d in graph.adj[prec].items():
            if d.get("relation") == "DETECTED_IN" and sample in sample_to_idx:
                V[sample_to_idx[sample], i] = float(d["intensity"])
    return V


def build_h_mask(graph: nx.Graph, proteins: List[str], precursors: List[str]) -> np.ndarray:
    """Builds the binary structural feasibility mask M mapping proteins to precursors."""
    p_idx = {p: i for i, p in enumerate(proteins)}
    mask = np.zeros((len(proteins), len(precursors)))
    for i, prec in enumerate(precursors):
        peps = graph.predecessors(prec) if graph.is_directed() else graph.neighbors(prec)
        for pep in peps:
            if graph.nodes[pep].get("type") == "Peptide":
                prots = graph.predecessors(pep) if graph.is_directed() else graph.neighbors(pep)
                for prot in prots:
                    if prot in p_idx:
                        mask[p_idx[prot], i] = 1.0
    return mask


def group_isoforms(proteins: List[str], mask: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Aggregates structurally identical proteins sharing the exact same peptide evidence signature."""
    prots = [str(p) for p in proteins]
    m: Dict[Tuple[float, ...], List[str]] = {}
    for i, p in enumerate(prots):
        m.setdefault(tuple(mask[i, :]), []).append(p)
    return ["; ".join(plist) for plist in m.values()], np.array(list(m.keys()))
