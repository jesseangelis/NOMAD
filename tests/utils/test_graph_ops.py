"""Unit tests for nomad.utils.graph_ops."""

import networkx as nx
import numpy as np
import pytest

from nomad.utils import graph_ops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prunable_graph():
    """Builds a graph where Pre3 has only 2 detections (below threshold of 3)."""
    g = nx.Graph()
    for node, ntype in [
        ("P1", "Protein"), ("P2", "Protein"),
        ("Pep1", "Peptide"), ("Pep2", "Peptide"), ("Pep3", "Peptide"),
        ("Pre1", "Precursor"), ("Pre2", "Precursor"), ("Pre3", "Precursor"),
        ("S1", "Sample"), ("S2", "Sample"), ("S3", "Sample"),
    ]:
        g.add_node(node, type=ntype)

    g.add_edge("P1", "Pep1", relation="PRODUCES")
    g.add_edge("P2", "Pep2", relation="PRODUCES")
    g.add_edge("P2", "Pep3", relation="PRODUCES")
    g.add_edge("Pep1", "Pre1", relation="HAS_PRECURSOR")
    g.add_edge("Pep2", "Pre2", relation="HAS_PRECURSOR")
    g.add_edge("Pep3", "Pre3", relation="HAS_PRECURSOR")

    # Pre1 and Pre2 have 3 detections each (above threshold)
    for s in ["S1", "S2", "S3"]:
        g.add_edge("Pre1", s, relation="DETECTED_IN", intensity=100.0)
        g.add_edge("Pre2", s, relation="DETECTED_IN", intensity=100.0)

    # Pre3 has only 2 detections (below threshold of 3)
    g.add_edge("Pre3", "S1", relation="DETECTED_IN", intensity=100.0)
    g.add_edge("Pre3", "S2", relation="DETECTED_IN", intensity=100.0)

    return g


# ---------------------------------------------------------------------------
# prune_graph
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_prune_graph_removes_under_detected_precursors():
    """Verifies that prune_graph removes precursors with fewer than 3 detections."""
    g = _make_prunable_graph()
    pruned = graph_ops.prune_graph(g)

    assert "Pre3" not in pruned.nodes
    assert "Pre1" in pruned.nodes
    assert "Pre2" in pruned.nodes


@pytest.mark.unit
def test_prune_graph_preserves_samples():
    """Verifies that Sample nodes are always retained regardless of pruning."""
    g = _make_prunable_graph()
    pruned = graph_ops.prune_graph(g)

    for s in ["S1", "S2", "S3"]:
        assert s in pruned.nodes


@pytest.mark.unit
def test_prune_graph_returns_copy():
    """Verifies that prune_graph does not mutate the original graph."""
    g = _make_prunable_graph()
    original_node_count = g.number_of_nodes()
    graph_ops.prune_graph(g)

    assert g.number_of_nodes() == original_node_count


# ---------------------------------------------------------------------------
# get_protein_nodes / get_precursor_nodes
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_protein_nodes_returns_sorted_list(graph_simple):
    """Verifies that get_protein_nodes returns a sorted list of protein node IDs."""
    proteins = graph_ops.get_protein_nodes(graph_simple)

    assert proteins == sorted(proteins)
    assert set(proteins) == {"P1", "P2"}


@pytest.mark.unit
def test_get_precursor_nodes_returns_sorted_list(graph_simple):
    """Verifies that get_precursor_nodes returns a sorted list of precursor node IDs."""
    precursors = graph_ops.get_precursor_nodes(graph_simple)

    assert precursors == sorted(precursors)
    assert set(precursors) == {"Pre1", "Pre2", "Pre3"}


# ---------------------------------------------------------------------------
# identify_components
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_identify_components_single_component(graph_simple):
    """Verifies that a fully connected graph yields a single component."""
    components = graph_ops.identify_components(graph_simple)

    assert len(components) == 1


@pytest.mark.unit
def test_identify_components_two_isolated_proteins():
    """Verifies that two disconnected proteins each yield their own component."""
    g = nx.Graph()
    for node, ntype in [
        ("PA", "Protein"), ("PepA", "Peptide"), ("PreA", "Precursor"),
        ("PB", "Protein"), ("PepB", "Peptide"), ("PreB", "Precursor"),
    ]:
        g.add_node(node, type=ntype)
    g.add_edge("PA", "PepA", relation="PRODUCES")
    g.add_edge("PepA", "PreA", relation="HAS_PRECURSOR")
    g.add_edge("PB", "PepB", relation="PRODUCES")
    g.add_edge("PepB", "PreB", relation="HAS_PRECURSOR")

    components = graph_ops.identify_components(g)

    assert len(components) == 2


# ---------------------------------------------------------------------------
# build_v_matrix
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_v_matrix_shape_and_values(graph_simple):
    """Verifies build_v_matrix shape and spot-checks known intensity values."""
    precursors = ["Pre1", "Pre2", "Pre3"]
    s2i = {"S1": 0, "S2": 1}

    v_mat = graph_ops.build_v_matrix(graph_simple, precursors, s2i)

    assert v_mat.shape == (2, 3)
    assert v_mat[0, 0] == 100.0   # Pre1 in S1
    assert v_mat[1, 2] == 0.0    # Pre3 absent in S2


# ---------------------------------------------------------------------------
# build_h_mask
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_h_mask_shape_and_values(graph_simple):
    """Verifies build_h_mask shape and spot-checks known connectivity values."""
    proteins = ["P1", "P2"]
    precursors = ["Pre1", "Pre2", "Pre3"]

    h_mask = graph_ops.build_h_mask(graph_simple, proteins, precursors)

    assert h_mask.shape == (2, 3)
    assert h_mask[0, 0] == 1.0   # P1 → Pep1 → Pre1
    assert h_mask[0, 2] == 0.0   # P1 does not produce Pre3


# ---------------------------------------------------------------------------
# group_isoforms
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_group_isoforms_distinct_proteins_stay_separate():
    """Verifies that proteins with different masks are kept as separate groups."""
    proteins = ["P1", "P2"]
    mask = np.array([[1.0, 0.0], [0.0, 1.0]])

    groups, grouped_mask = graph_ops.group_isoforms(proteins, mask)

    assert len(groups) == 2


@pytest.mark.unit
def test_group_isoforms_identical_masks_are_merged():
    """Verifies that proteins sharing the same peptide mask are merged into one group."""
    proteins = ["P1", "P2"]
    mask = np.array([[1.0, 1.0], [1.0, 1.0]])

    groups, grouped_mask = graph_ops.group_isoforms(proteins, mask)

    assert len(groups) == 1
    assert "P1" in groups[0]
    assert "P2" in groups[0]
