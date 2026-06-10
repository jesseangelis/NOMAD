"""Integration tests for nomad.utils.nmf.single_fit.fit_single_component."""

import networkx as nx
import polars as pl
import pytest
import torch

from nomad.utils.nmf.single_fit import fit_single_component


@pytest.fixture
def simple_component():
    """Builds a minimal graph component for single_fit testing.

    Returns:
        A tuple of (component_set, graph, sample_to_idx, sample_list, device).
    """
    g = nx.Graph()
    nodes = {
        "P1": "Protein", "P2": "Protein",
        "Pep1": "Peptide", "Pep2": "Peptide", "Pep3": "Peptide",
        "Pre1": "Precursor", "Pre2": "Precursor", "Pre3": "Precursor",
        "S1": "Sample", "S2": "Sample", "S3": "Sample",
    }
    for node, ntype in nodes.items():
        g.add_node(node, type=ntype)

    g.add_edge("P1", "Pep1", relation="PRODUCES")
    g.add_edge("P1", "Pep2", relation="PRODUCES")
    g.add_edge("P2", "Pep2", relation="PRODUCES")
    g.add_edge("P2", "Pep3", relation="PRODUCES")
    g.add_edge("Pep1", "Pre1", relation="HAS_PRECURSOR")
    g.add_edge("Pep2", "Pre2", relation="HAS_PRECURSOR")
    g.add_edge("Pep3", "Pre3", relation="HAS_PRECURSOR")

    for s in ["S1", "S2", "S3"]:
        g.add_edge("Pre1", s, relation="DETECTED_IN", intensity=100.0)
        g.add_edge("Pre2", s, relation="DETECTED_IN", intensity=200.0)
        g.add_edge("Pre3", s, relation="DETECTED_IN", intensity=150.0)

    cc = {"P1", "P2", "Pep1", "Pep2", "Pep3", "Pre1", "Pre2", "Pre3"}
    s2i = {"S1": 0, "S2": 1, "S3": 2}
    sample_list = ["S1", "S2", "S3"]
    device = torch.device("cpu")

    return cc, g, s2i, sample_list, device


@pytest.mark.integration
@pytest.mark.slow
def test_fit_single_component_returns_quant_rows(simple_component):
    """Verifies that fit_single_component returns non-empty quant rows for a valid component."""
    cc, g, s2i, nodes, device = simple_component

    q_rows, e_rows, cv_p, rep_p, rs_p = fit_single_component(
        cc, g, s2i, nodes, device,
        scale=1.0, l_reg=1e-3, g_dr=0.0, b_rep=0.0,
        avg_m={}, r_grp=[],
    )

    assert q_rows is not None
    assert len(q_rows) > 0


@pytest.mark.integration
@pytest.mark.slow
def test_fit_single_component_returns_emission_rows(simple_component):
    """Verifies that fit_single_component returns non-empty emission probability rows."""
    cc, g, s2i, nodes, device = simple_component

    _, e_rows, _, _, _ = fit_single_component(
        cc, g, s2i, nodes, device,
        scale=1.0, l_reg=1e-3, g_dr=0.0, b_rep=0.0,
        avg_m={}, r_grp=[],
    )

    assert e_rows is not None
    assert len(e_rows) > 0


@pytest.mark.integration
@pytest.mark.slow
def test_fit_single_component_returns_loo_tuple(simple_component):
    """Verifies that fit_single_component returns a LOO CV tuple with correct length."""
    cc, g, s2i, nodes, device = simple_component

    _, _, cv_p, _, _ = fit_single_component(
        cc, g, s2i, nodes, device,
        scale=1.0, l_reg=1e-3, g_dr=0.0, b_rep=0.0,
        avg_m={}, r_grp=[],
    )

    if cv_p is not None:
        # Tuple: (actual, predicted, scale, protein, sample, precursor, is_specific)
        assert len(cv_p) >= 6


@pytest.mark.unit
def test_fit_single_component_empty_component_returns_none():
    """Verifies that an empty component set returns all None values."""
    g = nx.Graph()
    result = fit_single_component(
        set(), g, {}, [], torch.device("cpu"),
        scale=1.0, l_reg=1e-3, g_dr=0.0, b_rep=0.0,
        avg_m={}, r_grp=[],
    )

    assert result == (None, None, None, None, None)
