"""Unit tests for nomad.utils.digestor.InSilicoDigestor."""

import networkx as nx
import pytest

from nomad.utils.digestor import InSilicoDigestor


@pytest.mark.unit
def test_init_stores_graph_and_defaults(graph_with_proteins):
    """Verifies that InSilicoDigestor stores the graph and sets default attributes."""
    digestor = InSilicoDigestor(graph_with_proteins, enzyme="trypsin")

    assert digestor.g == graph_with_proteins
    assert digestor.missed_cleavages == 2


@pytest.mark.unit
def test_init_raises_on_invalid_enzyme(graph_with_proteins):
    """Verifies that an unsupported enzyme name raises ValueError."""
    with pytest.raises(ValueError):
        InSilicoDigestor(graph_with_proteins, enzyme="invalid_enzyme")


@pytest.mark.unit
def test_digest_adds_peptide_nodes_and_produces_edges(graph_with_proteins):
    """Verifies that digest() populates the graph with Peptide nodes and PRODUCES edges."""
    digestor = InSilicoDigestor(graph_with_proteins, enzyme="trypsin", min_pep_len=2)
    digestor.digest()

    peptides = [
        n for n, d in graph_with_proteins.nodes(data=True) if d.get("type") == "Peptide"
    ]
    assert len(peptides) > 0

    for u, v, d in graph_with_proteins.edges(data=True):
        assert d["relation"] == "PRODUCES"
        assert graph_with_proteins.nodes[u]["type"] == "Protein"
        assert graph_with_proteins.nodes[v]["type"] == "Peptide"


@pytest.mark.unit
def test_digest_empty_graph_is_noop():
    """Verifies that digesting an empty graph leaves it empty."""
    g = nx.DiGraph()
    InSilicoDigestor(g).digest()

    assert len(g.nodes) == 0


@pytest.mark.unit
def test_digest_with_met_excision_produces_cleaved_peptides():
    """Verifies that methionine excision generates both original and M-cleaved peptides."""
    g = nx.DiGraph()
    g.add_node("P1", type="Protein", sequence="MAKAAAR")
    InSilicoDigestor(g, enzyme="trypsin", min_pep_len=1, allow_met_excision=True).digest()

    peps = {n for n, d in g.nodes(data=True) if d.get("type") == "Peptide"}
    # Original tryptic peptides: MAK, AAAR, MAKAAAR
    # Met-excised peptides:       AK,  AAAR, AKAAAR
    assert "MAK" in peps
    assert "AK" in peps
    assert "AAAR" in peps
    assert "MAKAAAR" in peps
    assert "AKAAAR" in peps


@pytest.mark.unit
def test_digest_without_met_excision_suppresses_cleaved_peptides():
    """Verifies that disabling met excision suppresses M-cleaved peptide nodes."""
    g = nx.DiGraph()
    g.add_node("P1", type="Protein", sequence="MAKAAAR")
    InSilicoDigestor(g, enzyme="trypsin", min_pep_len=1, allow_met_excision=False).digest()

    peps = {n for n, d in g.nodes(data=True) if d.get("type") == "Peptide"}
    assert "MAK" in peps
    assert "AAAR" in peps
    assert "MAKAAAR" in peps
    assert "AK" not in peps
    assert "AKAAAR" not in peps


@pytest.mark.unit
def test_digest_met_excision_small_side_chain_rule():
    """Verifies the small-side-chain rule: Ala (A) allows excision, Gln (Q) does not."""
    # Alanine (A) has a small side chain — excision allowed
    g1 = nx.DiGraph()
    g1.add_node("P1", type="Protein", sequence="MAKAAAR")
    InSilicoDigestor(g1, enzyme="trypsin", min_pep_len=1, allow_met_excision=True).digest()
    peps1 = {n for n, d in g1.nodes(data=True) if d.get("type") == "Peptide"}
    assert "AK" in peps1

    # Glutamine (Q) has a large side chain — excision suppressed
    g2 = nx.DiGraph()
    g2.add_node("P2", type="Protein", sequence="MQKAAAR")
    InSilicoDigestor(g2, enzyme="trypsin", min_pep_len=1, allow_met_excision=True).digest()
    peps2 = {n for n, d in g2.nodes(data=True) if d.get("type") == "Peptide"}
    assert "MQK" in peps2
    assert "QK" not in peps2
