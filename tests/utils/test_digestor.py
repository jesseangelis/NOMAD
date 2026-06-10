
import pytest
import networkx as nx
from nomad.utils.digestor import InSilicoDigestor

@pytest.fixture
def graph_with_proteins():
    g = nx.DiGraph()
    g.add_node("P1", type="Protein", sequence="MKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF")
    g.add_node("P2", type="Protein", sequence="MKAWLLLLLLVGLQSWYSGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF")
    return g

def test_digestor_init(graph_with_proteins):
    digestor = InSilicoDigestor(graph_with_proteins, enzyme="trypsin")
    assert digestor.g == graph_with_proteins
    assert digestor.missed_cleavages == 2

def test_digestor_init_invalid_enzyme(graph_with_proteins):
    with pytest.raises(ValueError):
        InSilicoDigestor(graph_with_proteins, enzyme="invalid_enzyme")

def test_digest(graph_with_proteins):
    digestor = InSilicoDigestor(graph_with_proteins, enzyme="trypsin", min_pep_len=2)
    digestor.digest()
    
    # Check if peptides were added
    peptides = [n for n, d in graph_with_proteins.nodes(data=True) if d.get("type") == "Peptide"]
    assert len(peptides) > 0
    
    # Check edges
    edges = list(graph_with_proteins.edges(data=True))
    assert len(edges) > 0
    for u, v, d in edges:
        assert d["relation"] == "PRODUCES"
        assert graph_with_proteins.nodes[u]["type"] == "Protein"
        assert graph_with_proteins.nodes[v]["type"] == "Peptide"

def test_empty_graph():
    g = nx.DiGraph()
    digestor = InSilicoDigestor(g)
    digestor.digest()
    assert len(g.nodes) == 0

def test_digest_with_met_excision():
    g = nx.DiGraph()
    # Sequence starts with 'M'.
    g.add_node("P1", type="Protein", sequence="MAKAAAR")
    # Digestion with trypsin:
    # MAKAAAR -> MAK AAAR (with M)
    # AKAAAR -> AK AAAR (without M)
    
    # 1. With Met excision (default)
    digestor = InSilicoDigestor(g, enzyme="trypsin", min_pep_len=1, allow_met_excision=True)
    digestor.digest()
    peps = {n for n, d in g.nodes(data=True) if d.get("type") == "Peptide"}
    # MAKAAAR digested:
    # Original: MAK, AAAR, MAKAAAR
    # Met excised: AK, AAAR, AKAAAR
    # Expected: 'MAK', 'AK', 'AAAR', 'MAKAAAR', 'AKAAAR'
    assert "MAK" in peps
    assert "AK" in peps
    assert "AAAR" in peps
    assert "MAKAAAR" in peps
    assert "AKAAAR" in peps

def test_digest_without_met_excision():
    g = nx.DiGraph()
    g.add_node("P1", type="Protein", sequence="MAKAAAR")
    
    # 2. Without Met excision
    digestor = InSilicoDigestor(g, enzyme="trypsin", min_pep_len=1, allow_met_excision=False)
    digestor.digest()
    peps = {n for n, d in g.nodes(data=True) if d.get("type") == "Peptide"}
    # Expected: 'MAK', 'AAAR', 'MAKAAAR'
    assert "MAK" in peps
    assert "AAAR" in peps
    assert "MAKAAAR" in peps
    assert "AK" not in peps
    assert "AKAAAR" not in peps

def test_digest_met_excision_small_side_chain_rule():
    # 1. Allowed (Alanine - A)
    g1 = nx.DiGraph()
    g1.add_node("P1", type="Protein", sequence="MAKAAAR")
    digestor1 = InSilicoDigestor(g1, enzyme="trypsin", min_pep_len=1, allow_met_excision=True)
    digestor1.digest()
    peps1 = {n for n, d in g1.nodes(data=True) if d.get("type") == "Peptide"}
    assert "AK" in peps1

    # 2. Not allowed (Glutamine - Q)
    g2 = nx.DiGraph()
    g2.add_node("P2", type="Protein", sequence="MQKAAAR")
    digestor2 = InSilicoDigestor(g2, enzyme="trypsin", min_pep_len=1, allow_met_excision=True)
    digestor2.digest()
    peps2 = {n for n, d in g2.nodes(data=True) if d.get("type") == "Peptide"}
    # Original: MQK, AAAR, MQKAAAR
    # Met excised (not allowed): QK, AAAR, QKAAAR
    assert "MQK" in peps2
    assert "QK" not in peps2
