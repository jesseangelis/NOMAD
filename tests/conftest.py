
import pytest
import networkx as nx
import polars as pl
import numpy as np

@pytest.fixture
def graph_simple():
    """Provides a simple NetworkX graph for testing."""
    g = nx.Graph()
    # Samples
    g.add_node("S1", type="Sample")
    g.add_node("S2", type="Sample")
    
    # Proteins
    g.add_node("P1", type="Protein")
    g.add_node("P2", type="Protein")
    
    # Peptides
    g.add_node("Pep1", type="Peptide")
    g.add_node("Pep2", type="Peptide")
    g.add_node("Pep3", type="Peptide")
    
    # Precursors
    g.add_node("Pre1", type="Precursor")
    g.add_node("Pre2", type="Precursor")
    g.add_node("Pre3", type="Precursor")
    
    # Edges PRODUCES
    g.add_edge("P1", "Pep1", relation="PRODUCES")
    g.add_edge("P1", "Pep2", relation="PRODUCES")
    g.add_edge("P2", "Pep2", relation="PRODUCES")
    g.add_edge("P2", "Pep3", relation="PRODUCES")
    
    # Edges HAS_PRECURSOR
    g.add_edge("Pep1", "Pre1", relation="HAS_PRECURSOR")
    g.add_edge("Pep2", "Pre2", relation="HAS_PRECURSOR")
    g.add_edge("Pep3", "Pre3", relation="HAS_PRECURSOR")
    
    # Edges DETECTED_IN
    g.add_edge("Pre1", "S1", relation="DETECTED_IN", intensity=100.0)
    g.add_edge("Pre2", "S1", relation="DETECTED_IN", intensity=200.0)
    g.add_edge("Pre3", "S1", relation="DETECTED_IN", intensity=300.0)
    
    g.add_edge("Pre1", "S2", relation="DETECTED_IN", intensity=150.0)
    g.add_edge("Pre2", "S2", relation="DETECTED_IN", intensity=250.0)
    # Pre3 is missing in S2
    
    return g

@pytest.fixture
def mock_quant_df():
    """Provides a mock quantification DataFrame."""
    return pl.DataFrame({
        "Protein": ["P1", "P2"],
        "S1": [10.0, 20.0],
        "S2": [12.0, 22.0],
        "S1_stderr": [1.0, 2.0],
        "S2_stderr": [1.2, 2.2]
    })
