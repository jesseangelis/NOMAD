"""Shared fixtures for tests/utils/nmf/ test package."""

import networkx as nx
import polars as pl
import pytest


@pytest.fixture
def meta_df():
    """Provides mock metadata with two drugs and three doses each."""
    return pl.DataFrame({
        "file": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "name": ["DrugA", "DrugA", "DrugA", "DrugB", "DrugB", "DrugB"],
        "dose": [1.0, 10.0, 100.0, 1.0, 10.0, 100.0],
    })


@pytest.fixture
def graph_undirected():
    """Provides a minimal undirected graph with one protein, peptide, and precursor."""
    g = nx.Graph()
    g.add_node("P1", type="Protein")
    g.add_node("Pep1", type="Peptide")
    g.add_node("Pre1", type="Precursor")
    g.add_edge("P1", "Pep1", relation="PRODUCES")
    g.add_edge("Pep1", "Pre1", relation="HAS_PRECURSOR")

    g.add_edge("Pre1", "S1", relation="DETECTED_IN", intensity=10.0)
    g.add_edge("Pre1", "S2", relation="DETECTED_IN", intensity=100.0)
    g.add_edge("Pre1", "S3", relation="DETECTED_IN", intensity=1000.0)

    return g
