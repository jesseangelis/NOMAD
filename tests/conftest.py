"""Shared top-level fixtures for the NOMAD test suite."""

import pandas as pd
import networkx as nx
import numpy as np
import polars as pl
import pytest


@pytest.fixture
def graph_simple():
    """Provides a simple NetworkX graph for testing.

    The graph contains two proteins (P1, P2), three peptides, three
    precursors, and two samples with one missing detection (Pre3 absent
    in S2).
    """
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

    # PRODUCES edges
    g.add_edge("P1", "Pep1", relation="PRODUCES")
    g.add_edge("P1", "Pep2", relation="PRODUCES")
    g.add_edge("P2", "Pep2", relation="PRODUCES")
    g.add_edge("P2", "Pep3", relation="PRODUCES")

    # HAS_PRECURSOR edges
    g.add_edge("Pep1", "Pre1", relation="HAS_PRECURSOR")
    g.add_edge("Pep2", "Pre2", relation="HAS_PRECURSOR")
    g.add_edge("Pep3", "Pre3", relation="HAS_PRECURSOR")

    # DETECTED_IN edges
    g.add_edge("Pre1", "S1", relation="DETECTED_IN", intensity=100.0)
    g.add_edge("Pre2", "S1", relation="DETECTED_IN", intensity=200.0)
    g.add_edge("Pre3", "S1", relation="DETECTED_IN", intensity=300.0)
    g.add_edge("Pre1", "S2", relation="DETECTED_IN", intensity=150.0)
    g.add_edge("Pre2", "S2", relation="DETECTED_IN", intensity=250.0)
    # Pre3 is intentionally absent in S2

    return g


@pytest.fixture
def mock_quant_df():
    """Provides a mock wide-format quantification DataFrame."""
    return pl.DataFrame({
        "Protein": ["P1", "P2"],
        "S1": [10.0, 20.0],
        "S2": [12.0, 22.0],
        "S1_stderr": [1.0, 2.0],
        "S2_stderr": [1.2, 2.2],
    })


@pytest.fixture
def mock_data(tmp_path):
    """Creates dummy FASTA and quantification files for pipeline tests.

    Generates two synthetic proteins with monotone dose-response trends
    across twelve samples (four dose levels, three replicates each).

    Args:
        tmp_path: Pytest-provided temporary directory.

    Yields:
        A tuple of (fasta_path, quant_path, samples, doses).
    """
    fasta_path = tmp_path / "test_proteins.fasta"
    quant_path = tmp_path / "test_quantification.txt"

    with open(fasta_path, "w") as f:
        f.write(
            ">sp|P12345|PROT1_HUMAN Protein 1 GN=G1 "
            "OS=Homo sapiens OX=9606 PE=1 SV=1\n"
        )
        f.write("MRLEPTIDEONER\n")
        f.write(
            ">sp|P67890|PROT2_HUMAN Protein 2 GN=G1 "
            "OS=Homo sapiens OX=9606 PE=1 SV=1\n"
        )
        f.write("MRLEPTIDETWOR\n")

    samples = [f"S{i}" for i in range(1, 13)]
    doses = [0, 0, 0, 10, 10, 10, 50, 50, 50, 100, 100, 100]

    data = {
        "Peptide Sequence": ["LEPTIDEONER", "LEPTIDETWOR"],
        "Charge": [2, 2],
    }
    for i, s in enumerate(samples):
        data[f"{s} Intensity"] = [
            10000.0 * (1.0 + doses[i] / 10.0),
            10000.0 * (1.0 + doses[i] / 100.0),
        ]

    pd.DataFrame(data).to_csv(quant_path, sep="\t", index=False)

    return fasta_path, quant_path, samples, doses
