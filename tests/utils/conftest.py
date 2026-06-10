"""Shared fixtures for all tests under tests/utils/."""

import os
import tempfile

import networkx as nx
import pandas as pd
import pytest


@pytest.fixture
def graph_with_proteins():
    """Provides a directed graph pre-populated with two protein nodes.

    Both proteins have realistic tryptic sequences for digestor testing.
    """
    g = nx.DiGraph()
    g.add_node(
        "P1",
        type="Protein",
        sequence="MKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF",
    )
    g.add_node(
        "P2",
        type="Protein",
        sequence="MKAWLLLLLLVGLQSWYSGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF",
    )
    return g


@pytest.fixture
def empty_graph():
    """Provides an empty directed graph for parser tests."""
    return nx.DiGraph()


@pytest.fixture
def graph_with_peptides():
    """Provides a directed graph with two peptide nodes for quantification parser tests."""
    g = nx.DiGraph()
    g.add_node("PEPTIDE1", type="Peptide")
    g.add_node("PEPTIDE2", type="Peptide")
    return g


@pytest.fixture
def fragpipe_tsv(tmp_path):
    """Writes a minimal FragPipe combined_ion.tsv to a temporary path.

    Includes intensities across three samples so precursor detection
    thresholds (≥ 3) are satisfied.
    """
    data = {
        "Peptide Sequence": ["PEPTIDE1", "PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
        "Charge": [2, 2, 3, 2],
        "SampleA Intensity": [1000, 500, 3000, 4000],
        "SampleB Intensity": [2000, 0, 3000, 4000],
        "SampleC Intensity": [1500, 1500, 3000, 4000],
    }
    path = tmp_path / "combined_ion.tsv"
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)
    return str(path)
