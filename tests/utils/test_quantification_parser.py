
import os
import tempfile

import networkx as nx
import pandas as pd
import polars as pl
import pytest

from nomad.utils.quantification_parser import QuantificationParser


@pytest.fixture
def graph_with_peptides():
    g = nx.DiGraph()
    g.add_node("PEPTIDE1", type="Peptide")
    g.add_node("PEPTIDE2", type="Peptide")
    return g


@pytest.fixture
def fragpipe_file():
    # Provide intensities across 3 distinct samples to satisfy >=3 precursor detection bounds
    data = {
        "Peptide Sequence": ["PEPTIDE1", "PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
        "Charge": [2, 2, 3, 2], # Duplicate sequence+charge for aggregation test
        "SampleA Intensity": [1000, 500, 3000, 4000],
        "SampleB Intensity": [2000, 0, 3000, 4000],
        "SampleC Intensity": [1500, 1500, 3000, 4000],
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        df.to_csv(f, sep="\t", index=False)
        path = f.name
    yield path
    os.remove(path)


def test_init_filenotfound(graph_with_peptides):
    with pytest.raises(FileNotFoundError):
        QuantificationParser(graph_with_peptides, "non_existent.tsv")


def test_invalid_format(graph_with_peptides):
    df = pd.DataFrame({"Invalid Column": [1, 2, 3]})
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        df.to_csv(f, sep="\t", index=False)
        path = f.name

    parser = QuantificationParser(graph_with_peptides, path)
    with pytest.raises(ValueError, match="Only FragPipe combined_ion.tsv format is supported."):
        parser.parse()

    os.remove(path)


def test_parse_fragpipe_combined(graph_with_peptides, fragpipe_file):
    parser = QuantificationParser(graph_with_peptides, fragpipe_file)
    parser.parse()

    # Verify Samples added
    assert "SampleA" in parser.g.nodes
    assert parser.g.nodes["SampleA"]["type"] == "Sample"
    assert "SampleB" in parser.g.nodes
    assert "SampleC" in parser.g.nodes

    # Verify Precursors purely by sequence and charge
    assert "PEPTIDE1_2" in parser.g.nodes
    assert parser.g.nodes["PEPTIDE1_2"]["type"] == "Precursor"
    assert "PEPTIDE2_3" in parser.g.nodes

    # PEPTIDE3 should be filtered out (not in original graph)
    assert "PEPTIDE3_2" not in parser.g.nodes

    # Verify edges and normalized aggregated intensities
    # SampleA Median: median(500, 1000, 3000, 4000) = 2000
    # SampleB Median: median(2000, 3000, 4000) = 3000 (0 is dropped)
    # SampleC Median: median(1500, 1500, 3000, 4000) = 2250
    # Global target: median(2000, 3000, 2250) = 2250
    # Scale A: 2250 / 2000 = 1.125
    # Scale B: 2250 / 3000 = 0.75
    # PEPTIDE1_2 in SampleA: (1000 * 1.125) + (500 * 1.125) = 1687.5
    # PEPTIDE1_2 in SampleB: (2000 * 0.75) + 0 = 1500.0
    import numpy as np
    assert ("PEPTIDE1", "PEPTIDE1_2") in parser.g.edges
    assert np.isclose(parser.g.edges[("PEPTIDE1_2", "SampleA")]["intensity"], 1687.5)
    assert np.isclose(parser.g.edges[("PEPTIDE1_2", "SampleB")]["intensity"], 1500.0)


def test_metadata_filtering(graph_with_peptides, fragpipe_file):
    # Restrict metadata to only SampleA and SampleB
    metadata = pl.DataFrame({"sample": ["SampleA", "SampleB"]})
    parser = QuantificationParser(graph_with_peptides, fragpipe_file, metadata=metadata)
    parser.parse()

    # Precursors will have <3 detections (only 2 valid samples left), so parsing yields empty graph updates
    assert "SampleC" not in parser.g.nodes
    assert "PEPTIDE1_2" not in parser.g.nodes
