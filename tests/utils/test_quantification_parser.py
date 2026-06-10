"""Unit tests for nomad.utils.quantification_parser.QuantificationParser."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nomad.utils.quantification_parser import QuantificationParser


@pytest.mark.unit
def test_init_raises_on_missing_file(graph_with_peptides):
    """Verifies that QuantificationParser raises FileNotFoundError for a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        QuantificationParser(graph_with_peptides, "non_existent.tsv")


@pytest.mark.unit
def test_parse_raises_on_invalid_format(graph_with_peptides, tmp_path):
    """Verifies that parse() raises ValueError for an unrecognised column layout."""
    tsv_file = tmp_path / "bad.tsv"
    pd.DataFrame({"Invalid Column": [1, 2, 3]}).to_csv(tsv_file, sep="\t", index=False)

    parser = QuantificationParser(graph_with_peptides, str(tsv_file))
    with pytest.raises(ValueError, match="Only FragPipe combined_ion.tsv format is supported."):
        parser.parse()


@pytest.mark.unit
def test_parse_fragpipe_adds_sample_nodes(graph_with_peptides, fragpipe_tsv):
    """Verifies that parse() adds Sample nodes for each detected sample column."""
    QuantificationParser(graph_with_peptides, fragpipe_tsv).parse()

    assert "SampleA" in graph_with_peptides.nodes
    assert graph_with_peptides.nodes["SampleA"]["type"] == "Sample"
    assert "SampleB" in graph_with_peptides.nodes
    assert "SampleC" in graph_with_peptides.nodes


@pytest.mark.unit
def test_parse_fragpipe_adds_precursor_nodes(graph_with_peptides, fragpipe_tsv):
    """Verifies that parse() creates Precursor nodes with sequence_charge labels."""
    QuantificationParser(graph_with_peptides, fragpipe_tsv).parse()

    assert "PEPTIDE1_2" in graph_with_peptides.nodes
    assert graph_with_peptides.nodes["PEPTIDE1_2"]["type"] == "Precursor"
    assert "PEPTIDE2_3" in graph_with_peptides.nodes


@pytest.mark.unit
def test_parse_fragpipe_filters_unknown_peptides(graph_with_peptides, fragpipe_tsv):
    """Verifies that peptides absent from the graph are not added as precursors."""
    QuantificationParser(graph_with_peptides, fragpipe_tsv).parse()

    # PEPTIDE3 is not in graph_with_peptides, so no precursor node should exist
    assert "PEPTIDE3_2" not in graph_with_peptides.nodes


@pytest.mark.unit
def test_parse_fragpipe_intensities_are_normalised(graph_with_peptides, fragpipe_tsv):
    """Verifies that per-sample median normalisation is applied to edge intensities.

    Expected normalisation (see fixture for raw values):
      SampleA median: 2000  → scale 2250/2000 = 1.125
      SampleB median: 3000  → scale 2250/3000 = 0.75
      PEPTIDE1_2 in SampleA: (1000 + 500) * 1.125 = 1687.5
      PEPTIDE1_2 in SampleB: 2000 * 0.75           = 1500.0
    """
    QuantificationParser(graph_with_peptides, fragpipe_tsv).parse()

    assert ("PEPTIDE1", "PEPTIDE1_2") in graph_with_peptides.edges
    assert np.isclose(
        graph_with_peptides.edges[("PEPTIDE1_2", "SampleA")]["intensity"], 1687.5
    )
    assert np.isclose(
        graph_with_peptides.edges[("PEPTIDE1_2", "SampleB")]["intensity"], 1500.0
    )


@pytest.mark.unit
def test_parse_with_metadata_restricts_samples(graph_with_peptides, fragpipe_tsv):
    """Verifies that passing metadata restricts parsing to the listed samples only."""
    metadata = pl.DataFrame({"sample": ["SampleA", "SampleB"]})
    parser = QuantificationParser(graph_with_peptides, fragpipe_tsv, metadata=metadata)
    parser.parse()

    # Only two valid samples remain → fewer than three detections per precursor
    assert "SampleC" not in graph_with_peptides.nodes
    assert "PEPTIDE1_2" not in graph_with_peptides.nodes
