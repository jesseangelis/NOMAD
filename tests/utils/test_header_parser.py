"""Unit tests for nomad.utils.header_parser."""

import pytest

from nomad.utils.header_parser import (
    parse_group_header_details,
    parse_header_details,
    parse_uniprot_header,
)


# ---------------------------------------------------------------------------
# parse_uniprot_header
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("header,expected_id,expected_entry", [
    (">sp|P12345|ALBU_HUMAN Serum albumin", "P12345", "ALBU_HUMAN"),
    (">tr|A0A000|ENTRY_MOUSE Protein", "A0A000", "ENTRY_MOUSE"),
    ("sp|Q9Y6K9|GENE_HUMAN Description", "Q9Y6K9", "GENE_HUMAN"),
])
def test_parse_uniprot_header_standard(header, expected_id, expected_entry):
    """Verifies parse_uniprot_header extracts the correct ID and entry name."""
    protein_id, entry_name = parse_uniprot_header(header)

    assert protein_id == expected_id
    assert entry_name == expected_entry


@pytest.mark.unit
def test_parse_uniprot_header_generic_format():
    """Verifies that a non-UniProt header returns the full token as protein_id."""
    protein_id, entry_name = parse_uniprot_header(">generic_protein")

    assert protein_id == "generic_protein"
    assert entry_name == ""


@pytest.mark.unit
@pytest.mark.parametrize("bad_input", [None, "", 123])
def test_parse_uniprot_header_invalid_input(bad_input):
    """Verifies that invalid input is handled gracefully without raising."""
    result = parse_uniprot_header(bad_input)

    assert isinstance(result, tuple)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# parse_header_details
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_parse_header_details_full_uniprot():
    """Verifies parse_header_details extracts all four fields from a full UniProt header."""
    header = ">sp|P12345|ALBU_HUMAN Serum albumin GN=ALB OS=Homo sapiens OX=9606"
    protein_id, entry_name, gene_symbol, description = parse_header_details(header)

    assert protein_id == "P12345"
    assert entry_name == "ALBU_HUMAN"
    assert gene_symbol == "ALB"
    assert "Serum albumin" in description


@pytest.mark.unit
def test_parse_header_details_no_gn_tag():
    """Verifies that a header without a GN= tag returns an empty gene_symbol."""
    header = ">sp|P12345|ALBU_HUMAN Serum albumin OS=Homo sapiens"
    _, _, gene_symbol, _ = parse_header_details(header)

    assert gene_symbol == ""


@pytest.mark.unit
def test_parse_header_details_generic_header():
    """Verifies that a generic (non-UniProt) header is handled without errors."""
    protein_id, entry_name, gene_symbol, description = parse_header_details(
        ">GENERIC_ID some description here"
    )

    assert protein_id == "GENERIC_ID"
    assert entry_name == ""
    assert gene_symbol == ""
    assert "some description" in description


@pytest.mark.unit
def test_parse_header_details_empty_string():
    """Verifies that an empty string input returns four empty strings."""
    result = parse_header_details("")

    assert result == ("", "", "", "")


# ---------------------------------------------------------------------------
# parse_group_header_details
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_parse_group_header_details_single_protein():
    """Verifies that a single-protein group header is parsed identically to parse_header_details."""
    header = ">sp|P12345|ALBU_HUMAN Serum albumin GN=ALB OS=Homo sapiens OX=9606"
    accessions, entries, genes, descs = parse_group_header_details(header)

    assert "P12345" in accessions
    assert "ALBU_HUMAN" in entries
    assert "ALB" in genes


@pytest.mark.unit
def test_parse_group_header_details_two_proteins():
    """Verifies that a two-protein group header concatenates accessions with '; '."""
    group = (
        ">sp|P12345|ALBU_HUMAN Albumin GN=ALB OS=Homo sapiens OX=9606; "
        ">sp|P67890|GLOB_HUMAN Globin GN=HBB OS=Homo sapiens OX=9606"
    )
    accessions, entries, genes, _ = parse_group_header_details(group)

    assert "P12345" in accessions
    assert "P67890" in accessions
    assert "ALB" in genes
    assert "HBB" in genes


@pytest.mark.unit
def test_parse_group_header_details_deduplicates_identical_genes():
    """Verifies that duplicate gene symbols within a group are deduplicated."""
    group = (
        ">sp|P00001|A_HUMAN Protein A GN=GENE1 OS=Homo sapiens OX=9606; "
        ">sp|P00002|B_HUMAN Protein B GN=GENE1 OS=Homo sapiens OX=9606"
    )
    _, _, genes, _ = parse_group_header_details(group)

    # GENE1 should appear only once
    assert genes.count("GENE1") == 1
