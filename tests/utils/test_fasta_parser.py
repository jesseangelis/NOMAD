"""Unit tests for nomad.utils.fasta_parser.FastaParser."""

import gzip

import networkx as nx
import pytest

from nomad.utils.fasta_parser import FastaParser


@pytest.mark.unit
def test_init_resolves_path(empty_graph, tmp_path):
    """Verifies that FastaParser resolves the given path to an absolute path."""
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(">sp|P1|NAME\nAAAA\n")

    parser = FastaParser(empty_graph, str(fasta_file))

    assert parser.file_path == parser.file_path.resolve()


@pytest.mark.unit
def test_init_raises_on_missing_file(empty_graph):
    """Verifies that FastaParser raises FileNotFoundError for a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        FastaParser(empty_graph, "non_existent.fasta")


@pytest.mark.unit
def test_extract_id_uniprot_format(tmp_path):
    """Verifies _extract_id parses a UniProt sp| header correctly."""
    fasta_file = tmp_path / "dummy.fasta"
    fasta_file.write_text("")
    parser = FastaParser(nx.DiGraph(), str(fasta_file))

    assert parser._extract_id(">sp|P12345|PROT1_HUMAN Protein 1") == "P12345"


@pytest.mark.unit
def test_extract_id_generic_format(tmp_path):
    """Verifies _extract_id returns the first token for non-UniProt headers."""
    fasta_file = tmp_path / "dummy.fasta"
    fasta_file.write_text("")
    parser = FastaParser(nx.DiGraph(), str(fasta_file))

    assert parser._extract_id(">GeneID:12345 Description") == "GeneID:12345"


@pytest.mark.unit
def test_parse_adds_protein_nodes(empty_graph, tmp_path):
    """Verifies that parse() adds correctly typed Protein nodes to the graph."""
    fasta_file = tmp_path / "proteins.fasta"
    fasta_file.write_text(
        ">sp|P12345|PROT1_HUMAN Protein 1\n"
        "MKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF\n"
        ">sp|P67890|PROT2_HUMAN Protein 2\n"
        "MKAWLLLLLLVGLQSWYSGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF\n"
    )

    FastaParser(empty_graph, str(fasta_file)).parse()

    header1 = ">sp|P12345|PROT1_HUMAN Protein 1"
    assert header1 in empty_graph.nodes
    assert empty_graph.nodes[header1]["type"] == "Protein"
    assert "MKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF" in (
        empty_graph.nodes[header1]["sequence"]
    )


@pytest.mark.unit
def test_parse_gzip_file(empty_graph, tmp_path):
    """Verifies that parse() handles gzip-compressed FASTA files correctly."""
    content = (
        ">sp|P12345|PROT1_HUMAN Protein 1\n"
        "MKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF\n"
        ">sp|P67890|PROT2_HUMAN Protein 2\n"
        "MKAWLLLLLLVGLQSWYSGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF\n"
    )
    gz_file = tmp_path / "proteins.fasta.gz"
    gz_file.write_bytes(gzip.compress(content.encode("utf-8")))

    FastaParser(empty_graph, str(gz_file)).parse()

    assert ">sp|P12345|PROT1_HUMAN Protein 1" in empty_graph.nodes
    assert ">sp|P67890|PROT2_HUMAN Protein 2" in empty_graph.nodes


@pytest.mark.unit
def test_parse_duplicate_headers_different_sequences(empty_graph, tmp_path):
    """Verifies that duplicate headers with different sequences get disambiguated."""
    fasta_file = tmp_path / "dupes.fasta"
    fasta_file.write_text(
        ">sp|P1|NAME\nAAAAAA\n"
        ">sp|P1|NAME\nBBBBBB\n"
    )

    FastaParser(empty_graph, str(fasta_file)).parse()

    assert ">sp|P1|NAME" in empty_graph.nodes
    assert ">sp|P1$2|NAME" in empty_graph.nodes
    assert empty_graph.nodes[">sp|P1|NAME"]["sequence"] == "AAAAAA"
    assert empty_graph.nodes[">sp|P1$2|NAME"]["sequence"] == "BBBBBB"


@pytest.mark.unit
def test_parse_duplicate_headers_same_sequence(empty_graph, tmp_path):
    """Verifies that exact duplicate entries (same header and sequence) are deduplicated."""
    fasta_file = tmp_path / "exact_dupes.fasta"
    fasta_file.write_text(
        ">sp|P1|NAME\nAAAAAA\n"
        ">sp|P1|NAME\nAAAAAA\n"
    )

    FastaParser(empty_graph, str(fasta_file)).parse()

    assert len(empty_graph.nodes) == 1
    assert ">sp|P1|NAME" in empty_graph.nodes
