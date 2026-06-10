
import pytest
import networkx as nx
import tempfile
import os
from nomad.utils.fasta_parser import FastaParser

@pytest.fixture
def empty_graph():
    return nx.DiGraph()

@pytest.fixture
def sample_fasta():
    content = """>sp|P12345|PROT1_HUMAN Protein 1
MKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF
>sp|P67890|PROT2_HUMAN Protein 2
MKAWLLLLLLVGLQSWYSGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as f:
        f.write(content)
        path = f.name
    yield path
    os.remove(path)

def test_fasta_parser_init(empty_graph, sample_fasta):
    parser = FastaParser(empty_graph, sample_fasta)
    assert parser.g == empty_graph
    assert parser.file_path == parser.file_path.resolve()

def test_fasta_parser_init_filenotfound(empty_graph):
    with pytest.raises(FileNotFoundError):
        FastaParser(empty_graph, "non_existent.fasta")

def test_extract_id_uniprot():
    with tempfile.NamedTemporaryFile(suffix='.fasta', delete=False) as tmp:
        tmp_name = tmp.name
    try:
        parser = FastaParser(nx.DiGraph(), tmp_name)
        header = ">sp|P12345|PROT1_HUMAN Protein 1"
        assert parser._extract_id(header) == "P12345"
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def test_extract_id_generic():
    with tempfile.NamedTemporaryFile(suffix='.fasta', delete=False) as tmp:
        tmp_name = tmp.name
    try:
        parser = FastaParser(nx.DiGraph(), tmp_name)
        header = ">GeneID:12345 Description"
        assert parser._extract_id(header) == "GeneID:12345"
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def test_parse(empty_graph, sample_fasta):
    parser = FastaParser(empty_graph, sample_fasta)
    parser.parse()
    
    header1 = ">sp|P12345|PROT1_HUMAN Protein 1"
    header2 = ">sp|P67890|PROT2_HUMAN Protein 2"
    
    assert header1 in empty_graph.nodes
    assert header2 in empty_graph.nodes
    assert empty_graph.nodes[header1]["type"] == "Protein"
    assert "MKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF" in empty_graph.nodes[header1]["sequence"]

def test_parse_gzip(empty_graph):
    import gzip
    content = """>sp|P12345|PROT1_HUMAN Protein 1
MKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF
>sp|P67890|PROT2_HUMAN Protein 2
MKAWLLLLLLVGLQSWYSGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF
"""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.fasta.gz') as f:
        f.write(gzip.compress(content.encode('utf-8')))
        path = f.name
        
    try:
        parser = FastaParser(empty_graph, path)
        parser.parse()
        
        header1 = ">sp|P12345|PROT1_HUMAN Protein 1"
        header2 = ">sp|P67890|PROT2_HUMAN Protein 2"
        
        assert header1 in empty_graph.nodes
        assert header2 in empty_graph.nodes
        assert empty_graph.nodes[header1]["type"] == "Protein"
    finally:
        if os.path.exists(path):
            os.remove(path)

def test_parse_duplicates_different_sequences(empty_graph):
    content = """>sp|P1|NAME
AAAAAA
>sp|P1|NAME
BBBBBB
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as f:
        f.write(content)
        path = f.name
        
    try:
        parser = FastaParser(empty_graph, path)
        parser.parse()
        
        assert ">sp|P1|NAME" in empty_graph.nodes
        assert ">sp|P1$2|NAME" in empty_graph.nodes
        assert empty_graph.nodes[">sp|P1|NAME"]["sequence"] == "AAAAAA"
        assert empty_graph.nodes[">sp|P1$2|NAME"]["sequence"] == "BBBBBB"
    finally:
        if os.path.exists(path):
            os.remove(path)

def test_parse_duplicates_same_sequence(empty_graph):
    content = """>sp|P1|NAME
AAAAAA
>sp|P1|NAME
AAAAAA
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as f:
        f.write(content)
        path = f.name
        
    try:
        parser = FastaParser(empty_graph, path)
        parser.parse()
        
        assert len(empty_graph.nodes) == 1
        assert ">sp|P1|NAME" in empty_graph.nodes
    finally:
        if os.path.exists(path):
            os.remove(path)
