import os
import subprocess
import sys
import pytest
import pandas as pd
import polars as pl
import numpy as np
import plotly.graph_objects as go
from nomad.nomad import Nomad

@pytest.fixture
def mock_data(tmp_path):
    """Create dummy data with significant trends for testing."""
    fasta_path = tmp_path / "test_proteins.fasta"
    quant_path = tmp_path / "test_quantification.txt"
    
    with open(fasta_path, "w") as f:
        f.write(">sp|P12345|PROT1_HUMAN Protein 1 GN=G1 OS=Homo sapiens OX=9606 PE=1 SV=1\n")
        f.write("MRLEPTIDEONER\n")
        f.write(">sp|P67890|PROT2_HUMAN Protein 2 GN=G1 OS=Homo sapiens OX=9606 PE=1 SV=1\n")
        f.write("MRLEPTIDETWOR\n")

    samples = [f"S{i}" for i in range(1, 13)]
    doses = [0, 0, 0, 10, 10, 10, 50, 50, 50, 100, 100, 100]
    
    data = {
        "Peptide Sequence": ["LEPTIDEONER", "LEPTIDETWOR"],
        "Charge": [2, 2],
    }
    for i, s in enumerate(samples):
        data[f"{s} Intensity"] = [
            10000.0 * (1.0 + doses[i]/10.0),
            10000.0 * (1.0 + doses[i]/100.0)
        ]
        
    df = pd.DataFrame(data)
    df.to_csv(quant_path, sep="\t", index=False)
    
    return fasta_path, quant_path, samples, doses

def test_hierarchical_workflow(mock_data, tmp_path):
    fasta_path, quant_path, samples, doses = mock_data
    db_path = str(tmp_path / "test.sqlite")
    
    nm = Nomad(num_workers=1, db_path=db_path)
    nm.load_fasta(str(fasta_path))
    nm.digest(enzyme="trypsin", min_pep_len=3)
    nm.load_quantification(str(quant_path))
    
    meta_path = tmp_path / "metadata.csv"
    meta_df = pl.DataFrame({
        "name": ["DrugA"] * 12,
        "file": samples,
        "dose": doses,
        "unit": ["nM"] * 12
    })
    meta_df.write_csv(meta_path)
    
    nm.set_metadata(str(meta_path))
    nm.fit()
    assert nm.quant_df is not None
    
    # Check that new columns are present
    expected_cols = {"protein", "gene_symbol", "entry_name", "description", "n_proteins"}
    assert expected_cols.issubset(set(nm.quant_df.columns))
    assert nm.quant_df.height == 2

    # Check database load_intensities
    from nomad.utils import db
    loaded_intensities = db.load_intensities(db_path)
    assert not loaded_intensities.is_empty()
    db_expected_cols = {"protein", "gene_symbol", "entry_name", "description"}
    assert db_expected_cols.issubset(set(loaded_intensities.columns))


def test_cli_scripts(mock_data, tmp_path):
    fasta_path, quant_path, samples, doses = mock_data
    
    # Write metadata
    meta_path = tmp_path / "metadata.csv"
    meta_df = pl.DataFrame({
        "name": ["DrugA"] * 12,
        "file": samples,
        "dose": doses,
        "unit": ["nM"] * 12
    })
    meta_df.write_csv(meta_path)
    
    quant_tsv = str(tmp_path / "quantification.tsv")
    db_path = str(tmp_path / "test_cli.sqlite")
    
    # 1. Run nomad module entry point
    cmd_quant = [
        sys.executable,
        "-m",
        "nomad.nomad",
        "--fasta", str(fasta_path),
        "--evidence", str(quant_path),
        "--metadata", str(meta_path),
        "--out", quant_tsv,
        "--db", db_path,
        "--workers", "1"
    ]
    subprocess.run(cmd_quant, check=True)
    assert os.path.exists(quant_tsv)
    
    quant_df = pl.read_csv(quant_tsv, separator="\t")
    expected_cols = {"protein", "gene_symbol", "entry_name", "description", "n_proteins"}
    assert expected_cols.issubset(set(quant_df.columns))



def test_plotting_api(mock_data, tmp_path):
    fasta_path, quant_path, samples, doses = mock_data
    nm = Nomad(num_workers=1, db_path=str(tmp_path / "test_plot.sqlite"))
    nm.load_fasta(str(fasta_path))
    nm.digest(enzyme="trypsin", min_pep_len=3)
    nm.load_quantification(str(quant_path))
    
    meta_path = tmp_path / "metadata_plot.csv"
    meta_df = pl.DataFrame({
        "name": ["DrugA"] * 12,
        "file": samples,
        "dose": doses,
        "unit": ["nM"] * 12
    })
    meta_df.write_csv(meta_path)
    nm.set_metadata(str(meta_path))
    nm.fit()
    
    # Structural
    fig_struct = nm.structural_evidence("P12345")
    assert fig_struct is not None
    assert isinstance(fig_struct, go.Figure)

    # Diagnostics Plot
    fig_diag = nm.plot_performance(output_dir=str(tmp_path / "plots"))
    assert fig_diag is not None
    assert isinstance(fig_diag, go.Figure)


def test_diagnostic_loo_database(mock_data, tmp_path):
    import sqlite3
    fasta_path, quant_path, samples, doses = mock_data
    db_path = str(tmp_path / "test_loo.sqlite")
    nm = Nomad(num_workers=1, db_path=db_path)
    nm.load_fasta(str(fasta_path))
    nm.digest(enzyme="trypsin", min_pep_len=3)
    nm.load_quantification(str(quant_path))
    
    meta_path = tmp_path / "metadata_loo.csv"
    meta_df = pl.DataFrame({
        "name": ["DrugA"] * 12,
        "file": samples,
        "dose": doses,
        "unit": ["nM"] * 12
    })
    meta_df.write_csv(meta_path)
    nm.set_metadata(str(meta_path))
    nm.fit()
    
    # Query database to verify diagnostic_loo table has entries
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='diagnostic_loo'")
    assert cursor.fetchone() is not None
    
    cursor.execute("SELECT * FROM diagnostic_loo")
    rows = cursor.fetchall()
    assert len(rows) > 0
    # Columns should be (protein, sample, precursor, actual, predicted, scale)
    row = rows[0]
    assert len(row) == 6
    assert isinstance(row[0], str)  # protein
    assert isinstance(row[1], str)  # sample
    assert isinstance(row[2], str)  # precursor
    assert isinstance(row[3], float) # actual
    assert isinstance(row[4], float) # predicted
    assert isinstance(row[5], float) # scale
    conn.close()
