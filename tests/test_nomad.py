"""Integration tests for the top-level Nomad pipeline API and CLI."""

import os
import subprocess
import sys

import plotly.graph_objects as go
import polars as pl
import pytest

from nomad.nomad import Nomad
from nomad.utils import db
from nomad.utils.db.db_write import save_dose_response
from nomad.utils.plotting.structural import StructuralEvidencePlot
from nomad.utils.plotting.volcano import VolcanoPlot


@pytest.mark.integration
@pytest.mark.slow
def test_hierarchical_workflow(mock_data, tmp_path):
    """Tests the full Nomad API pipeline end-to-end.

    Verifies that loading FASTA, digesting, loading quantification,
    setting metadata, and fitting all succeed, and that the resulting
    quant_df has the expected columns and the database is populated.
    """
    fasta_path, quant_path, samples, doses = mock_data
    db_path = str(tmp_path / "test.sqlite")

    nm = Nomad(num_workers=1, db_path=db_path)
    nm.load_fasta(str(fasta_path))
    nm.digest(enzyme="trypsin", min_pep_len=3)
    nm.load_quantification(str(quant_path))

    meta_path = tmp_path / "metadata.csv"
    pl.DataFrame({
        "name": ["DrugA"] * 12,
        "file": samples,
        "dose": doses,
        "unit": ["nM"] * 12,
    }).write_csv(meta_path)

    nm.set_metadata(str(meta_path))
    nm.fit()

    assert nm.quant_df is not None

    expected_cols = {"protein", "gene_symbol", "entry_name", "description", "n_proteins"}
    assert expected_cols.issubset(set(nm.quant_df.columns))
    assert nm.quant_df.height == 2

    loaded_intensities = db.load_intensities(db_path)
    assert not loaded_intensities.is_empty()
    assert {"protein", "gene_symbol", "entry_name", "description"}.issubset(
        set(loaded_intensities.columns)
    )


@pytest.mark.integration
@pytest.mark.slow
def test_cli_entry_point(mock_data, tmp_path):
    """Tests the nomad module CLI entry point produces a valid TSV output."""
    fasta_path, quant_path, samples, doses = mock_data

    meta_path = tmp_path / "metadata.csv"
    pl.DataFrame({
        "name": ["DrugA"] * 12,
        "file": samples,
        "dose": doses,
        "unit": ["nM"] * 12,
    }).write_csv(meta_path)

    quant_tsv = str(tmp_path / "quantification.tsv")
    db_path = str(tmp_path / "test_cli.sqlite")

    subprocess.run(
        [
            sys.executable, "-m", "nomad.nomad",
            "--fasta", str(fasta_path),
            "--evidence", str(quant_path),
            "--metadata", str(meta_path),
            "--out", quant_tsv,
            "--db", db_path,
            "--workers", "1",
        ],
        check=True,
    )

    assert os.path.exists(quant_tsv)
    quant_df = pl.read_csv(quant_tsv, separator="\t")
    expected_cols = {"protein", "gene_symbol", "entry_name", "description", "n_proteins"}
    assert expected_cols.issubset(set(quant_df.columns))


@pytest.mark.integration
@pytest.mark.slow
def test_plotting_api(mock_data, tmp_path):
    """Tests that the Nomad plotting API returns valid Plotly Figure objects."""
    fasta_path, quant_path, samples, doses = mock_data

    nm = Nomad(num_workers=1, db_path=str(tmp_path / "test_plot.sqlite"))
    nm.load_fasta(str(fasta_path))
    nm.digest(enzyme="trypsin", min_pep_len=3)
    nm.load_quantification(str(quant_path))

    meta_path = tmp_path / "metadata_plot.csv"
    pl.DataFrame({
        "name": ["DrugA"] * 12,
        "file": samples,
        "dose": doses,
        "unit": ["nM"] * 12,
    }).write_csv(meta_path)
    nm.set_metadata(str(meta_path))
    nm.fit()

    fig_struct = nm.structural_evidence("P12345")
    assert fig_struct is not None
    assert isinstance(fig_struct, go.Figure)

    fig_diag = nm.plot_performance(output_dir=str(tmp_path / "plots"))
    assert fig_diag is not None
    assert isinstance(fig_diag, go.Figure)


@pytest.mark.integration
@pytest.mark.slow
def test_db_only_plotting(mock_data, tmp_path):
    """Tests that plotting classes can render directly from a database path."""
    fasta_path, quant_path, samples, doses = mock_data
    db_path = str(tmp_path / "test_db_plotting.sqlite")

    meta_path = tmp_path / "metadata_plot.csv"
    meta_df = pl.DataFrame({
        "name": ["DrugA"] * 12,
        "file": samples,
        "dose": doses,
        "unit": ["nM"] * 12,
    })
    meta_df.write_csv(meta_path)

    nm = Nomad(num_workers=1, db_path=db_path)
    nm.load_fasta(str(fasta_path))
    nm.digest(enzyme="trypsin", min_pep_len=3)
    nm.load_quantification(str(quant_path))
    nm.set_metadata(str(meta_path))
    nm.fit()

    fig_struct = StructuralEvidencePlot.plot(db_path, "P12345", metadata=meta_df)
    assert fig_struct is not None
    assert isinstance(fig_struct, go.Figure)

    save_dose_response(
        db_path,
        pl.DataFrame({
            "protein": ["P12345"],
            "drug": ["DrugA"],
            "log2fc": [1.5],
            "relevance_score": [3.0],
            "regulation": ["up"],
            "p_val": [0.001],
            "gene_symbol": ["PROT1"],
        }),
    )

    fig_volcano = VolcanoPlot.plot(db_path, output_dir=str(tmp_path / "volcano_out"))
    assert fig_volcano is not None
    assert isinstance(fig_volcano, go.Figure)


@pytest.mark.integration
@pytest.mark.slow
def test_diagnostic_loo_database(mock_data, tmp_path):
    """Tests that the diagnostic_loo table is populated with correctly typed rows."""
    import sqlite3

    fasta_path, quant_path, samples, doses = mock_data
    db_path = str(tmp_path / "test_loo.sqlite")

    meta_path = tmp_path / "metadata_loo.csv"
    pl.DataFrame({
        "name": ["DrugA"] * 12,
        "file": samples,
        "dose": doses,
        "unit": ["nM"] * 12,
    }).write_csv(meta_path)

    nm = Nomad(num_workers=1, db_path=db_path)
    nm.load_fasta(str(fasta_path))
    nm.digest(enzyme="trypsin", min_pep_len=3)
    nm.load_quantification(str(quant_path))
    nm.set_metadata(str(meta_path))
    nm.fit()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='diagnostic_loo'"
        )
        assert cursor.fetchone() is not None

        cursor.execute("SELECT * FROM diagnostic_loo")
        rows = cursor.fetchall()

    assert len(rows) > 0
    row = rows[0]
    # Columns: protein, sample, precursor, actual, predicted, scale
    assert len(row) == 6
    assert isinstance(row[0], str)    # protein
    assert isinstance(row[1], str)    # sample
    assert isinstance(row[2], str)    # precursor
    assert isinstance(row[3], float)  # actual
    assert isinstance(row[4], float)  # predicted
    assert isinstance(row[5], float)  # scale
