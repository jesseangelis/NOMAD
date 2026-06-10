"""Unit tests for nomad.utils.db.db_write."""

import sqlite3

import polars as pl
import pytest

from nomad.utils.db.db_write import (
    init_db,
    save_diagnostic_loo,
    save_dose_response,
    save_emissions,
    save_intensities,
)


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_init_db_creates_all_tables(tmp_path):
    """Verifies that init_db creates the four required tables in the database."""
    path = str(tmp_path / "test.sqlite")
    init_db(path)

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

    assert {"intensities", "emissions", "diagnostic_loo", "dose_response"}.issubset(tables)


@pytest.mark.unit
def test_init_db_is_idempotent(tmp_path):
    """Verifies that calling init_db twice does not raise or duplicate tables."""
    path = str(tmp_path / "test.sqlite")
    init_db(path)
    init_db(path)  # Should not raise

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='intensities'"
        )
        assert cursor.fetchone()[0] == 1


@pytest.mark.unit
def test_init_db_creates_parent_directories(tmp_path):
    """Verifies that init_db creates missing parent directories."""
    path = str(tmp_path / "nested" / "dir" / "test.sqlite")
    init_db(path)

    import os
    assert os.path.exists(path)


# ---------------------------------------------------------------------------
# save_intensities
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_save_intensities_inserts_rows(tmp_path):
    """Verifies that save_intensities writes one row per (protein, sample) pair."""
    path = str(tmp_path / "test.sqlite")
    init_db(path)

    quant_df = pl.DataFrame({
        "protein": ["P1"],
        "gene_symbol": ["GeneA"],
        "entry_name": ["PROA_HUMAN"],
        "description": ["Protein A"],
        "n_proteins": [1],
        "S1": [100.0],
        "S2": [150.0],
        "S1_stderr": [5.0],
        "S2_stderr": [7.0],
        "S1_supported": [True],
        "S2_supported": [True],
    })
    save_intensities(path, quant_df)

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM intensities")
        count = cursor.fetchone()[0]

    assert count == 2  # One row for S1, one for S2


@pytest.mark.unit
def test_save_intensities_upserts_on_conflict(tmp_path):
    """Verifies that save_intensities replaces existing rows on primary key conflict."""
    path = str(tmp_path / "test.sqlite")
    init_db(path)

    base_df = pl.DataFrame({
        "protein": ["P1"],
        "gene_symbol": ["GeneA"],
        "entry_name": ["PROA_HUMAN"],
        "description": ["Protein A"],
        "n_proteins": [1],
        "S1": [100.0],
        "S1_stderr": [5.0],
        "S1_supported": [True],
    })
    save_intensities(path, base_df)

    updated_df = base_df.with_columns(pl.lit(999.0).alias("S1"))
    save_intensities(path, updated_df)

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT intensity FROM intensities WHERE protein='P1' AND sample='S1'")
        intensity = cursor.fetchone()[0]

    assert intensity == 999.0


# ---------------------------------------------------------------------------
# save_emissions
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_save_emissions_inserts_rows(tmp_path):
    """Verifies that save_emissions writes one row per (protein, precursor) pair."""
    path = str(tmp_path / "test.sqlite")
    init_db(path)

    emissions_df = pl.DataFrame({
        "protein": ["P1", "P1"],
        "precursor": ["Pre1", "Pre2"],
        "probability": [0.9, 0.8],
    })
    save_emissions(path, emissions_df)

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM emissions")
        count = cursor.fetchone()[0]

    assert count == 2


# ---------------------------------------------------------------------------
# save_diagnostic_loo
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_save_diagnostic_loo_inserts_rows(tmp_path):
    """Verifies that save_diagnostic_loo writes LOO diagnostics to the database."""
    path = str(tmp_path / "test.sqlite")
    init_db(path)

    loo_data = [
        (100.0, 95.0, 200.0, "P1", "S1", "Pre1", True),
    ]
    save_diagnostic_loo(path, loo_data)

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM diagnostic_loo")
        rows = cursor.fetchall()

    assert len(rows) == 1
    # Schema: protein, sample, precursor, actual, predicted, scale
    assert rows[0][0] == "P1"
    assert rows[0][3] == 100.0


@pytest.mark.unit
def test_save_diagnostic_loo_empty_data_is_noop(tmp_path):
    """Verifies that passing an empty list to save_diagnostic_loo does not raise."""
    path = str(tmp_path / "test.sqlite")
    init_db(path)

    save_diagnostic_loo(path, [])  # Should not raise

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM diagnostic_loo")
        assert cursor.fetchone()[0] == 0


# ---------------------------------------------------------------------------
# save_dose_response
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_save_dose_response_inserts_rows(tmp_path):
    """Verifies that save_dose_response writes results to the dose_response table."""
    path = str(tmp_path / "test.sqlite")

    dose_df = pl.DataFrame({
        "protein": ["P1"],
        "drug": ["DrugA"],
        "log2fc": [1.5],
        "relevance_score": [3.0],
        "regulation": ["up"],
        "p_val": [0.001],
        "gene_symbol": ["GeneA"],
    })
    save_dose_response(path, dose_df)

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT protein, log2fc FROM dose_response")
        row = cursor.fetchone()

    assert row[0] == "P1"
    assert abs(row[1] - 1.5) < 1e-6


@pytest.mark.unit
def test_save_dose_response_handles_missing_optional_columns(tmp_path):
    """Verifies that save_dose_response fills in defaults for missing columns."""
    path = str(tmp_path / "test.sqlite")

    # Minimal DataFrame without regulation or p_val
    dose_df = pl.DataFrame({
        "protein": ["P2"],
        "drug": ["DrugB"],
        "log2fc": [0.5],
        "relevance_score": [1.0],
    })
    save_dose_response(path, dose_df)  # Should not raise

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT regulation, p_val FROM dose_response WHERE protein='P2'")
        row = cursor.fetchone()

    assert row is not None
