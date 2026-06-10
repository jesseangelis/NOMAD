"""Unit tests for nomad.utils.db.db_read."""

import pytest

from nomad.utils.db.db_read import (
    load_dose_response,
    load_emissions,
    load_intensities,
    load_raw_intensities,
)


@pytest.mark.unit
def test_load_intensities_returns_wide_dataframe(db_path):
    """Verifies that load_intensities returns a wide-format DataFrame with sample columns."""
    df = load_intensities(db_path)

    assert not df.is_empty()
    assert "protein" in df.columns
    # Wide-format: each sample should have its own column
    assert "S1" in df.columns
    assert "S2" in df.columns


@pytest.mark.unit
def test_load_intensities_includes_stderr_and_supported_columns(db_path):
    """Verifies that load_intensities pivots stderr and supported columns correctly."""
    df = load_intensities(db_path)

    assert "S1_stderr" in df.columns
    assert "S2_stderr" in df.columns
    assert "S1_supported" in df.columns
    assert "S2_supported" in df.columns


@pytest.mark.unit
def test_load_intensities_with_protein_filter(db_path):
    """Verifies that load_intensities correctly filters by a protein list."""
    df = load_intensities(db_path, proteins=["P1"])

    assert not df.is_empty()
    assert all(p == "P1" for p in df["protein"].to_list())


@pytest.mark.unit
def test_load_intensities_empty_on_missing_protein(db_path):
    """Verifies that load_intensities returns an empty DataFrame for unknown proteins."""
    df = load_intensities(db_path, proteins=["NONEXISTENT"])

    assert df.is_empty()


@pytest.mark.unit
def test_load_emissions_returns_rows(db_path):
    """Verifies that load_emissions returns a DataFrame with protein and precursor columns."""
    df = load_emissions(db_path)

    assert not df.is_empty()
    assert "protein" in df.columns
    assert "precursor" in df.columns
    assert "probability" in df.columns


@pytest.mark.unit
def test_load_emissions_with_protein_filter(db_path):
    """Verifies that load_emissions correctly filters by a protein list."""
    df = load_emissions(db_path, proteins=["P1"])

    assert not df.is_empty()
    assert all(p == "P1" for p in df["protein"].to_list())


@pytest.mark.unit
def test_load_raw_intensities_returns_long_format(db_path):
    """Verifies that load_raw_intensities returns a long-format DataFrame."""
    df = load_raw_intensities(db_path)

    assert not df.is_empty()
    assert "sample" in df.columns
    assert "intensity" in df.columns
    # Long format: one row per (protein, sample) pair
    assert df.height == 4  # 2 proteins × 2 samples


@pytest.mark.unit
def test_load_dose_response_returns_rows(db_path):
    """Verifies that load_dose_response returns dose-response fitting results."""
    df = load_dose_response(db_path)

    assert not df.is_empty()
    assert "protein" in df.columns
    assert "log2fc" in df.columns
    assert "relevance_score" in df.columns
