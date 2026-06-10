"""Shared fixtures for the tests/utils/db/ test package."""

import polars as pl
import pytest

from nomad.utils.db.db_write import (
    init_db,
    save_diagnostic_loo,
    save_dose_response,
    save_emissions,
    save_intensities,
)


@pytest.fixture
def db_path(tmp_path):
    """Creates an initialised, populated SQLite database for DB tests.

    All four tables (intensities, emissions, diagnostic_loo, dose_response)
    are pre-populated with minimal synthetic data.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        Path string to the populated database file.
    """
    path = str(tmp_path / "test.sqlite")
    init_db(path)

    quant_df = pl.DataFrame({
        "protein": ["P1", "P2"],
        "gene_symbol": ["GeneA", "GeneB"],
        "entry_name": ["PROA_HUMAN", "PROB_HUMAN"],
        "description": ["Protein A", "Protein B"],
        "n_proteins": [1, 1],
        "S1": [100.0, 200.0],
        "S2": [150.0, 250.0],
        "S1_stderr": [5.0, 10.0],
        "S2_stderr": [7.0, 12.0],
        "S1_supported": [True, True],
        "S2_supported": [True, False],
    })
    save_intensities(path, quant_df)

    emissions_df = pl.DataFrame({
        "protein": ["P1", "P1", "P2"],
        "precursor": ["Pre1", "Pre2", "Pre2"],
        "probability": [0.9, 0.8, 0.7],
    })
    save_emissions(path, emissions_df)

    loo_data = [
        (100.0, 95.0, 200.0, "P1", "S1", "Pre1", True),
        (150.0, 148.0, 200.0, "P2", "S2", "Pre2", False),
    ]
    save_diagnostic_loo(path, loo_data)

    dose_resp_df = pl.DataFrame({
        "protein": ["P1"],
        "drug": ["DrugA"],
        "log2fc": [1.5],
        "relevance_score": [3.0],
        "regulation": ["up"],
        "p_val": [0.001],
        "gene_symbol": ["GeneA"],
    })
    save_dose_response(path, dose_resp_df)

    return path
