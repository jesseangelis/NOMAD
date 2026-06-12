"""Unit tests for nomad.utils.plotting.structural.StructuralEvidencePlot."""

import polars as pl
import plotly.graph_objects as go
import pytest

from nomad.utils.db.db_write import (
    init_db,
    save_diagnostic_loo,
    save_emissions,
    save_intensities,
)
from nomad.utils.plotting.structural import StructuralEvidencePlot


@pytest.fixture
def structural_db(tmp_path):
    """Creates an SQLite database populated for structural plot testing.

    Contains one protein (P12345) with two precursors and two samples.
    """
    db_path = str(tmp_path / "structural.sqlite")
    init_db(db_path)

    quant_df = pl.DataFrame({
        "protein": ["P12345"],
        "gene_symbol": ["PROT1"],
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
    save_intensities(db_path, quant_df)

    emissions_df = pl.DataFrame({
        "protein": ["P12345", "P12345"],
        "precursor": ["Pre1", "Pre2"],
        "probability": [0.9, 0.8],
    })
    save_emissions(db_path, emissions_df)

    loo_data = [
        (100.0, 95.0, 200.0, "P12345", "S1", "Pre1", True),
        (150.0, 148.0, 200.0, "P12345", "S2", "Pre2", False),
    ]
    save_diagnostic_loo(db_path, loo_data)

    return db_path


@pytest.fixture
def simple_metadata():
    """Provides minimal sample metadata matching the structural_db fixture."""
    return pl.DataFrame({
        "file": ["S1", "S2"],
        "name": ["DrugA", "DrugA"],
        "dose": [0.0, 10.0],
    })


@pytest.mark.unit
def test_plot_returns_figure(structural_db, simple_metadata):
    """Verifies that StructuralEvidencePlot.plot returns a non-None Plotly Figure."""
    fig = StructuralEvidencePlot.plot(
        structural_db, "P12345", metadata=simple_metadata
    )

    assert fig is not None
    assert isinstance(fig, go.Figure)


@pytest.mark.unit
def test_plot_returns_none_for_missing_db(tmp_path):
    """Verifies that plot returns None when the database file does not exist."""
    fig = StructuralEvidencePlot.plot(
        str(tmp_path / "nonexistent.sqlite"), "P12345"
    )

    assert fig is None


@pytest.mark.unit
def test_plot_contains_drug_selector_dropdown(structural_db, simple_metadata):
    """Verifies that the figure contains a drug-selection dropdown menu."""
    fig = StructuralEvidencePlot.plot(
        structural_db, "P12345", metadata=simple_metadata
    )

    assert len(fig.layout.updatemenus) > 0
    all_labels = [
        btn.label
        for menu in fig.layout.updatemenus
        for btn in menu.buttons
    ]
    assert "All Drugs" in all_labels


@pytest.mark.unit
def test_plot_residuals_mode_contains_log2fc_colorbar(structural_db, simple_metadata):
    """Verifies that residuals mode produces a trace with the Log2 Fold Change colorbar."""
    fig = StructuralEvidencePlot.plot(
        structural_db, "P12345", metadata=simple_metadata, bottom_right="residuals"
    )

    resid_trace = next(
        (t for t in fig.data if t.name == "Deconvolution View"), None
    )

    assert resid_trace is not None
    assert resid_trace.zmin == -2.0
    assert resid_trace.zmax == 2.0


@pytest.mark.unit
def test_plot_view_switcher_has_log2_fold_change_button(structural_db, simple_metadata):
    """Verifies that the view-switcher dropdown contains a 'Log2 Fold Change' button."""
    fig = StructuralEvidencePlot.plot(
        structural_db, "P12345", metadata=simple_metadata
    )

    all_labels = [
        btn.label
        for menu in fig.layout.updatemenus
        for btn in menu.buttons
    ]
    assert "Log2 Fold Change" in all_labels
