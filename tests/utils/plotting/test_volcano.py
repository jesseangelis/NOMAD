"""Unit tests for nomad.utils.plotting.volcano.VolcanoPlot."""

import polars as pl
import plotly.graph_objects as go
import pytest

from nomad.utils.plotting.volcano import VolcanoPlot


@pytest.fixture
def dose_response_df():
    """Provides a minimal dose-response DataFrame for volcano plot testing."""
    return pl.DataFrame({
        "protein": ["P1", "P2", "P3", "P4"],
        "drug": ["DrugA", "DrugA", "DrugA", "DrugA"],
        "log2fc": [2.5, -1.8, 0.1, -3.0],
        "relevance_score": [4.0, 3.5, 0.5, 5.0],
        "regulation": ["up", "down", "insignificant", "down"],
        "p_val": [0.001, 0.005, 0.8, 0.0001],
        "gene_symbol": ["GeneA", "GeneB", "GeneC", "GeneD"],
    })


@pytest.mark.unit
def test_plot_from_results_returns_figure(dose_response_df):
    """Verifies that a non-empty dose-response DataFrame produces a Plotly Figure."""
    fig = VolcanoPlot._plot_from_results(dose_response_df)

    assert fig is not None
    assert isinstance(fig, go.Figure)


@pytest.mark.unit
def test_plot_from_results_empty_df_returns_none():
    """Verifies that an empty DataFrame returns None rather than raising."""
    fig = VolcanoPlot._plot_from_results(pl.DataFrame())

    assert fig is None


@pytest.mark.unit
def test_plot_from_results_writes_html_file(dose_response_df, tmp_path):
    """Verifies that providing output_dir causes an HTML file to be written."""
    output_dir = str(tmp_path / "volcano_out")
    VolcanoPlot._plot_from_results(dose_response_df, output_dir=output_dir)

    import os
    assert os.path.exists(os.path.join(output_dir, "volcano_plot_all_drugs.html"))


@pytest.mark.unit
def test_plot_from_results_has_drug_selector_buttons(dose_response_df):
    """Verifies that the figure contains dropdown buttons for drug selection."""
    fig = VolcanoPlot._plot_from_results(dose_response_df)

    assert len(fig.layout.updatemenus) > 0
    buttons = fig.layout.updatemenus[0].buttons
    drug_labels = {btn.label for btn in buttons}
    assert "DrugA" in drug_labels


@pytest.mark.unit
def test_plot_from_results_multi_drug_generates_one_button_per_drug():
    """Verifies that multi-drug data produces one selector button per drug."""
    df = pl.DataFrame({
        "protein": ["P1", "P2"],
        "drug": ["DrugA", "DrugB"],
        "log2fc": [2.0, -1.5],
        "relevance_score": [3.0, 2.5],
        "regulation": ["up", "down"],
        "p_val": [0.01, 0.02],
        "gene_symbol": ["GeneA", "GeneB"],
    })

    fig = VolcanoPlot._plot_from_results(df)
    buttons = fig.layout.updatemenus[0].buttons

    drug_labels = {btn.label for btn in buttons}
    assert "DrugA" in drug_labels
    assert "DrugB" in drug_labels


@pytest.mark.unit
def test_plot_returns_none_for_missing_db(tmp_path):
    """Verifies that VolcanoPlot.plot returns None when the database file is absent."""
    fig = VolcanoPlot.plot(str(tmp_path / "nonexistent.sqlite"))

    assert fig is None
