"""Unit tests for nomad.utils.plotting.diagnostics.DiagnosticsPlot."""

import numpy as np
import plotly.graph_objects as go
import pytest

from nomad.utils.plotting.diagnostics import DiagnosticsPlot


@pytest.mark.unit
def test_plot_nmf_performance_legacy_2_tuples_returns_figure(tmp_path):
    """Verifies that the legacy 2-tuple format (actual, predicted) produces a Figure."""
    cv_data = [(float(i), float(i + np.random.normal(0, 1))) for i in range(100)]
    rep_data = [(float(i), float(i + np.random.normal(0, 0.5))) for i in range(50)]

    fig = DiagnosticsPlot.plot_nmf_performance(cv_data, rep_data, output_dir=str(tmp_path))

    assert isinstance(fig, go.Figure)


@pytest.mark.unit
def test_plot_nmf_performance_legacy_2_tuples_writes_html(tmp_path):
    """Verifies that the legacy format writes an HTML file to the output directory."""
    cv_data = [(float(i), float(i + np.random.normal(0, 1))) for i in range(100)]
    rep_data = [(float(i), float(i + np.random.normal(0, 0.5))) for i in range(50)]

    DiagnosticsPlot.plot_nmf_performance(cv_data, rep_data, output_dir=str(tmp_path))

    assert (tmp_path / "nmf_diagnostics.html").exists()


@pytest.mark.unit
def test_plot_nmf_performance_new_3_tuples_returns_figure(tmp_path):
    """Verifies that the 3-tuple format (actual, predicted, scale) produces a Figure."""
    cv_data = [(float(i), float(i + np.random.normal(0, 1)), 150.0) for i in range(100)]
    rep_data = [(float(i), float(i + np.random.normal(0, 0.5))) for i in range(50)]
    rs_data = [(float(i), float(i + np.random.normal(0, 0.2)), 150.0) for i in range(100)]

    fig = DiagnosticsPlot.plot_nmf_performance(
        cv_data, rep_data, rs_data, output_dir=str(tmp_path)
    )

    assert isinstance(fig, go.Figure)


@pytest.mark.unit
def test_plot_nmf_performance_new_3_tuples_has_sqrt_scaled_titles(tmp_path):
    """Verifies that 3-tuple mode produces subplot titles referencing sqrt-scaled axes."""
    cv_data = [(float(i), float(i + np.random.normal(0, 1)), 150.0) for i in range(100)]
    rep_data = [(float(i), float(i + np.random.normal(0, 0.5))) for i in range(50)]
    rs_data = [(float(i), float(i + np.random.normal(0, 0.2)), 150.0) for i in range(100)]

    fig = DiagnosticsPlot.plot_nmf_performance(
        cv_data, rep_data, rs_data, output_dir=str(tmp_path)
    )

    title_texts = [str(ann) for ann in fig.layout.annotations]
    assert any("Sqrt(WH/scale)" in t or "Sqrt(V/scale)" in t for t in title_texts)


@pytest.mark.unit
def test_plot_nmf_performance_3_tuples_writes_html(tmp_path):
    """Verifies that 3-tuple mode writes an HTML file to the output directory."""
    cv_data = [(float(i), float(i + np.random.normal(0, 1)), 150.0) for i in range(100)]
    rep_data = [(float(i), float(i + np.random.normal(0, 0.5))) for i in range(50)]
    rs_data = [(float(i), float(i + np.random.normal(0, 0.2)), 150.0) for i in range(100)]

    DiagnosticsPlot.plot_nmf_performance(
        cv_data, rep_data, rs_data, output_dir=str(tmp_path)
    )

    assert (tmp_path / "nmf_diagnostics.html").exists()
