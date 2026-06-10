"""Tests for the NMF deconvolution diagnostics plotting logic."""

import pytest
import numpy as np
import plotly.graph_objects as go
from nomad.utils.plotting.diagnostics import DiagnosticsPlot


def test_plot_nmf_performance_legacy_2_tuples(tmp_path):
    """Test generating diagnostics plot with the legacy 2-tuple format (Actual, Predicted)."""
    cv_data = [(float(i), float(i + np.random.normal(0, 1))) for i in range(100)]
    rep_data = [(float(i), float(i + np.random.normal(0, 0.5))) for i in range(50)]

    fig = DiagnosticsPlot.plot_nmf_performance(cv_data, rep_data, output_dir=str(tmp_path))

    assert isinstance(fig, go.Figure)
    # Check that diagnostic plot file is written
    expected_file = tmp_path / "nmf_diagnostics.html"
    assert expected_file.exists()


def test_plot_nmf_performance_new_3_tuples(tmp_path):
    """Test generating diagnostics plot with the new 3-tuple format (Actual, Predicted, Scale) and RS data."""
    # Simulate actual, predicted, and scale values
    cv_data = [(float(i), float(i + np.random.normal(0, 1)), 150.0) for i in range(100)]
    rep_data = [(float(i), float(i + np.random.normal(0, 0.5))) for i in range(50)]
    rs_data = [(float(i), float(i + np.random.normal(0, 0.2)), 150.0) for i in range(100)]

    fig = DiagnosticsPlot.plot_nmf_performance(cv_data, rep_data, rs_data, output_dir=str(tmp_path))

    assert isinstance(fig, go.Figure)
    # Verify titles
    assert any("Sqrt(WH/scale)" in str(title) or "Sqrt(V/scale)" in str(title) for title in fig.layout.annotations)
    # Check that diagnostic plot file is written
    expected_file = tmp_path / "nmf_diagnostics.html"
    assert expected_file.exists()
