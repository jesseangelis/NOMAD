"""Unit tests for nomad.utils.plotting.structural.StructuralEvidencePlot."""

import networkx as nx
import polars as pl
import pytest

from nomad.utils.plotting import StructuralEvidencePlot


@pytest.fixture
def simple_protein_graph():
    """Provides a minimal single-protein graph for structural plot testing."""
    g = nx.Graph()
    g.add_node("Prot1", type="Protein", label="Prot1")
    g.add_node("Pep1", type="Peptide")
    g.add_node("Prec1", type="Precursor", label="Prec1")
    g.add_node("Samp1", type="Sample")
    g.add_edge("Prot1", "Pep1", relation="PRODUCES")
    g.add_edge("Pep1", "Prec1", relation="HAS_PRECURSOR")
    g.add_edge("Prec1", "Samp1", relation="DETECTED_IN", intensity=100.0)
    return g


@pytest.fixture
def simple_metadata():
    """Provides minimal sample metadata matching the simple_protein_graph fixture."""
    return pl.DataFrame({
        "file": ["Samp1"],
        "name": ["DrugA"],
        "dose": [10.0],
    })


@pytest.mark.unit
def test_plot_from_graph_returns_figure(simple_protein_graph, simple_metadata):
    """Verifies that plot_from_graph returns a non-None Plotly Figure."""
    import plotly.graph_objects as go

    fig = StructuralEvidencePlot.plot_from_graph(
        simple_protein_graph, "Prot1", simple_metadata, bottom_right="residuals"
    )

    assert fig is not None
    assert isinstance(fig, go.Figure)


@pytest.mark.unit
def test_plot_from_graph_residual_subplot_title(simple_protein_graph, simple_metadata):
    """Verifies that the residuals panel title contains 'Log2 Fold Change'."""
    fig = StructuralEvidencePlot.plot_from_graph(
        simple_protein_graph, "Prot1", simple_metadata, bottom_right="residuals"
    )

    found = any(
        "Log2 Fold Change" in ann.text
        for ann in fig.layout.annotations
    )
    assert found, "Expected 'Log2 Fold Change' annotation not found in figure"


@pytest.mark.unit
def test_plot_from_graph_residual_button_exists(simple_protein_graph, simple_metadata):
    """Verifies that an update button labelled 'Log2 Fold Change' exists in the figure."""
    fig = StructuralEvidencePlot.plot_from_graph(
        simple_protein_graph, "Prot1", simple_metadata, bottom_right="residuals"
    )

    found = any(
        "Log2 Fold Change" in button.label
        for menu in fig.layout.updatemenus
        for button in menu.buttons
    )
    assert found, "Expected percentage residual button not found in figure"


@pytest.mark.unit
def test_plot_from_graph_deconvolution_trace_colorbar(simple_protein_graph, simple_metadata):
    """Verifies that the Deconvolution View trace has the correct colorbar and scale."""
    fig = StructuralEvidencePlot.plot_from_graph(
        simple_protein_graph, "Prot1", simple_metadata, bottom_right="residuals"
    )

    resid_trace = next(
        (t for t in fig.data if t.name == "Deconvolution View"), None
    )

    assert resid_trace is not None
    assert resid_trace.zmin == -2
    assert resid_trace.zmax == 2
    assert resid_trace.colorbar.title.text == "Log2 Fold Change (WH/V)"
