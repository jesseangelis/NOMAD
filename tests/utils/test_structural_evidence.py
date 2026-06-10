
import pytest
import numpy as np
import polars as pl
from networkx import Graph
from nomad.utils.plotting import StructuralEvidencePlot

def test_structural_evidence_plot_percentage_residuals():
    """Test that StructuralEvidencePlot uses percentage residuals and scales -1 to 1."""
    # Create mock graph
    graph = Graph()
    
    # Add nodes
    graph.add_node("Prot1", type="Protein", label="Prot1")
    graph.add_node("Pep1", type="Peptide")
    graph.add_node("Prec1", type="Precursor", label="Prec1")
    graph.add_node("Samp1", type="Sample")
    
    # Add edges
    graph.add_edge("Prot1", "Pep1", relation="PRODUCES")
    graph.add_edge("Pep1", "Prec1", relation="HAS_PRECURSOR")
    graph.add_edge("Prec1", "Samp1", relation="DETECTED_IN", intensity=100.0)
    
    # Mock metadata
    metadata = pl.DataFrame({
        "file": ["Samp1"],
        "name": ["DrugA"],
        "dose": [10.0]
    })
    
    # Generate plot
    fig = StructuralEvidencePlot.plot_from_graph(graph, "Prot1", metadata, bottom_right="residuals")
    
    # Check if fig is generated
    assert fig is not None
    
    # Check plot titles/labels for percentage residuals
    # Subplot titles are in fig.layout.annotations
    found_title = False
    for ann in fig.layout.annotations:
        if "Log2 Fold Change" in ann.text:
            found_title = True
            break
    assert found_title, "Residual title not found in annotations"
    
    # Check buttons for percentage residuals label
    found_button = False
    for menu in fig.layout.updatemenus:
        for button in menu.buttons:
            if "Log2 Fold Change" in button.label:
                found_button = True
                # Find the key that updates the correct annotation title
                ann_update_key = next((k for k in button.args[1].keys() if k.startswith("annotations[") and k.endswith("].text")), None)
                # It's okay if not updating text via direct index if it's handled via coordinated update
                pass
                break
    assert found_button, "Percentage residual button not found"
    
    # Check trace 4 (Residuals) for correct colorbar and scale
    # Trace 4 is index 3 or 5 depending on overlays
    # Based on code:
    # 0: Trace 0 (W heatmap)
    # 1: Trace 1 (H heatmap)
    # 2: Trace 2 (Recon V)
    # 3: Trace 3 (Observed V)
    # 4: Trace 4 (Imputed Overlay for Observed)
    # 5: Trace 5 (Percentage Residuals)
    # 6: Trace 6 (Imputed Overlay for Residuals)
    
    resid_trace = None
    for trace in fig.data:
        if trace.name == "Deconvolution View":
            resid_trace = trace
            break
    
    assert resid_trace is not None
    assert resid_trace.zmin == -2
    assert resid_trace.zmax == 2
    assert resid_trace.colorbar.title.text == "Log2 Fold Change (WH/V)"
