import pytest
import numpy as np
import torch
import polars as pl
import networkx as nx
from nomad.utils.nmf import NMFFit

@pytest.fixture
def meta_df():
    """Provides mock metadata with 3 doses per drug to test efficiency."""
    return pl.DataFrame({
        "file": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "name": ["DrugA", "DrugA", "DrugA", "DrugB", "DrugB", "DrugB"],
        "dose": [1.0, 10.0, 100.0, 1.0, 10.0, 100.0]
    })

@pytest.fixture
def graph_undirected():
    """Provides a simple undirected Graph for testing components."""
    g = nx.Graph()
    g.add_node("P1", type="Protein")
    g.add_node("Pep1", type="Peptide")
    g.add_node("Pre1", type="Precursor")
    g.add_edge("P1", "Pep1", relation="PRODUCES")
    g.add_edge("Pep1", "Pre1", relation="HAS_PRECURSOR")
    
    # Intensities in Samples (S1-S3 for P1)
    g.add_edge("Pre1", "S1", relation="DETECTED_IN", intensity=10.0)
    g.add_edge("Pre1", "S2", relation="DETECTED_IN", intensity=100.0)
    g.add_edge("Pre1", "S3", relation="DETECTED_IN", intensity=1000.0)
    
    return g

def test_component_identification(graph_undirected, meta_df):
    from nomad.utils import graph_ops
    components = graph_ops.identify_components(graph_undirected)
    assert len(components) == 1

def test_constraint_efficiency(meta_df):
    fitter = NMFFit(nx.Graph(), meta_df)
    from nomad.utils.nmf import optimizer
    
    w_smooth = torch.tensor([[3.16], [10.0], [31.6]], dtype=torch.float32, device=fitter.device)
    V = w_smooth**2
    M = torch.ones((1, 1), device=fitter.device)
    h = torch.ones((1, 1), device=fitter.device)
    
    A = torch.sparse_csr_tensor(
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        size=(3, 3), device=fitter.device
    )
    
    loss_smooth, _ = optimizer.compute_nmf_loss(w_smooth, h, V, M, 0.0, 10.0, 0.0, {"DrugA": A}, [])
    
    w_jagged = torch.tensor([[3.16], [31.6], [3.16]], dtype=torch.float32, device=fitter.device)
    loss_jagged, _ = optimizer.compute_nmf_loss(w_jagged, h, V, M, 0.0, 10.0, 0.0, {"DrugA": A}, [])
    
    assert loss_smooth.item() < loss_jagged.item()

def test_run_convergence(meta_df):
    # Simple 1x1 NMF
    v_matrix = np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]])
    h_mask = np.array([[1.0]])
    
    fitter = NMFFit(nx.Graph(), meta_df)
    from nomad.utils.nmf import optimizer
    w_fit, h_fit, w_se, mp = optimizer.optimize_component(v_matrix, h_mask, fitter.device, 1e-3, 0.0, 0.0, {}, [])
    
    assert np.allclose(w_fit @ h_fit, v_matrix, rtol=0.2)
    assert np.all(w_fit >= 0)

def test_full_fit_orchestration(graph_undirected, meta_df):
    fitter = NMFFit(graph_undirected, meta_df)
    quant_df, stats = fitter.fit()
    
    assert not quant_df.is_empty()
    assert "protein" in quant_df.columns
    assert "S1" in quant_df.columns
    assert quant_df.height == 1 # 1 component with 1 protein group
