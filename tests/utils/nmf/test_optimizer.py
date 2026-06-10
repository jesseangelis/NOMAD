"""Unit tests for nomad.utils.nmf.optimizer."""

import networkx as nx
import numpy as np
import torch
import pytest

from nomad.utils.nmf import NMFFit
from nomad.utils.nmf import optimizer


@pytest.mark.unit
def test_constraint_efficiency_smooth_loss_is_lower(meta_df):
    """Verifies that a monotone dose-response incurs a lower loss than a jagged one."""
    fitter = NMFFit(nx.Graph(), meta_df)

    w_smooth = torch.tensor(
        [[3.16], [10.0], [31.6]], dtype=torch.float32, device=fitter.device
    )
    V = w_smooth ** 2
    M = torch.ones((1, 1), device=fitter.device)
    h = torch.ones((1, 1), device=fitter.device)

    A = torch.sparse_csr_tensor(
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        size=(3, 3),
        device=fitter.device,
    )

    loss_smooth, _ = optimizer.compute_nmf_loss(
        w_smooth, h, V, M, 0.0, 10.0, 0.0, {"DrugA": A}, []
    )

    w_jagged = torch.tensor(
        [[3.16], [31.6], [3.16]], dtype=torch.float32, device=fitter.device
    )
    loss_jagged, _ = optimizer.compute_nmf_loss(
        w_jagged, h, V, M, 0.0, 10.0, 0.0, {"DrugA": A}, []
    )

    assert loss_smooth.item() < loss_jagged.item()


@pytest.mark.unit
def test_run_convergence_recovers_input_matrix(meta_df):
    """Verifies that optimize_component fits a simple 1×1 NMF problem accurately."""
    v_matrix = np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]])
    h_mask = np.array([[1.0]])

    fitter = NMFFit(nx.Graph(), meta_df)
    w_fit, h_fit, w_se, mp = optimizer.optimize_component(
        v_matrix, h_mask, fitter.device, 1e-3, 0.0, 0.0, {}, []
    )

    assert np.allclose(w_fit @ h_fit, v_matrix, rtol=0.2)
    assert np.all(w_fit >= 0)


@pytest.mark.unit
def test_compute_nmf_loss_recon_only_is_nonnegative(meta_df):
    """Verifies that the reconstruction-only NMF loss is non-negative."""
    fitter = NMFFit(nx.Graph(), meta_df)

    w = torch.ones((3, 1), device=fitter.device)
    h = torch.ones((1, 2), device=fitter.device)
    V = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=fitter.device)
    M = torch.ones((1, 2), device=fitter.device)

    loss, recon_loss = optimizer.compute_nmf_loss(w, h, V, M, 0.0, 0.0, 0.0, {}, [])

    assert loss.item() >= 0
    assert recon_loss >= 0
