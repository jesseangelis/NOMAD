"""Unit tests for nomad.utils.nmf.uncertainty."""

import numpy as np
import pytest
import torch

from nomad.utils.nmf import uncertainty


@pytest.mark.unit
def test_calculate_uncertainty_returns_finite_se_for_small_problem():
    """Verifies that SE estimates are finite for a minimal (2-sample, 1-protein) problem."""
    n_s, n_prot, n_p = 3, 1, 2
    # Simple V: each sample observes both precursors
    v_vals = [[100.0, 200.0], [150.0, 250.0], [120.0, 220.0]]
    V = torch.tensor(v_vals, dtype=torch.float32)
    M = torch.ones((n_prot, n_p), dtype=torch.float32)

    w_opt = torch.sqrt(torch.tensor([[100.0], [150.0], [120.0]], dtype=torch.float32))
    h_opt = torch.sqrt(torch.tensor([[1.0, 1.0]], dtype=torch.float32))

    w_se, merge_pair = uncertainty.calculate_uncertainty(w_opt, h_opt, V, M, lambda_reg=1e-3)

    assert w_se.shape == (n_s, n_prot)
    assert not np.any(np.isnan(w_se))
    assert merge_pair is None  # Single protein, no merge possible


@pytest.mark.unit
def test_calculate_uncertainty_skips_large_component():
    """Verifies that components with >8000 parameters skip SE and return NaN."""
    n_s, n_prot, n_p = 8001, 1, 1  # params.shape[0] = 8001 + 1 = 8002 > 8000
    V = torch.zeros((n_s, n_p), dtype=torch.float32)
    M = torch.ones((n_prot, n_p), dtype=torch.float32)
    w_opt = torch.ones((n_s, n_prot), dtype=torch.float32)
    h_opt = torch.ones((1, n_p), dtype=torch.float32)

    w_se, merge_pair = uncertainty.calculate_uncertainty(w_opt, h_opt, V, M, lambda_reg=1e-3)

    assert w_se.shape == (n_s, n_prot)
    assert np.all(np.isnan(w_se))
    assert merge_pair is None


@pytest.mark.unit
def test_calculate_uncertainty_detects_merge_pair_for_collinear_proteins():
    """Verifies that two proteins with identical evidence columns are flagged for merging."""
    n_s, n_p = 4, 2
    # Two identical precursor columns → two proteins with perfectly correlated evidence
    V = torch.tensor(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]],
        dtype=torch.float32,
    )
    M = torch.ones((2, n_p), dtype=torch.float32)
    w_opt = torch.sqrt(torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]], dtype=torch.float32))
    h_opt = torch.sqrt(torch.ones((1, n_p), dtype=torch.float32))

    _, merge_pair = uncertainty.calculate_uncertainty(w_opt, h_opt, V, M, lambda_reg=1e-3)

    # With perfectly collinear proteins, a merge pair should be suggested
    assert merge_pair is not None
    assert isinstance(merge_pair, tuple)
    assert len(merge_pair) == 2
