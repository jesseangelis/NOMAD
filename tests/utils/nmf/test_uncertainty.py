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
    """Verifies that two proteins with highly correlated uncertainties are flagged for merging."""
    M = torch.tensor([[0.12145286798477173, 0.5355274081230164, 0.15016603469848633], [0.4821828007698059, 0.18927443027496338, 0.6983774900436401]])
    V = torch.tensor([[9.616996765136719, 7.094485282897949, 5.132293224334717], [7.242255687713623, 5.464740753173828, 0.05975782871246338], [9.32310962677002, 4.101711273193359, 7.291594505310059]])
    w_opt = torch.tensor([[4.4590349197387695, 1.334875226020813], [0.5674946308135986, 4.362870216369629], [2.090116500854492, 0.42673230171203613]])
    h_opt = torch.tensor([[3.6656389236450195, 4.869297981262207, 2.746314764022827]])
    l_reg = 1.2156139146603279e-09

    _, merge_pair = uncertainty.calculate_uncertainty(
        w_opt, h_opt, V, M, lambda_reg=l_reg
    )

    assert merge_pair is not None
    assert isinstance(merge_pair, tuple)
    assert len(merge_pair) == 2


