"""Gauss-Newton Jacobian uncertainty estimators and merge detection."""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)


def calculate_uncertainty(
    w_opt: torch.Tensor,
    h_opt: torch.Tensor,
    V: torch.Tensor,
    M: torch.Tensor,
    lambda_reg: float,
    scale: float = 1.0,
    P: Optional[torch.Tensor] = None,
    gamma_prior: float = 0.0,
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """Estimates standard errors using the Gauss-Newton approximation.

    Detects highly positively correlated off-diagonal elements to merge
    indifferentiable protein isoforms.
    """
    n_s, n_p = V.shape
    n_prot = M.shape[0]
    params = torch.cat([w_opt.view(-1), h_opt.view(-1)])

    if params.shape[0] > 8000:
        logger.warning("Component too large for exact Hessian calculation. Skipping SE.")
        return np.full((n_s, n_prot), np.nan), None

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        w_in = p[: n_s * n_prot].view(n_s, n_prot)
        h_in = p[n_s * n_prot :].view(1, n_p)
        w_sq, h_sq = w_in**2, h_in**2
        recon = torch.matmul(w_sq, h_sq * M)
        recon_sq = torch.sqrt(recon + 1e-8)

        res = recon_sq - torch.sqrt(V)
        res_flat = res[V > 0].view(-1)

        reg_weight = float(np.sqrt(lambda_reg))
        reg_res = reg_weight * w_in.reshape(-1)
        
        if P is not None and gamma_prior > 0.0:
            prior_weight = float(np.sqrt(gamma_prior))
            prior_res = prior_weight * torch.sqrt(P) * w_in
            return torch.cat([res_flat, reg_res, prior_res.reshape(-1)])
            
        return torch.cat([res_flat, reg_res])

    try:
        J = torch.autograd.functional.jacobian(residual_fn, params)
        H = 2.0 * torch.matmul(J.T, J)

        res_vals = residual_fn(params)
        loss_val = torch.sum(res_vals**2).item()
        df = max(1, (V > 0).sum().item() - params.shape[0])
        sigma_sq = loss_val / df

        ridge = torch.eye(H.shape[0], device=H.device) * 1e-5
        Cov = sigma_sq * torch.linalg.inv(H + ridge)

        diag_cov = torch.diag(Cov)
        se_flat = torch.sqrt(torch.clamp(diag_cov[: n_s * n_prot], min=0))
        w_se = (2.0 * torch.abs(w_opt) * se_flat.view(n_s, n_prot)).cpu().numpy()

        merge_pair: Optional[Tuple[int, int]] = None
        if n_prot > 1:
            W_Cov = Cov[: n_s * n_prot, : n_s * n_prot].view(n_s, n_prot, n_s, n_prot)
            sum_cov = W_Cov.sum(dim=(0, 2))
            std = torch.sqrt(torch.clamp(torch.diag(sum_cov), min=1e-9))
            corr = sum_cov / (std.unsqueeze(1) @ std.unsqueeze(0))

            # CORR-01 Fix: Look for high positive off-diagonal correlation
            mask_diag = torch.eye(n_prot, device=Cov.device, dtype=torch.bool)
            corr_masked = corr.masked_fill(mask_diag, -float("inf"))
            max_corr, max_idx = torch.max(corr_masked.view(-1), dim=0)

            if max_corr > 0.9:
                i = int(max_idx.item() // n_prot)
                j = int(max_idx.item() % n_prot)
                if i != j:
                    merge_pair = (i, j)

        return w_se, merge_pair

    except (torch.cuda.OutOfMemoryError, torch.linalg.LinAlgError, RuntimeError) as e:
        logger.debug("Uncertainty estimation step failed: %s", e)
        return np.full((n_s, n_prot), np.nan), None
