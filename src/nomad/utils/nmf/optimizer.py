"""Core Non-negative Matrix Factorization loss evaluator and numerical solver."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from nomad.utils.nmf import uncertainty

logger = logging.getLogger(__name__)


def compute_nmf_loss(
    w: torch.Tensor, h: torch.Tensor, V: torch.Tensor, M: torch.Tensor,
    l_reg: float, g_dr: float, b_rep: float,
    avg_m: Dict[str, torch.Tensor], rep_g: List[torch.Tensor],
    scale: float = 1.0,
    P: Optional[torch.Tensor] = None,
    gamma_prior: float = 0.0,
) -> Tuple[torch.Tensor, float]:
    w_sq, h_sq = w**2, h**2
    recon = torch.matmul(w_sq, h_sq * M)
    res = torch.sqrt(recon + 1e-8) - torch.sqrt(V)
    recon_loss = torch.sum((res * (V > 0)) ** 2)
    reg_loss = l_reg * torch.sum(w_sq)

    smooth_loss = torch.tensor(0.0, device=V.device)
    if g_dr > 0 and avg_m:
        for A in avg_m.values():
            w_dose = torch.sparse.mm(A, w_sq)
            diffs = torch.diff(w_dose, dim=0)
            if diffs.shape[0] > 1:
                smooth_loss = smooth_loss + torch.sum(torch.relu(-diffs[1:] * diffs[:-1]))

    rep_loss = torch.tensor(0.0, device=V.device)
    if b_rep > 0 and rep_g:
        for idxs in rep_g:
            w_reps = w_sq[idxs, :]
            rep_loss = rep_loss + torch.sum((w_reps - w_reps.mean(dim=0, keepdim=True)) ** 2)

    total_loss = recon_loss + reg_loss + (g_dr * smooth_loss) + (b_rep * rep_loss)
    if P is not None and gamma_prior > 0.0:
        total_loss = total_loss + gamma_prior * torch.sum(P * w_sq)

    return total_loss, recon_loss.item()


def _run_core(
    v_norm: np.ndarray, h_mask: np.ndarray, dev: torch.device, l_reg: float, g_dr: float,
    b_rep: float, avg_m: Dict[str, torch.Tensor], r_grp: List[torch.Tensor], max_iter: int,
    scale: float = 1.0,
    init_w: Optional[np.ndarray] = None,
    init_h: Optional[np.ndarray] = None,
    compute_se: bool = True,
    lbfgs_only: bool = False,
    P: Optional[np.ndarray] = None,
    gamma_prior: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    V, M = torch.as_tensor(v_norm, device=dev, dtype=torch.float32), torch.as_tensor(h_mask, device=dev, dtype=torch.float32)
    P_tensor = torch.as_tensor(P, device=dev, dtype=torch.float32) if P is not None else None
    
    if init_w is not None and init_h is not None:
        w_init_tensor = torch.tensor(np.sqrt(np.maximum(init_w, 0.0)), device=dev, dtype=torch.float32)
        h_init_tensor = torch.tensor(np.sqrt(np.maximum(init_h, 0.0)), device=dev, dtype=torch.float32)
        w = torch.nn.Parameter(w_init_tensor)
        h = torch.nn.Parameter(h_init_tensor)
    else:
        v_obs = V[V > 0]
        w_init = float(np.sqrt(v_obs.median().item() if v_obs.numel() > 0 else 1.0))
        w = torch.nn.Parameter(torch.full((V.shape[0], M.shape[0]), w_init, device=dev))
        h = torch.nn.Parameter(torch.ones((1, V.shape[1]), device=dev))

    if not lbfgs_only:
        optimizer, prev_loss = torch.optim.Adam([w, h], lr=2e-2), float("inf")
        for _ in range(max_iter):
            optimizer.zero_grad()
            loss, _ = compute_nmf_loss(w, h, V, M, l_reg, g_dr, b_rep, avg_m, r_grp, scale, P_tensor, gamma_prior)
            l_val = loss.item()
            if abs(prev_loss - l_val) / (abs(prev_loss) + 1e-9) < 1e-5: break
            prev_loss = l_val
            loss.backward(); optimizer.step()

    with torch.no_grad():
        h_max = h.max().clamp(min=1e-6)
        h.copy_(h / h_max); w.copy_(w * h_max)

    if lbfgs_only or (w.numel() + h.numel() < 5000):
        lbfgs = torch.optim.LBFGS([w, h], max_iter=30, line_search_fn="strong_wolfe")
        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            ls, _ = compute_nmf_loss(w, h, V, M, l_reg, g_dr, b_rep, avg_m, r_grp, scale, P_tensor, gamma_prior)
            ls.backward(); return ls
        try: lbfgs.step(closure)
        except RuntimeError as e: logger.debug("L-BFGS polish failed: %s", e)

    with torch.no_grad():
        h_sq_val = h**2
        h_sq_max = h_sq_val.max().clamp(min=1e-8)
        h_scale = torch.sqrt(h_sq_max)
        h.copy_(h / h_scale)
        w.copy_(w * h_scale)

    if compute_se or h_mask.shape[0] > 1:
        w_se, mp = uncertainty.calculate_uncertainty(w.detach(), h.detach(), V, M, l_reg, scale, P_tensor, gamma_prior)
    else:
        w_se, mp = np.zeros((V.shape[0], M.shape[0])), None
    return (w.detach() ** 2).cpu().numpy(), (h.detach() ** 2).cpu().numpy(), w_se, mp


def optimize_component(
    v_norm: np.ndarray, h_mask: np.ndarray, device: torch.device, l_reg: float, g_dr: float,
    b_rep: float, avg_m: Dict[str, torch.Tensor], rep_g: List[torch.Tensor], max_iter: int = 1000,
    scale: float = 1.0,
    init_w: Optional[np.ndarray] = None,
    init_h: Optional[np.ndarray] = None,
    compute_se: bool = True,
    lbfgs_only: bool = False,
    P: Optional[np.ndarray] = None,
    gamma_prior: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    """Optimizes NMF parameters using Adam and L-BFGS with CPU fallback."""
    try:
        return _run_core(v_norm, h_mask, device, l_reg, g_dr, b_rep, avg_m, rep_g, max_iter, scale, init_w, init_h, compute_se, lbfgs_only, P, gamma_prior)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM during NMF fitting. Falling back to CPU.")
        torch.cuda.empty_cache()
        cpu_dev = torch.device("cpu")
        return _run_core(v_norm, h_mask, cpu_dev, l_reg, g_dr, b_rep, {k: v.to(cpu_dev) for k, v in avg_m.items()}, [t.to(cpu_dev) for t in rep_g], max_iter, scale, init_w, init_h, compute_se, lbfgs_only, P, gamma_prior)
