"""Multiprocessing worker initializers and batch processing targets."""

from __future__ import annotations

import logging
import multiprocessing
from typing import Any, List, Optional, Tuple
import polars as pl
import torch

logger = logging.getLogger(__name__)

# Process-local cache object holding the initialized engine instance
_worker_engine: Optional[Any] = None


def init_nmf_worker(
    n_gpus: int, graph: Any, meta_df: pl.DataFrame,
    lambda_reg: float, gamma_dr: float, beta_rep: float,
    scale_factor: float, num_threads: int = 1,
    gamma_prior: float = 10.0,
) -> None:
    """Initializes a worker process with specific GPU mapping and engine allocation."""
    global _worker_engine
    try:
        proc = multiprocessing.current_process()
        proc_idx = int(proc._identity[0] - 1) if proc._identity else 0
        gpu_id = proc_idx % n_gpus if n_gpus > 0 else -1
    except (AttributeError, IndexError):
        gpu_id = -1

    try:
        if gpu_id >= 0 and torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        from nomad.utils.nmf.engine import NMFFit
        _worker_engine = NMFFit(
            graph, meta_df, lambda_reg=lambda_reg, gamma_dr=gamma_dr,
            beta_rep=beta_rep, device=device, num_workers=num_threads,
            gamma_prior=gamma_prior,
        )
        _worker_engine.scale_factor = scale_factor
    except (ImportError, RuntimeError, torch.cuda.CudaError) as e:
        logger.error("Worker engine allocation failed: %s", e)


def run_worker_batch(
    batch_components: List[Any],
) -> Tuple[List[Any], List[Any], int, List[Any], List[Any], List[Any]]:
    """Executes NMF optimization across a component batch via the localized engine."""
    global _worker_engine
    if _worker_engine is None:
        return [], [], len(batch_components), [], [], []
    try:
        q_list, e_list, cv_list, rep_list, rs_list = _worker_engine._fit_subset(batch_components, show_progress=False)
        return q_list, e_list, len(batch_components), cv_list, rep_list, rs_list
    except RuntimeError as e:
        logger.error("Worker batch optimization failed: %s", e)
        return [], [], len(batch_components), [], [], []
