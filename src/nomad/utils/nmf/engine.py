"""NMF solver orchestration engine layer."""

from __future__ import annotations

from concurrent.futures import as_completed, ThreadPoolExecutor
import logging
import multiprocessing
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm
from nomad.utils import graph_ops
from nomad.utils.nmf import averaging, single_fit, worker
from nomad.utils.plotting.diagnostics import DiagnosticsPlot

logger = logging.getLogger(__name__)


class NMFFit:
    """GPU-accelerated Joint NMF fitting engine orchestrator."""

    def __init__(
        self, graph: Any, meta_df: pl.DataFrame, lambda_reg: float = 1e-3,
        gamma_dr: float = 10.0, beta_rep: float = 0.1, device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        self.graph = graph
        id_col = "sample" if "sample" in meta_df.columns else "file"
        self.metadata = meta_df.unique(subset=[id_col])
        self.lambda_reg, self.gamma_dr, self.beta_rep = lambda_reg, gamma_dr, beta_rep
        self.gamma_prior = float(kwargs.get("gamma_prior", 10.0))
        self.num_workers = int(kwargs.get("num_workers", 4))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.samples = sorted(self.metadata[id_col].to_list())
        ints = [d["intensity"] for _, _, d in self.graph.edges(data=True) if "intensity" in d]
        self.scale_factor = float(np.mean(ints)) if ints else 1.0

        self.drug_averaging_matrices = averaging.prepare_averaging_matrices(self.metadata, self.device)
        self.replicate_groups = averaging.prepare_replicate_groups(self.metadata, self.device)
        self.quant_df, self.emissions_df = pl.DataFrame(), pl.DataFrame()
        self.cv_data, self.rep_data, self.rs_data = [], [], []

    def fit(self, **kwargs: Any) -> Tuple[pl.DataFrame, Dict[str, float]]:
        comps = graph_ops.identify_components(self.graph)
        if not comps: return self.quant_df, {}

        n_gpus = torch.cuda.device_count() if (torch.cuda.is_available() and self.device.type == "cuda") else 0
        if n_gpus > 1 and self.num_workers > 1:
            logger.info("[*] Distributing %d components across %d GPUs", len(comps), n_gpus)
            ctx, batches = multiprocessing.get_context("spawn"), [[c] for c in comps]
            init_args = (n_gpus, self.graph, self.metadata, self.lambda_reg, self.gamma_dr, self.beta_rep, self.scale_factor, 1, self.gamma_prior)

            q_rows, e_rows, cv_p, rep_p, rs_p = [], [], [], [], []
            with ctx.Pool(processes=self.num_workers, initializer=worker.init_nmf_worker, initargs=init_args) as pool:
                with tqdm(total=len(comps), desc=" [*] Fitting on GPUs") as pbar:
                    for ql, el, nb, cvl, repl, rsl in pool.imap_unordered(worker.run_worker_batch, batches):
                        if ql: q_rows.extend(ql); e_rows.extend(el); cv_p.extend(cvl); rep_p.extend(repl); rs_p.extend(rsl)
                        pbar.update(nb)
            stats = self._print_cv_metrics(cv_p, rep_p, rs_p) if cv_p else {}
        else:
            logger.info("[*] Fitting %d components locally using threads", len(comps))
            q_rows, e_rows, cv_p, rep_p, rs_p = self._fit_subset(comps, show_progress=True)
            stats = self._print_cv_metrics(cv_p, rep_p, rs_p) if cv_p else {}

        self.quant_df, self.emissions_df = pl.DataFrame(q_rows), pl.DataFrame(e_rows)
        self.cv_data, self.rep_data, self.rs_data = cv_p, rep_p, rs_p
        return self.quant_df, stats

    def _fit_subset(self, comps: List[Set[Any]], show_progress: bool = False) -> Tuple[List[Any], List[Any], List[Any], List[Any], List[Any]]:
        id_col = "sample" if "sample" in self.metadata.columns else "file"
        nodes = self.metadata[id_col].to_list()
        s2i = {name: i for i, name in enumerate(nodes)}
        q_rows, e_rows, cv_p, rep_p, rs_p = [], [], [], [], []

        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                futs = [ex.submit(single_fit.fit_single_component, c, self.graph, s2i, nodes, self.device, self.scale_factor, self.lambda_reg, self.gamma_dr, self.beta_rep, self.drug_averaging_matrices, self.replicate_groups, self.gamma_prior) for c in comps]
                it = tqdm(as_completed(futs), total=len(futs), desc=" [*] Fitting") if show_progress else as_completed(futs)
                for f in it:
                    try:
                        q, e, cv, rep, rs = f.result()
                        if q: q_rows.extend(q); e_rows.extend(e)
                        if cv: cv_p.append(cv)
                        if rep: rep_p.append(rep)
                        if rs: rs_p.append(rs)
                    except RuntimeError as exc: logger.error("Fitting thread error: %s", exc)
        else:
            it = tqdm(comps, desc=" [*] Fitting") if show_progress else comps
            for c in it:
                q, e, cv, rep, rs = single_fit.fit_single_component(c, self.graph, s2i, nodes, self.device, self.scale_factor, self.lambda_reg, self.gamma_dr, self.beta_rep, self.drug_averaging_matrices, self.replicate_groups, self.gamma_prior)
                if q: q_rows.extend(q); e_rows.extend(e)
                if cv: cv_p.append(cv)
                if rep: rep_p.append(rep)
                if rs: rs_p.append(rs)
        return q_rows, e_rows, cv_p, rep_p, rs_p

    def _print_cv_metrics(self, cv_p: List[Tuple[float, float]], rep_p: Optional[List[Tuple[float, float]]], rs_p: Optional[List[Tuple[float, float]]] = None) -> Dict[str, float]:
        if not cv_p: return {}
        acts, preds = np.array([p[0] for p in cv_p]), np.array([p[1] for p in cv_p])
        
        pearson_raw = float(np.corrcoef(acts, preds)[0, 1]) if len(acts) > 1 else 0.0
        
        min_val = 10.0
        pearson_log = float(np.corrcoef(np.log10(np.maximum(acts, min_val)), np.log10(np.maximum(preds, min_val)))[0, 1]) if len(acts) > 1 else 0.0
        if cv_p and len(cv_p[0]) >= 3:
            scales = np.array([p[2] for p in cv_p])
            pearson_sqrt = float(np.corrcoef(np.sqrt(np.maximum(acts / scales, 0.0)), np.sqrt(np.maximum(preds / scales, 0.0)))[0, 1]) if len(acts) > 1 else 0.0
        else:
            pearson_sqrt = float(np.corrcoef(np.sqrt(np.maximum(acts, 0.0)), np.sqrt(np.maximum(preds, 0.0)))[0, 1]) if len(acts) > 1 else 0.0
        
        from scipy.stats import spearmanr
        rho = float(spearmanr(acts, preds).statistic) if len(acts) > 1 else 0.0
        
        stats = {
            "pearson_raw": pearson_raw,
            "pearson_log10": pearson_log,
            "pearson_sqrt": pearson_sqrt,
            "spearman_rho": rho
        }

        # Specific-precursor only subsets
        cv_spec_idx = [i for i, p in enumerate(cv_p) if len(p) >= 7 and p[6]]
        if len(cv_spec_idx) > 1:
            stats["pearson_spec_raw"] = float(np.corrcoef(acts[cv_spec_idx], preds[cv_spec_idx])[0, 1])
            stats["pearson_spec_log10"] = float(np.corrcoef(np.log10(np.maximum(acts[cv_spec_idx], min_val)), np.log10(np.maximum(preds[cv_spec_idx], min_val)))[0, 1])
        else:
            stats["pearson_spec_raw"] = 0.0
            stats["pearson_spec_log10"] = 0.0
        
        if rep_p:
            r_acts, r_preds = np.array([p[0] for p in rep_p]), np.array([p[1] for p in rep_p])
            if len(r_acts) > 1:
                stats["rep_pearson_raw"] = float(np.corrcoef(r_acts, r_preds)[0, 1])
                stats["rep_pearson_log10"] = float(np.corrcoef(np.log10(np.maximum(r_acts, min_val)), np.log10(np.maximum(r_preds, min_val)))[0, 1])
                if rep_p and len(rep_p[0]) >= 3:
                    rep_scales = np.array([p[2] for p in rep_p])
                    stats["rep_pearson_sqrt"] = float(np.corrcoef(np.sqrt(np.maximum(r_acts / rep_scales, 0.0)), np.sqrt(np.maximum(r_preds / rep_scales, 0.0)))[0, 1])
                else:
                    stats["rep_pearson_sqrt"] = float(np.corrcoef(np.sqrt(np.maximum(r_acts, 0.0)), np.sqrt(np.maximum(r_preds, 0.0)))[0, 1])
                stats["rep_spearman_rho"] = float(spearmanr(r_acts, r_preds).statistic)

                rep_spec_idx = [i for i, p in enumerate(rep_p) if len(p) >= 4 and p[3]]
                if len(rep_spec_idx) > 1:
                    stats["rep_spec_pearson_raw"] = float(np.corrcoef(r_acts[rep_spec_idx], r_preds[rep_spec_idx])[0, 1])
                    stats["rep_spec_pearson_log10"] = float(np.corrcoef(np.log10(np.maximum(r_acts[rep_spec_idx], min_val)), np.log10(np.maximum(r_preds[rep_spec_idx], min_val)))[0, 1])
                else:
                    stats["rep_spec_pearson_raw"] = 0.0
                    stats["rep_spec_pearson_log10"] = 0.0
                
        if rs_p:
            rs_acts, rs_preds = np.array([p[0] for p in rs_p]), np.array([p[1] for p in rs_p])
            if len(rs_acts) > 1:
                stats["rs_pearson_raw"] = float(np.corrcoef(rs_acts, rs_preds)[0, 1])
                stats["rs_pearson_log10"] = float(np.corrcoef(np.log10(np.maximum(rs_acts, min_val)), np.log10(np.maximum(rs_preds, min_val)))[0, 1])
                if rs_p and len(rs_p[0]) >= 3:
                    rs_scales = np.array([p[2] for p in rs_p])
                    stats["rs_pearson_sqrt"] = float(np.corrcoef(np.sqrt(np.maximum(rs_acts / rs_scales, 0.0)), np.sqrt(np.maximum(rs_preds / rs_scales, 0.0)))[0, 1])
                else:
                    stats["rs_pearson_sqrt"] = float(np.corrcoef(np.sqrt(np.maximum(rs_acts, 0.0)), np.sqrt(np.maximum(rs_preds, 0.0)))[0, 1])
                stats["rs_spearman_rho"] = float(spearmanr(rs_acts, rs_preds).statistic)

                rs_spec_idx = [i for i, p in enumerate(rs_p) if len(p) >= 4 and p[3]]
                if len(rs_spec_idx) > 1:
                    stats["rs_spec_pearson_raw"] = float(np.corrcoef(rs_acts[rs_spec_idx], rs_preds[rs_spec_idx])[0, 1])
                    stats["rs_spec_pearson_log10"] = float(np.corrcoef(np.log10(np.maximum(rs_acts[rs_spec_idx], min_val)), np.log10(np.maximum(rs_preds[rs_spec_idx], min_val)))[0, 1])
                else:
                    stats["rs_spec_pearson_raw"] = 0.0
                    stats["rs_spec_pearson_log10"] = 0.0

        logger.info("[*] CV Stats: Raw Pearson=%.4f, Log10 Pearson=%.4f, Sqrt Pearson=%.4f, Spearman=%.4f", pearson_raw, pearson_log, pearson_sqrt, rho)
        logger.info("[*] CV Specific Precursors Stats: Raw Pearson=%.4f, Log10 Pearson=%.4f", stats["pearson_spec_raw"], stats["pearson_spec_log10"])
        if rs_p and len(rs_acts) > 1:
            logger.info("[*] RS Stats: Raw Pearson=%.4f, Log10 Pearson=%.4f, Sqrt Pearson=%.4f, Spearman=%.4f", stats["rs_pearson_raw"], stats["rs_pearson_log10"], stats["rs_pearson_sqrt"], stats["rs_spearman_rho"])
            logger.info("[*] RS Specific Precursors Stats: Raw Pearson=%.4f, Log10 Pearson=%.4f", stats["rs_spec_pearson_raw"], stats["rs_spec_pearson_log10"])
        
        # Generate diagnostic plots
        DiagnosticsPlot.plot_nmf_performance(cv_p, rep_p, rs_p)
        
        return stats
