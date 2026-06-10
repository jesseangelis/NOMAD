"""Single component adaptive outlier pruning and merging pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from nomad.utils import graph_ops, header_parser
from nomad.utils.nmf import optimizer

logger = logging.getLogger(__name__)


def fit_single_component(
    cc: Set[Any], graph: Any, s2i: Dict[str, int], nodes: List[str], dev: torch.device,
    scale: float, l_reg: float, g_dr: float, b_rep: float, avg_m: Dict[str, torch.Tensor],
    r_grp: List[torch.Tensor], gamma_prior: float = 0.0,
) -> Tuple[Any, Any, Any, Any, Any]:
    try:
        prots, precs = [n for n in cc if graph.nodes[n].get("type") == "Protein"], [n for n in cc if graph.nodes[n].get("type") == "Precursor"]
        if not prots or not precs: return None, None, None, None, None
        v_mat_raw = graph_ops.build_v_matrix(graph, precs, s2i)
        scale = float(np.max(v_mat_raw)) if np.max(v_mat_raw) > 0 else 1.0
        v_mat = v_mat_raw / scale
        if np.sum(v_mat) == 0: return None, None, None, None, None

        grps, h_msk = graph_ops.group_isoforms(prots, graph_ops.build_h_mask(graph, prots, precs))
        has_p, o_msk, s_msk = False, v_mat > 0, ((v_mat > 0) @ h_msk.T) > 0

        # --- Phase 1: Mask a value (LOO) and copy the matrix ---
        o_idx = np.argwhere(v_mat > 0)
        valid_loo_indices = []
        for s, p in o_idx:
            connected_prots = np.where(h_msk[:, p] > 0)[0]
            if len(connected_prots) == 0:
                continue
            valid = True
            for g in connected_prots:
                other_precs = np.where(h_msk[g, :] > 0)[0]
                other_precs = other_precs[other_precs != p]
                if not any(v_mat[s, op] > 0 for op in other_precs):
                    valid = False
                    break
            if valid:
                valid_loo_indices.append([s, p])
        
        loo_pool = np.array(valid_loo_indices) if len(valid_loo_indices) > 0 else o_idx

        cv_p, cv_act, v_curr = None, None, v_mat.copy()
        s_cv, p_cv = None, None
        if len(loo_pool) > 0:
            c_i = int(np.random.randint(len(loo_pool)))
            s_cv, p_cv = int(loo_pool[c_i][0]), int(loo_pool[c_i][1])
            cv_act = float(v_mat[s_cv, p_cv] * scale)
            v_curr[s_cv, p_cv] = 0.0  # Masked V for Phase 2

        # --- Phase 2: Fit on masked V, remove outliers, and iteratively merge ---
        P_matrix = None
        while True:
            # Construct Prior Penalty Matrix P based on specific precursors of current groups
            if gamma_prior > 0.0:
                S_len, G_len = v_mat.shape[0], h_msk.shape[0]
                P_matrix = np.zeros((S_len, G_len))
                for g_idx in range(G_len):
                    spec_precs = []
                    for p_idx in range(h_msk.shape[1]):
                        if h_msk[g_idx, p_idx] == 1.0:
                            is_spec = True
                            for other_g in range(G_len):
                                if other_g != g_idx and h_msk[other_g, p_idx] == 1.0:
                                    is_spec = False
                                    break
                            if is_spec:
                                spec_precs.append(p_idx)
                    
                    if len(spec_precs) > 0:
                        for s_idx in range(S_len):
                            n_det = sum(1 for p_idx in spec_precs if v_mat[s_idx, p_idx] > 0.0)
                            det_frac = n_det / len(spec_precs)
                            if det_frac < 0.2:
                                P_matrix[s_idx, g_idx] = 1.0 - (det_frac / 0.2)

            wf, hf, wse, mp = optimizer.optimize_component(
                v_curr, h_msk, dev, l_reg, g_dr, b_rep, avg_m, r_grp,
                scale=scale, compute_se=False, P=P_matrix, gamma_prior=gamma_prior
            )
            if not has_p:
                res = np.abs(np.sqrt(wf @ (hf * h_msk) + 1e-8) - np.sqrt(v_curr))
                if np.any(o_msk):
                    sigma = 1.4826 * float(np.median(np.abs(res[o_msk] - np.median(res[o_msk])))) + 1e-6
                    outs = (res > 5.0 * sigma) & (res > 0.5) & o_msk
                    if np.any(outs) and np.mean(outs) < 0.05:
                        v_curr[outs] = 0.0
                        for si, pi in zip(*np.where(outs)):
                            for pri in np.where(h_msk[:, pi] > 0)[0]: s_msk[si, pri] = False
                        has_p = True; continue

            if mp is not None:
                i, j = mp
                grps.append(f"{grps[i]}; {grps[j]}")
                h_msk = np.vstack([h_msk[[k for k in range(len(grps)-1) if k not in (i, j)], :], np.maximum(h_msk[i, :], h_msk[j, :])])
                s_msk = np.column_stack([s_msk[:, [k for k in range(len(grps)-1) if k not in (i, j)]], s_msk[:, i] & s_msk[:, j]])
                grps = [grps[k] for k in range(len(grps)-1) if k not in (i, j)] + [grps[-1]]
                continue
            break

        def _clean_group(g: str) -> str:
            """Converts a raw FASTA group key to clean accession IDs (e.g. 'O15360-3; O15360-5')."""
            return "; ".join(header_parser.parse_uniprot_header(p.strip())[0] for p in g.split("; "))

        clean_grps = [_clean_group(g) for g in grps]

        # Save the diagnostics from this Phase 2 (masked) fit
        if cv_act is not None:
            pred_val = float((wf @ (hf * h_msk))[s_cv, p_cv] * scale)
            is_spec = bool(np.sum(h_msk[:, p_cv]) == 1.0)
            cv_p = (cv_act, pred_val, float(scale), "; ".join(clean_grps), nodes[s_cv], precs[p_cv], is_spec)

        # Randomly selected value (RS)
        rs_p = None
        # Find positive entries in v_curr (non-zero, non-outlier, not masked)
        rs_idx = np.argwhere(v_curr > 0)
        if len(rs_idx) > 0:
            r_i = int(np.random.randint(len(rs_idx)))
            s_rs, p_rs = int(rs_idx[r_i][0]), int(rs_idx[r_i][1])
            rs_act = float(v_curr[s_rs, p_rs] * scale)
            rs_pred = float((wf @ (hf * h_msk))[s_rs, p_rs] * scale)
            is_spec = bool(np.sum(h_msk[:, p_rs]) == 1.0)
            rs_p = (rs_act, rs_pred, float(scale), is_spec)

        # --- Phase 3: Unmask the LOO value. Fit again using the weights from phase 2 as initialization. ---
        if s_cv is not None and p_cv is not None:
            v_final = v_curr.copy()
            # Restore the LOO value to V
            v_final[s_cv, p_cv] = cv_act / scale
            
            try:
                wf, hf, wse, _ = optimizer.optimize_component(
                    v_final, h_msk, dev, l_reg, g_dr, b_rep, avg_m, r_grp,
                    scale=scale, init_w=wf, init_h=hf, lbfgs_only=True,
                    P=P_matrix, gamma_prior=gamma_prior
                )
            except Exception as e:
                logger.debug("Phase 3 unmasked fit failed: %s", e)

        wf *= scale; wse *= scale
        q_rows = []
        for idx, g in enumerate(grps):
            accessions, entry_names, genes, descriptions = header_parser.parse_group_header_details(g)
            row = {
                "protein": accessions,
                "gene_symbol": genes,
                "entry_name": entry_names,
                "description": descriptions,
                "n_proteins": len(accessions.split("; ")) if accessions else 0,
                **{s: float(wf[si, idx]) for si, s in enumerate(nodes)},
                **{f"{s}_stderr": float(wse[si, idx]) for si, s in enumerate(nodes)},
                **{f"{s}_supported": bool(s_msk[si, idx]) for si, s in enumerate(nodes)}
            }
            q_rows.append(row)

        e_rows = [{"protein": clean_grps[idx], "precursor": precs[j], "probability": float(hf[0, j] * h_msk[idx, j])} for j in range(len(precs)) for idx in range(len(grps)) if hf[0, j] * h_msk[idx, j] > 1e-6]

        rep_p = None
        if r_grp and len(o_idx) > 0:
            np.random.shuffle(o_idx)
            for si, pi in o_idx:
                for gt in r_grp:
                    gl = gt.tolist()
                    if si in gl and len(gl) > 1 and any(x != si and v_curr[x, pi] > 0 for x in gl):
                        is_spec = bool(np.sum(h_msk[:, pi]) == 1.0)
                        rep_p = (float(v_curr[si, pi] * scale), float(v_curr[np.random.choice([x for x in gl if x != si and v_curr[x, pi] > 0]), pi] * scale), float(scale), is_spec)
                        break
                if rep_p: break

        return q_rows, e_rows, cv_p, rep_p, rs_p
    except RuntimeError as e: logger.error("Pipeline optimization error: %s", e); return None, None, None, None, None
