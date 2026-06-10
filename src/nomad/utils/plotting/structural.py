"""Premium visualization of the NMF structural deconvolution mapping evidence matrices."""

from __future__ import annotations
import os, sqlite3
from typing import Any, Optional
import networkx as nx, numpy as np, plotly.graph_objects as go, polars as pl
from plotly.subplots import make_subplots
import torch
from nomad.utils import graph_ops
from nomad.utils.header_parser import parse_uniprot_header
from nomad.utils.nmf import averaging, optimizer
from nomad.utils.plotting._helpers import clean_prot

class StructuralEvidencePlot:
    @staticmethod
    def plot_from_graph(
        graph: Any, protein_match: str, metadata: Optional[pl.DataFrame] = None,
        bottom_right: str = "reconstruction", lambda_reg: float = 0.1, db_path: Optional[str] = None,
    ) -> Optional[go.Figure]:
        t_ids = protein_match.split("; ")
        t_node = next((n for n, d in graph.nodes(data=True) if d.get("type") == "Protein" and (any(i in n for i in t_ids) or parse_uniprot_header(n)[0] in t_ids)), None)
        if not t_node: return None

        sub = graph.subgraph([n for n in graph.nodes if graph.nodes[n].get("type") in ("Protein", "Precursor", "Peptide")])
        cc = next((c for c in nx.connected_components(sub) if t_node in c), [])
        prots, precs = sorted([n for n in cc if graph.nodes[n].get("type") == "Protein"]), sorted([n for n in cc if graph.nodes[n].get("type") == "Precursor"])
        if not prots or not precs: return None

        s_col = "sample" if (metadata is not None and "sample" in metadata.columns) else "file"
        samples = sorted(metadata[s_col].unique().to_list()) if metadata is not None else sorted({v for n in precs for v in graph.neighbors(n) if graph.nodes[v].get("type") == "Sample"})
        s2i = {s: i for i, s in enumerate(samples)}
        
        scale = float(np.mean([d["intensity"] for _, _, d in graph.edges(data=True) if "intensity" in d] or [1.0]))
        v_raw = graph_ops.build_v_matrix(graph, precs, s2i)
        grps, h_msk = graph_ops.group_isoforms(prots, graph_ops.build_h_mask(graph, prots, precs))
        p_keys = ["; ".join([parse_uniprot_header(pi)[0] for pi in g.split("; ")]) for g in grps]

        w_fit, h_fit = np.zeros((len(samples), len(grps))), np.zeros_like(h_msk, dtype=float)
        if db_path and os.path.exists(db_path):
            with sqlite3.connect(db_path) as conn:
                p_map, pr_map = {k: i for i, k in enumerate(p_keys)}, {p: j for j, p in enumerate(precs)}
                for pk, pr, prb in conn.execute(f"SELECT protein, precursor, probability FROM emissions WHERE protein IN ({','.join(['?']*len(p_keys))})", p_keys).fetchall():
                    if pk in p_map and pr in pr_map: h_fit[p_map[pk], pr_map[pr]] = prb
                for pk, sm, intn in conn.execute(f"SELECT protein, sample, intensity FROM intensities WHERE protein IN ({','.join(['?']*len(p_keys))})", p_keys).fetchall():
                    if pk in p_map and sm in s2i: w_fit[s2i[sm], p_map[pk]] = intn
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            wf, hf, *_ = optimizer.optimize_component(v_raw / scale, h_msk, dev, lambda_reg, 10.0, 0.1, averaging.prepare_averaging_matrices(metadata, dev) if metadata is not None else {}, averaging.prepare_replicate_groups(metadata, dev) if metadata is not None else [], scale=scale)
            w_fit, h_fit = wf * scale, h_msk * hf

        wh_fit, p_labels, p_order = w_fit @ h_fit, [clean_prot(g) for g in grps], list(range(len(grps)))
        if len(grps) > 1:
            from scipy.spatial.distance import pdist; from scipy.cluster.hierarchy import linkage, leaves_list
            try: p_order = leaves_list(linkage(pdist(h_msk, metric="jaccard"), method="average")).tolist()
            except Exception: pass

        p_labels, h_msk, h_fit, w_fit = [p_labels[i] for i in p_order], h_msk[p_order, :], h_fit[p_order, :], w_fit[:, p_order]
        h_plot, drugs = np.where(h_msk > 0, h_fit, np.nan), ["All Drugs"] + (metadata["name"].unique().to_list() if metadata is not None else [])
        m_vals = [v_raw[v_raw > 0], wh_fit[wh_fit > 0]]
        z_max = np.log10(max(1.0, np.nanmax([np.nanmax(m) for m in m_vals if m.size > 0]) if any(m.size > 0 for m in m_vals) else 10.0))
        
        drug_data = {}
        for dn in drugs:
            sm = np.ones(len(samples), dtype=bool) if dn == "All Drugs" else np.array([s in metadata.filter(pl.col("name") == dn)[s_col].unique().to_list() for s in samples])
            if not np.any(sm): continue
            s_sub, v_sub, w_sub, wh_sub = [samples[i] for i in range(len(sm)) if sm[i]], v_raw[sm, :], w_fit[sm, :], wh_fit[sm, :]
            s_labels = s_sub
            if metadata is not None and dn != "All Drugs":
                sub_m = metadata.filter(pl.col(s_col).is_in(s_sub)).unique(s_col).sort(["name", "dose", s_col])
                s_map = {s: i for i, s in enumerate(s_sub)}
                v_sub, w_sub, wh_sub = v_sub[[s_map[s] for s in sub_m[s_col]], :], w_sub[[s_map[s] for s in sub_m[s_col]], :], wh_sub[[s_map[s] for s in sub_m[s_col]], :]
                s_labels = [f"{r['name']} @ {r['dose']}" for r in sub_m.to_dicts()]
            
            with np.errstate(divide="ignore", invalid="ignore"):
                res_raw = np.log2(np.where(wh_sub > 1.0, wh_sub, 1.0)) - np.log2(np.where(v_sub > 1.0, v_sub, 1.0))
                res_raw[v_sub <= 1.0] = np.nan
            
            drug_data[dn] = {"W": np.log10(np.where(w_sub > 1.0, w_sub, 1.0)), "labels": s_labels, "reconstruction": np.log10(np.where(wh_sub > 1.0, wh_sub, 1.0)), "observed": np.where(v_sub <= 1.0, np.nan, np.log10(np.where(v_sub > 1.0, v_sub, 1.0))), "residuals": res_raw, "residuals_raw": res_raw, "sparsity": np.where(np.isnan(v_sub) | (v_sub <= 1.0), 1.0, np.nan)}

        titles = {"reconstruction": "Reconstruction (log10 WH)", "observed": "Observed Data (log10 V)", "residuals": "Deconvolution Detail: Log2 Fold Change (log2 WH/V)", "residuals_raw": "Deconvolution Detail: Log2 Fold Change (log2 WH/V)"}
        fig = make_subplots(rows=2, cols=2, subplot_titles=(" ", "H: Structural Mapping", "W: Isoform Intensities", titles.get(bottom_right, titles["reconstruction"])), vertical_spacing=0.12, horizontal_spacing=0.1, column_widths=[0.3, 0.7], row_heights=[0.4, 0.6])
        init = drug_data["All Drugs"]

        fig.add_trace(go.Heatmap(z=h_plot, x=precs, y=p_labels, colorscale="Viridis", name="H", colorbar=dict(title="Prob", x=1.02, y=0.75, len=0.4)), row=1, col=2)
        fig.add_trace(go.Heatmap(z=init["W"], x=p_labels, y=init["labels"], colorscale="Blues", name="W", colorbar=dict(title="Log10 W", x=-0.1, y=0.75, len=0.4)), row=2, col=1)
        
        is_res = "residuals" in bottom_right
        fig.add_trace(go.Heatmap(z=init[bottom_right], x=precs, y=init["labels"], name="Deconvolution View", colorscale=[[0, "blue"], [0.5, "white"], [1, "red"]] if is_res else [[0, "white"], [1, "#27AE60"]], zmin=-2.0 if is_res else 0.0, zmax=2.0 if is_res else z_max, colorbar=dict(title="Log2 Fold Change (WH/V)" if is_res else "Log10", x=1.02, y=0.25, len=0.4)), row=2, col=2)
        fig.add_trace(go.Heatmap(z=init["sparsity"], x=precs, y=init["labels"], colorscale=[[0, "#D3D3D3"], [1, "black"]], showscale=False, hoverinfo="skip"), row=2, col=2)

        d_btns = [dict(label=dn, method="update", args=[{"z": [h_plot, dm["W"], dm[bottom_right], dm["sparsity"]], "y": [p_labels, dm["labels"], dm["labels"], dm["labels"]]}, {"title": f"Structural Deconvolution: {p_labels[0]} ({dn})"}]) for dn, dm in drug_data.items()]
        v_btns = [dict(label=vl, method="update", args=[{"z": [init[vk], init["sparsity"]], "colorscale": [[[0, "blue"], [0.5, "white"], [1, "red"]] if "residuals" in vk else [[0, "white"], [1, "#27AE60"]], [[0, "#D3D3D3"], [1, "black"]]], "zmin": [-2.0 if "residuals" in vk else 0.0, 1], "zmax": [2.0 if "residuals" in vk else z_max, 2], "colorbar.title.text": ["Log2 Fold Change (WH/V)" if "residuals" in vk else "Log10", None]}, {"annotations[3].text": titles[vk]}, [2, 3]]) for vl, vk in [("Reconstruction", "reconstruction"), ("Observed", "observed"), ("Log2 Fold Change", "residuals")]]

        fig.update_layout(updatemenus=[dict(type="dropdown", x=0.01, xanchor="left", y=1.2, buttons=d_btns), dict(type="dropdown", x=0.25, xanchor="left", y=1.2, buttons=v_btns)], title=dict(text=f"Structural Deconvolution: {p_labels[0]}", x=0.5, font=dict(size=24)), template="simple_white", height=900, width=1100, showlegend=False)
        fig.update_xaxes(row=1, col=2, showticklabels=False); fig.update_yaxes(autorange="reversed"); fig.update_yaxes(row=2, col=2, showticklabels=False)
        return fig
