"""Premium interactive Volcano plots highlighting pharmacological significance thresholds."""

from __future__ import annotations

import os
from typing import Optional
import numpy as np
import plotly.graph_objects as go
import polars as pl
from nomad.utils.header_parser import parse_uniprot_header


class VolcanoPlot:
    """Generates premium volcano plots for dose-response results."""

    @staticmethod
    def _plot_from_results(
        protein_df: pl.DataFrame, output_dir: Optional[str] = None, alpha: float = 0.05, lfc_threshold: float = 1.0,
    ) -> Optional[go.Figure]:
        if protein_df.is_empty(): return None
        drugs = protein_df["drug"].unique().to_list()
        fig, drug_trace_indices, total_traces = go.Figure(), {}, 0

        for d_name in drugs:
            p_df = protein_df.filter(pl.col("drug") == d_name).unique(subset=["protein"])
            if p_df.is_empty(): continue

            y_col = "relevance_score" if "relevance_score" in p_df.columns else None
            if "significant_trend" in p_df.columns: is_sig = p_df["significant_trend"].cast(pl.Boolean)
            elif y_col: is_sig = (p_df["relevance_score"] >= -np.log10(alpha)) & (p_df["log2fc"].abs() >= lfc_threshold)
            else: is_sig = (p_df["p_val"] <= alpha) & (p_df["log2fc"].abs() >= lfc_threshold)

            is_pos, is_neg = is_sig & (pl.col("log2fc") > 0), is_sig & (pl.col("log2fc") < 0)
            p_df = p_df.with_columns([pl.when(is_pos).then(pl.lit("positive")).when(is_neg).then(pl.lit("negative")).otherwise(pl.lit("insignificant")).alias("trend_type")])

            n_t, drug_indices, is_vis = p_df.height, [], (d_name == drugs[0])
            for t_type, b_name, c_hex, sz in [("insignificant", "Insignificant", "#BDC3C7", 6), ("positive", "Positive Trend", "#E74C3C", 9), ("negative", "Negative Trend", "#3498DB", 9)]:
                sub_df = p_df.filter(pl.col("trend_type") == t_type)
                if not sub_df.is_empty():
                    y_sub = sub_df["relevance_score"] if y_col else -np.log10(sub_df["p_val"] + 1e-300)
                    # Show only the protein accession(s), not the full FASTA header
                    prot_labels = pl.Series(
                        "protein_label",
                        [
                            "; ".join(parse_uniprot_header(part.strip())[0] for part in p.split("; "))
                            for p in sub_df["protein"].to_list()
                        ],
                    )
                    hov = prot_labels + "<br>Log2FC: " + sub_df["log2fc"].round(2).cast(pl.Utf8)
                    if y_col: hov = hov + "<br>RS: " + sub_df["relevance_score"].round(2).cast(pl.Utf8)
                    elif "p_val" in sub_df.columns: hov = hov + "<br>P-val: " + sub_df["p_val"].map_elements(lambda x: f"{x:.2e}", return_dtype=pl.Utf8)
                    if "n_proteins" in sub_df.columns: hov = hov + "<br>Group Size: " + sub_df["n_proteins"].cast(pl.Utf8)

                    m_dict = dict(color=c_hex, size=sz, opacity=0.4 if t_type == "insignificant" else 0.8)
                    if t_type != "insignificant": m_dict["line"] = dict(width=0.5, color="white")
                    fig.add_trace(go.Scattergl(x=sub_df["log2fc"], y=y_sub, text=hov.to_list(), hoverinfo="text", mode="markers", name=f"{b_name} ({sub_df.height}, {(sub_df.height/n_t)*100 if n_t > 0 else 0:.1f}%)", marker=m_dict, visible=is_vis))
                    drug_indices.append(total_traces)
                    total_traces += 1
            drug_trace_indices[d_name] = drug_indices

        buttons = []
        for d_name in drugs:
            v_mask = [False] * total_traces
            for idx in drug_trace_indices.get(d_name, []): v_mask[idx] = True
            buttons.append(dict(label=d_name, method="update", args=[{"visible": v_mask}, {"title": f"Dose-Response Volcano: {d_name}"}]))

        fig.add_hline(y=-np.log10(alpha), line_dash="dash", line_color="black", opacity=0.5)
        fig.add_vline(x=lfc_threshold, line_dash="dash", line_color="black", opacity=0.5)
        fig.add_vline(x=-lfc_threshold, line_dash="dash", line_color="black", opacity=0.5)

        fig.update_layout(
            updatemenus=[dict(active=0, buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.15)],
            title=dict(text=f"Dose-Response Volcano: {drugs[0]}", x=0.5, font=dict(size=24, weight="bold")),
            xaxis_title="Log2 Fold Change", yaxis_title="Relevance Score" if "relevance_score" in protein_df.columns else "-Log10 P-value",
            template="simple_white", width=900, height=750, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.write_html(os.path.join(output_dir, "volcano_plot_all_drugs.html"))
        return fig

    @staticmethod
    def plot(
        db_path: str, output_dir: Optional[str] = None, alpha: float = 0.05, lfc_threshold: float = 1.0
    ) -> Optional[go.Figure]:
        """Generates premium volcano plots by loading dose-response results from the SQLite database."""
        from nomad.utils import db
        if not os.path.exists(db_path):
            return None
        protein_df = db.load_dose_response(db_path)
        return VolcanoPlot._plot_from_results(protein_df, output_dir, alpha, lfc_threshold)
