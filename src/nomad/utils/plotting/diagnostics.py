"""Diagnostic plots for NMF deconvolution quality."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class DiagnosticsPlot:
    """Generates performance plots for NMF deconvolution."""

    @staticmethod
    def plot_nmf_performance(
        cv_data: List[Tuple[float, float]],
        rep_data: Optional[List[Tuple[float, float]]] = None,
        rs_data: Optional[List[Tuple[float, float]]] = None,
        output_dir: Optional[str] = "artifacts/plots",
    ) -> go.Figure:
        """Creates a 3x4 grid of scatter plots for NMF diagnostics including specific-precursor subsets.

        Args:
            cv_data: List of LOO cross-validation values.
            rep_data: Optional list of replicate intensity values.
            rs_data: Optional list of randomly selected entry values.
            output_dir: Directory to save the resulting HTML plot.

        Returns:
            Plotly Figure object.
        """
        if not cv_data:
            logger.warning("No CV data provided for diagnostic plots.")
            return go.Figure()

        min_val = 10.0  # Baseline as used in stats calculation

        # Correlation string helpers
        def get_pearson_str(x, y, is_log=False):
            if x is None or y is None or len(x) <= 1 or len(y) <= 1:
                return "R=N/A"
            if is_log:
                x = np.log10(np.maximum(x, min_val))
                y = np.log10(np.maximum(y, min_val))
            try:
                corr = np.corrcoef(x, y)[0, 1]
                return f"R={corr:.3f}" if not np.isnan(corr) else "R=N/A"
            except Exception:
                return "R=N/A"

        # Extract data & subsets
        # LOO
        acts, preds = np.array([p[0] for p in cv_data]), np.array([p[1] for p in cv_data])
        cv_spec_mask = np.array([p[6] for p in cv_data]) if len(cv_data[0]) >= 7 else np.zeros_like(acts, dtype=bool)
        acts_spec, preds_spec = acts[cv_spec_mask], preds[cv_spec_mask]

        # RS
        rs_acts, rs_preds = None, None
        rs_acts_spec, rs_preds_spec = None, None
        if rs_data:
            rs_acts, rs_preds = np.array([p[0] for p in rs_data]), np.array([p[1] for p in rs_data])
            rs_spec_mask = np.array([p[3] for p in rs_data]) if len(rs_data[0]) >= 4 else np.zeros_like(rs_acts, dtype=bool)
            rs_acts_spec, rs_preds_spec = rs_acts[rs_spec_mask], rs_preds[rs_spec_mask]

        # Replicates
        r1, r2 = None, None
        r1_spec, r2_spec = None, None
        if rep_data:
            r1, r2 = np.array([p[0] for p in rep_data]), np.array([p[1] for p in rep_data])
            rep_spec_mask = np.array([p[3] for p in rep_data]) if len(rep_data[0]) >= 4 else np.zeros_like(r1, dtype=bool)
            r1_spec, r2_spec = r1[rep_spec_mask], r2[rep_spec_mask]

        # Set up subplot titles containing Pearson correlations
        fig = make_subplots(
            rows=3, cols=4,
            subplot_titles=(
                f"RS: Observed vs Recon (Raw) ({get_pearson_str(rs_acts, rs_preds)})",
                f"RS: Observed vs Recon (Log10) ({get_pearson_str(rs_acts, rs_preds, is_log=True)})",
                f"RS: Log10 (Specific Only) ({get_pearson_str(rs_acts_spec, rs_preds_spec, is_log=True)})",
                "RS: Sqrt(WH/scale) vs Sqrt(V/scale)",
                
                f"LOO: Predicted vs Masked (Raw) ({get_pearson_str(acts, preds)})",
                f"LOO: Predicted vs Masked (Log10) ({get_pearson_str(acts, preds, is_log=True)})",
                f"LOO: Log10 (Specific Only) ({get_pearson_str(acts_spec, preds_spec, is_log=True)})",
                "LOO: Sqrt(WH/scale) vs Sqrt(V/scale)",
                
                f"Replicate Correlation (Raw) ({get_pearson_str(r2, r1)})",
                f"Replicate Correlation (Log10) ({get_pearson_str(r2, r1, is_log=True)})",
                f"Replicate Correlation (Log10 Specific) ({get_pearson_str(r2_spec, r1_spec, is_log=True)})",
                "Replicate Correlation (Sqrt/scale)" if rep_data and len(rep_data[0]) >= 3 else "Replicate Correlation (Sqrt)"
            ),
            vertical_spacing=0.10,
            horizontal_spacing=0.06
        )

        # Row 1: RS
        if rs_data:
            # Col 1: Raw
            fig.add_trace(
                go.Scatter(
                    x=rs_acts, y=rs_preds, mode='markers',
                    marker=dict(color='#10b981', opacity=0.4, size=4),
                    name="RS Raw"
                ), row=1, col=1
            )
            # Col 2: Log10
            l_rs_acts = np.log10(np.maximum(rs_acts, min_val))
            l_rs_preds = np.log10(np.maximum(rs_preds, min_val))
            fig.add_trace(
                go.Scatter(
                    x=l_rs_acts, y=l_rs_preds, mode='markers',
                    marker=dict(color='#34d399', opacity=0.4, size=4),
                    name="RS Log10"
                ), row=1, col=2
            )
            # Col 3: Log10 Specific
            if len(rs_acts_spec) > 0:
                l_rs_acts_spec = np.log10(np.maximum(rs_acts_spec, min_val))
                l_rs_preds_spec = np.log10(np.maximum(rs_preds_spec, min_val))
                fig.add_trace(
                    go.Scatter(
                        x=l_rs_acts_spec, y=l_rs_preds_spec, mode='markers',
                        marker=dict(color='#06b6d4', opacity=0.5, size=4.5),
                        name="RS Specific Log10"
                    ), row=1, col=3
                )
            # Col 4: Sqrt
            if len(rs_data[0]) >= 3:
                rs_scales = np.array([p[2] for p in rs_data])
                s_rs_acts = np.sqrt(np.maximum(rs_acts / rs_scales, 0.0))
                s_rs_preds = np.sqrt(np.maximum(rs_preds / rs_scales, 0.0))
            else:
                s_rs_acts, s_rs_preds = np.sqrt(np.maximum(rs_acts, 0.0)), np.sqrt(np.maximum(rs_preds, 0.0))
            fig.add_trace(
                go.Scatter(
                    x=s_rs_acts, y=s_rs_preds, mode='markers',
                    marker=dict(color='#059669', opacity=0.4, size=4),
                    name="RS Sqrt"
                ), row=1, col=4
            )

        # Row 2: LOO
        # Col 1: Raw
        fig.add_trace(
            go.Scatter(
                x=acts, y=preds, mode='markers',
                marker=dict(color='#3b82f6', opacity=0.4, size=4),
                name="LOO Raw"
            ), row=2, col=1
        )
        # Col 2: Log10
        l_acts, l_preds = np.log10(np.maximum(acts, min_val)), np.log10(np.maximum(preds, min_val))
        fig.add_trace(
            go.Scatter(
                x=l_acts, y=l_preds, mode='markers',
                marker=dict(color='#60a5fa', opacity=0.4, size=4),
                name="LOO Log10"
            ), row=2, col=2
        )
        # Col 3: Log10 Specific
        if len(acts_spec) > 0:
            l_acts_spec = np.log10(np.maximum(acts_spec, min_val))
            l_preds_spec = np.log10(np.maximum(preds_spec, min_val))
            fig.add_trace(
                go.Scatter(
                    x=l_acts_spec, y=l_preds_spec, mode='markers',
                    marker=dict(color='#06b6d4', opacity=0.5, size=4.5),
                    name="LOO Specific Log10"
                ), row=2, col=3
            )
        # Col 4: Sqrt
        if len(cv_data[0]) >= 3:
            scales = np.array([p[2] for p in cv_data])
            s_acts = np.sqrt(np.maximum(acts / scales, 0.0))
            s_preds = np.sqrt(np.maximum(preds / scales, 0.0))
        else:
            s_acts, s_preds = np.sqrt(np.maximum(acts, 0.0)), np.sqrt(np.maximum(preds, 0.0))
        fig.add_trace(
            go.Scatter(
                x=s_acts, y=s_preds, mode='markers',
                marker=dict(color='#818cf8', opacity=0.4, size=4),
                name="LOO Sqrt"
            ), row=2, col=4
        )

        # Row 3: Replicates
        if rep_data:
            # Col 1: Raw
            fig.add_trace(
                go.Scatter(
                    x=r2, y=r1, mode='markers',
                    marker=dict(color='#f59e0b', opacity=0.4, size=4),
                    name="Rep Raw"
                ), row=3, col=1
            )
            # Col 2: Log10
            lr1, lr2 = np.log10(np.maximum(r1, min_val)), np.log10(np.maximum(r2, min_val))
            fig.add_trace(
                go.Scatter(
                    x=lr2, y=lr1, mode='markers',
                    marker=dict(color='#fbbf24', opacity=0.4, size=4),
                    name="Rep Log10"
                ), row=3, col=2
            )
            # Col 3: Log10 Specific
            if len(r1_spec) > 0:
                lr1_spec = np.log10(np.maximum(r1_spec, min_val))
                lr2_spec = np.log10(np.maximum(r2_spec, min_val))
                fig.add_trace(
                    go.Scatter(
                        x=lr2_spec, y=lr1_spec, mode='markers',
                        marker=dict(color='#06b6d4', opacity=0.5, size=4.5),
                        name="Rep Specific Log10"
                    ), row=3, col=3
                )
            # Col 4: Sqrt
            if len(rep_data[0]) >= 3:
                rep_scales = np.array([p[2] for p in rep_data])
                sr1 = np.sqrt(np.maximum(r1 / rep_scales, 0.0))
                sr2 = np.sqrt(np.maximum(r2 / rep_scales, 0.0))
            else:
                sr1, sr2 = np.sqrt(np.maximum(r1, 0.0)), np.sqrt(np.maximum(r2, 0.0))
            fig.add_trace(
                go.Scatter(
                    x=sr2, y=sr1, mode='markers',
                    marker=dict(color='#f97316', opacity=0.4, size=4),
                    name="Rep Sqrt"
                ), row=3, col=4
            )

        # Add unity lines
        for r in [1, 2, 3]:
            if r == 1 and not rs_data:
                continue
            if r == 3 and not rep_data:
                continue
            for c in [1, 2, 3, 4]:
                if c == 1:
                    if r == 1:
                        x1 = rs_acts.max()
                    elif r == 2:
                        x1 = acts.max()
                    else:
                        x1 = r2.max()
                    x0, x1 = min_val, x1
                elif c == 2:
                    if r == 1:
                        x1 = l_rs_acts.max()
                    elif r == 2:
                        x1 = l_acts.max()
                    else:
                        x1 = lr2.max()
                    x0, x1 = np.log10(min_val), x1
                elif c == 3:
                    if r == 1:
                        x1 = l_rs_acts_spec.max() if len(rs_acts_spec) > 0 else np.log10(min_val)
                    elif r == 2:
                        x1 = l_acts_spec.max() if len(acts_spec) > 0 else np.log10(min_val)
                    else:
                        x1 = lr2_spec.max() if len(r2_spec) > 0 else np.log10(min_val)
                    x0, x1 = np.log10(min_val), max(x1, np.log10(min_val) + 0.1)
                else:
                    if r == 1 and rs_data and len(rs_data[0]) >= 3:
                        x0, x1 = 0.0, 1.0
                    elif r == 2 and cv_data and len(cv_data[0]) >= 3:
                        x0, x1 = 0.0, 1.0
                    elif r == 3 and rep_data and len(rep_data[0]) >= 3:
                        x0, x1 = 0.0, 1.0
                    else:
                        if r == 1:
                            x1 = s_rs_acts.max()
                        elif r == 2:
                            x1 = s_acts.max()
                        else:
                            x1 = sr2.max()
                        x0, x1 = np.sqrt(min_val), x1
                
                fig.add_shape(
                    type="line", x0=x0, y0=x0, x1=x1, y1=x1,
                    line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dash"),
                    row=r, col=c
                )

        fig.update_layout(
            template="plotly_dark",
            title_text="NMF Deconvolution Performance Diagnostics",
            height=1200, width=1600,
            showlegend=False
        )
        
        # Labels
        # Row 1 (RS)
        fig.update_xaxes(title_text="Observed Intensity", row=1, col=1)
        fig.update_yaxes(title_text="Reconstructed Intensity", row=1, col=1)
        fig.update_xaxes(title_text="Log10 Observed", row=1, col=2)
        fig.update_yaxes(title_text="Log10 Reconstructed", row=1, col=2)
        fig.update_xaxes(title_text="Log10 Observed (Specific Only)", row=1, col=3)
        fig.update_yaxes(title_text="Log10 Reconstructed (Specific Only)", row=1, col=3)
        fig.update_xaxes(title_text="Sqrt(V/scale)", row=1, col=4)
        fig.update_yaxes(title_text="Sqrt(WH/scale)", row=1, col=4)

        # Row 2 (LOO)
        fig.update_xaxes(title_text="Actual Intensity", row=2, col=1)
        fig.update_yaxes(title_text="Predicted Intensity", row=2, col=1)
        fig.update_xaxes(title_text="Log10 Actual", row=2, col=2)
        fig.update_yaxes(title_text="Log10 Predicted", row=2, col=2)
        fig.update_xaxes(title_text="Log10 Actual (Specific Only)", row=2, col=3)
        fig.update_yaxes(title_text="Log10 Predicted (Specific Only)", row=2, col=3)
        fig.update_xaxes(title_text="Sqrt(V/scale)", row=2, col=4)
        fig.update_yaxes(title_text="Sqrt(WH/scale)", row=2, col=4)
        
        # Row 3 (Replicates)
        fig.update_xaxes(title_text="Replicate 2", row=3, col=1)
        fig.update_yaxes(title_text="Replicate 1", row=3, col=1)
        fig.update_xaxes(title_text="Log10 Replicate 2", row=3, col=2)
        fig.update_yaxes(title_text="Log10 Replicate 1", row=3, col=2)
        fig.update_xaxes(title_text="Log10 Replicate 2 (Specific Only)", row=3, col=3)
        fig.update_yaxes(title_text="Log10 Replicate 1 (Specific Only)", row=3, col=3)
        fig.update_xaxes(title_text="Sqrt(Replicate 2 / scale)" if rep_data and len(rep_data[0]) >= 3 else "Sqrt Replicate 2", row=3, col=4)
        fig.update_yaxes(title_text="Sqrt(Replicate 1 / scale)" if rep_data and len(rep_data[0]) >= 3 else "Sqrt Replicate 1", row=3, col=4)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "nmf_diagnostics.html")
            fig.write_html(path)
            logger.info("[*] Diagnostic plots saved to %s", path)

        return fig
