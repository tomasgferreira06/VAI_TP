"""
Gráficos para a View 4: Fairness.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

from src.config.settings import COLORS, MODEL_COLORS, MODEL_NAMES
from src.utils.helpers import hex_to_rgba


# ═══════════════════════════════════════════════════════════════════════════════
# HORIZON GRAPH — Advanced Fairness Visualization
# ═══════════════════════════════════════════════════════════════════════════════

HORIZON_METRIC_CONFIG = {
    "FNR": {
        "rgb": (239, 68, 68),
        "label": "False Negative Rate",
        "desc": "Proportion of actual positives missed by the model",
    },
    "FPR": {
        "rgb": (245, 158, 11),
        "label": "False Positive Rate",
        "desc": "Proportion of actual negatives incorrectly flagged",
    },
    "Recall": {
        "rgb": (20, 184, 166),
        "label": "Recall (True Positive Rate)",
        "desc": "Proportion of actual positives correctly detected",
    },
}

BAND_OPACITIES = [0.18, 0.38, 0.58, 0.82]


def compute_fairness_metrics_grid(
    df: pd.DataFrame,
    sensitive_col: str,
    thresholds: np.ndarray = None,
) -> pd.DataFrame:
    """
    Build a metrics dataframe with FNR, FPR, Recall for every
    (model, subgroup, threshold) combination.

    For race, values are binarized to White / Non-White.

    Args:
        df: Evaluation DataFrame with y_true, y_proba, model, sensitive cols
        sensitive_col: Column name for the sensitive attribute
        thresholds: Array of threshold values to evaluate

    Returns:
        DataFrame with columns: model, subgroup, threshold, FNR, FPR, Recall,
        TP, FP, TN, FN
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    temp = df.copy()
    if sensitive_col == "race":
        temp["_grp"] = temp["race"].apply(
            lambda x: "White" if x == "White" else "Non-White"
        )
    else:
        temp["_grp"] = temp[sensitive_col]

    rows = []
    for model in sorted(temp["model"].unique()):
        mdf = temp[temp["model"] == model]
        for grp in sorted(mdf["_grp"].unique()):
            gdf = mdf[mdf["_grp"] == grp]
            yt = gdf["y_true"].values
            yp = gdf["y_proba"].values
            for t in thresholds:
                pred = (yp >= t).astype(int)
                cm = confusion_matrix(yt, pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                rows.append({
                    "model": model,
                    "subgroup": grp,
                    "threshold": round(t, 4),
                    "FNR": fn / (fn + tp) if (fn + tp) else 0.0,
                    "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
                    "Recall": tp / (tp + fn) if (tp + fn) else 0.0,
                    "TP": int(tp),
                    "FP": int(fp),
                    "TN": int(tn),
                    "FN": int(fn),
                })
    return pd.DataFrame(rows)


def create_fairness_sunburst(
    df: pd.DataFrame,
    sensitive_col: str,
    threshold: float = 0.5,
    model_focus: str = "both"
) -> go.Figure:
    """
    Sunburst Chart for hierarchical error analysis by demographic groups.
    
    Args:
        df: DataFrame de avaliação
        sensitive_col: Sensitive attribute column ("sex" or "race")
        threshold: Decision threshold
        model_focus: "logreg", "rf", or "both"
        
    Returns:
        Figura Plotly
    """
    color_map = {
        "Correct": COLORS["success"],
        "False Positive": COLORS["error"],
        "False Negative": COLORS["warning"]
    }
    
    def build_sunburst_data(model_df, sensitive_col):
        """Helper function to build sunburst data for a single model."""
        # Apply threshold
        model_df = model_df.copy()
        model_df["y_pred"] = (model_df["y_proba"] >= threshold).astype(int)
        
        # Create error type classification
        model_df["error_type"] = "Correct"
        model_df.loc[(model_df["y_true"] == 1) & (model_df["y_pred"] == 0), "error_type"] = "False Negative"
        model_df.loc[(model_df["y_true"] == 0) & (model_df["y_pred"] == 1), "error_type"] = "False Positive"
        
        # Handle race grouping (White / Non-White)
        if sensitive_col == "race":
            model_df["_group"] = model_df["race"].apply(
                lambda x: "White" if x == "White" else "Non-White"
            )
        else:
            model_df["_group"] = model_df[sensitive_col]
        
        # Group by error type and demographic group
        sunburst_data = model_df.groupby(["error_type", "_group"]).size().reset_index(name="count")
        
        # Prepare sunburst data structure
        labels = ["All Predictions"]
        parents = [""]
        values = [len(model_df)]
        colors = [COLORS["primary"]]
        
        # Add error type level
        for error_type in ["Correct", "False Positive", "False Negative"]:
            type_data = sunburst_data[sunburst_data["error_type"] == error_type]
            type_total = type_data["count"].sum()
            
            if type_total > 0:
                labels.append(error_type)
                parents.append("All Predictions")
                values.append(type_total)
                colors.append(color_map.get(error_type, COLORS["primary"]))
                
                # Add demographic group level
                for _, row in type_data.iterrows():
                    labels.append(f"{row['_group']}")
                    parents.append(error_type)
                    values.append(row["count"])
                    colors.append(hex_to_rgba(color_map.get(error_type, COLORS["primary"]), 0.7))
        
        return labels, parents, values, colors
    
    # Case 1: Both models - create side by side subplots
    if model_focus == "both":
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "domain"}]],
            subplot_titles=[MODEL_NAMES["logreg"], MODEL_NAMES["rf"]],
            horizontal_spacing=0.05
        )
        
        # Logistic Regression
        lr_df = df[df["model"] == "logreg"]
        lr_labels, lr_parents, lr_values, lr_colors = build_sunburst_data(lr_df, sensitive_col)
        
        fig.add_trace(go.Sunburst(
            labels=lr_labels,
            parents=lr_parents,
            values=lr_values,
            branchvalues="total",
            marker=dict(
                colors=lr_colors,
                line=dict(color=COLORS["bg_card"], width=2)
            ),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.1%}<extra></extra>",
            textfont=dict(size=10, color="#FFFFFF"),
            insidetextorientation="radial",
            domain=dict(column=0)
        ), row=1, col=1)
        
        # Random Forest
        rf_df = df[df["model"] == "rf"]
        rf_labels, rf_parents, rf_values, rf_colors = build_sunburst_data(rf_df, sensitive_col)
        
        fig.add_trace(go.Sunburst(
            labels=rf_labels,
            parents=rf_parents,
            values=rf_values,
            branchvalues="total",
            marker=dict(
                colors=rf_colors,
                line=dict(color=COLORS["bg_card"], width=2)
            ),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.1%}<extra></extra>",
            textfont=dict(size=10, color="#FFFFFF"),
            insidetextorientation="radial",
            domain=dict(column=1)
        ), row=1, col=2)
        
        fig.update_layout(
            title=dict(
                text=f"Hierarchical Error Distribution — Model Comparison<br>"
                     f"<span style='font-size:12px;color:{COLORS['text_secondary']}'>"
                     f"By {sensitive_col.title()} — Threshold: {threshold:.2f}</span>",
                font=dict(size=15)
            ),
            height=550,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=90, l=20, r=20, b=20),
            transition=dict(duration=500, easing="cubic-in-out")
        )
        
        # Style subplot titles
        for annotation in fig.layout.annotations:
            annotation.font.size = 13
            annotation.font.color = COLORS["text_primary"]
    
    # Case 2: Single model
    else:
        model_df = df[df["model"] == model_focus]
        model_label = MODEL_NAMES.get(model_focus, model_focus)
        
        labels, parents, values, colors = build_sunburst_data(model_df, sensitive_col)
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=colors,
                line=dict(color=COLORS["bg_card"], width=2)
            ),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.1%}<extra></extra>",
            textfont=dict(size=11, color="#FFFFFF"),
            insidetextorientation="radial"
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Hierarchical Error Distribution — {model_label}<br>"
                     f"<span style='font-size:12px;color:{COLORS['text_secondary']}'>"
                     f"By {sensitive_col.title()} — Threshold: {threshold:.2f}</span>",
                font=dict(size=15)
            ),
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=80, l=20, r=20, b=20),
            transition=dict(duration=500, easing="cubic-in-out")
        )
    
    return fig


def create_fairness_horizon_chart(
    df: pd.DataFrame,
    sensitive_col: str,
    current_threshold: float = 0.5,
    metric_name: str = "FNR",
    model_focus: str = "both",
    n_bands: int = 4,
) -> go.Figure:
    """
    Horizon Graph for fairness analysis.

    Shows how error rates for different demographic groups evolve as the
    decision threshold changes.  Darker / more intense bands indicate higher
    metric values, enabling visual detection of systematic disparities.

    The bottom row shows the absolute disparity (gap) between groups,
    with a 5 % reference line and annotations for large gaps.

    Args:
        df: Evaluation DataFrame
        sensitive_col: Sensitive attribute column ("sex" or "race")
        current_threshold: Current global decision threshold
        metric_name: One of "FNR", "FPR", "Recall"
        model_focus: "logreg", "rf", or "both"
        n_bands: Number of horizon bands (default 4)

    Returns:
        Plotly Figure
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    grid = compute_fairness_metrics_grid(df, sensitive_col, thresholds)

    # ── Model filtering ──────────────────────────────────────────────────
    if model_focus == "both":
        models = sorted(grid["model"].unique())
    else:
        grid = grid[grid["model"] == model_focus]
        models = [model_focus]

    subgroups = sorted(grid["subgroup"].unique())
    n_groups = len(subgroups)
    n_cols = len(models)
    n_rows = n_groups + 1  # +1 for disparity row

    # ── Band height (global max → consistent scale) ──────────────────────
    gmax = max(grid[metric_name].max(), 0.01)
    band_h = max(np.ceil(gmax / n_bands * 100) / 100, 0.02)

    # ── Pre-compute disparity per model ──────────────────────────────────
    all_gaps: dict = {}
    for m in models:
        mg = grid[grid["model"] == m]
        gaps = []
        for t in thresholds:
            td = mg[mg["threshold"] == round(t, 4)]
            if len(td) >= 2:
                vs = td[metric_name].values
                gaps.append(vs.max() - vs.min())
            else:
                gaps.append(0.0)
        all_gaps[m] = np.array(gaps)
    global_max_gap = max(
        max(g.max() for g in all_gaps.values()) if all_gaps else 0.05, 0.06
    )

    # ── Subplot titles ───────────────────────────────────────────────────
    titles = []
    for sg in subgroups:
        for m in models:
            lbl = sg
            if n_cols > 1:
                lbl += f"  —  {MODEL_NAMES.get(m, m)}"
            titles.append(lbl)
    for m in models:
        lbl = "Disparity (|Δ|)"
        if n_cols > 1:
            lbl += f"  —  {MODEL_NAMES.get(m, m)}"
        titles.append(lbl)

    row_h = [1.0] * n_groups + [0.65]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.10 if n_cols > 1 else 0.0,
        subplot_titles=titles,
        row_heights=row_h,
    )

    # ── Colour config ────────────────────────────────────────────────────
    cfg = HORIZON_METRIC_CONFIG.get(metric_name, HORIZON_METRIC_CONFIG["FNR"])
    r, g, b = cfg["rgb"]
    opacities = BAND_OPACITIES[:n_bands]

    # Helper: axis reference strings for annotations
    def _axref(row, col):
        idx = (row - 1) * n_cols + col
        return (f"x{idx}" if idx > 1 else "x", f"y{idx}" if idx > 1 else "y")

    # ── Draw horizon bands ───────────────────────────────────────────────
    for ci, model in enumerate(models):
        for ri, sg in enumerate(subgroups):
            cell = grid[
                (grid["model"] == model) & (grid["subgroup"] == sg)
            ].sort_values("threshold")
            t_arr = cell["threshold"].values
            vals = cell[metric_name].values

            # Draw bands bottom → top (later traces paint over earlier)
            for bi in range(n_bands):
                lo = bi * band_h
                clipped = np.clip(vals - lo, 0, band_h)
                opa = opacities[bi]

                fig.add_trace(
                    go.Scatter(
                        x=t_arr,
                        y=clipped,
                        fill="tozeroy",
                        mode="none",
                        fillcolor=f"rgba({r},{g},{b},{opa})",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=ri + 1,
                    col=ci + 1,
                )

            # Outline + hover trace (actual value, not clipped)
            hover_texts = [
                (
                    f"<b>{MODEL_NAMES.get(model, model)}</b> · {sg}<br>"
                    f"Threshold: {tv:.2f}<br>"
                    f"<b>{metric_name}: {vv:.1%}</b><br>"
                    f"TP={tp}  FP={fp}<br>"
                    f"TN={tn}  FN={fn}"
                )
                for tv, vv, tp, fp, tn, fn in zip(
                    t_arr,
                    vals,
                    cell["TP"].values,
                    cell["FP"].values,
                    cell["TN"].values,
                    cell["FN"].values,
                )
            ]
            fig.add_trace(
                go.Scatter(
                    x=t_arr,
                    y=np.minimum(vals, band_h),
                    mode="lines",
                    line=dict(color=f"rgba({r},{g},{b},0.65)", width=1.3),
                    text=hover_texts,
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=ri + 1,
                col=ci + 1,
            )

        # ── Disparity row ────────────────────────────────────────────────
        disp_row = n_rows
        gaps = all_gaps[model]

        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=gaps,
                fill="tozeroy",
                mode="lines",
                fillcolor=f"rgba({r},{g},{b},0.22)",
                line=dict(color=f"rgba({r},{g},{b},0.70)", width=1.5),
                hovertemplate=(
                    f"<b>{MODEL_NAMES.get(model, model)}</b><br>"
                    f"Threshold: %{{x:.2f}}<br>"
                    f"{metric_name} Gap: %{{y:.1%}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=disp_row,
            col=ci + 1,
        )

        # 5 % disparity reference line
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color=COLORS["warning"],
            line_width=1,
            annotation_text="5 %",
            annotation_font=dict(size=9, color=COLORS["warning"]),
            row=disp_row,
            col=ci + 1,
        )

        # Annotation at peak if disparity > 5 %
        peak_idx = int(np.argmax(gaps))
        if gaps[peak_idx] > 0.05:
            xr, yr = _axref(disp_row, ci + 1)
            fig.add_annotation(
                x=thresholds[peak_idx],
                y=gaps[peak_idx],
                text="⚠ Disparity",
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                arrowcolor=COLORS["error"],
                font=dict(size=9, color=COLORS["error"]),
                ax=0,
                ay=-25,
                xref=xr,
                yref=yr,
            )

    # ── Vertical threshold indicator ─────────────────────────────────────
    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            fig.add_vline(
                x=current_threshold,
                line_dash="solid",
                line_color=COLORS["text_primary"],
                line_width=1.5,
                opacity=0.6,
                row=row,
                col=col,
            )
    # Threshold label on disparity row - positioned at mid-height to avoid title overlap
    for col in range(1, n_cols + 1):
        xr, yr = _axref(n_rows, col)
        fig.add_annotation(
            x=current_threshold,
            y=global_max_gap * 0.5,  # Mid-height of disparity plot
            text=f"t={current_threshold:.2f}",
            showarrow=False,
            font=dict(size=9, color=COLORS["text_primary"]),
            bgcolor="rgba(30,41,59,0.9)",
            bordercolor=COLORS["text_primary"],
            borderwidth=1,
            borderpad=3,
            xref=xr,
            yref=yr,
            xanchor="left",
            xshift=5,
        )

    # ── Y-axis ranges ────────────────────────────────────────────────────
    for row in range(1, n_groups + 1):
        for col in range(1, n_cols + 1):
            fig.update_yaxes(
                range=[0, band_h * 1.08],
                tickformat=".0%",
                showgrid=False,
                row=row,
                col=col,
            )
    for col in range(1, n_cols + 1):
        fig.update_yaxes(
            range=[0, global_max_gap * 1.25],
            tickformat=".1%",
            showgrid=False,
            row=n_rows,
            col=col,
        )

    # ── X-axis labels (bottom row only) ──────────────────────────────────
    for col in range(1, n_cols + 1):
        fig.update_xaxes(
            title_text="Decision Threshold",
            row=n_rows,
            col=col,
            range=[0.03, 0.97],
        )

    # ── Title ────────────────────────────────────────────────────────────
    title = f"Horizon Graph — {cfg['label']} by {sensitive_col.title()}"
    if model_focus != "both":
        title += f"  ({MODEL_NAMES.get(model_focus, model_focus)})"

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=max(180 * n_rows + 50, 450),
        showlegend=False,
        margin=dict(l=60, r=30, t=80, b=50),
        # Force full figure replacement on every callback update
        uirevision=f"{sensitive_col}-{metric_name}-{model_focus}",
        datarevision=f"{sensitive_col}-{current_threshold}-{metric_name}-{model_focus}",
    )

    # Style subplot titles (only the first len(titles) annotations)
    n_titles = len(titles)
    for i, ann in enumerate(fig.layout.annotations):
        if i < n_titles:
            ann.font.size = 11
            ann.font.color = COLORS.get("text_secondary", "#94A3B8")

    return fig
