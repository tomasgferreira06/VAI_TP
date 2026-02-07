"""
Gráficos para a View 4: Fairness.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix

from src.config.settings import COLORS, MODEL_COLORS, MODEL_NAMES
from src.models.training import recompute_with_threshold


def create_fairness_accuracy_chart(df: pd.DataFrame, sensitive_col: str, threshold: float = 0.5) -> go.Figure:
    """
    Accuracy por grupo sensível.
    
    Args:
        df: DataFrame de avaliação
        sensitive_col: Coluna sensível para análise de fairness
        threshold: Limiar de decisão
        
    Returns:
        Figura Plotly
    """
    temp_df = recompute_with_threshold(df, threshold)
    
    fairness_data = (
        temp_df
        .groupby(["model", sensitive_col])
        .apply(lambda x: accuracy_score(x["y_true"], x["y_pred"]))
        .reset_index(name="accuracy")
    )
    
    fig = go.Figure()
    
    for model in fairness_data["model"].unique():
        model_data = fairness_data[fairness_data["model"] == model]
        fig.add_trace(go.Bar(
            name=MODEL_NAMES.get(model, model),
            x=model_data[sensitive_col],
            y=model_data["accuracy"],
            marker_color=MODEL_COLORS.get(model, COLORS["primary"]),
            text=[f"{v:.1%}" for v in model_data["accuracy"]],
            textposition="outside",
            textfont={"size": 11},
            hovertemplate=f"{sensitive_col}: %{{x}}<br>Accuracy: %{{y:.2%}}<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"Accuracy por {sensitive_col.title()}",
        barmode="group",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.15],
        xaxis_title=sensitive_col.title(),
        yaxis_title="Accuracy",
        height=400,
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig


def create_fairness_rates_chart(df: pd.DataFrame, sensitive_col: str, threshold: float = 0.5) -> go.Figure:
    """
    FPR e FNR por grupo sensível.
    
    Args:
        df: DataFrame de avaliação
        sensitive_col: Coluna sensível para análise de fairness
        threshold: Limiar de decisão
        
    Returns:
        Figura Plotly
    """
    temp_df = recompute_with_threshold(df, threshold)
    
    def compute_rates(group):
        tn, fp, fn, tp = confusion_matrix(group["y_true"], group["y_pred"]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        return pd.Series({"FPR": fpr, "FNR": fnr})
    
    fairness_data = (
        temp_df
        .groupby(["model", sensitive_col])
        .apply(compute_rates)
        .reset_index()
    )
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["False Positive Rate (FPR)", "False Negative Rate (FNR)"],
        horizontal_spacing=0.12
    )
    
    for i, (rate, color) in enumerate([("FPR", COLORS["error"]), ("FNR", COLORS["warning"])]):
        for model in fairness_data["model"].unique():
            model_data = fairness_data[fairness_data["model"] == model]
            fig.add_trace(
                go.Bar(
                    name=MODEL_NAMES.get(model, model),
                    x=model_data[sensitive_col],
                    y=model_data[rate],
                    marker_color=MODEL_COLORS.get(model, COLORS["primary"]),
                    text=[f"{v:.1%}" for v in model_data[rate]],
                    textposition="outside",
                    textfont={"size": 10},
                    showlegend=(i == 0),
                    hovertemplate=f"{rate}: %{{y:.2%}}<extra></extra>"
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(
        title=f"Taxas de Erro por {sensitive_col.title()}",
        barmode="group",
        height=400,
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    fig.update_yaxes(tickformat=".0%", range=[0, 0.5], row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 0.5], row=1, col=2)
    
    return fig


def create_fairness_disparity_chart(df: pd.DataFrame, sensitive_col: str, threshold: float = 0.5) -> go.Figure:
    """
    Disparidade entre grupos (gap).
    
    Args:
        df: DataFrame de avaliação
        sensitive_col: Coluna sensível para análise de fairness
        threshold: Limiar de decisão
        
    Returns:
        Figura Plotly
    """
    temp_df = recompute_with_threshold(df, threshold)
    
    fairness_data = (
        temp_df
        .groupby(["model", sensitive_col])
        .apply(lambda x: accuracy_score(x["y_true"], x["y_pred"]))
        .reset_index(name="accuracy")
    )
    
    # Calcular disparidade (max - min accuracy por modelo)
    disparity = (
        fairness_data
        .groupby("model")
        .agg(
            max_acc=("accuracy", "max"),
            min_acc=("accuracy", "min")
        )
        .reset_index()
    )
    disparity["gap"] = disparity["max_acc"] - disparity["min_acc"]
    disparity["model_name"] = disparity["model"].map(MODEL_NAMES)
    
    fig = go.Figure()
    
    for _, row in disparity.iterrows():
        fig.add_trace(go.Bar(
            x=[row["model_name"]],
            y=[row["gap"]],
            marker_color=MODEL_COLORS.get(row["model"], COLORS["primary"]),
            text=[f"{row['gap']:.1%}"],
            textposition="outside",
            textfont={"size": 12},
            name=row["model_name"],
            hovertemplate=f"Max Acc: {row['max_acc']:.2%}<br>Min Acc: {row['min_acc']:.2%}<br>Gap: {row['gap']:.2%}<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"Disparidade de Accuracy por {sensitive_col.title()} (Gap = Max - Min)",
        yaxis_tickformat=".1%",
        yaxis_range=[0, max(disparity["gap"]) * 1.5],
        yaxis_title="Accuracy Gap",
        showlegend=False,
        height=350,
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    # Adicionar linha de referência (fairness ideal = 0)
    fig.add_hline(
        y=0.05, 
        line_dash="dash", 
        line_color=COLORS["success"],
        annotation_text="Target (< 5%)",
        annotation_position="right"
    )
    
    return fig


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
    # Threshold label on first row
    for col in range(1, n_cols + 1):
        xr, yr = _axref(1, col)
        fig.add_annotation(
            x=current_threshold,
            y=band_h,
            text=f"t={current_threshold:.2f}",
            showarrow=False,
            font=dict(size=9, color=COLORS["text_primary"]),
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor=COLORS["text_primary"],
            borderwidth=1,
            borderpad=2,
            xref=xr,
            yref=yr,
            yshift=14,
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
