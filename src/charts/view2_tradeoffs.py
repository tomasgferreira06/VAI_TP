"""
Gráficos para a View 2: Trade-offs (Precision vs Recall).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

from src.config.settings import COLORS, MODEL_COLORS, MODEL_NAMES, CHART_LEGEND_CONFIG
from src.models.training import global_metrics, recompute_with_threshold


# ═══════════════════════════════════════════════════════════════════════════════
# DECISION MODE CONFIGURATION FOR PR CURVE
# ═══════════════════════════════════════════════════════════════════════════════

PR_DECISION_MODE_CONFIG = {
    "balanced": {
        "label": "Balanced / Max F1",
        "highlight_region": None,
        "emphasis": "f1",
        "annotation": "Operating point (Balanced / max F1)"
    },
    "precision": {
        "label": "Precision-focused",
        "highlight_region": {"metric": "precision", "threshold": 0.8},
        "emphasis": "precision",
        "annotation": "Operating point for Precision-focused mode"
    },
    "recall": {
        "label": "Recall-focused",
        "highlight_region": {"metric": "recall", "threshold": 0.8},
        "emphasis": "recall",
        "annotation": "Operating point for Recall-focused mode"
    }
}


def _find_closest_threshold_idx(thresholds: np.ndarray, target_threshold: float) -> int:
    """Find the index of the closest threshold to the target."""
    if len(thresholds) == 0:
        return 0
    # Thresholds array is one element shorter than precision/recall arrays
    idx = np.argmin(np.abs(thresholds - target_threshold))
    return idx


def _compute_f1_scores(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """Compute F1 scores for each precision-recall pair."""
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1, nan=0.0)
    return f1


def _find_optimal_point(precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray, 
                        mode: str) -> dict:
    """Find the optimal operating point based on decision mode."""
    f1_scores = _compute_f1_scores(precision[:-1], recall[:-1])  # Exclude last point (threshold undefined)
    
    if mode == "balanced":
        # Max F1 point
        idx = np.argmax(f1_scores)
    elif mode == "precision":
        # Max recall where precision >= 0.8
        valid_mask = precision[:-1] >= 0.8
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            # Among valid points, find max recall
            recall_values = recall[:-1][valid_mask]
            best_valid_idx = np.argmax(recall_values)
            idx = valid_indices[best_valid_idx]
        else:
            # Fallback to highest precision point
            idx = np.argmax(precision[:-1])
    else:  # recall
        # Max precision where recall >= 0.8
        valid_mask = recall[:-1] >= 0.8
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            # Among valid points, find max precision
            precision_values = precision[:-1][valid_mask]
            best_valid_idx = np.argmax(precision_values)
            idx = valid_indices[best_valid_idx]
        else:
            # Fallback to highest recall point
            idx = np.argmax(recall[:-1])
    
    return {
        "idx": idx,
        "precision": precision[idx],
        "recall": recall[idx],
        "threshold": thresholds[idx] if idx < len(thresholds) else thresholds[-1],
        "f1": f1_scores[idx]
    }


def _compute_confusion_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, 
                                     threshold: float) -> dict:
    """Compute confusion matrix components at a specific threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def create_precision_recall_curve_enhanced(
    df: pd.DataFrame,
    threshold: float = 0.5,
    decision_mode: str = "balanced",
    show_area: bool = False,
    analysis_focus: str = "global"
) -> tuple:
    """
    Enhanced Precision-Recall curve with decision-oriented interactions.
    
    Args:
        df: DataFrame de avaliação
        threshold: Current global threshold
        decision_mode: "balanced", "precision", or "recall"
        show_area: Whether to fill area under curve
        analysis_focus: "global", "sex", or "race" - demographic analysis mode
        
    Returns:
        Tuple of (figure, delta_ap_text)
    """
    fig = go.Figure()
    
    mode_config = PR_DECISION_MODE_CONFIG.get(decision_mode, PR_DECISION_MODE_CONFIG["balanced"])
    
    # Store AP values for delta calculation
    ap_values = {}
    
    # Configure demographic groups based on analysis_focus
    if analysis_focus == "sex":
        demographic_groups = [
            ("Male", df[df["sex"] == "Male"], "solid"),
            ("Female", df[df["sex"] == "Female"], "dash")
        ]
        title_demographic = " by Sex"
    elif analysis_focus == "race":
        demographic_groups = [
            ("White", df[df["race"] == "White"], "solid"),
            ("Non-White", df[df["race"] != "White"], "dash")
        ]
        title_demographic = " by Race"
    else:
        demographic_groups = [("Global", df, "solid")]
        title_demographic = ""
    
    # ═══════════════════════════════════════════════════════════════════════════
    # D) HIGHLIGHT REGIONS OF INTEREST (background shapes)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # High Precision region (Precision > 0.8)
    precision_region_opacity = 0.15 if decision_mode == "precision" else 0.05
    fig.add_shape(
        type="rect",
        x0=0, x1=1, y0=0.8, y1=1,
        fillcolor=COLORS["success"],
        opacity=precision_region_opacity,
        layer="below",
        line_width=0
    )
    
    # High Recall region (Recall > 0.8)
    recall_region_opacity = 0.15 if decision_mode == "recall" else 0.05
    fig.add_shape(
        type="rect",
        x0=0.8, x1=1, y0=0, y1=1,
        fillcolor=COLORS["accent"],
        opacity=recall_region_opacity,
        layer="below",
        line_width=0
    )
    
    # Region labels
    if decision_mode == "precision":
        fig.add_annotation(
            x=0.5, y=0.92,
            text="High Precision Zone",
            showarrow=False,
            font=dict(size=9, color=COLORS["success"]),
            opacity=0.8
        )
    elif decision_mode == "recall":
        fig.add_annotation(
            x=0.92, y=0.5,
            text="High Recall Zone",
            showarrow=False,
            font=dict(size=9, color=COLORS["accent"]),
            textangle=-90,
            opacity=0.8
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN PR CURVES FOR EACH MODEL AND DEMOGRAPHIC GROUP
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Store first model's optimal point for annotation (avoid recalculating later)
    first_model_optimal = None
    
    for model in df["model"].unique():
        model_color = MODEL_COLORS.get(model, COLORS["primary"])
        model_name = MODEL_NAMES.get(model, model)
        
        for group_name, group_df, line_dash in demographic_groups:
            model_group_df = group_df[group_df["model"] == model]
            
            if len(model_group_df) < 10:
                continue
                
            y_true = model_group_df["y_true"].values
            y_proba = model_group_df["y_proba"].values
            
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            ap = average_precision_score(y_true, y_proba)
            
            # Store AP with group info for delta calculation
            ap_key = f"{model}_{group_name}" if analysis_focus != "global" else model
            ap_values[ap_key] = ap
            
            # Build legend name based on analysis focus
            if analysis_focus == "global":
                legend_name = f"{model_name} (AP = {ap:.3f})"
                legend_group = model
            else:
                legend_name = f"{model_name} ({group_name}) — AP: {ap:.3f}"
                legend_group = f"{model}_{group_name}"
            
            # ═══════════════════════════════════════════════════════════════════════
            # G) OPTIONAL: FILLED AREA UNDER CURVE
            # ═══════════════════════════════════════════════════════════════════════
            
            if show_area and analysis_focus == "global":  # Only show area for global view
                fig.add_trace(go.Scatter(
                    x=recall,
                    y=precision,
                    mode="none",
                    fill="tozeroy",
                    fillcolor=f"rgba{tuple(list(int(model_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}",
                    name=f"{model_name} Area",
                    legendgroup=legend_group,
                    showlegend=False,
                    hoverinfo="skip"
                ))
        
            # ═══════════════════════════════════════════════════════════════════════
            # B) RICH TOOLTIP ALONG THE CURVE (OPTIMIZED)
            # ═══════════════════════════════════════════════════════════════════════
            
            # Prepare custom data for rich tooltips
            # Note: thresholds array is 1 shorter than precision/recall
            custom_thresholds = np.append(thresholds, thresholds[-1] if len(thresholds) > 0 else 0.5)
            f1_scores = _compute_f1_scores(precision, recall)
            
            # OPTIMIZATION: Don't compute confusion matrix for every point
            # Only show basic metrics in hover - confusion matrix only for marker points
            hover_texts = []
            display_name = f"{model_name} ({group_name})" if analysis_focus != "global" else model_name
            for i in range(len(precision)):
                t = custom_thresholds[i] if i < len(custom_thresholds) else custom_thresholds[-1]
                
                hover_text = (
                    f"<b>{display_name}</b><br>"
                    f"Threshold: {t:.3f}<br>"
                    f"Precision: {precision[i]:.3f}<br>"
                    f"Recall: {recall[i]:.3f}<br>"
                    f"F1: {f1_scores[i]:.3f}"
                )
                hover_texts.append(hover_text)
            
            # Determine curve opacity based on decision mode
            curve_opacity = 1.0
            if decision_mode == "precision":
                # Emphasize high-precision portion
                pass  # We'll use full opacity but can dim in other ways
            elif decision_mode == "recall":
                # Emphasize high-recall portion
                pass
            
            # Main PR curve trace
            fig.add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=legend_name,
                line={"color": model_color, "width": 2.5, "dash": line_dash},
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
                opacity=curve_opacity,
                legendgroup=legend_group,
                showlegend=True
            ))
            
            # ═══════════════════════════════════════════════════════════════════════
            # A) THRESHOLD MARKER (LINKED TO GLOBAL SLIDER)
            # ═══════════════════════════════════════════════════════════════════════
            
            # Find closest point to current threshold
            thresh_idx = _find_closest_threshold_idx(thresholds, threshold)
            thresh_precision = precision[thresh_idx]
            thresh_recall = recall[thresh_idx]
            actual_threshold = thresholds[thresh_idx] if thresh_idx < len(thresholds) else threshold
            
            # Compute confusion at current threshold
            cm_at_thresh = _compute_confusion_at_threshold(y_true, y_proba, actual_threshold)
            
            marker_hover = (
                f"<b>Current Threshold — {display_name}</b><br>"
                f"Threshold: {actual_threshold:.3f}<br>"
                f"Precision: {thresh_precision:.3f}<br>"
                f"Recall: {thresh_recall:.3f}<br>"
                f"─────────────<br>"
                f"TP: {cm_at_thresh['tp']:,} | FP: {cm_at_thresh['fp']:,}<br>"
                f"FN: {cm_at_thresh['fn']:,}"
            )
            
            fig.add_trace(go.Scatter(
                x=[thresh_recall],
                y=[thresh_precision],
                mode="markers",
                name=f"{display_name} @ t={actual_threshold:.2f}",
                marker=dict(
                    size=14,
                    color=model_color,
                    symbol="circle",
                    line=dict(color="white", width=3)
                ),
                hovertemplate=marker_hover + "<extra></extra>",
                legendgroup=legend_group,
                showlegend=False
            ))
            
            # ═══════════════════════════════════════════════════════════════════════
            # E) & F) OPTIMAL POINT ACCORDING TO DECISION MODE
            # ═══════════════════════════════════════════════════════════════════════
            
            optimal = _find_optimal_point(precision, recall, thresholds, decision_mode)
            
            # Store first model's optimal for annotation later
            if first_model_optimal is None:
                first_model_optimal = optimal
            
            # Different marker for optimal point
            fig.add_trace(go.Scatter(
                x=[optimal["recall"]],
                y=[optimal["precision"]],
                mode="markers",
                name=f"{display_name} Optimal",
                marker=dict(
                    size=12,
                    color=model_color,
                    symbol="star",
                    line=dict(color="white", width=2)
                ),
                hovertemplate=(
                    f"<b>Optimal ({mode_config['label']}) — {display_name}</b><br>"
                    f"Threshold: {optimal['threshold']:.3f}<br>"
                    f"Precision: {optimal['precision']:.3f}<br>"
                    f"Recall: {optimal['recall']:.3f}<br>"
                    f"F1: {optimal['f1']:.3f}"
                    "<extra></extra>"
                ),
                legendgroup=legend_group,
                showlegend=False
            ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEGEND MARKERS: Add dummy traces to explain symbols in legend
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Add legend entry for optimal point (star) - informative only, not interactive
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=12, color="gray", symbol="star", line=dict(color="white", width=2)),
        name=f"Optimal ({mode_config['label']})",
        showlegend=True,
        hoverinfo="skip"
    ))
    
    # Add legend entry for current threshold marker (circle) - informative only, not interactive
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=14, color="gray", symbol="circle", line=dict(color="white", width=3)),
        name=f"Current Threshold ({threshold:.2f})",
        showlegend=True,
        hoverinfo="skip"
    ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BASELINE
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Baseline: proportion of positives
    baseline_precision = df["y_true"].mean()
    fig.add_hline(
        y=baseline_precision,
        line_dash="dot",
        line_color=COLORS["text_muted"],
        line_width=1
    )
    
    fig.add_annotation(
        x=0.12,
        y=baseline_precision + 0.04,
        text=f"Random ({baseline_precision:.1%})",
        showarrow=False,
        font=dict(size=9, color=COLORS["text_muted"])
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LAYOUT
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Build title with mode indication and demographic info
    title_suffix = f" • Mode: {mode_config['label']}"
    
    fig.update_layout(
        title=dict(
            text=f"Precision-Recall Curves{title_demographic}{title_suffix}",
            font=dict(size=14)
        ),
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1.02],
        yaxis_range=[0, 1.05],
        height=480,
        legend=CHART_LEGEND_CONFIG,
        margin=dict(b=130),
        transition=dict(duration=400, easing="cubic-in-out")
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # G) DELTA AP CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    delta_ap_text = ""
    if len(ap_values) == 2:
        models = list(ap_values.keys())
        ap1, ap2 = ap_values[models[0]], ap_values[models[1]]
        delta = ap2 - ap1
        model_name1 = MODEL_NAMES.get(models[0], models[0])
        model_name2 = MODEL_NAMES.get(models[1], models[1])
        
        if delta > 0:
            delta_ap_text = f"ΔAP ({model_name2} − {model_name1}): +{delta:.4f}"
        else:
            delta_ap_text = f"ΔAP ({model_name2} − {model_name1}): {delta:.4f}"
    
    return fig, delta_ap_text


def create_precision_recall_curve(df: pd.DataFrame) -> go.Figure:
    """
    Curva Precision-Recall para ambos os modelos.
    Legacy function for backward compatibility.
    
    Args:
        df: DataFrame de avaliação
        
    Returns:
        Figura Plotly
    """
    fig, _ = create_precision_recall_curve_enhanced(df)
    return fig


def create_threshold_analysis(df: pd.DataFrame, selected_model: str) -> go.Figure:
    """
    Análise de métricas por threshold com anotação do threshold ótimo.
    
    Args:
        df: DataFrame de avaliação
        selected_model: Modelo selecionado
        
    Returns:
        Figura Plotly
    """
    model_df = df[df["model"] == selected_model]
    
    thresholds = np.linspace(0.1, 0.9, 17)
    metrics_by_thresh = []
    
    for t in thresholds:
        temp_df = recompute_with_threshold(model_df, t)
        m = global_metrics(temp_df)
        m["threshold"] = t
        metrics_by_thresh.append(m)
    
    thresh_df = pd.DataFrame(metrics_by_thresh)
    
    fig = go.Figure()
    
    metric_config = [
        ("precision", "Precision", COLORS["primary"]),
        ("recall", "Recall", COLORS["accent"]),
        ("f1", "F1-Score", COLORS["secondary"]),
    ]
    
    for metric, label, color in metric_config:
        fig.add_trace(go.Scatter(
            x=thresh_df["threshold"],
            y=thresh_df[metric],
            mode="lines+markers",
            name=label,
            line={"color": color, "width": 2},
            marker={"size": 6},
            hovertemplate=f"Threshold: %{{x:.2f}}<br>{label}: %{{y:.2%}}<extra></extra>"
        ))
    
    # Encontrar threshold ótimo (máximo F1)
    optimal_idx = thresh_df["f1"].idxmax()
    optimal_thresh = thresh_df.loc[optimal_idx, "threshold"]
    optimal_f1 = thresh_df.loc[optimal_idx, "f1"]
    
    # Linha vertical no threshold ótimo
    fig.add_vline(
        x=optimal_thresh,
        line_dash="dash",
        line_color=COLORS["success"],
        line_width=2
    )
    
    # Anotação do threshold ótimo
    fig.add_annotation(
        x=optimal_thresh,
        y=optimal_f1 + 0.08,
        text=f"Optimal F1<br>t={optimal_thresh:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["success"],
        font=dict(size=10, color=COLORS["success"]),
        bgcolor=COLORS["bg_card"],
        borderpad=4
    )
    
    # Anotação baseline (threshold=0.5)
    fig.add_vline(
        x=0.5,
        line_dash="dot",
        line_color=COLORS["text_muted"],
        line_width=1
    )
    
    fig.add_annotation(
        x=0.5,
        y=0.15,
        text="Default (0.5)",
        showarrow=False,
        font=dict(size=9, color=COLORS["text_muted"]),
        textangle=-90
    )
    
    fig.update_layout(
        title=f"Metricas vs Threshold - {MODEL_NAMES.get(selected_model, selected_model)}",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        xaxis_range=[0.05, 0.95],
        yaxis_range=[0, 1.05],
        yaxis_tickformat=".0%",
        height=400,
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD ANALYSIS DECISION MODE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

THRESHOLD_DECISION_MODE_CONFIG = {
    "balanced": {
        "label": "Balanced / Max F1",
        "emphasis": "f1",
        "annotation": "Operating point (Balanced / max F1)",
        "highlight": {"metric": "f1", "description": "Max F1 region"}
    },
    "precision": {
        "label": "Precision-focused",
        "emphasis": "precision",
        "annotation": "Operating point (Precision-focused)",
        "highlight": {"metric": "precision", "threshold": 0.8, "description": "High Precision region"}
    },
    "recall": {
        "label": "Recall-focused",
        "emphasis": "recall",
        "annotation": "Operating point (Recall-focused)",
        "highlight": {"metric": "recall", "threshold": 0.8, "description": "High Recall region"}
    }
}


def _find_optimal_threshold_for_mode(thresh_df: pd.DataFrame, mode: str) -> dict:
    """
    Find the optimal operating point threshold based on decision mode.
    
    Args:
        thresh_df: DataFrame with threshold and metrics columns
        mode: "balanced", "precision", or "recall"
        
    Returns:
        dict with idx, threshold, precision, recall, f1
    """
    if mode == "balanced":
        # Max F1 point
        idx = thresh_df["f1"].idxmax()
    elif mode == "precision":
        # Highest threshold where precision is maximized
        # (or max recall where precision >= 0.8)
        valid_mask = thresh_df["precision"] >= 0.8
        if valid_mask.any():
            valid_df = thresh_df[valid_mask]
            # Among valid points, find max recall
            idx = valid_df["recall"].idxmax()
        else:
            # Fallback to max precision
            idx = thresh_df["precision"].idxmax()
    else:  # recall
        # Lowest threshold where recall is maximized
        # (or max precision where recall >= 0.8)
        valid_mask = thresh_df["recall"] >= 0.8
        if valid_mask.any():
            valid_df = thresh_df[valid_mask]
            # Among valid points, find max precision
            idx = valid_df["precision"].idxmax()
        else:
            # Fallback to max recall
            idx = thresh_df["recall"].idxmax()
    
    return {
        "idx": idx,
        "threshold": thresh_df.loc[idx, "threshold"],
        "precision": thresh_df.loc[idx, "precision"],
        "recall": thresh_df.loc[idx, "recall"],
        "f1": thresh_df.loc[idx, "f1"]
    }


def create_threshold_analysis_enhanced(
    df: pd.DataFrame,
    selected_model: str = None,
    threshold: float = 0.5,
    decision_mode: str = "balanced",
    show_precision: bool = True,
    show_recall: bool = True,
    show_f1: bool = True,
    overlay_models: bool = False,
    analysis_focus: str = "global"
) -> go.Figure:
    """
    Enhanced Metrics vs Threshold analysis with decision-oriented interactions.
    
    Features:
    - A) Link to global threshold slider with highlighted markers
    - B) Rich tooltips for any point on curves
    - C) Decision Mode integration from global controls
    - D) Highlight regions of interest based on mode
    - E) Toggle metrics visibility (Precision/Recall/F1)
    - F) Support for multiple models (overlay or single)
    
    Args:
        df: DataFrame de avaliação
        selected_model: Modelo selecionado (None = all models if overlay_models)
        threshold: Current global threshold value
        decision_mode: "balanced", "precision", or "recall"
        show_precision: Whether to show Precision curve
        show_recall: Whether to show Recall curve
        show_f1: Whether to show F1 curve
        overlay_models: If True, overlay both models; if False, show only selected_model
        analysis_focus: "global", "sex", or "race" - demographic analysis mode
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    mode_config = THRESHOLD_DECISION_MODE_CONFIG.get(decision_mode, THRESHOLD_DECISION_MODE_CONFIG["balanced"])
    
    # Configure demographic groups based on analysis focus
    if analysis_focus == "sex":
        demographic_groups = [
            ("Male", df[df["sex"] == "Male"], "solid"),
            ("Female", df[df["sex"] == "Female"], "dash")
        ]
        title_demographic = " by Sex"
    elif analysis_focus == "race":
        demographic_groups = [
            ("White", df[df["race"] == "White"], "solid"),
            ("Non-White", df[df["race"] != "White"], "dash")
        ]
        title_demographic = " by Race"
    else:
        demographic_groups = [("All Data", df, "solid")]
        title_demographic = ""
    
    # Determine which models to plot
    if overlay_models:
        models_to_plot = df["model"].unique().tolist()
    else:
        models_to_plot = [selected_model] if selected_model else df["model"].unique().tolist()[:1]
    
    # Define thresholds range
    thresholds = np.linspace(0.1, 0.9, 33)  # More granular for smoother curves
    
    # Metric configuration
    metric_config = [
        ("precision", "Precision", COLORS["primary"], show_precision),
        ("recall", "Recall", COLORS["accent"], show_recall),
        ("f1", "F1-Score", COLORS["secondary"], show_f1),
    ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # D) HIGHLIGHT REGIONS OF INTEREST (background shapes)
    # ═══════════════════════════════════════════════════════════════════════════
    
    if decision_mode == "precision":
        # Highlight high precision region (right side of plot where threshold is higher)
        # Precision generally increases with higher thresholds
        fig.add_shape(
            type="rect",
            x0=0.6, x1=0.95, y0=0, y1=1.05,
            fillcolor=COLORS["primary"],
            opacity=0.08,
            layer="below",
            line_width=0
        )
        fig.add_annotation(
            x=0.78, y=0.98,
            text="High Precision Zone",
            showarrow=False,
            font=dict(size=9, color=COLORS["primary"]),
            opacity=0.7
        )
    elif decision_mode == "recall":
        # Highlight high recall region (left side where threshold is lower)
        # Recall generally increases with lower thresholds
        fig.add_shape(
            type="rect",
            x0=0.1, x1=0.4, y0=0, y1=1.05,
            fillcolor=COLORS["accent"],
            opacity=0.08,
            layer="below",
            line_width=0
        )
        fig.add_annotation(
            x=0.25, y=0.98,
            text="High Recall Zone",
            showarrow=False,
            font=dict(size=9, color=COLORS["accent"]),
            opacity=0.7
        )
    else:  # balanced
        # We'll highlight the F1 peak region after computing metrics
        pass
    
    # Store data for each model and group combination
    all_model_data = {}
    
    for model in models_to_plot:
        model_name = MODEL_NAMES.get(model, model)
        model_color = MODEL_COLORS.get(model, COLORS["primary"])
        
        for group_name, group_df, line_dash in demographic_groups:
            model_group_df = group_df[group_df["model"] == model]
            
            if len(model_group_df) < 10:
                continue
                
            y_true = model_group_df["y_true"].values
            y_proba = model_group_df["y_proba"].values
            
            # Build display name and legend group
            if analysis_focus == "global":
                display_name = model_name
                legend_group_base = model
            else:
                display_name = f"{model_name} ({group_name})"
                legend_group_base = f"{model}_{group_name}"
            
            # Compute metrics at each threshold
            metrics_by_thresh = []
            for t in thresholds:
                temp_df = recompute_with_threshold(model_group_df, t)
                m = global_metrics(temp_df)
                m["threshold"] = t
                
                # Compute confusion matrix for rich tooltips
                y_pred_at_t = (y_proba >= t).astype(int)
                cm = confusion_matrix(y_true, y_pred_at_t)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
                m["tp"] = tp
                m["fp"] = fp
                m["fn"] = fn
                m["tn"] = tn
                
                metrics_by_thresh.append(m)
            
            thresh_df = pd.DataFrame(metrics_by_thresh)
            all_model_data[legend_group_base] = thresh_df
            
            # Define line styles for each metric when in overlay mode or demographic
            metric_line_styles = {
                "precision": "solid",
                "recall": "dash",
                "f1": "dot"
            }
            
            # ═══════════════════════════════════════════════════════════════════════
            # PLOT METRIC CURVES WITH RICH TOOLTIPS (B)
            # ═══════════════════════════════════════════════════════════════════════
            
            for metric_key, metric_label, metric_color, is_visible in metric_config:
                if not is_visible:
                    continue
                
                # Build rich hover text for each point (B)
                hover_texts = []
                for _, row in thresh_df.iterrows():
                    hover_text = (
                        f"<b>{metric_label} — {display_name}</b><br>"
                        f"Threshold: {row['threshold']:.3f}<br>"
                        f"─────────────<br>"
                        f"Precision: {row['precision']:.3f}<br>"
                        f"Recall: {row['recall']:.3f}<br>"
                        f"F1: {row['f1']:.3f}<br>"
                        f"─────────────<br>"
                        f"TP: {row['tp']:,.0f} | FP: {row['fp']:,.0f}<br>"
                        f"FN: {row['fn']:,.0f} | TN: {row['tn']:,.0f}"
                    )
                    hover_texts.append(hover_text)
                
                # Adjust color and line style based on context
                if overlay_models or analysis_focus != "global":
                    curve_color = model_color
                    metric_line_dash = line_dash if analysis_focus != "global" else metric_line_styles.get(metric_key, "solid")
                    trace_name = f"{metric_label} ({display_name})"
                else:
                    curve_color = metric_color
                    metric_line_dash = "solid"
                    trace_name = metric_label
                
                fig.add_trace(go.Scatter(
                    x=thresh_df["threshold"],
                    y=thresh_df[metric_key],
                    mode="lines+markers",
                    name=trace_name,
                    line={"color": curve_color, "width": 2.5, "dash": metric_line_dash},
                    marker={"size": 5},
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_texts,
                    legendgroup=f"{legend_group_base}_{metric_key}",
                    visible=True
                ))
            
            # ═══════════════════════════════════════════════════════════════════════
            # A) CURRENT THRESHOLD MARKER (LINKED TO GLOBAL SLIDER)
            # ═══════════════════════════════════════════════════════════════════════
            
            # Find closest threshold in our computed range
            thresh_idx = np.argmin(np.abs(thresholds - threshold))
            current_row = thresh_df.iloc[thresh_idx]
            
            # Add markers at intersection points for each visible metric
            for metric_key, metric_label, metric_color, is_visible in metric_config:
                if not is_visible:
                    continue
                
                marker_color = model_color if (overlay_models or analysis_focus != "global") else metric_color
                
                # Rich tooltip for current threshold marker (A)
                marker_hover = (
                    f"<b>Current Threshold — {display_name}</b><br>"
                    f"Threshold: {current_row['threshold']:.3f}<br>"
                    f"─────────────<br>"
                    f"Precision: {current_row['precision']:.3f}<br>"
                    f"Recall: {current_row['recall']:.3f}<br>"
                    f"F1: {current_row['f1']:.3f}<br>"
                    f"─────────────<br>"
                    f"TP: {current_row['tp']:,.0f} | FP: {current_row['fp']:,.0f}<br>"
                    f"FN: {current_row['fn']:,.0f} | TN: {current_row['tn']:,.0f}"
                )
                
                fig.add_trace(go.Scatter(
                    x=[current_row["threshold"]],
                    y=[current_row[metric_key]],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color=marker_color,
                        symbol="circle",
                        line=dict(color="white", width=3)
                    ),
                    hovertemplate=marker_hover + "<extra></extra>",
                    showlegend=False,
                    legendgroup=f"{legend_group_base}_{metric_key}"
                ))
            
            # ═══════════════════════════════════════════════════════════════════════
            # C) & E) OPTIMAL OPERATING POINT BASED ON DECISION MODE
            # ═══════════════════════════════════════════════════════════════════════
            
            optimal = _find_optimal_threshold_for_mode(thresh_df, decision_mode)
            optimal_row = thresh_df.loc[optimal["idx"]]
            
            # Add star markers at optimal point for each visible metric
            for metric_key, metric_label, metric_color, is_visible in metric_config:
                if not is_visible:
                    continue
                
                marker_color = model_color if (overlay_models or analysis_focus != "global") else COLORS["success"]
                
                # Optimal point tooltip
                optimal_hover = (
                    f"<b>{mode_config['annotation']} — {display_name}</b><br>"
                    f"Threshold: {optimal_row['threshold']:.3f}<br>"
                    f"─────────────<br>"
                    f"Precision: {optimal_row['precision']:.3f}<br>"
                    f"Recall: {optimal_row['recall']:.3f}<br>"
                    f"F1: {optimal_row['f1']:.3f}<br>"
                    f"─────────────<br>"
                    f"TP: {optimal_row['tp']:,.0f} | FP: {optimal_row['fp']:,.0f}<br>"
                    f"FN: {optimal_row['fn']:,.0f} | TN: {optimal_row['tn']:,.0f}"
                )
                
                fig.add_trace(go.Scatter(
                    x=[optimal_row["threshold"]],
                    y=[optimal_row[metric_key]],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=marker_color,
                        symbol="star",
                        line=dict(color="white", width=2)
                    ),
                    hovertemplate=optimal_hover + "<extra></extra>",
                    showlegend=False,
                    legendgroup=f"{legend_group_base}_{metric_key}"
                ))
    
    # Add vertical lines for current threshold and optimal (only once outside the loops)
    fig.add_vline(
        x=threshold,
        line_dash="solid",
        line_color=COLORS["warning"],
        line_width=2.5,
        opacity=0.9
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # D) BALANCED MODE: Highlight F1 peak region
    # ═══════════════════════════════════════════════════════════════════════════
    
    if decision_mode == "balanced" and len(all_model_data) > 0:
        # Get F1 data from first entry in all_model_data
        first_key = list(all_model_data.keys())[0]
        thresh_df = all_model_data[first_key]
        optimal = _find_optimal_threshold_for_mode(thresh_df, "balanced")
        
        # Highlight region around optimal F1 (±0.1)
        opt_t = optimal["threshold"]
        x0 = max(0.1, opt_t - 0.1)
        x1 = min(0.9, opt_t + 0.1)
        
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=0, y1=1.05,
            fillcolor=COLORS["secondary"],
            opacity=0.08,
            layer="below",
            line_width=0
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=0.98,
            text="Max F1 Zone",
            showarrow=False,
            font=dict(size=9, color=COLORS["secondary"]),
            opacity=0.7
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEGEND ENTRIES FOR SYMBOLS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Add legend entry for current threshold marker
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=14, color=COLORS["warning"], symbol="circle", line=dict(color="white", width=3)),
        name=f"● Current Threshold ({threshold:.2f})",
        showlegend=True,
        hoverinfo="skip"
    ))
    
    # Add legend entry for optimal point
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=12, color=COLORS["success"], symbol="star", line=dict(color="white", width=2)),
        name=f"★ Optimal ({mode_config['label']})",
        showlegend=True,
        hoverinfo="skip"
    ))
    
    # Add legend entries for line styles when in overlay mode
    if overlay_models:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="gray", width=2, dash="solid"),
            name="─── Precision",
            showlegend=True,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="gray", width=2, dash="dash"),
            name="- - - Recall",
            showlegend=True,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="gray", width=2, dash="dot"),
            name="··· F1-Score",
            showlegend=True,
            hoverinfo="skip"
        ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LAYOUT
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Build title with demographic info
    if overlay_models:
        title_text = f"Metrics vs Threshold{title_demographic} (All Models) • Mode: {mode_config['label']}"
    else:
        model_display = MODEL_NAMES.get(selected_model, selected_model) if selected_model else ""
        title_text = f"Metrics vs Threshold{title_demographic} - {model_display} • Mode: {mode_config['label']}"
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=14)
        ),
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        xaxis_range=[0.08, 0.92],
        yaxis_range=[0, 1.08],
        yaxis_tickformat=".0%",
        height=550 if overlay_models else 480,
        legend=CHART_LEGEND_CONFIG,
        margin=dict(b=180 if overlay_models else 130),
        transition=dict(duration=400, easing="cubic-in-out")
    )
    
    return fig


def create_fp_fn_evolution_chart(df: pd.DataFrame, selected_model: str) -> go.Figure:
    """
    Gráfico de evolução de FP/FN com threshold.
    
    Args:
        df: DataFrame de avaliação
        selected_model: Modelo selecionado
        
    Returns:
        Figura Plotly
    """
    model_df = df[df["model"] == selected_model]
    
    thresholds = np.linspace(0.1, 0.9, 17)
    evolution_data = []
    
    for t in thresholds:
        temp_df = recompute_with_threshold(model_df, t)
        tn, fp, fn, tp = confusion_matrix(temp_df["y_true"], temp_df["y_pred"]).ravel()
        
        # Calcular taxas
        total = len(temp_df)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        evolution_data.append({
            "threshold": t,
            "FP": fp,
            "FN": fn,
            "FPR": fpr,
            "FNR": fnr,
            "Total Errors": fp + fn,
            "Error Rate": (fp + fn) / total
        })
    
    evo_df = pd.DataFrame(evolution_data)
    
    # Criar gráfico com eixo duplo
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Contagem de Erros vs Threshold",
            "Taxas de Erro vs Threshold"
        ],
        horizontal_spacing=0.12
    )
    
    # Subplot 1: Contagem absoluta de FP e FN
    fig.add_trace(
        go.Scatter(
            x=evo_df["threshold"],
            y=evo_df["FP"],
            mode="lines+markers",
            name="False Positives",
            line={"color": COLORS["error"], "width": 2.5},
            marker={"size": 7, "symbol": "circle"},
            hovertemplate="Threshold: %{x:.2f}<br>FP: %{y:,}<extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=evo_df["threshold"],
            y=evo_df["FN"],
            mode="lines+markers",
            name="False Negatives",
            line={"color": COLORS["warning"], "width": 2.5},
            marker={"size": 7, "symbol": "diamond"},
            hovertemplate="Threshold: %{x:.2f}<br>FN: %{y:,}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Subplot 2: Taxas FPR e FNR
    fig.add_trace(
        go.Scatter(
            x=evo_df["threshold"],
            y=evo_df["FPR"],
            mode="lines+markers",
            name="FPR",
            line={"color": COLORS["error"], "width": 2.5, "dash": "solid"},
            marker={"size": 7, "symbol": "circle"},
            showlegend=False,
            hovertemplate="Threshold: %{x:.2f}<br>FPR: %{y:.1%}<extra></extra>"
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=evo_df["threshold"],
            y=evo_df["FNR"],
            mode="lines+markers",
            name="FNR",
            line={"color": COLORS["warning"], "width": 2.5, "dash": "solid"},
            marker={"size": 7, "symbol": "diamond"},
            showlegend=False,
            hovertemplate="Threshold: %{x:.2f}<br>FNR: %{y:.1%}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Encontrar threshold onde FP ≈ FN
    min_diff_idx = np.argmin(np.abs(evo_df["FP"].values - evo_df["FN"].values))
    optimal_thresh = evo_df["threshold"].iloc[min_diff_idx]
    
    fig.add_vline(
        x=optimal_thresh,
        line_dash="dash",
        line_color=COLORS["success"],
        annotation_text=f"Balance ≈ {optimal_thresh:.2f}",
        annotation_position="top",
        row=1, col=1
    )
    
    fig.update_layout(
        title=f"Evolução de Erros vs Threshold - {MODEL_NAMES.get(selected_model, selected_model)}",
        height=380,
        legend=CHART_LEGEND_CONFIG,
        margin=dict(b=100),
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    fig.update_xaxes(title_text="Threshold", row=1, col=1)
    fig.update_xaxes(title_text="Threshold", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Rate", tickformat=".0%", row=1, col=2)
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FP/FN EVOLUTION DECISION MODE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

FP_FN_DECISION_MODE_CONFIG = {
    "balanced": {
        "label": "Balanced",
        "annotation": "Operating point (Balanced)",
        "description": "Minimizes total errors (FP + FN)"
    },
    "precision": {
        "label": "Precision-focused",
        "annotation": "Operating point (Precision-focused)",
        "description": "Minimizes False Positives"
    },
    "recall": {
        "label": "Recall-focused",
        "annotation": "Operating point (Recall-focused)",
        "description": "Minimizes False Negatives"
    }
}


def _find_optimal_error_threshold(evo_df: pd.DataFrame, mode: str) -> dict:
    """
    Find the optimal operating point for errors based on decision mode.
    
    Args:
        evo_df: DataFrame with threshold, FP, FN, FPR, FNR columns
        mode: "balanced", "precision", or "recall"
        
    Returns:
        dict with idx, threshold, FP, FN, FPR, FNR, total_errors
    """
    if mode == "balanced":
        # Minimize total errors (FP + FN)
        idx = evo_df["Total Errors"].idxmin()
    elif mode == "precision":
        # Minimize False Positives
        idx = evo_df["FP"].idxmin()
    else:  # recall
        # Minimize False Negatives
        idx = evo_df["FN"].idxmin()
    
    row = evo_df.loc[idx]
    return {
        "idx": idx,
        "threshold": row["threshold"],
        "FP": row["FP"],
        "FN": row["FN"],
        "FPR": row["FPR"],
        "FNR": row["FNR"],
        "total_errors": row["Total Errors"],
        "error_rate": row["Error Rate"]
    }


def _compute_sensitivity_regions(evo_df: pd.DataFrame, mode: str) -> dict:
    """
    Compute regions of high sensitivity where small threshold changes cause large error changes.
    
    Args:
        evo_df: DataFrame with threshold and error data
        mode: Decision mode to determine which metric to analyze
        
    Returns:
        dict with x0, x1 bounds of sensitive region
    """
    # Compute derivatives (rate of change)
    fp_diff = np.abs(np.diff(evo_df["FP"].values))
    fn_diff = np.abs(np.diff(evo_df["FN"].values))
    thresholds = evo_df["threshold"].values
    
    if mode == "precision":
        # Focus on FP sensitivity
        sensitivity = fp_diff
    elif mode == "recall":
        # Focus on FN sensitivity
        sensitivity = fn_diff
    else:  # balanced
        # Combined sensitivity
        sensitivity = fp_diff + fn_diff
    
    # Find region with highest sensitivity (top 30%)
    threshold_cutoff = np.percentile(sensitivity, 70)
    high_sensitivity_mask = sensitivity >= threshold_cutoff
    
    # Get the threshold range
    if np.any(high_sensitivity_mask):
        indices = np.where(high_sensitivity_mask)[0]
        x0 = thresholds[indices[0]]
        x1 = thresholds[min(indices[-1] + 1, len(thresholds) - 1)]
    else:
        # Fallback to middle region
        x0, x1 = 0.3, 0.6
    
    return {"x0": x0, "x1": x1}


def create_fp_fn_evolution_enhanced(
    df: pd.DataFrame,
    selected_model: str,
    threshold: float = 0.5,
    decision_mode: str = "balanced",
    show_counts: bool = True,
    analysis_focus: str = "global"
) -> go.Figure:
    """
    Enhanced FP/FN Evolution chart with decision-oriented interactions.
    
    Features:
    - A) Link to global threshold slider with highlighted markers
    - B) Rich tooltips for any point on curves
    - C) Decision Mode integration from global controls
    - D) Highlight regions of high sensitivity
    - E) Toggle between counts and rates
    
    Args:
        df: DataFrame de avaliação
        selected_model: Modelo selecionado
        threshold: Current global threshold value
        decision_mode: "balanced", "precision", or "recall"
        show_counts: True for absolute counts, False for rates
        analysis_focus: "global", "sex", or "race" - demographic analysis mode
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    mode_config = FP_FN_DECISION_MODE_CONFIG.get(decision_mode, FP_FN_DECISION_MODE_CONFIG["balanced"])
    
    # Configure demographic groups based on analysis focus
    if analysis_focus == "sex":
        demographic_groups = [
            ("Male", df[(df["model"] == selected_model) & (df["sex"] == "Male")], "solid"),
            ("Female", df[(df["model"] == selected_model) & (df["sex"] == "Female")], "dash")
        ]
        title_demographic = " by Sex"
    elif analysis_focus == "race":
        demographic_groups = [
            ("White", df[(df["model"] == selected_model) & (df["race"] == "White")], "solid"),
            ("Non-White", df[(df["model"] == selected_model) & (df["race"] != "White")], "dash")
        ]
        title_demographic = " by Race"
    else:
        demographic_groups = [("All Data", df[df["model"] == selected_model], "solid")]
        title_demographic = ""
    
    model_name = MODEL_NAMES.get(selected_model, selected_model)
    model_color = MODEL_COLORS.get(selected_model, COLORS["primary"])
    
    # Compute error evolution at each threshold
    thresholds = np.linspace(0.1, 0.9, 33)
    
    # Store data and max values for all groups
    all_group_data = {}
    max_fp_value = 0
    max_fn_value = 0
    
    for group_name, group_df, line_dash in demographic_groups:
        if len(group_df) < 10:
            continue
            
        y_true = group_df["y_true"].values
        y_proba = group_df["y_proba"].values
        total_samples = len(group_df)
        
        evolution_data = []
        
        for t in thresholds:
            y_pred_at_t = (y_proba >= t).astype(int)
            cm = confusion_matrix(y_true, y_pred_at_t)
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
            
            # Compute rates
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            evolution_data.append({
                "threshold": t,
                "FP": fp,
                "FN": fn,
                "FPR": fpr,
                "FNR": fnr,
                "Total Errors": fp + fn,
                "Error Rate": (fp + fn) / total_samples if total_samples > 0 else 0,
                "FP_pct": fp / total_samples if total_samples > 0 else 0,
                "FN_pct": fn / total_samples if total_samples > 0 else 0
            })
        
        evo_df = pd.DataFrame(evolution_data)
        all_group_data[group_name] = {"df": evo_df, "line_dash": line_dash}
        
        # Track max values for layout
        if show_counts:
            max_fp_value = max(max_fp_value, evo_df["FP"].max())
            max_fn_value = max(max_fn_value, evo_df["FN"].max())
        else:
            max_fp_value = max(max_fp_value, evo_df["FPR"].max())
            max_fn_value = max(max_fn_value, evo_df["FNR"].max())
    
    # Determine label format
    if show_counts:
        y_label = "Count"
        y_format = ",.0f"
        fp_base_label = "False Positives"
        fn_base_label = "False Negatives"
        fp_key = "FP"
        fn_key = "FN"
    else:
        y_label = "Rate"
        y_format = ".1%"
        fp_base_label = "FP Rate"
        fn_base_label = "FN Rate"
        fp_key = "FPR"
        fn_key = "FNR"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # D) HIGHLIGHT REGIONS OF HIGH SENSITIVITY
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Use first group's data for sensitivity region
    first_group = list(all_group_data.keys())[0] if all_group_data else None
    if first_group:
        first_evo_df = all_group_data[first_group]["df"]
        sensitivity_region = _compute_sensitivity_regions(first_evo_df, decision_mode)
    else:
        sensitivity_region = {"x0": 0.3, "x1": 0.6}
    
    # Add sensitivity region highlight
    if decision_mode == "precision":
        region_color = COLORS["error"]
        region_label = "High FP Sensitivity"
    elif decision_mode == "recall":
        region_color = COLORS["warning"]
        region_label = "High FN Sensitivity"
    else:
        region_color = COLORS["secondary"]
        region_label = "High Error Sensitivity"
    
    max_y_value = max(max_fp_value, max_fn_value) * 1.1
    
    fig.add_shape(
        type="rect",
        x0=sensitivity_region["x0"],
        x1=sensitivity_region["x1"],
        y0=0,
        y1=max_y_value,
        fillcolor=region_color,
        opacity=0.1,
        layer="below",
        line_width=0
    )
    
    fig.add_annotation(
        x=(sensitivity_region["x0"] + sensitivity_region["x1"]) / 2,
        y=max_y_value * 0.95,
        text=region_label,
        showarrow=False,
        font=dict(size=9, color=region_color),
        opacity=0.8
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PLOT FP AND FN CURVES FOR EACH DEMOGRAPHIC GROUP
    # ═══════════════════════════════════════════════════════════════════════════
    
    for group_name, group_data in all_group_data.items():
        evo_df = group_data["df"]
        line_dash = group_data["line_dash"]
        
        # Build display name
        if analysis_focus == "global":
            fp_label = fp_base_label
            fn_label = fn_base_label
        else:
            fp_label = f"{fp_base_label} ({group_name})"
            fn_label = f"{fn_base_label} ({group_name})"
        
        # Build rich hover texts
        fp_hover_texts = []
        fn_hover_texts = []
        
        for _, row in evo_df.iterrows():
            group_info = f" — {group_name}" if analysis_focus != "global" else ""
            fp_hover = (
                f"<b>False Positives{group_info}</b><br>"
                f"Threshold: {row['threshold']:.3f}<br>"
                f"─────────────<br>"
                f"FP: {row['FP']:,.0f} ({row['FP_pct']:.1%})<br>"
                f"FN: {row['FN']:,.0f} ({row['FN_pct']:.1%})<br>"
                f"─────────────<br>"
                f"FPR: {row['FPR']:.2%}<br>"
                f"FNR: {row['FNR']:.2%}<br>"
                f"Total Errors: {row['Total Errors']:,.0f}"
            )
            fn_hover = (
                f"<b>False Negatives{group_info}</b><br>"
                f"Threshold: {row['threshold']:.3f}<br>"
                f"─────────────<br>"
                f"FP: {row['FP']:,.0f} ({row['FP_pct']:.1%})<br>"
                f"FN: {row['FN']:,.0f} ({row['FN_pct']:.1%})<br>"
                f"─────────────<br>"
                f"FPR: {row['FPR']:.2%}<br>"
                f"FNR: {row['FNR']:.2%}<br>"
                f"Total Errors: {row['Total Errors']:,.0f}"
            )
            fp_hover_texts.append(fp_hover)
            fn_hover_texts.append(fn_hover)
        
        # Add FP curve
        fig.add_trace(go.Scatter(
            x=evo_df["threshold"],
            y=evo_df[fp_key],
            mode="lines+markers",
            name=fp_label,
            line={"color": COLORS["error"], "width": 2.5, "dash": line_dash},
            marker={"size": 6, "symbol": "circle"},
            hovertemplate="%{customdata}<extra></extra>",
            customdata=fp_hover_texts,
            legendgroup=f"fp_{group_name}"
        ))
        
        # Add FN curve
        fig.add_trace(go.Scatter(
            x=evo_df["threshold"],
            y=evo_df[fn_key],
            mode="lines+markers",
            name=fn_label,
            line={"color": COLORS["warning"], "width": 2.5, "dash": line_dash},
            marker={"size": 6, "symbol": "diamond"},
            hovertemplate="%{customdata}<extra></extra>",
            customdata=fn_hover_texts,
            legendgroup=f"fn_{group_name}"
        ))
        
        # ═══════════════════════════════════════════════════════════════════════
        # A) CURRENT THRESHOLD MARKER (LINKED TO GLOBAL SLIDER)
        # ═══════════════════════════════════════════════════════════════════════
        
        # Find closest threshold in our computed range
        thresh_idx = np.argmin(np.abs(thresholds - threshold))
        current_row = evo_df.iloc[thresh_idx]
        
        # Current threshold values
        current_fp = current_row[fp_key]
        current_fn = current_row[fn_key]
        
        # Rich tooltip for current threshold markers
        group_info = f" — {group_name}" if analysis_focus != "global" else ""
        current_hover = (
            f"<b>Current Threshold{group_info}</b><br>"
            f"Threshold: {current_row['threshold']:.3f}<br>"
            f"─────────────<br>"
            f"FP: {current_row['FP']:,.0f} ({current_row['FP_pct']:.1%})<br>"
            f"FN: {current_row['FN']:,.0f} ({current_row['FN_pct']:.1%})<br>"
            f"─────────────<br>"
            f"FPR: {current_row['FPR']:.2%}<br>"
            f"FNR: {current_row['FNR']:.2%}<br>"
            f"Total Errors: {current_row['Total Errors']:,.0f}"
        )
        
        # Add markers at current threshold
        fig.add_trace(go.Scatter(
            x=[current_row["threshold"]],
            y=[current_fp],
            mode="markers",
            marker=dict(size=14, color=COLORS["error"], symbol="circle", line=dict(color="white", width=3)),
            hovertemplate=current_hover + "<extra></extra>",
            showlegend=False,
            legendgroup=f"fp_{group_name}"
        ))
        
        fig.add_trace(go.Scatter(
            x=[current_row["threshold"]],
            y=[current_fn],
            mode="markers",
            marker=dict(size=14, color=COLORS["warning"], symbol="diamond", line=dict(color="white", width=3)),
            hovertemplate=current_hover + "<extra></extra>",
            showlegend=False,
            legendgroup=f"fn_{group_name}"
        ))
        
        # ═══════════════════════════════════════════════════════════════════════
        # C) OPTIMAL OPERATING POINT BASED ON DECISION MODE
        # ═══════════════════════════════════════════════════════════════════════
        
        optimal = _find_optimal_error_threshold(evo_df, decision_mode)
        
        # Optimal values for markers
        optimal_fp = optimal[fp_key] if fp_key in optimal else optimal["FP"]
        optimal_fn = optimal[fn_key] if fn_key in optimal else optimal["FN"]
        
        # Optimal point tooltip
        optimal_hover = (
            f"<b>{mode_config['annotation']}{group_info}</b><br>"
            f"Threshold: {optimal['threshold']:.3f}<br>"
            f"─────────────<br>"
            f"FP: {optimal['FP']:,.0f}<br>"
            f"FN: {optimal['FN']:,.0f}<br>"
            f"─────────────<br>"
            f"FPR: {optimal['FPR']:.2%}<br>"
            f"FNR: {optimal['FNR']:.2%}<br>"
            f"Total Errors: {optimal['total_errors']:,.0f}"
        )
        
        # Add star markers at optimal point
        fig.add_trace(go.Scatter(
            x=[optimal["threshold"]],
            y=[optimal_fp],
            mode="markers",
            marker=dict(size=12, color=COLORS["error"], symbol="star", line=dict(color="white", width=2)),
            hovertemplate=optimal_hover + "<extra></extra>",
            showlegend=False,
            legendgroup=f"fp_{group_name}"
        ))
        
        fig.add_trace(go.Scatter(
            x=[optimal["threshold"]],
            y=[optimal_fn],
            mode="markers",
            marker=dict(size=12, color=COLORS["warning"], symbol="star", line=dict(color="white", width=2)),
            hovertemplate=optimal_hover + "<extra></extra>",
            showlegend=False,
            legendgroup=f"fn_{group_name}"
        ))
    
    # Add vertical lines for current threshold and optimal (only once)
    fig.add_vline(
        x=threshold,
        line_dash="solid",
        line_color=COLORS["primary"],
        line_width=2.5,
        opacity=0.9
    )
    
    # Get optimal from first group for vertical line
    if first_group:
        first_evo_df = all_group_data[first_group]["df"]
        optimal = _find_optimal_error_threshold(first_evo_df, decision_mode)
        fig.add_vline(
            x=optimal["threshold"],
            line_dash="dash",
            line_color=COLORS["success"],
            line_width=2,
            opacity=0.8
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEGEND ENTRIES FOR SYMBOLS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Add legend entry for current threshold marker
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=14, color=COLORS["primary"], symbol="circle", line=dict(color="white", width=3)),
        name=f"Current Threshold ({threshold:.2f})",
        showlegend=True,
        hoverinfo="skip"
    ))
    
    # Add legend entry for optimal point
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=12, color=COLORS["success"], symbol="star", line=dict(color="white", width=2)),
        name=f"★ Optimal ({mode_config['label']})",
        showlegend=True,
        hoverinfo="skip"
    ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LAYOUT
    # ═══════════════════════════════════════════════════════════════════════════
    
    mode_suffix = f"{'Counts' if show_counts else 'Rates'} • Mode: {mode_config['label']}"
    
    fig.update_layout(
        title=dict(
            text=f"Errors vs Threshold{title_demographic} - {model_name} • {mode_suffix}",
            x=0.01,
            xanchor="left",
            font=dict(size=14)
        ),
        xaxis_title="Decision Threshold",
        yaxis_title=y_label,
        xaxis_range=[0.08, 0.92],
        yaxis_tickformat=y_format if not show_counts else ",",
        height=420,
        legend=CHART_LEGEND_CONFIG,
        margin=dict(t=80, b=100),
        transition=dict(duration=400, easing="cubic-in-out")
    )
    
    return fig


def create_threshold_impact_bars(df: pd.DataFrame, threshold: float = 0.5) -> go.Figure:
    """
    Impacto do threshold nas predições.
    Legacy function for backward compatibility.
    
    Args:
        df: DataFrame de avaliação
        threshold: Limiar de decisão
        
    Returns:
        Figura Plotly
    """
    return create_prediction_distribution_enhanced(df, threshold)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION DISTRIBUTION DECISION MODE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PREDICTION_DIST_MODE_CONFIG = {
    "balanced": {
        "label": "Balanced",
        "annotation": "Balanced operating point",
        "description": "Standard threshold optimized for F1 balance"
    },
    "precision": {
        "label": "Precision-focused",
        "annotation": "Precision-focused: fewer positives, higher confidence",
        "description": "Higher threshold → fewer but more confident positive predictions"
    },
    "recall": {
        "label": "Recall-focused",
        "annotation": "Recall-focused: more positives, higher coverage",
        "description": "Lower threshold → more positive predictions for coverage"
    }
}


def _compute_prediction_distribution(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Compute prediction distribution for all models at a given threshold.
    
    Args:
        df: DataFrame with model predictions
        threshold: Decision threshold
        
    Returns:
        DataFrame with model, predicted positive/negative counts and percentages
    """
    results = []
    total_samples = len(df[df["model"] == df["model"].unique()[0]])
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        temp_df = recompute_with_threshold(model_df, threshold)
        
        n_positive = (temp_df["y_pred"] == 1).sum()
        n_negative = (temp_df["y_pred"] == 0).sum()
        
        results.append({
            "model": model,
            "model_name": MODEL_NAMES.get(model, model),
            "positive": n_positive,
            "negative": n_negative,
            "positive_pct": n_positive / total_samples,
            "negative_pct": n_negative / total_samples,
            "total": total_samples
        })
    
    return pd.DataFrame(results)


def _detect_large_change(prev_dist: pd.DataFrame, curr_dist: pd.DataFrame, 
                          change_threshold: float = 0.05) -> dict:
    """
    Detect if there's a large change in prediction distribution between thresholds.
    
    Args:
        prev_dist: Previous distribution dataframe
        curr_dist: Current distribution dataframe
        change_threshold: Percentage change to consider "large"
        
    Returns:
        dict with change detection info per model
    """
    changes = {}
    
    for model in curr_dist["model"].unique():
        curr_row = curr_dist[curr_dist["model"] == model].iloc[0]
        prev_row = prev_dist[prev_dist["model"] == model].iloc[0] if len(prev_dist) > 0 else None
        
        if prev_row is not None:
            delta_positive = abs(curr_row["positive_pct"] - prev_row["positive_pct"])
            delta_negative = abs(curr_row["negative_pct"] - prev_row["negative_pct"])
            
            is_large_change = delta_positive >= change_threshold or delta_negative >= change_threshold
            
            changes[model] = {
                "is_large": is_large_change,
                "delta_positive": curr_row["positive"] - prev_row["positive"],
                "delta_negative": curr_row["negative"] - prev_row["negative"],
                "delta_positive_pct": delta_positive,
                "delta_negative_pct": delta_negative
            }
        else:
            changes[model] = {"is_large": False, "delta_positive": 0, "delta_negative": 0}
    
    return changes


def create_prediction_distribution_enhanced(
    df: pd.DataFrame,
    threshold: float = 0.5,
    decision_mode: str = "balanced",
    show_delta_view: bool = False,
    previous_threshold: float = None
) -> go.Figure:
    """
    Enhanced Prediction Distribution chart with decision-oriented interactions.
    
    Features:
    - A) Full link to global threshold slider with real-time updates
    - B) Rich tooltips on bars with counts, percentages, and deltas
    - C) Decision Mode integration from global controls
    - D) Highlight large changes in distribution when threshold shifts
    - F) Delta view between models (RF - LR difference)
    
    Args:
        df: DataFrame de avaliação
        threshold: Current global threshold value
        decision_mode: "balanced", "precision", or "recall"
        show_delta_view: True to show difference between models, False for absolute
        previous_threshold: Previous threshold for change detection (optional)
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    mode_config = PREDICTION_DIST_MODE_CONFIG.get(decision_mode, PREDICTION_DIST_MODE_CONFIG["balanced"])
    
    # Compute current distribution
    curr_dist = _compute_prediction_distribution(df, threshold)
    total_samples = curr_dist.iloc[0]["total"]
    
    # Compute previous distribution for change detection (D)
    if previous_threshold is not None and previous_threshold != threshold:
        prev_dist = _compute_prediction_distribution(df, previous_threshold)
        changes = _detect_large_change(prev_dist, curr_dist)
    else:
        # Use slightly different threshold for sensitivity indication
        prev_threshold_check = max(0.1, threshold - 0.05)
        prev_dist = _compute_prediction_distribution(df, prev_threshold_check)
        changes = _detect_large_change(prev_dist, curr_dist, change_threshold=0.03)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # F) DELTA VIEW BETWEEN MODELS
    # ═══════════════════════════════════════════════════════════════════════════
    
    if show_delta_view and len(curr_dist) == 2:
        # Compute delta between models (second - first, typically RF - LR)
        model1 = curr_dist.iloc[0]
        model2 = curr_dist.iloc[1]
        
        delta_positive = model2["positive"] - model1["positive"]
        delta_negative = model2["negative"] - model1["negative"]
        delta_positive_pct = model2["positive_pct"] - model1["positive_pct"]
        delta_negative_pct = model2["negative_pct"] - model1["negative_pct"]
        
        # Build hover texts for delta view
        positive_hover = (
            f"<b>Δ Predicted Positive (>50K)</b><br>"
            f"Threshold: {threshold:.3f}<br>"
            f"─────────────<br>"
            f"{model2['model_name']}: {model2['positive']:,} ({model2['positive_pct']:.1%})<br>"
            f"{model1['model_name']}: {model1['positive']:,} ({model1['positive_pct']:.1%})<br>"
            f"─────────────<br>"
            f"<b>Difference: {'+' if delta_positive > 0 else ''}{delta_positive:,}</b><br>"
            f"({'+' if delta_positive_pct > 0 else ''}{delta_positive_pct:.1%} of dataset)"
        )
        
        negative_hover = (
            f"<b>Δ Predicted Negative (≤50K)</b><br>"
            f"Threshold: {threshold:.3f}<br>"
            f"─────────────<br>"
            f"{model2['model_name']}: {model2['negative']:,} ({model2['negative_pct']:.1%})<br>"
            f"{model1['model_name']}: {model1['negative']:,} ({model1['negative_pct']:.1%})<br>"
            f"─────────────<br>"
            f"<b>Difference: {'+' if delta_negative > 0 else ''}{delta_negative:,}</b><br>"
            f"({'+' if delta_negative_pct > 0 else ''}{delta_negative_pct:.1%} of dataset)"
        )
        
        # Determine bar colors based on direction
        positive_color = COLORS["success"] if delta_positive >= 0 else COLORS["error"]
        negative_color = COLORS["text_muted"] if delta_negative <= 0 else COLORS["warning"]
        
        # Create diverging bar chart
        categories = ["Predicted Positive", "Predicted Negative"]
        deltas = [delta_positive, delta_negative]
        hover_texts = [positive_hover, negative_hover]
        bar_colors = [positive_color, negative_color]
        
        for i, (cat, delta, hover, color) in enumerate(zip(categories, deltas, hover_texts, bar_colors)):
            fig.add_trace(go.Bar(
                name=cat,
                x=[cat],
                y=[delta],
                marker_color=color,
                text=[f"{'+' if delta > 0 else ''}{delta:,}"],
                textposition="outside" if abs(delta) < total_samples * 0.1 else "inside",
                textfont={"size": 12, "color": "white" if abs(delta) >= total_samples * 0.1 else color},
                hovertemplate=hover + "<extra></extra>",
                showlegend=False
            ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color=COLORS["text_muted"], line_width=1.5)
        
        # Add interpretation annotation
        if delta_positive > 0:
            interp_text = f"{model2['model_name']} predicts {abs(delta_positive):,} more positives"
        else:
            interp_text = f"{model1['model_name']} predicts {abs(delta_positive):,} more positives"
        
        fig.add_annotation(
            x=0.5, y=1.12,
            xref="paper", yref="paper",
            text=interp_text,
            showarrow=False,
            font=dict(size=10, color=COLORS["text_secondary"]),
            bgcolor="rgba(128, 128, 128, 0.3)",
            borderpad=4
        )
        
        # Layout for delta view
        fig.update_layout(
            title=dict(
                text=f"Model Difference: {model2['model_name']} − {model1['model_name']} (Threshold = {threshold:.2f})",
                font=dict(size=14)
            ),
            xaxis_title="",
            yaxis_title="Difference in Predictions",
            yaxis_zeroline=True,
            height=400,
            showlegend=False,
            transition=dict(duration=400, easing="cubic-in-out")
        )
        
    else:
        # ═══════════════════════════════════════════════════════════════════════
        # ABSOLUTE DISTRIBUTION VIEW (STACKED BARS)
        # ═══════════════════════════════════════════════════════════════════════
        
        # D) Check for large changes and prepare highlighting
        any_large_change = any(changes[m]["is_large"] for m in changes)
        
        # Build rich hover texts (B)
        positive_hovers = []
        negative_hovers = []
        
        for _, row in curr_dist.iterrows():
            model = row["model"]
            change_info = changes.get(model, {"delta_positive": 0, "delta_negative": 0})
            
            # Positive bar hover
            pos_hover = (
                f"<b>{row['model_name']}</b><br>"
                f"Prediction: Positive (>50K)<br>"
                f"─────────────<br>"
                f"Count: {row['positive']:,}<br>"
                f"Share: {row['positive_pct']:.1%} of dataset<br>"
            )
            if change_info["delta_positive"] != 0:
                pos_hover += (
                    f"─────────────<br>"
                    f"Δ from prev: {'+' if change_info['delta_positive'] > 0 else ''}{change_info['delta_positive']:,}"
                )
            positive_hovers.append(pos_hover)
            
            # Negative bar hover
            neg_hover = (
                f"<b>{row['model_name']}</b><br>"
                f"Prediction: Negative (≤50K)<br>"
                f"─────────────<br>"
                f"Count: {row['negative']:,}<br>"
                f"Share: {row['negative_pct']:.1%} of dataset<br>"
            )
            if change_info["delta_negative"] != 0:
                neg_hover += (
                    f"─────────────<br>"
                    f"Δ from prev: {'+' if change_info['delta_negative'] > 0 else ''}{change_info['delta_negative']:,}"
                )
            negative_hovers.append(neg_hover)
        
        # D) Determine bar styling based on sensitivity
        positive_line_width = []
        negative_line_width = []
        positive_line_color = []
        negative_line_color = []
        
        for _, row in curr_dist.iterrows():
            model = row["model"]
            if changes.get(model, {}).get("is_large", False):
                positive_line_width.append(3)
                negative_line_width.append(3)
                positive_line_color.append(COLORS["warning"])
                negative_line_color.append(COLORS["warning"])
            else:
                positive_line_width.append(0)
                negative_line_width.append(0)
                positive_line_color.append("rgba(0,0,0,0)")
                negative_line_color.append("rgba(0,0,0,0)")
        
        # Create stacked bar chart
        fig.add_trace(go.Bar(
            name="Predicted Positive (>50K)",
            x=curr_dist["model_name"],
            y=curr_dist["positive"],
            marker=dict(
                color=COLORS["success"],
                line=dict(width=positive_line_width, color=positive_line_color)
            ),
            text=[f"{p:,}" for p in curr_dist["positive"]],
            textposition="inside",
            textfont={"color": "white", "size": 11},
            hovertemplate="%{customdata}<extra></extra>",
            customdata=positive_hovers
        ))
        
        fig.add_trace(go.Bar(
            name="Predicted Negative (≤50K)",
            x=curr_dist["model_name"],
            y=curr_dist["negative"],
            marker=dict(
                color=COLORS["text_muted"],
                line=dict(width=negative_line_width, color=negative_line_color)
            ),
            text=[f"{n:,}" for n in curr_dist["negative"]],
            textposition="inside",
            textfont={"color": "white", "size": 11},
            hovertemplate="%{customdata}<extra></extra>",
            customdata=negative_hovers
        ))
        
        # ═══════════════════════════════════════════════════════════════════════
        # C) DECISION MODE ANNOTATION
        # ═══════════════════════════════════════════════════════════════════════
        
        # Add mode annotation
        fig.add_annotation(
            x=0.5, y=1.08,
            xref="paper", yref="paper",
            text=f"<b>{mode_config['annotation']}</b>",
            showarrow=False,
            font=dict(size=10, color=COLORS["primary"]),
            bgcolor="rgba(99, 102, 241, 0.1)",
            borderpad=6,
            bordercolor="rgba(99, 102, 241, 0.2)",
            borderwidth=1
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # D) SENSITIVITY ANNOTATION (if large change detected)
        # ═══════════════════════════════════════════════════════════════════════
        
        if any_large_change:
            # Find which model has the large change
            large_change_models = [m for m in changes if changes[m]["is_large"]]
            
            fig.add_annotation(
                x=0.5, y=-0.18,
                xref="paper", yref="paper",
                text="⚠ Sensitive threshold region: small changes cause large shifts in predictions",
                showarrow=False,
                font=dict(size=9, color=COLORS["warning"]),
                bgcolor="rgba(245, 158, 11, 0.1)",
                borderpad=4
            )
        
        # Layout for absolute view
        fig.update_layout(
            title=dict(
                text=f"Prediction Distribution (Threshold = {threshold:.2f})",
                font=dict(size=14)
            ),
            barmode="stack",
            xaxis_title="",
            yaxis_title="Number of Samples",
            height=400,
            legend=CHART_LEGEND_CONFIG,
            margin=dict(t=80, b=100),
            transition=dict(duration=400, easing="cubic-in-out")
        )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL COORDINATES PLOT - OPERATING POINTS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

PCP_DECISION_MODE_CONFIG = {
    "balanced": {
        "label": "Balanced / Max F1",
        "optimize": "f1",
        "constraint": None,
        "description": "Operating point with highest F1 score"
    },
    "precision": {
        "label": "Precision-focused",
        "optimize": "precision",
        "constraint": {"metric": "recall", "min": 0.5},
        "description": "Highest precision with recall ≥ 50%"
    },
    "recall": {
        "label": "Recall-focused",
        "optimize": "recall",
        "constraint": {"metric": "precision", "min": 0.5},
        "description": "Highest recall with precision ≥ 50%"
    }
}


def build_operating_points_df(
    df: pd.DataFrame,
    models_selected: list = None,
    thresholds: np.ndarray = None,
    subgroup_mode: str = "Global",
    subgroup_attr: str = "sex",
    subgroup_pairs: dict = None
) -> pd.DataFrame:
    """
    Build a dataframe of operating points for parallel coordinates visualization.
    
    Each row represents one operating point: (model, threshold, [subgroup]) with metrics.
    
    Args:
        df: Evaluation DataFrame with columns: model, y_true, y_proba, sex, race
        models_selected: List of model names to include (None = all)
        thresholds: Array of thresholds to evaluate (None = default range)
        subgroup_mode: "Global", "Sex", or "Race"
        subgroup_attr: Attribute for subgroup analysis ("sex" or "race")
        subgroup_pairs: Dict mapping attribute to (group_a, group_b) tuple
        
    Returns:
        DataFrame with columns: model, threshold, precision, recall, f1, fpr, fnr,
                               calibration_error, fairness_gap, tp, fp, tn, fn, subgroup
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    
    if subgroup_pairs is None:
        subgroup_pairs = {
            "sex": ("Male", "Female"),
            "race": ("White", "Non-White")
        }
    
    if models_selected is None:
        models_selected = df["model"].unique().tolist()
    
    operating_points = []
    
    for model in models_selected:
        model_df = df[df["model"] == model]
        model_name = MODEL_NAMES.get(model, model)
        
        y_true_all = model_df["y_true"].values
        y_proba_all = model_df["y_proba"].values
        
        for t in thresholds:
            if subgroup_mode == "Global":
                # Compute global metrics
                y_pred = (y_proba_all >= t).astype(int)
                cm = confusion_matrix(y_true_all, y_pred)
                
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
                
                # Compute metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                
                # Calibration error: use samples near threshold
                # Simple approach: mean predicted prob near threshold vs actual positive rate
                mask_near = (y_proba_all >= t - 0.05) & (y_proba_all <= t + 0.05)
                if mask_near.sum() > 10:
                    actual_rate = y_true_all[mask_near].mean()
                    pred_rate = y_proba_all[mask_near].mean()
                    calibration_error = abs(actual_rate - pred_rate)
                else:
                    # Fallback: global calibration at this threshold
                    calibration_error = abs(y_true_all.mean() - t)
                
                operating_points.append({
                    "model": model,
                    "model_name": model_name,
                    "threshold": t,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "fpr": fpr,
                    "fnr": fnr,
                    "calibration_error": calibration_error,
                    "fairness_gap": 0.0,  # No fairness gap for global
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "subgroup": "Global",
                    "metric_group_a": None,
                    "metric_group_b": None
                })
            
            else:
                # Subgroup mode: compute per-group metrics and fairness gap
                attr = subgroup_attr.lower()
                if attr not in subgroup_pairs:
                    continue
                    
                group_a_name, group_b_name = subgroup_pairs[attr]
                
                # Handle "Non-White" special case
                if group_b_name == "Non-White":
                    group_a_mask = model_df[attr] == group_a_name
                    group_b_mask = model_df[attr] != group_a_name
                else:
                    group_a_mask = model_df[attr] == group_a_name
                    group_b_mask = model_df[attr] == group_b_name
                
                # Compute metrics for each group
                group_metrics = {}
                for group_name, mask in [(group_a_name, group_a_mask), (group_b_name, group_b_mask)]:
                    if mask.sum() == 0:
                        continue
                    
                    y_true_g = model_df.loc[mask, "y_true"].values
                    y_proba_g = model_df.loc[mask, "y_proba"].values
                    y_pred_g = (y_proba_g >= t).astype(int)
                    
                    cm_g = confusion_matrix(y_true_g, y_pred_g)
                    if cm_g.shape == (2, 2):
                        tn_g, fp_g, fn_g, tp_g = cm_g.ravel()
                    else:
                        tn_g, fp_g, fn_g, tp_g = 0, 0, 0, 0
                    
                    prec_g = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else 0
                    rec_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
                    f1_g = 2 * prec_g * rec_g / (prec_g + rec_g) if (prec_g + rec_g) > 0 else 0
                    fpr_g = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0
                    fnr_g = fn_g / (fn_g + tp_g) if (fn_g + tp_g) > 0 else 0
                    
                    group_metrics[group_name] = {
                        "precision": prec_g,
                        "recall": rec_g,
                        "f1": f1_g,
                        "fpr": fpr_g,
                        "fnr": fnr_g,
                        "tp": tp_g, "fp": fp_g, "tn": tn_g, "fn": fn_g
                    }
                
                if len(group_metrics) < 2:
                    continue
                
                # Compute fairness gap (using FNR difference)
                fnr_a = group_metrics[group_a_name]["fnr"]
                fnr_b = group_metrics[group_b_name]["fnr"]
                fairness_gap = abs(fnr_a - fnr_b)
                
                # Add operating point for each group
                for group_name in [group_a_name, group_b_name]:
                    gm = group_metrics[group_name]
                    
                    # Calibration error per group
                    mask_group = group_a_mask if group_name == group_a_name else group_b_mask
                    y_true_g = model_df.loc[mask_group, "y_true"].values
                    y_proba_g = model_df.loc[mask_group, "y_proba"].values
                    
                    mask_near = (y_proba_g >= t - 0.05) & (y_proba_g <= t + 0.05)
                    if mask_near.sum() > 5:
                        actual_rate = y_true_g[mask_near].mean()
                        pred_rate = y_proba_g[mask_near].mean()
                        calibration_error = abs(actual_rate - pred_rate)
                    else:
                        calibration_error = abs(y_true_g.mean() - t)
                    
                    operating_points.append({
                        "model": model,
                        "model_name": model_name,
                        "threshold": t,
                        "precision": gm["precision"],
                        "recall": gm["recall"],
                        "f1": gm["f1"],
                        "fpr": gm["fpr"],
                        "fnr": gm["fnr"],
                        "calibration_error": calibration_error,
                        "fairness_gap": fairness_gap,
                        "tp": gm["tp"],
                        "fp": gm["fp"],
                        "tn": gm["tn"],
                        "fn": gm["fn"],
                        "subgroup": group_name,
                        "metric_group_a": fnr_a,
                        "metric_group_b": fnr_b
                    })
    
    return pd.DataFrame(operating_points)


def _find_recommended_operating_point(ops_df: pd.DataFrame, decision_mode: str) -> dict:
    """
    Find the recommended operating point based on decision mode.
    
    Args:
        ops_df: Operating points dataframe
        decision_mode: "balanced", "precision", or "recall"
        
    Returns:
        dict with the recommended row data or None
    """
    if len(ops_df) == 0:
        return None
    
    mode_config = PCP_DECISION_MODE_CONFIG.get(decision_mode, PCP_DECISION_MODE_CONFIG["balanced"])
    optimize_metric = mode_config["optimize"]
    constraint = mode_config["constraint"]
    
    # Apply constraint if exists
    filtered_df = ops_df.copy()
    if constraint is not None:
        mask = filtered_df[constraint["metric"]] >= constraint["min"]
        if mask.any():
            filtered_df = filtered_df[mask]
        # If no rows pass constraint, use full df
    
    if len(filtered_df) == 0:
        filtered_df = ops_df
    
    # Find best by optimize metric
    best_idx = filtered_df[optimize_metric].idxmax()
    best_row = filtered_df.loc[best_idx].to_dict()
    best_row["_index"] = best_idx
    
    return best_row


def create_parallel_coordinates_operating_points(
    ops_df: pd.DataFrame,
    current_threshold: float = 0.5,
    decision_mode: str = "balanced",
    color_by: str = "model",
    selected_indices: list = None,
    highlight_threshold_range: float = 0.05
) -> go.Figure:
    """
    Create a Parallel Coordinates Plot for operating points analysis.
    
    Args:
        ops_df: Operating points dataframe from build_operating_points_df
        current_threshold: Current global threshold for highlighting
        decision_mode: "balanced", "precision", or "recall"
        color_by: "model" or "subgroup"
        selected_indices: List of indices to highlight (from brushing)
        highlight_threshold_range: Range around current threshold to highlight
        
    Returns:
        Plotly Figure with parallel coordinates
    """
    if len(ops_df) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No operating points data available",
            showarrow=False,
            font=dict(size=14, color=COLORS["text_muted"])
        )
        fig.update_layout(height=400)
        return fig
    
    mode_config = PCP_DECISION_MODE_CONFIG.get(decision_mode, PCP_DECISION_MODE_CONFIG["balanced"])
    
    # Find recommended operating point
    recommended = _find_recommended_operating_point(ops_df, decision_mode)
    
    # Create color encoding
    if color_by == "model":
        # Map models to numeric values for color scale
        unique_models = ops_df["model"].unique()
        model_to_num = {m: i for i, m in enumerate(unique_models)}
        color_values = ops_df["model"].map(model_to_num).values
        
        # Create colorscale
        if len(unique_models) == 2:
            colorscale = [
                [0, MODEL_COLORS.get(unique_models[0], COLORS["primary"])],
                [1, MODEL_COLORS.get(unique_models[1], COLORS["accent"])]
            ]
        else:
            colorscale = "Viridis"
        
        colorbar_title = "Model"
        colorbar_tickvals = list(range(len(unique_models)))
        colorbar_ticktext = [MODEL_NAMES.get(m, m) for m in unique_models]
    else:
        # Color by subgroup
        unique_subgroups = ops_df["subgroup"].unique()
        subgroup_to_num = {s: i for i, s in enumerate(unique_subgroups)}
        color_values = ops_df["subgroup"].map(subgroup_to_num).values
        
        colorscale = [
            [0, COLORS["primary"]],
            [0.5, COLORS["secondary"]],
            [1, COLORS["accent"]]
        ]
        colorbar_title = "Subgroup"
        colorbar_tickvals = list(range(len(unique_subgroups)))
        colorbar_ticktext = list(unique_subgroups)
    
    # Highlight lines near current threshold
    near_threshold_mask = (
        (ops_df["threshold"] >= current_threshold - highlight_threshold_range) &
        (ops_df["threshold"] <= current_threshold + highlight_threshold_range)
    )
    
    # Build dimensions for parallel coordinates
    dimensions = [
        dict(
            label="Threshold",
            values=ops_df["threshold"],
            range=[0.1, 0.9],
            tickvals=[0.1, 0.3, 0.5, 0.7, 0.9],
            ticktext=["0.1", "0.3", "0.5", "0.7", "0.9"]
        ),
        dict(
            label="Precision",
            values=ops_df["precision"],
            range=[0, 1],
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0%", "25%", "50%", "75%", "100%"]
        ),
        dict(
            label="Recall",
            values=ops_df["recall"],
            range=[0, 1],
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0%", "25%", "50%", "75%", "100%"]
        ),
        dict(
            label="F1 Score",
            values=ops_df["f1"],
            range=[0, 1],
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0%", "25%", "50%", "75%", "100%"]
        ),
        dict(
            label="FPR",
            values=ops_df["fpr"],
            range=[0, max(0.5, ops_df["fpr"].max() * 1.1)],
            tickformat=".0%"
        ),
        dict(
            label="Calib. Error",
            values=ops_df["calibration_error"],
            range=[0, max(0.3, ops_df["calibration_error"].max() * 1.1)],
            tickformat=".1%"
        ),
        dict(
            label="Fairness Gap",
            values=ops_df["fairness_gap"],
            range=[0, max(0.3, ops_df["fairness_gap"].max() * 1.1) if ops_df["fairness_gap"].max() > 0 else 0.1],
            tickformat=".1%"
        )
    ]
    
    # Create the parallel coordinates trace
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=color_values,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=colorbar_title,
                    tickvals=colorbar_tickvals,
                    ticktext=colorbar_ticktext,
                    len=0.5,
                    y=0.75
                )
            ),
            dimensions=dimensions,
            labelangle=-30,
            labelfont=dict(size=10),
            tickfont=dict(size=9)
        )
    )
    
    
    
    # Add decision mode annotation
    fig.add_annotation(
        x=0.0, y=-0.15,
        xref="paper", yref="paper",
        text=f"Decision Mode: <b>{mode_config['label']}</b> — {mode_config['description']}",
        showarrow=False,
        font=dict(size=9, color=COLORS["text_secondary"]),
        xanchor="left"
    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text="Parallel Coordinates: Operating Points Analysis",
            font=dict(size=14),
            x=0.0,
            y=0.97,
            xanchor="left"
        ),
        height=450,
        margin=dict(l=80, r=80, t=100, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig


def create_selected_operating_points_table(
    ops_df: pd.DataFrame,
    selected_indices: list = None,
    current_threshold: float = 0.5,
    decision_mode: str = "balanced",
    max_rows: int = 5
) -> pd.DataFrame:
    """
    Create a summary table of selected or top operating points.
    
    Args:
        ops_df: Operating points dataframe
        selected_indices: Indices selected via brushing (None = show top by F1)
        current_threshold: Current threshold for proximity filtering
        decision_mode: Decision mode for sorting
        max_rows: Maximum rows to return
        
    Returns:
        DataFrame formatted for display
    """
    if len(ops_df) == 0:
        return pd.DataFrame()
    
    mode_config = PCP_DECISION_MODE_CONFIG.get(decision_mode, PCP_DECISION_MODE_CONFIG["balanced"])
    optimize_metric = mode_config["optimize"]
    
    if selected_indices is not None and len(selected_indices) > 0:
        # Filter to selected indices
        display_df = ops_df.loc[ops_df.index.isin(selected_indices)].copy()
    else:
        # Show operating points near current threshold, sorted by optimize metric
        near_threshold = ops_df[
            (ops_df["threshold"] >= current_threshold - 0.1) &
            (ops_df["threshold"] <= current_threshold + 0.1)
        ].copy()
        
        if len(near_threshold) > 0:
            display_df = near_threshold.nlargest(max_rows, optimize_metric)
        else:
            display_df = ops_df.nlargest(max_rows, optimize_metric)
    
    # Format for display
    display_df = display_df.head(max_rows)
    
    result = display_df[[
        "model_name", "threshold", "precision", "recall", "f1", 
        "fpr", "calibration_error", "fairness_gap"
    ]].copy()
    
    result.columns = [
        "Model", "Threshold", "Precision", "Recall", "F1",
        "FPR", "Calib. Error", "Fairness Gap"
    ]
    
    # Format percentages
    for col in ["Precision", "Recall", "F1", "FPR", "Calib. Error", "Fairness Gap"]:
        result[col] = result[col].apply(lambda x: f"{x:.1%}")
    
    result["Threshold"] = result["Threshold"].apply(lambda x: f"{x:.2f}")
    
    return result.reset_index(drop=True)


def get_operating_point_details(ops_df: pd.DataFrame, index: int) -> dict:
    """
    Get detailed information for a specific operating point.
    
    Args:
        ops_df: Operating points dataframe
        index: Row index
        
    Returns:
        dict with all details formatted for tooltip/panel display
    """
    if index not in ops_df.index:
        return None
    
    row = ops_df.loc[index]
    
    details = {
        "Model": row["model_name"],
        "Threshold": f"{row['threshold']:.3f}",
        "Subgroup": row["subgroup"],
        "Precision": f"{row['precision']:.2%}",
        "Recall": f"{row['recall']:.2%}",
        "F1 Score": f"{row['f1']:.2%}",
        "FPR": f"{row['fpr']:.2%}",
        "FNR": f"{row['fnr']:.2%}",
        "Calibration Error": f"{row['calibration_error']:.2%}",
        "Fairness Gap (ΔFNR)": f"{row['fairness_gap']:.2%}",
        "TP": f"{row['tp']:,}",
        "FP": f"{row['fp']:,}",
        "TN": f"{row['tn']:,}",
        "FN": f"{row['fn']:,}"
    }
    
    return details
