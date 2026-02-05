"""
Gráficos para a View 2: Trade-offs (Precision vs Recall).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

from src.config.settings import COLORS, MODEL_COLORS, MODEL_NAMES
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
    show_area: bool = False
) -> tuple:
    """
    Enhanced Precision-Recall curve with decision-oriented interactions.
    
    Args:
        df: DataFrame de avaliação
        threshold: Current global threshold
        decision_mode: "balanced", "precision", or "recall"
        show_area: Whether to fill area under curve
        
    Returns:
        Tuple of (figure, delta_ap_text)
    """
    fig = go.Figure()
    
    mode_config = PR_DECISION_MODE_CONFIG.get(decision_mode, PR_DECISION_MODE_CONFIG["balanced"])
    
    # Store AP values for delta calculation
    ap_values = {}
    
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
    # MAIN PR CURVES FOR EACH MODEL
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Store first model's optimal point for annotation (avoid recalculating later)
    first_model_optimal = None
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        y_true = model_df["y_true"].values
        y_proba = model_df["y_proba"].values
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ap_values[model] = ap
        
        model_color = MODEL_COLORS.get(model, COLORS["primary"])
        model_name = MODEL_NAMES.get(model, model)
        
        # ═══════════════════════════════════════════════════════════════════════
        # G) OPTIONAL: FILLED AREA UNDER CURVE
        # ═══════════════════════════════════════════════════════════════════════
        
        if show_area:
            fig.add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode="none",
                fill="tozeroy",
                fillcolor=f"rgba{tuple(list(int(model_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}",
                name=f"{model_name} Area",
                legendgroup=model,
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
        for i in range(len(precision)):
            t = custom_thresholds[i] if i < len(custom_thresholds) else custom_thresholds[-1]
            
            hover_text = (
                f"<b>{model_name}</b><br>"
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
            name=f"{model_name} (AP = {ap:.3f})",
            line={"color": model_color, "width": 2.5},
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts,
            opacity=curve_opacity,
            legendgroup=model,
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
            f"<b>Current Threshold</b><br>"
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
            name=f"{model_name} @ t={actual_threshold:.2f}",
            marker=dict(
                size=14,
                color=model_color,
                symbol="circle",
                line=dict(color="white", width=3)
            ),
            hovertemplate=marker_hover + "<extra></extra>",
            legendgroup=model,
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
            name=f"{model_name} Optimal",
            marker=dict(
                size=12,
                color=model_color,
                symbol="star",
                line=dict(color="white", width=2)
            ),
            hovertemplate=(
                f"<b>Optimal ({mode_config['label']})</b><br>"
                f"Threshold: {optimal['threshold']:.3f}<br>"
                f"Precision: {optimal['precision']:.3f}<br>"
                f"Recall: {optimal['recall']:.3f}<br>"
                f"F1: {optimal['f1']:.3f}"
                "<extra></extra>"
            ),
            legendgroup=model,
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
    
    # Build title with mode indication
    title_suffix = f" • Mode: {mode_config['label']}"
    
    fig.update_layout(
        title=dict(
            text=f"Precision-Recall Curves{title_suffix}",
            font=dict(size=14)
        ),
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1.02],
        yaxis_range=[0, 1.05],
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
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
    overlay_models: bool = False
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
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    mode_config = THRESHOLD_DECISION_MODE_CONFIG.get(decision_mode, THRESHOLD_DECISION_MODE_CONFIG["balanced"])
    
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
    
    # Store data for each model
    all_model_data = {}
    
    for model in models_to_plot:
        model_df = df[df["model"] == model]
        y_true = model_df["y_true"].values
        y_proba = model_df["y_proba"].values
        
        # Compute metrics at each threshold
        metrics_by_thresh = []
        for t in thresholds:
            temp_df = recompute_with_threshold(model_df, t)
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
        all_model_data[model] = thresh_df
        
        model_name = MODEL_NAMES.get(model, model)
        model_color = MODEL_COLORS.get(model, COLORS["primary"])
        
        # Use different line styles for overlay mode
        line_dash = "solid"
        if overlay_models and model != models_to_plot[0]:
            line_dash = "dash"
        
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
                    f"<b>{metric_label}</b><br>"
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
            
            # Adjust color for overlay mode
            if overlay_models:
                curve_color = model_color
                trace_name = f"{metric_label} ({model_name})"
            else:
                curve_color = metric_color
                trace_name = metric_label
            
            fig.add_trace(go.Scatter(
                x=thresh_df["threshold"],
                y=thresh_df[metric_key],
                mode="lines+markers",
                name=trace_name,
                line={"color": curve_color, "width": 2.5, "dash": line_dash},
                marker={"size": 5},
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
                legendgroup=f"{model}_{metric_key}",
                visible=True
            ))
        
        # ═══════════════════════════════════════════════════════════════════════
        # A) CURRENT THRESHOLD MARKER (LINKED TO GLOBAL SLIDER)
        # ═══════════════════════════════════════════════════════════════════════
        
        # Find closest threshold in our computed range
        thresh_idx = np.argmin(np.abs(thresholds - threshold))
        current_row = thresh_df.iloc[thresh_idx]
        
        # Add vertical line for current threshold
        if model == models_to_plot[0]:  # Only add once
            fig.add_vline(
                x=threshold,
                line_dash="solid",
                line_color=COLORS["warning"],
                line_width=2.5,
                opacity=0.9
            )
        
        # Add markers at intersection points for each visible metric
        for metric_key, metric_label, metric_color, is_visible in metric_config:
            if not is_visible:
                continue
            
            marker_color = model_color if overlay_models else metric_color
            
            # Rich tooltip for current threshold marker (A)
            marker_hover = (
                f"<b>Current Threshold</b><br>"
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
                legendgroup=f"{model}_{metric_key}"
            ))
        
        # ═══════════════════════════════════════════════════════════════════════
        # C) & E) OPTIMAL OPERATING POINT BASED ON DECISION MODE
        # ═══════════════════════════════════════════════════════════════════════
        
        optimal = _find_optimal_threshold_for_mode(thresh_df, decision_mode)
        optimal_row = thresh_df.loc[optimal["idx"]]
        
        # Add optimal threshold vertical line (dashed)
        if model == models_to_plot[0]:  # Only add once
            fig.add_vline(
                x=optimal["threshold"],
                line_dash="dash",
                line_color=COLORS["success"],
                line_width=2,
                opacity=0.8
            )
        
        # Add star markers at optimal point for each visible metric
        for metric_key, metric_label, metric_color, is_visible in metric_config:
            if not is_visible:
                continue
            
            marker_color = model_color if overlay_models else COLORS["success"]
            
            # Optimal point tooltip
            optimal_hover = (
                f"<b>{mode_config['annotation']}</b><br>"
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
                legendgroup=f"{model}_{metric_key}"
            ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # D) BALANCED MODE: Highlight F1 peak region
    # ═══════════════════════════════════════════════════════════════════════════
    
    if decision_mode == "balanced" and len(all_model_data) > 0:
        # Get F1 data from first model
        first_model = models_to_plot[0]
        thresh_df = all_model_data[first_model]
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
    
    # Build title
    if overlay_models:
        title_text = f"Metrics vs Threshold (All Models) • Mode: {mode_config['label']}"
    else:
        model_display = MODEL_NAMES.get(selected_model, selected_model) if selected_model else ""
        title_text = f"Metrics vs Threshold - {model_display} • Mode: {mode_config['label']}"
    
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
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(b=130),
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
        title=f"Evolucao de Erros vs Threshold - {MODEL_NAMES.get(selected_model, selected_model)}",
        height=380,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.25
        ),
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    fig.update_xaxes(title_text="Threshold", row=1, col=1)
    fig.update_xaxes(title_text="Threshold", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Rate", tickformat=".0%", row=1, col=2)
    
    return fig


def create_threshold_impact_bars(df: pd.DataFrame, threshold: float = 0.5) -> go.Figure:
    """
    Impacto do threshold nas predições.
    
    Args:
        df: DataFrame de avaliação
        threshold: Limiar de decisão
        
    Returns:
        Figura Plotly
    """
    results = []
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        temp_df = recompute_with_threshold(model_df, threshold)
        
        n_positive = (temp_df["y_pred"] == 1).sum()
        n_negative = (temp_df["y_pred"] == 0).sum()
        
        results.append({
            "model": MODEL_NAMES.get(model, model),
            "Predicted Positive (>50K)": n_positive,
            "Predicted Negative (≤50K)": n_negative
        })
    
    result_df = pd.DataFrame(results)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Pred. Positive (>50K)",
        x=result_df["model"],
        y=result_df["Predicted Positive (>50K)"],
        marker_color=COLORS["success"],
        text=result_df["Predicted Positive (>50K)"],
        textposition="inside",
        textfont={"color": "white", "size": 11}
    ))
    
    fig.add_trace(go.Bar(
        name="Pred. Negative (≤50K)",
        x=result_df["model"],
        y=result_df["Predicted Negative (≤50K)"],
        marker_color=COLORS["text_muted"],
        text=result_df["Predicted Negative (≤50K)"],
        textposition="inside",
        textfont={"color": "white", "size": 11}
    ))
    
    fig.update_layout(
        title=f"Distribuicao de Predicoes (Threshold = {threshold:.2f})",
        barmode="stack",
        xaxis_title="",
        yaxis_title="Numero de Amostras",
        height=350,
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig
