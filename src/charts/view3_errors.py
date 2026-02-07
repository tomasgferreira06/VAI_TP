"""
Gráficos para a View 3: Análise de Erros.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

from src.config.settings import COLORS, MODEL_COLORS, MODEL_NAMES
from src.models.training import recompute_with_threshold


def compute_confusion_matrix_data(df: pd.DataFrame, model: str, threshold: float) -> dict:
    """
    Compute confusion matrix with all normalization variants.
    
    Returns dict with:
        - counts: raw counts
        - pct_total: percentage of total
        - pct_row: row-normalized (by actual class)
        - pct_col: column-normalized (by predicted class)
        - labels: cell labels (TN, FP, FN, TP)
    """
    model_df = df[df["model"] == model]
    temp_df = recompute_with_threshold(model_df, threshold)
    
    cm = confusion_matrix(temp_df["y_true"], temp_df["y_pred"])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    # Row sums (actual class totals)
    actual_neg = tn + fp
    actual_pos = fn + tp
    
    # Column sums (predicted class totals)
    pred_neg = tn + fn
    pred_pos = fp + tp
    
    return {
        "counts": np.array([[tn, fp], [fn, tp]]),
        "pct_total": np.array([
            [tn / total * 100, fp / total * 100],
            [fn / total * 100, tp / total * 100]
        ]),
        "pct_row": np.array([
            [tn / actual_neg * 100 if actual_neg > 0 else 0, fp / actual_neg * 100 if actual_neg > 0 else 0],
            [fn / actual_pos * 100 if actual_pos > 0 else 0, tp / actual_pos * 100 if actual_pos > 0 else 0]
        ]),
        "pct_col": np.array([
            [tn / pred_neg * 100 if pred_neg > 0 else 0, fp / pred_pos * 100 if pred_pos > 0 else 0],
            [fn / pred_neg * 100 if pred_neg > 0 else 0, tp / pred_pos * 100 if pred_pos > 0 else 0]
        ]),
        "raw": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "total": total},
        "totals": {
            "actual_neg": actual_neg, "actual_pos": actual_pos,
            "pred_neg": pred_neg, "pred_pos": pred_pos
        }
    }


def create_advanced_confusion_matrix(
    df: pd.DataFrame,
    selected_model: str,
    threshold: float = 0.5,
    norm_mode: str = "counts",
    comparison_mode: str = "single"
) -> go.Figure:
    """
    Advanced Confusion Matrix with multiple normalization modes and comparison options.
    
    Args:
        df: DataFrame de avaliação
        selected_model: Modelo selecionado (for single mode)
        threshold: Limiar de decisão
        norm_mode: 'counts', 'pct_total', 'pct_row', 'pct_col'
        comparison_mode: 'single', 'side_by_side', 'delta'
        
    Returns:
        Figura Plotly
    """
    labels = ["Negative (≤50K)", "Positive (>50K)"]
    cell_names = [["TN", "FP"], ["FN", "TP"]]
    cell_descriptions = [
        ["True Negatives", "False Positives"],
        ["False Negatives", "True Positives"]
    ]
    
    # Mode descriptions
    mode_titles = {
        "counts": "Raw Counts",
        "pct_total": "% of Total",
        "pct_row": "Row-Normalized (by Actual)",
        "pct_col": "Column-Normalized (by Predicted)"
    }
    
    if comparison_mode == "single":
        return _create_single_confusion_matrix(
            df, selected_model, threshold, norm_mode, 
            labels, cell_names, cell_descriptions, mode_titles
        )
    elif comparison_mode == "side_by_side":
        return _create_side_by_side_confusion_matrix(
            df, threshold, norm_mode,
            labels, cell_names, cell_descriptions, mode_titles
        )
    else:  # delta
        return _create_delta_confusion_matrix(
            df, threshold, norm_mode,
            labels, cell_names, cell_descriptions, mode_titles
        )


def _create_single_confusion_matrix(
    df, selected_model, threshold, norm_mode,
    labels, cell_names, cell_descriptions, mode_titles
) -> go.Figure:
    """Create single model confusion matrix with rich tooltips."""
    
    data = compute_confusion_matrix_data(df, selected_model, threshold)
    
    # Select the data to display based on norm_mode
    if norm_mode == "counts":
        z_values = data["counts"]
        value_format = lambda v: f"{int(v):,}"
    else:
        z_values = data[norm_mode]
        value_format = lambda v: f"{v:.1f}%"
    
    # Create rich text annotations with cell names
    text = []
    custom_data = []
    for i in range(2):
        row_text = []
        row_custom = []
        for j in range(2):
            cell_name = cell_names[i][j]
            value = z_values[i][j]
            
            # Display text
            if norm_mode == "counts":
                display_text = f"<b>{cell_name}</b><br>{int(value):,}"
            else:
                display_text = f"<b>{cell_name}</b><br>{value:.1f}%"
            row_text.append(display_text)
            
            # Custom data for rich tooltip
            row_custom.append({
                "cell_type": cell_descriptions[i][j],
                "cell_name": cell_name,
                "count": int(data["counts"][i][j]),
                "pct_total": data["pct_total"][i][j],
                "pct_row": data["pct_row"][i][j],
                "pct_col": data["pct_col"][i][j]
            })
        text.append(row_text)
        custom_data.append(row_custom)
    
    # Create hovertemplate for rich tooltips
    hover_template = (
        "<b>%{customdata.cell_type}</b> (%{customdata.cell_name})<br><br>"
        "Count: %{customdata.count:,}<br>"
        "% of Total: %{customdata.pct_total:.1f}%<br>"
        "% of Actual Class (Row): %{customdata.pct_row:.1f}%<br>"
        "% of Predicted Class (Col): %{customdata.pct_col:.1f}%"
        "<extra></extra>"
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=data["counts"],  # Always use counts for color intensity
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        customdata=custom_data,
        hovertemplate=hover_template,
        colorscale=[
            [0, COLORS["bg_hover"]],
            [0.5, "rgba(99, 102, 241, 0.5)"],
            [1, COLORS["primary"]]
        ],
        showscale=False
    ))
    
    # Calculate metrics
    raw = data["raw"]
    accuracy = (raw["tn"] + raw["tp"]) / raw["total"]
    precision = raw["tp"] / (raw["tp"] + raw["fp"]) if (raw["tp"] + raw["fp"]) > 0 else 0
    recall = raw["tp"] / (raw["tp"] + raw["fn"]) if (raw["tp"] + raw["fn"]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Add metrics annotation
    metrics_text = (
        f"<b>Accuracy:</b> {accuracy:.1%}<br>"
        f"<b>Precision:</b> {precision:.1%}<br>"
        f"<b>Recall:</b> {recall:.1%}<br>"
        f"<b>F1:</b> {f1:.1%}"
    )
    
    fig.add_annotation(
        x=1.45,
        y=0.5,
        text=metrics_text,
        showarrow=False,
        font=dict(size=11, color=COLORS["text_primary"]),
        bgcolor=COLORS["bg_hover"],
        borderpad=10,
        align="left"
    )
    
    model_name = MODEL_NAMES.get(selected_model, selected_model)
    fig.update_layout(
        title=dict(
            text=f"Confusion Matrix — {model_name}<br>"
                 f"<span style='font-size:12px;color:{COLORS['text_secondary']}'>"
                 f"Mode: {mode_titles[norm_mode]} | Threshold: {threshold:.2f}</span>",
            font=dict(size=15)
        ),
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=420,
        yaxis_autorange="reversed",
        margin=dict(r=120),
        transition=dict(duration=400, easing="cubic-in-out")
    )
    
    return fig


def _create_side_by_side_confusion_matrix(
    df, threshold, norm_mode,
    labels, cell_names, cell_descriptions, mode_titles
) -> go.Figure:
    """Create side-by-side comparison of both models."""
    
    models = ["logreg", "rf"]
    model_data = {m: compute_confusion_matrix_data(df, m, threshold) for m in models}
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[MODEL_NAMES["logreg"], MODEL_NAMES["rf"]],
        horizontal_spacing=0.12
    )
    
    # Find global max for consistent color scale
    max_count = max(
        model_data["logreg"]["counts"].max(),
        model_data["rf"]["counts"].max()
    )
    
    for idx, model in enumerate(models):
        data = model_data[model]
        col = idx + 1
        
        # Select the data to display based on norm_mode
        if norm_mode == "counts":
            z_values = data["counts"]
        else:
            z_values = data[norm_mode]
        
        # Create text annotations
        text = []
        custom_data = []
        for i in range(2):
            row_text = []
            row_custom = []
            for j in range(2):
                cell_name = cell_names[i][j]
                value = z_values[i][j]
                
                if norm_mode == "counts":
                    display_text = f"<b>{cell_name}</b><br>{int(value):,}"
                else:
                    display_text = f"<b>{cell_name}</b><br>{value:.1f}%"
                row_text.append(display_text)
                
                row_custom.append({
                    "cell_type": cell_descriptions[i][j],
                    "cell_name": cell_name,
                    "count": int(data["counts"][i][j]),
                    "pct_total": data["pct_total"][i][j],
                    "pct_row": data["pct_row"][i][j],
                    "pct_col": data["pct_col"][i][j]
                })
            text.append(row_text)
            custom_data.append(row_custom)
        
        hover_template = (
            f"<b>{MODEL_NAMES[model]}</b><br>"
            "<b>%{customdata.cell_type}</b> (%{customdata.cell_name})<br><br>"
            "Count: %{customdata.count:,}<br>"
            "% of Total: %{customdata.pct_total:.1f}%<br>"
            "% of Actual (Row): %{customdata.pct_row:.1f}%<br>"
            "% of Predicted (Col): %{customdata.pct_col:.1f}%"
            "<extra></extra>"
        )
        
        fig.add_trace(
            go.Heatmap(
                z=data["counts"],
                x=labels,
                y=labels,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 12, "color": "white"},
                customdata=custom_data,
                hovertemplate=hover_template,
                colorscale=[
                    [0, COLORS["bg_hover"]],
                    [0.5, "rgba(99, 102, 241, 0.5)"],
                    [1, COLORS["primary"]]
                ],
                zmin=0,
                zmax=max_count,
                showscale=False
            ),
            row=1, col=col
        )
    
    fig.update_yaxes(autorange="reversed")
    
    fig.update_layout(
        title=dict(
            text=f"Confusion Matrix Comparison<br>"
                 f"<span style='font-size:12px;color:{COLORS['text_secondary']}'>"
                 f"Mode: {mode_titles[norm_mode]} | Threshold: {threshold:.2f}</span>",
            font=dict(size=15)
        ),
        height=420,
        showlegend=False,
        transition=dict(duration=400, easing="cubic-in-out")
    )
    
    # Update subplot titles style
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13, color=COLORS["text_primary"])
    
    return fig


def _create_delta_confusion_matrix(
    df, threshold, norm_mode,
    labels, cell_names, cell_descriptions, mode_titles
) -> go.Figure:
    """Create delta confusion matrix (RF - LR) with diverging color scale."""
    
    lr_data = compute_confusion_matrix_data(df, "logreg", threshold)
    rf_data = compute_confusion_matrix_data(df, "rf", threshold)
    
    # Compute delta based on norm_mode
    if norm_mode == "counts":
        delta = rf_data["counts"] - lr_data["counts"]
        value_format = lambda v: f"{int(v):+,}" if v != 0 else "0"
    else:
        delta = rf_data[norm_mode] - lr_data[norm_mode]
        value_format = lambda v: f"{v:+.1f}%" if abs(v) > 0.05 else "0%"
    
    # Create text and custom data
    text = []
    custom_data = []
    for i in range(2):
        row_text = []
        row_custom = []
        for j in range(2):
            cell_name = cell_names[i][j]
            d_val = delta[i][j]
            
            if norm_mode == "counts":
                if d_val > 0:
                    display_text = f"<b>{cell_name}</b><br>+{int(d_val):,}"
                elif d_val < 0:
                    display_text = f"<b>{cell_name}</b><br>{int(d_val):,}"
                else:
                    display_text = f"<b>{cell_name}</b><br>0"
            else:
                if abs(d_val) > 0.05:
                    display_text = f"<b>{cell_name}</b><br>{d_val:+.1f}%"
                else:
                    display_text = f"<b>{cell_name}</b><br>0%"
            
            row_text.append(display_text)
            
            row_custom.append({
                "cell_type": cell_descriptions[i][j],
                "cell_name": cell_name,
                "lr_count": int(lr_data["counts"][i][j]),
                "rf_count": int(rf_data["counts"][i][j]),
                "delta_count": int(rf_data["counts"][i][j] - lr_data["counts"][i][j]),
                "lr_pct": lr_data["pct_total"][i][j],
                "rf_pct": rf_data["pct_total"][i][j],
                "delta_pct": rf_data["pct_total"][i][j] - lr_data["pct_total"][i][j]
            })
        text.append(row_text)
        custom_data.append(row_custom)
    
    hover_template = (
        "<b>%{customdata.cell_type}</b> (%{customdata.cell_name})<br><br>"
        "<b>Random Forest:</b> %{customdata.rf_count:,} (%{customdata.rf_pct:.1f}%)<br>"
        "<b>Logistic Regression:</b> %{customdata.lr_count:,} (%{customdata.lr_pct:.1f}%)<br><br>"
        "<b>Δ (RF − LR):</b> %{customdata.delta_count:+,} (%{customdata.delta_pct:+.1f}%)"
        "<extra></extra>"
    )
    
    # Find max absolute value for symmetric color scale
    max_abs = max(abs(delta.min()), abs(delta.max()))
    if max_abs == 0:
        max_abs = 1
    
    # Diverging color scale (blue for negative, red for positive)
    fig = go.Figure(data=go.Heatmap(
        z=delta,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        customdata=custom_data,
        hovertemplate=hover_template,
        colorscale=[
            [0, COLORS["primary"]],      # Negative (LR better)
            [0.5, COLORS["bg_hover"]],   # Zero
            [1, COLORS["accent"]]         # Positive (RF better)
        ],
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Δ (RF−LR)",
                side="right",
                font=dict(color=COLORS["text_secondary"])
            ),
            tickfont=dict(color=COLORS["text_secondary"])
        )
    ))
    
    # Add interpretation legend
    fig.add_annotation(
        x=1.35,
        y=0.85,
        xref="paper",
        yref="paper",
        text=(
            f"<span style='color:{COLORS['accent']}'>■</span> RF has more<br>"
            f"<span style='color:{COLORS['primary']}'>■</span> LR has more"
        ),
        showarrow=False,
        font=dict(size=10, color=COLORS["text_secondary"]),
        align="left"
    )
    
    fig.update_layout(
        title=dict(
            text=f"Delta Confusion Matrix (RF − LR)<br>"
                 f"<span style='font-size:12px;color:{COLORS['text_secondary']}'>"
                 f"Mode: {mode_titles[norm_mode]} | Threshold: {threshold:.2f}</span>",
            font=dict(size=15)
        ),
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=420,
        yaxis_autorange="reversed",
        margin=dict(r=130),
        transition=dict(duration=400, easing="cubic-in-out")
    )
    
    return fig


# Keep original function for backwards compatibility
def create_confusion_matrix_heatmap(df: pd.DataFrame, selected_model: str, threshold: float = 0.5) -> go.Figure:
    """
    Matriz de confusão estilizada com anotação de accuracy.
    (Legacy function - use create_advanced_confusion_matrix for full features)
    """
    return create_advanced_confusion_matrix(df, selected_model, threshold, "counts", "single")





def compute_error_tradeoff_data(df: pd.DataFrame, subgroup: str = "global") -> pd.DataFrame:
    """
    Compute FPR and FNR for all models across a range of thresholds.
    
    Args:
        df: DataFrame de avaliação
        subgroup: "global", "Male", "Female", "White", "Non-White"
        
    Returns:
        DataFrame with columns: model, threshold, subgroup, fpr, fnr, fp, fn, tp, tn, precision, recall
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    models = ["logreg", "rf"]
    results = []
    
    for model in models:
        model_df = df[df["model"] == model]
        
        # Filter by subgroup
        if subgroup == "global":
            filtered_df = model_df
            subgroup_label = "Global"
        elif subgroup in ["Male", "Female"]:
            filtered_df = model_df[model_df["sex"] == subgroup]
            subgroup_label = subgroup
        elif subgroup == "White":
            filtered_df = model_df[model_df["race"] == "White"]
            subgroup_label = "White"
        elif subgroup == "Non-White":
            filtered_df = model_df[model_df["race"] != "White"]
            subgroup_label = "Non-White"
        else:
            filtered_df = model_df
            subgroup_label = "Global"
        
        for t in thresholds:
            temp_df = recompute_with_threshold(filtered_df, t)
            
            if len(temp_df) == 0:
                continue
                
            cm = confusion_matrix(temp_df["y_true"], temp_df["y_pred"], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            results.append({
                "model": model,
                "model_name": MODEL_NAMES.get(model, model),
                "threshold": t,
                "subgroup": subgroup_label,
                "fpr": fpr,
                "fnr": fnr,
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "tn": int(tn),
                "precision": precision,
                "recall": recall
            })
    
    return pd.DataFrame(results)


def create_error_tradeoff_scatter(
    df: pd.DataFrame,
    current_threshold: float = 0.5,
    subgroup: str = "global"
) -> go.Figure:
    """
    Error Trade-off Scatter: FPR vs FNR trajectories for both models.
    
    Shows how each model moves through (FPR, FNR) space as threshold changes.
    This is a "dynamic data, static representation" visualization similar to
    Gapminder-style trajectories with marker size encoding threshold.
    
    Args:
        df: DataFrame de avaliação
        current_threshold: Current global threshold (highlighted on trajectories)
        subgroup: "global", "Male", "Female", "White", "Non-White"
        
    Returns:
        Figura Plotly
    """
    # Compute data for all thresholds
    data = compute_error_tradeoff_data(df, subgroup)
    
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected subgroup",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["text_secondary"])
        )
        fig.update_layout(height=450)
        return fig
    
    fig = go.Figure()
    
    # Define model styles - both use circles
    model_styles = {
        "logreg": {
            "color": MODEL_COLORS["logreg"],
            "color_light": "rgba(99, 102, 241, 0.4)",
            "name": MODEL_NAMES["logreg"],
            "symbol": "circle"
        },
        "rf": {
            "color": MODEL_COLORS["rf"],
            "color_light": "rgba(20, 184, 166, 0.4)",
            "name": MODEL_NAMES["rf"],
            "symbol": "circle"
        }
    }
    
    # Marker size range based on threshold (Gapminder style)
    # Lower threshold = smaller marker, higher threshold = larger marker
    min_size = 6
    max_size = 28
    
    def threshold_to_size(t):
        """Map threshold (0.05-0.95) to marker size."""
        normalized = (t - 0.05) / (0.95 - 0.05)
        return min_size + normalized * (max_size - min_size)
    
    # Track axis ranges for consistent scaling
    all_fpr = data["fpr"].values
    all_fnr = data["fnr"].values
    
    for model in ["logreg", "rf"]:
        model_data = data[data["model"] == model].sort_values("threshold")
        style = model_styles[model]
        
        if model_data.empty:
            continue
        
        # Calculate marker sizes based on threshold
        marker_sizes = [threshold_to_size(t) for t in model_data["threshold"]]
        
        # Create custom data for rich tooltips
        customdata = np.column_stack([
            model_data["threshold"],
            model_data["fp"],
            model_data["fn"],
            model_data["tp"],
            model_data["tn"],
            model_data["precision"] * 100,
            model_data["recall"] * 100,
            model_data["fpr"] * 100,
            model_data["fnr"] * 100
        ])
        
        # Hover template with all metrics
        hover_template = (
            f"<b>{style['name']}</b><br>"
            f"<b>Subgroup:</b> {model_data['subgroup'].iloc[0]}<br><br>"
            "<b>Threshold:</b> %{customdata[0]:.2f}<br>"
            "<b>FPR:</b> %{customdata[7]:.1f}%<br>"
            "<b>FNR:</b> %{customdata[8]:.1f}%<br><br>"
            "<b>Confusion Matrix:</b><br>"
            "  TP: %{customdata[3]:,} | FP: %{customdata[1]:,}<br>"
            "  FN: %{customdata[2]:,} | TN: %{customdata[4]:,}<br><br>"
            "<b>Precision:</b> %{customdata[5]:.1f}%<br>"
            "<b>Recall:</b> %{customdata[6]:.1f}%"
            "<extra></extra>"
        )
        
        # Trajectory line (thin, connecting the dots)
        fig.add_trace(go.Scatter(
            x=model_data["fpr"],
            y=model_data["fnr"],
            mode="lines",
            name=f"{style['name']} path",
            line=dict(color=style["color_light"], width=2),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=model
        ))
        
        # Main markers with size encoding threshold
        fig.add_trace(go.Scatter(
            x=model_data["fpr"],
            y=model_data["fnr"],
            mode="markers",
            name=style["name"],
            marker=dict(
                size=marker_sizes,
                color=style["color_light"],
                symbol=style["symbol"],
                line=dict(width=1.5, color=style["color"]),
                opacity=0.85
            ),
            customdata=customdata,
            hovertemplate=hover_template,
            legendgroup=model
        ))
        
        # Highlight current threshold point with pulsing effect
        # Reset index to ensure proper indexing
        model_data_reset = model_data.reset_index(drop=True)
        threshold_diffs = (model_data_reset["threshold"] - current_threshold).abs()
        closest_pos = threshold_diffs.idxmin()
        current_point = model_data_reset.iloc[closest_pos]
        current_size = threshold_to_size(current_point["threshold"])
        
        # Outer glow for current point
        fig.add_trace(go.Scatter(
            x=[current_point["fpr"]],
            y=[current_point["fnr"]],
            mode="markers",
            marker=dict(
                size=current_size + 14,
                color=style["color"],
                symbol=style["symbol"],
                opacity=0.25
            ),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=model
        ))
        
        # Current point marker (solid, highlighted)
        fig.add_trace(go.Scatter(
            x=[current_point["fpr"]],
            y=[current_point["fnr"]],
            mode="markers",
            name=f"Current (t={current_threshold:.2f})",
            marker=dict(
                size=current_size + 6,
                color=style["color"],
                symbol=style["symbol"],
                line=dict(width=3, color=COLORS["text_primary"]),
                opacity=1
            ),
            customdata=[[
                current_point["threshold"],
                current_point["fp"],
                current_point["fn"],
                current_point["tp"],
                current_point["tn"],
                current_point["precision"] * 100,
                current_point["recall"] * 100,
                current_point["fpr"] * 100,
                current_point["fnr"] * 100
            ]],
            hovertemplate=(
                f"<b>{style['name']} — CURRENT</b><br>"
                f"<b>Subgroup:</b> {current_point['subgroup']}<br><br>"
                "<b>Threshold:</b> %{customdata[0]:.2f}<br>"
                "<b>FPR:</b> %{customdata[7]:.1f}%<br>"
                "<b>FNR:</b> %{customdata[8]:.1f}%<br><br>"
                "<b>Confusion Matrix:</b><br>"
                "  TP: %{customdata[3]:,} | FP: %{customdata[1]:,}<br>"
                "  FN: %{customdata[2]:,} | TN: %{customdata[4]:,}<br><br>"
                "<b>Precision:</b> %{customdata[5]:.1f}%<br>"
                "<b>Recall:</b> %{customdata[6]:.1f}%"
                "<extra></extra>"
            ),
            showlegend=False,
            legendgroup=model
        ))
        
        # Add model name label near the trajectory
        mid_idx = len(model_data) // 2
        mid_point = model_data.iloc[mid_idx]
        
        # Offset for label positioning
        x_offset = 0.015 if model == "logreg" else -0.015
        y_offset = 0.02 if model == "logreg" else -0.02
        
        fig.add_annotation(
            x=mid_point["fpr"] + x_offset,
            y=mid_point["fnr"] + y_offset,
            text=f"<b>{style['name']}</b>",
            showarrow=False,
            font=dict(size=10, color=style["color"]),
            opacity=0.9
        )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # THRESHOLD SIZE LEGEND (Gapminder style - horizontal at top)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # We'll add the legend as a separate trace using a secondary y-axis
    # This allows proper positioning above the main chart
    legend_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    legend_sizes = [threshold_to_size(t) for t in legend_thresholds]
    
    # X positions for legend circles (evenly spaced)
    legend_x = list(range(len(legend_thresholds)))
    
    # Add connecting line for legend (behind markers)
    fig.add_trace(go.Scatter(
        x=legend_x,
        y=[0] * len(legend_thresholds),
        mode="lines",
        line=dict(color="rgba(148, 163, 184, 0.3)", width=2),
        hoverinfo="skip",
        showlegend=False,
        xaxis="x2",
        yaxis="y2"
    ))
    
    # Add legend markers on secondary axes
    fig.add_trace(go.Scatter(
        x=legend_x,
        y=[0] * len(legend_thresholds),
        mode="markers",
        marker=dict(
            size=legend_sizes,
            color="rgba(148, 163, 184, 0.4)",
            line=dict(width=1, color=COLORS["text_muted"]),
            symbol="circle"
        ),
        hoverinfo="skip",
        showlegend=False,
        xaxis="x2",
        yaxis="y2"
    ))
    
    # Add "0.1" label on left
    fig.add_annotation(
        x=-0.8,
        y=0,
        text="<b>0.1</b>",
        showarrow=False,
        font=dict(size=10, color=COLORS["text_secondary"]),
        xref="x2",
        yref="y2",
        xanchor="right",
        yanchor="middle"
    )
    
    # Add "0.9" label on right
    fig.add_annotation(
        x=len(legend_thresholds) - 0.2,
        y=0,
        text="<b>0.9</b>",
        showarrow=False,
        font=dict(size=10, color=COLORS["text_secondary"]),
        xref="x2",
        yref="y2",
        xanchor="left",
        yanchor="middle"
    )
    
    # Add current threshold annotation
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"<b>Current: t = {current_threshold:.2f}</b>",
        showarrow=False,
        font=dict(size=12, color=COLORS["text_primary"]),
        bgcolor=COLORS["bg_hover"],
        borderpad=8,
        align="left",
        xanchor="left",
        yanchor="top"
    )
    
    # Add direction annotation with arrow style
    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref="paper",
        yref="paper",
        text=(
            "◀ <i>Lower threshold</i><br>"
            "<i>Higher FPR, Lower FNR</i>"
        ),
        showarrow=False,
        font=dict(size=9, color=COLORS["text_muted"]),
        align="left",
        xanchor="left",
        yanchor="bottom"
    )
    
    fig.add_annotation(
        x=0.35,
        y=0.02,
        xref="paper",
        yref="paper",
        text=(
            "<i>Higher threshold</i> ▶<br>"
            "<i>Lower FPR, Higher FNR</i>"
        ),
        showarrow=False,
        font=dict(size=9, color=COLORS["text_muted"]),
        align="left",
        xanchor="left",
        yanchor="bottom"
    )
    
    # Calculate axis ranges with padding
    fpr_padding = (all_fpr.max() - all_fpr.min()) * 0.1
    fnr_padding = (all_fnr.max() - all_fnr.min()) * 0.1
    fpr_range = [max(0, all_fpr.min() - fpr_padding), min(1, all_fpr.max() + fpr_padding)]
    fnr_range = [max(0, all_fnr.min() - fnr_padding), min(1, all_fnr.max() + fnr_padding)]
    
    # Get actual subgroup label from the data
    subgroup_label = data["subgroup"].iloc[0] if not data.empty else "Global"
    
    fig.update_layout(
        title=dict(
            text=f"Error Trade-off Trajectories<br>"
                 f"<span style='font-size:12px;color:{COLORS['text_secondary']}'>"
                 f"Subgroup: {subgroup_label} — Marker size encodes threshold value</span>",
            font=dict(size=15),
            x=0.02,
            xanchor="left"
        ),
        # Main plot axes
        xaxis=dict(
            title=dict(text="False Positive Rate (FPR)", font=dict(size=12)),
            tickformat=".0%",
            range=fpr_range,
            gridcolor="rgba(51, 65, 85, 0.3)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor=COLORS["border"],
            zerolinewidth=1,
            domain=[0, 1]
        ),
        yaxis=dict(
            title=dict(text="False Negative Rate (FNR)", font=dict(size=12)),
            tickformat=".0%",
            range=fnr_range,
            gridcolor="rgba(51, 65, 85, 0.3)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor=COLORS["border"],
            zerolinewidth=1,
            domain=[0, 0.82]
        ),
        # Secondary axes for threshold legend (positioned at top)
        xaxis2=dict(
            domain=[0.5, 0.95],
            range=[-1.5, 9],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            anchor="y2"
        ),
        yaxis2=dict(
            domain=[0.88, 0.98],
            range=[-1.5, 1.5],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            anchor="x2"
        ),
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11)
        ),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        transition=dict(duration=400, easing="cubic-in-out"),
        margin=dict(t=120, r=40),
        uirevision=f"{current_threshold}-{subgroup}"  # Force update on threshold/subgroup change
    )
    
    return fig


# Legacy function kept for backwards compatibility
def create_error_rates_comparison(df: pd.DataFrame, threshold: float = 0.5) -> go.Figure:
    """
    Comparação de taxas de erro entre modelos.
    (Legacy - use create_error_tradeoff_scatter for advanced visualization)
    """
    return create_error_tradeoff_scatter(df, threshold, "global")
