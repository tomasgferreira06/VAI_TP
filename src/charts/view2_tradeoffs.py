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


def create_precision_recall_curve(df: pd.DataFrame) -> go.Figure:
    """
    Curva Precision-Recall para ambos os modelos.
    
    Args:
        df: DataFrame de avaliação
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        precision, recall, thresholds = precision_recall_curve(
            model_df["y_true"], model_df["y_proba"]
        )
        ap = average_precision_score(model_df["y_true"], model_df["y_proba"])
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines",
            name=f"{MODEL_NAMES.get(model, model)} (AP = {ap:.3f})",
            line={"color": MODEL_COLORS.get(model, COLORS["primary"]), "width": 2.5},
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
        ))
    
    # Adicionar linha de baseline
    baseline_precision = df["y_true"].mean()
    fig.add_hline(
        y=baseline_precision,
        line_dash="dot",
        line_color=COLORS["text_muted"],
        line_width=1
    )
    
    fig.add_annotation(
        x=0.85,
        y=baseline_precision + 0.05,
        text=f"Baseline ({baseline_precision:.1%})",
        showarrow=False,
        font=dict(size=9, color=COLORS["text_muted"])
    )
    
    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1.02],
        yaxis_range=[0, 1.02],
        height=400,
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
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
