"""
Gráficos para a View 1: Comparação Global.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.config.settings import COLORS, MODEL_COLORS, MODEL_NAMES
from src.utils.helpers import hex_to_rgba
from src.models.training import global_metrics, recompute_with_threshold


def create_metrics_comparison_chart(df: pd.DataFrame, threshold: float = 0.5) -> go.Figure:
    """
    Gráfico de barras comparando métricas entre modelos.
    
    Args:
        df: DataFrame de avaliação
        threshold: Limiar de decisão
        
    Returns:
        Figura Plotly
    """
    df_thresh = recompute_with_threshold(df, threshold)
    summary = df_thresh.groupby("model").apply(global_metrics).reset_index()
    
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    fig = go.Figure()
    
    for model in summary["model"].unique():
        model_data = summary[summary["model"] == model]
        values = [model_data[m].values[0] for m in metrics]
        
        # Tooltips mais informativos
        hover_texts = [
            f"<b>{metric_labels[i]}</b><br>" +
            f"Model: {MODEL_NAMES.get(model, model)}<br>" +
            f"Value: {values[i]:.2%}<br>" +
            f"Threshold: {threshold}<br>" +
            f"<i>{'Higher is better' if m != 'recall' else 'Trade-off with Precision'}</i>"
            for i, m in enumerate(metrics)
        ]
        
        fig.add_trace(go.Bar(
            name=MODEL_NAMES.get(model, model),
            x=metric_labels,
            y=values,
            marker_color=MODEL_COLORS.get(model, COLORS["primary"]),
            marker_line_width=0,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            textfont={"size": 11, "color": COLORS["text_secondary"]},
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts
        ))
    
    fig.update_layout(
        title="Comparacao de Metricas por Modelo",
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        yaxis_range=[0, 1.15],
        yaxis_tickformat=".0%",
        showlegend=True,
        height=400,
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig


def create_roc_curves(df: pd.DataFrame) -> go.Figure:
    """
    Curvas ROC para ambos os modelos com anotações.
    
    Args:
        df: DataFrame de avaliação
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # Linha diagonal (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line={"dash": "dash", "color": COLORS["text_muted"], "width": 1},
        name="Random Classifier",
        hoverinfo="skip"
    ))
    
    # Anotação explicativa na diagonal
    fig.add_annotation(
        x=0.65, y=0.35,
        text="Random Classifier<br>(AUC = 0.5)",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["text_muted"],
        ax=-30, ay=-30,
        font=dict(size=9, color=COLORS["text_muted"])
    )
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        fpr, tpr, _ = roc_curve(model_df["y_true"], model_df["y_proba"])
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{MODEL_NAMES.get(model, model)} (AUC = {roc_auc:.3f})",
            line={"color": MODEL_COLORS.get(model, COLORS["primary"]), "width": 2.5},
            fill="tozeroy",
            fillcolor=hex_to_rgba(MODEL_COLORS.get(model, COLORS["primary"]), 0.15),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
        height=400,
        legend={"yanchor": "bottom", "y": 0.02, "xanchor": "right", "x": 0.98},
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig


def create_feature_importance_chart(pipelines: dict, cat_cols: list, num_cols: list) -> go.Figure:
    """
    Gráfico de importância de features (explainability/faithfulness).
    
    Args:
        pipelines: Dicionário de pipelines treinados
        cat_cols: Lista de colunas categóricas
        num_cols: Lista de colunas numéricas
        
    Returns:
        Figura Plotly
    """
    # Extrair feature importances do Random Forest
    rf_pipe = pipelines["rf"]
    rf_model = rf_pipe.named_steps["model"]
    preprocessor = rf_pipe.named_steps["preprocess"]
    
    # Obter nomes das features após transformação
    feature_names = []
    
    # Features categóricas (one-hot encoded)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_features = cat_encoder.get_feature_names_out(cat_cols)
    feature_names.extend(cat_features)
    
    # Features numéricas
    feature_names.extend(num_cols)
    
    # Importâncias
    importances = rf_model.feature_importances_
    
    # Criar DataFrame e ordenar
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=True)
    
    # Top 15 features mais importantes
    top_features = importance_df.tail(15)
    
    # Criar gráfico
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_features["importance"],
        y=top_features["feature"],
        orientation="h",
        marker=dict(
            color=top_features["importance"],
            colorscale=[[0, COLORS["primary"]], [0.5, COLORS["accent"]], [1, COLORS["secondary"]]],
            line=dict(width=0)
        ),
        text=[f"{v:.3f}" for v in top_features["importance"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text_secondary"]),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    
    # Anotação explicativa
    fig.add_annotation(
        x=top_features["importance"].max() * 0.7,
        y=14,
        text="Features com maior<br>impacto nas predições",
        showarrow=False,
        font=dict(size=10, color=COLORS["text_muted"]),
        bgcolor=COLORS["bg_card"],
        borderpad=6
    )
    
    fig.update_layout(
        title="Feature Importance (Random Forest)",
        xaxis_title="Importance Score (MDI)",
        yaxis_title="",
        height=450,
        margin=dict(l=180, r=60, t=60, b=50),
        showlegend=False,
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig


def create_calibration_plot(eval_df: pd.DataFrame) -> go.Figure:
    """
    Calibration Plot (Reliability Diagram) - Robustness.
    
    Args:
        eval_df: DataFrame de avaliação
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # Linha diagonal (perfect calibration)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color=COLORS["text_muted"], width=1.5),
        name="Perfectly Calibrated",
        hoverinfo="skip"
    ))
    
    # Anotação na diagonal
    fig.add_annotation(
        x=0.85, y=0.78,
        text="Perfect<br>Calibration",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor=COLORS["text_muted"],
        ax=30, ay=30,
        font=dict(size=9, color=COLORS["text_muted"])
    )
    
    for model in eval_df["model"].unique():
        model_df = eval_df[eval_df["model"] == model]
        
        # Calcular curva de calibração
        prob_true, prob_pred = calibration_curve(
            model_df["y_true"], 
            model_df["y_proba"], 
            n_bins=10,
            strategy="uniform"
        )
        
        # Linha de calibração
        fig.add_trace(go.Scatter(
            x=prob_pred, y=prob_true,
            mode="lines+markers",
            name=MODEL_NAMES.get(model, model),
            line=dict(color=MODEL_COLORS.get(model, COLORS["primary"]), width=2.5),
            marker=dict(size=8, symbol="circle"),
            hovertemplate="Mean Predicted: %{x:.2f}<br>Fraction Positive: %{y:.2f}<extra></extra>"
        ))
        
        # Calcular Brier Score
        brier = brier_score_loss(model_df["y_true"], model_df["y_proba"])
        
        # Anotação com Brier Score
        y_pos = 0.15 if model == "logreg" else 0.08
        fig.add_annotation(
            x=0.02, y=y_pos,
            text=f"{MODEL_NAMES.get(model, model)}: Brier={brier:.3f}",
            showarrow=False,
            font=dict(size=10, color=MODEL_COLORS.get(model, COLORS["primary"])),
            xanchor="left"
        )
    
    fig.update_layout(
        title="Calibration Plot (Reliability Diagram)",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.55),
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig
