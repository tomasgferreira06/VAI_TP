"""
Gráficos para a View 4: Fairness.
"""
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
