"""
Gráficos para a View 3: Análise de Erros.
"""
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from src.config.settings import COLORS, MODEL_COLORS, MODEL_NAMES
from src.models.training import recompute_with_threshold


def create_confusion_matrix_heatmap(df: pd.DataFrame, selected_model: str, threshold: float = 0.5) -> go.Figure:
    """
    Matriz de confusão estilizada com anotação de accuracy.
    
    Args:
        df: DataFrame de avaliação
        selected_model: Modelo selecionado
        threshold: Limiar de decisão
        
    Returns:
        Figura Plotly
    """
    model_df = df[df["model"] == selected_model]
    temp_df = recompute_with_threshold(model_df, threshold)
    
    cm = confusion_matrix(temp_df["y_true"], temp_df["y_pred"])
    tn, fp, fn, tp = cm.ravel()
    
    # Normalizar para percentagens
    cm_pct = cm / cm.sum() * 100
    
    labels = ["≤50K (Negative)", ">50K (Positive)"]
    
    # Texto para cada célula
    text = [
        [f"TN<br>{tn:,}<br>({cm_pct[0,0]:.1f}%)", f"FP<br>{fp:,}<br>({cm_pct[0,1]:.1f}%)"],
        [f"FN<br>{fn:,}<br>({cm_pct[1,0]:.1f}%)", f"TP<br>{tp:,}<br>({cm_pct[1,1]:.1f}%)"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale=[
            [0, COLORS["bg_hover"]],
            [0.5, COLORS["primary"] + "88"],
            [1, COLORS["primary"]]
        ],
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>"
    ))
    
    # Calcular accuracy e adicionar como anotação
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    
    fig.add_annotation(
        x=1.35,
        y=0.5,
        text=f"<b>Accuracy</b><br>{accuracy:.1%}",
        showarrow=False,
        font=dict(size=12, color=COLORS["text_primary"]),
        bgcolor=COLORS["bg_hover"],
        borderpad=8
    )
    
    fig.update_layout(
        title=f"Confusion Matrix - {MODEL_NAMES.get(selected_model, selected_model)}",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=400,
        yaxis_autorange="reversed",
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig


def create_error_distribution_by_feature(df: pd.DataFrame, selected_model: str, feature: str, threshold: float = 0.5) -> go.Figure:
    """
    Distribuição de erros por feature categórica.
    
    Args:
        df: DataFrame de avaliação
        selected_model: Modelo selecionado
        feature: Nome da feature para agrupar
        threshold: Limiar de decisão
        
    Returns:
        Figura Plotly
    """
    model_df = df[df["model"] == selected_model]
    temp_df = recompute_with_threshold(model_df, threshold)
    
    # Identificar tipos de erro
    temp_df = temp_df.copy()
    temp_df["error_type"] = "Correct"
    temp_df.loc[(temp_df["y_true"] == 0) & (temp_df["y_pred"] == 1), "error_type"] = "False Positive"
    temp_df.loc[(temp_df["y_true"] == 1) & (temp_df["y_pred"] == 0), "error_type"] = "False Negative"
    
    # Agregar por feature
    error_summary = temp_df.groupby([feature, "error_type"]).size().unstack(fill_value=0)
    error_summary = error_summary.reset_index()
    
    fig = go.Figure()
    
    error_config = [
        ("Correct", COLORS["success"], "Correct"),
        ("False Positive", COLORS["error"], "False Positive"),
        ("False Negative", COLORS["warning"], "False Negative")
    ]
    
    for col, color, name in error_config:
        if col in error_summary.columns:
            fig.add_trace(go.Bar(
                name=name,
                x=error_summary[feature],
                y=error_summary[col],
                marker_color=color,
                hovertemplate=f"{name}<br>{feature}: %{{x}}<br>Count: %{{y:,}}<extra></extra>"
            ))
    
    fig.update_layout(
        title=f"Distribuicao de Erros por {feature.title()}",
        barmode="stack",
        xaxis_title=feature.title(),
        yaxis_title="Numero de Amostras",
        height=400,
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig


def create_error_rates_comparison(df: pd.DataFrame, threshold: float = 0.5) -> go.Figure:
    """
    Comparação de taxas de erro entre modelos.
    
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
        
        tn, fp, fn, tp = confusion_matrix(temp_df["y_true"], temp_df["y_pred"]).ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        error_rate = (fp + fn) / len(temp_df)
        
        results.append({
            "model": MODEL_NAMES.get(model, model),
            "model_key": model,
            "False Positive Rate": fpr,
            "False Negative Rate": fnr,
            "Overall Error Rate": error_rate
        })
    
    result_df = pd.DataFrame(results)
    
    fig = go.Figure()
    
    metrics = [
        ("False Positive Rate", COLORS["error"]),
        ("False Negative Rate", COLORS["warning"]),
    ]
    
    for metric, color in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=result_df["model"],
            y=result_df[metric],
            marker_color=color,
            text=[f"{v:.1%}" for v in result_df[metric]],
            textposition="outside",
            textfont={"size": 11}
        ))
    
    fig.update_layout(
        title="Taxas de Erro por Modelo",
        barmode="group",
        yaxis_tickformat=".0%",
        yaxis_range=[0, max(result_df["False Positive Rate"].max(), 
                          result_df["False Negative Rate"].max()) * 1.3],
        height=350,
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig
