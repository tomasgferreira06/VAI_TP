"""
Gráficos avançados: Parallel Coordinates, Radar Chart e Sunburst.
"""
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from src.config.settings import COLORS, MODEL_COLORS, MODEL_NAMES
from src.utils.helpers import hex_to_rgba


def create_parallel_coordinates(eval_df: pd.DataFrame, threshold: float = 0.5) -> go.Figure:
    """
    Parallel Coordinates para comparar múltiplas métricas entre modelos.
    
    Args:
        eval_df: DataFrame de avaliação
        threshold: Limiar de decisão
        
    Returns:
        Figura Plotly
    """
    metrics_data = []
    
    for model_name in ["logreg", "rf"]:
        model_df = eval_df[eval_df["model"] == model_name].copy()
        model_df["y_pred"] = (model_df["y_proba"] >= threshold).astype(int)
        
        metrics_data.append({
            "Model": MODEL_NAMES.get(model_name, model_name),
            "model_id": 0 if model_name == "logreg" else 1,
            "Accuracy": accuracy_score(model_df["y_true"], model_df["y_pred"]),
            "Precision": precision_score(model_df["y_true"], model_df["y_pred"], zero_division=0),
            "Recall": recall_score(model_df["y_true"], model_df["y_pred"], zero_division=0),
            "F1-Score": f1_score(model_df["y_true"], model_df["y_pred"], zero_division=0),
            "ROC-AUC": roc_auc_score(model_df["y_true"], model_df["y_proba"]),
            "Balanced Acc": (recall_score(model_df["y_true"], model_df["y_pred"], pos_label=1) + 
                           recall_score(model_df["y_true"], model_df["y_pred"], pos_label=0)) / 2
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df_metrics["model_id"],
                colorscale=[[0, MODEL_COLORS["logreg"]], [1, MODEL_COLORS["rf"]]],
                showscale=False
            ),
            dimensions=[
                dict(
                    range=[0, 1],
                    label="Accuracy",
                    values=df_metrics["Accuracy"],
                    tickvals=[0, 0.5, 1],
                    ticktext=["0%", "50%", "100%"]
                ),
                dict(
                    range=[0, 1],
                    label="Precision",
                    values=df_metrics["Precision"],
                    tickvals=[0, 0.5, 1],
                    ticktext=["0%", "50%", "100%"]
                ),
                dict(
                    range=[0, 1],
                    label="Recall",
                    values=df_metrics["Recall"],
                    tickvals=[0, 0.5, 1],
                    ticktext=["0%", "50%", "100%"]
                ),
                dict(
                    range=[0, 1],
                    label="F1-Score",
                    values=df_metrics["F1-Score"],
                    tickvals=[0, 0.5, 1],
                    ticktext=["0%", "50%", "100%"]
                ),
                dict(
                    range=[0, 1],
                    label="ROC-AUC",
                    values=df_metrics["ROC-AUC"],
                    tickvals=[0, 0.5, 1],
                    ticktext=["0%", "50%", "100%"]
                ),
                dict(
                    range=[0, 1],
                    label="Balanced Acc",
                    values=df_metrics["Balanced Acc"],
                    tickvals=[0, 0.5, 1],
                    ticktext=["0%", "50%", "100%"]
                )
            ],
            labelside="top",
            labelfont=dict(size=11, color=COLORS["text_primary"]),
            tickfont=dict(size=10, color=COLORS["text_muted"])
        )
    )
    
    fig.update_layout(
        title=dict(
            text="Parallel Coordinates - Comparacao Multidimensional de Metricas",
            x=0.5,
            xanchor="center"
        ),
        height=400,
        margin=dict(l=80, r=80, t=80, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    # Adicionar legenda manual
    fig.add_annotation(
        x=0.05, y=1.15, xref="paper", yref="paper",
        text=f"<span style='color:{MODEL_COLORS['logreg']}'>-</span> Logistic Regression",
        showarrow=False, font=dict(size=11)
    )
    fig.add_annotation(
        x=0.30, y=1.15, xref="paper", yref="paper",
        text=f"<span style='color:{MODEL_COLORS['rf']}'>-</span> Random Forest",
        showarrow=False, font=dict(size=11)
    )
    
    return fig


def create_radar_chart(eval_df: pd.DataFrame, threshold: float = 0.5) -> go.Figure:
    """
    Radar/Spider Chart para visualização polar das métricas.
    
    Args:
        eval_df: DataFrame de avaliação
        threshold: Limiar de decisão
        
    Returns:
        Figura Plotly
    """
    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "ROC-AUC"]
    
    radar_data = []
    
    for model_name in ["logreg", "rf"]:
        model_df = eval_df[eval_df["model"] == model_name].copy()
        model_df["y_pred"] = (model_df["y_proba"] >= threshold).astype(int)
        
        # Calcular especificidade (TNR)
        tn, fp, fn, tp = confusion_matrix(model_df["y_true"], model_df["y_pred"]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        values = [
            accuracy_score(model_df["y_true"], model_df["y_pred"]),
            precision_score(model_df["y_true"], model_df["y_pred"], zero_division=0),
            recall_score(model_df["y_true"], model_df["y_pred"], zero_division=0),
            f1_score(model_df["y_true"], model_df["y_pred"], zero_division=0),
            specificity,
            roc_auc_score(model_df["y_true"], model_df["y_proba"])
        ]
        
        radar_data.append({
            "model": model_name,
            "values": values + [values[0]]  # Fechar o polígono
        })
    
    categories_closed = categories + [categories[0]]
    
    fig = go.Figure()
    
    for data in radar_data:
        fig.add_trace(go.Scatterpolar(
            r=data["values"],
            theta=categories_closed,
            fill="toself",
            fillcolor=hex_to_rgba(MODEL_COLORS[data["model"]], 0.2),
            line=dict(color=MODEL_COLORS[data["model"]], width=2),
            name=MODEL_NAMES[data["model"]],
            hovertemplate="<b>%{theta}</b><br>Value: %{r:.2%}<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(
            text="Radar Chart - Perfil de Performance dos Modelos",
            x=0.5,
            xanchor="center"
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".0%",
                tickfont=dict(size=9, color=COLORS["text_muted"]),
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"]
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=COLORS["text_primary"]),
                linecolor=COLORS["border"],
                gridcolor=COLORS["border"]
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig


def create_sunburst_errors(eval_df: pd.DataFrame, threshold: float = 0.5, model: str = "logreg") -> go.Figure:
    """
    Sunburst Chart para análise hierárquica de erros.
    
    Args:
        eval_df: DataFrame de avaliação
        threshold: Limiar de decisão
        model: Modelo para análise
        
    Returns:
        Figura Plotly
    """
    model_df = eval_df[eval_df["model"] == model].copy()
    model_df["y_pred"] = (model_df["y_proba"] >= threshold).astype(int)
    
    # Criar dataframe com predições
    error_df = model_df.copy()
    error_df["error_type"] = "Correct"
    error_df.loc[(error_df["y_true"] == 1) & (error_df["y_pred"] == 0), "error_type"] = "False Negative"
    error_df.loc[(error_df["y_true"] == 0) & (error_df["y_pred"] == 1), "error_type"] = "False Positive"
    
    # Agrupar por tipo de erro e sexo
    sunburst_data = error_df.groupby(["error_type", "sex"]).size().reset_index(name="count")
    
    # Preparar dados para sunburst
    labels = ["All Predictions"]
    parents = [""]
    values = [len(error_df)]
    colors = [COLORS["primary"]]
    
    color_map = {
        "Correct": COLORS["success"],
        "False Positive": COLORS["error"],
        "False Negative": COLORS["warning"]
    }
    
    for error_type in ["Correct", "False Positive", "False Negative"]:
        type_data = sunburst_data[sunburst_data["error_type"] == error_type]
        type_total = type_data["count"].sum()
        
        labels.append(error_type)
        parents.append("All Predictions")
        values.append(type_total)
        colors.append(color_map.get(error_type, COLORS["primary"]))
        
        for _, row in type_data.iterrows():
            labels.append(f"{row['sex']}")
            parents.append(error_type)
            values.append(row["count"])
            colors.append(hex_to_rgba(color_map.get(error_type, COLORS["primary"]), 0.7))
    
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
        textfont=dict(size=11),
        insidetextorientation="radial"
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Sunburst - Distribuicao Hierarquica de Erros ({MODEL_NAMES[model]})",
            x=0.5,
            xanchor="center"
        ),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig
