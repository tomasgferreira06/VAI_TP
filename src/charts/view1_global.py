"""
Gráficos para a View 1: Comparação Global.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.config.settings import COLORS, MODEL_COLORS, MODEL_NAMES, SENSITIVE_COLUMNS, CHART_LEGEND_CONFIG
from src.utils.helpers import hex_to_rgba
from src.models.training import global_metrics, recompute_with_threshold


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES DO GRÁFICO DE MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_EXPLANATIONS = {
    "accuracy": "Accuracy shows how often the model predicts correctly. It can be misleading on imbalanced datasets.",
    "precision": "Precision shows how many predicted positives are actually correct. Higher precision means fewer false positives.",
    "recall": "Recall shows how many actual positives were found. Higher recall means fewer false negatives.",
    "f1": "F1-Score combines Precision and Recall into a single score, balancing both."
}

DECISION_MODE_CONFIG = {
    "balanced": {
        "highlight": ["accuracy", "f1"],
        "description": "Balanced view across all metrics"
    },
    "precision": {
        "highlight": ["precision", "accuracy"],
        "description": "Focus on reducing false positives"
    },
    "recall": {
        "highlight": ["recall", "f1"],
        "description": "Focus on reducing false negatives"
    }
}



def compute_subgroup_metrics(df: pd.DataFrame, subgroup: str = "global") -> pd.DataFrame:
    """
    Calcula métricas para um subgrupo específico.
    
    Args:
        df: DataFrame de avaliação
        subgroup: "global", "Male", "Female", "White", "Non-White", "sex", ou "race"
        
    Returns:
        DataFrame com métricas por modelo (e por grupo se sex/race)
    """
    if subgroup == "global":
        filtered_df = df
    elif subgroup in ["Male", "Female"]:
        filtered_df = df[df["sex"] == subgroup]
    elif subgroup == "White":
        filtered_df = df[df["race"] == "White"]
    elif subgroup == "Non-White":
        filtered_df = df[df["race"] != "White"]
    elif subgroup in ["sex", "race"]:
        # Retornar métricas para ambos os grupos
        filtered_df = df
    else:
        filtered_df = df
    
    return filtered_df.groupby("model").apply(global_metrics).reset_index()


def detect_fairness_disparity(df: pd.DataFrame, threshold: float = 0.05) -> dict:
    """
    Detecta disparidades de fairness entre subgrupos.
    
    Args:
        df: DataFrame de avaliação
        threshold: Limiar para considerar disparidade significativa
        
    Returns:
        Dicionário com alertas de disparidade por métrica
    """
    alerts = {}
    metrics = ["accuracy", "precision", "recall", "f1"]
    
    for metric in metrics:
        max_gap = 0
        groups_with_gap = []
        
        # Verificar por sexo
        for sex in df["sex"].unique():
            sex_df = df[df["sex"] == sex]
            if len(sex_df) > 0:
                for model in df["model"].unique():
                    model_df = sex_df[sex_df["model"] == model]
                    global_df = df[df["model"] == model]
                    if len(model_df) > 0 and len(global_df) > 0:
                        model_metrics = global_metrics(model_df)
                        global_model_metrics = global_metrics(global_df)
                        gap = abs(model_metrics[metric] - global_model_metrics[metric])
                        if gap > max_gap:
                            max_gap = gap
                            groups_with_gap = [sex]
        
        if max_gap > threshold:
            alerts[metric] = {
                "gap": max_gap,
                "groups": groups_with_gap
            }
    
    return alerts


def create_metrics_comparison_chart(
    df: pd.DataFrame, 
    threshold: float = 0.5,
    display_mode: str = "absolute",
    subgroup: str = "global",
    decision_mode: str = "balanced"
) -> go.Figure:
    """
    Gráfico de barras comparando métricas entre modelos com funcionalidades avançadas.
    Suporta comparação entre subgrupos quando subgroup é "sex" ou "race".
    
    Args:
        df: DataFrame de avaliação
        threshold: Limiar de decisão
        display_mode: "absolute" para valores ou "relative" para diferenças
        subgroup: "global", "sex", ou "race" - define o modo de análise
        decision_mode: Modo de decisão ("balanced", "precision", "recall")
        
    Returns:
        Figura Plotly
    """
    df_thresh = recompute_with_threshold(df, threshold)
    
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    # Configuração do modo de decisão
    highlighted_metrics = DECISION_MODE_CONFIG.get(decision_mode, {}).get("highlight", metrics)
    mode_label = DECISION_MODE_CONFIG.get(decision_mode, {}).get("description", "")
    
    fig = go.Figure()
    
    # === MODO COMPARAÇÃO POR SUBGRUPO (sex ou race) ===
    if subgroup in ["sex", "race"]:
        # Definir grupos a comparar
        if subgroup == "sex":
            groups = [("Male", df_thresh[df_thresh["sex"] == "Male"]),
                     ("Female", df_thresh[df_thresh["sex"] == "Female"])]
            group_colors = {"Male": "#4A90D9", "Female": "#E91E8C"}  # Azul e Rosa
        else:  # race
            groups = [("White", df_thresh[df_thresh["race"] == "White"]),
                     ("Non-White", df_thresh[df_thresh["race"] != "White"])]
            group_colors = {"White": "#8B5CF6", "Non-White": "#F59E0B"}  # Roxo e Laranja
        
        models = df_thresh["model"].unique()
        
        # Calcular métricas para cada combinação modelo x grupo
        all_metrics = {}
        for model in models:
            all_metrics[model] = {}
            for group_name, group_df in groups:
                model_group_df = group_df[group_df["model"] == model]
                if len(model_group_df) > 0:
                    all_metrics[model][group_name] = global_metrics(model_group_df)
        
        # Criar barras agrupadas: para cada métrica, mostrar grupos lado a lado por modelo
        # Layout: Metric -> (LR-Group1, LR-Group2, RF-Group1, RF-Group2)
        
        for model in models:
            model_name = MODEL_NAMES.get(model, model)
            base_color = MODEL_COLORS.get(model, COLORS["primary"])
            
            for group_name, _ in groups:
                if group_name not in all_metrics[model]:
                    continue
                    
                group_metrics = all_metrics[model][group_name]
                values = [group_metrics[m] for m in metrics]
                
                # Determinar opacidade baseada no decision mode
                opacities = [1.0 if m in highlighted_metrics else 0.65 for m in metrics]
                
                # Criar hover texts
                hover_texts = []
                for i, m in enumerate(metrics):
                    # Calcular diferença entre grupos para este modelo
                    other_group = [g for g, _ in groups if g != group_name][0]
                    if other_group in all_metrics[model]:
                        other_val = all_metrics[model][other_group][m]
                        diff = values[i] - other_val
                        diff_text = f"+{diff:.1%}" if diff > 0 else f"{diff:.1%}"
                        gap_info = f"<br><span style='color:{COLORS['accent']}'>{diff_text} vs {other_group}</span>"
                    else:
                        gap_info = ""
                    
                    tooltip = f"<b>{metric_labels[i]}</b><br>"
                    tooltip += f"Model: {model_name}<br>"
                    tooltip += f"Group: {group_name}<br>"
                    tooltip += f"Value: {values[i]:.2%}{gap_info}<br>"
                    tooltip += f"<br><i style='color:{COLORS['text_muted']}'>{METRIC_EXPLANATIONS.get(m, '')}</i>"
                    hover_texts.append(tooltip)
                
                # Usar padrão de cor: cor do modelo com intensidade variada por grupo
                # Primeiro grupo: mais escuro, segundo grupo: mais claro
                if group_name == groups[0][0]:
                    bar_color = base_color
                    pattern_shape = ""
                else:
                    bar_color = hex_to_rgba(base_color, 0.6)
                    pattern_shape = "/"
                
                fig.add_trace(go.Bar(
                    name=f"{model_name} ({group_name})",
                    x=metric_labels,
                    y=values,
                    marker=dict(
                        color=bar_color,
                        opacity=opacities,
                        pattern_shape=pattern_shape
                    ),
                    text=[f"{v:.1%}" for v in values],
                    textposition="outside",
                    textfont={"size": 10, "color": COLORS["text_secondary"]},
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_texts
                ))
        
        # Título para modo comparação
        group_label = "Sex (Male vs Female)" if subgroup == "sex" else "Race (White vs Non-White)"
        title_text = f"Metric Comparison by {group_label}"
        
    # === MODO GLOBAL (comportamento original) ===
    else:
        summary = compute_subgroup_metrics(df_thresh, subgroup)
        
        # Obter valores por modelo
        model_values = {}
        for model in summary["model"].unique():
            model_data = summary[summary["model"] == model]
            model_values[model] = {m: model_data[m].values[0] for m in metrics}
        
        # Identificar vencedor por métrica
        winners = {}
        for m in metrics:
            best_model = max(model_values.keys(), key=lambda x: model_values[x][m])
            winners[m] = best_model
        
        models = list(summary["model"].unique())
        
        for model in models:
            values = [model_values[model][m] for m in metrics]
            other_model = [m for m in models if m != model][0] if len(models) > 1 else model
            
            # Calcular diferenças percentuais
            diffs = []
            for m in metrics:
                if display_mode == "relative" and other_model != model:
                    diff = model_values[model][m] - model_values[other_model][m]
                    diffs.append(diff)
                else:
                    diffs.append(model_values[model][m])
            
            # Criar textos para as barras
            if display_mode == "relative":
                bar_texts = [f"{'+' if d > 0 else ''}{d:.1%}" for d in diffs]
                display_values = diffs
            else:
                bar_texts = [f"{v:.1%}" for v in values]
                display_values = values
            
            # Configurar cores e estilos por barra
            marker_colors = []
            marker_line_widths = []
            marker_line_colors = []
            opacities = []
            
            for i, m in enumerate(metrics):
                base_color = MODEL_COLORS.get(model, COLORS["primary"])
                is_winner = winners[m] == model
                is_highlighted = m in highlighted_metrics
                
                # Opacidade baseada no modo de decisão
                if is_highlighted:
                    opacities.append(1.0)
                else:
                    opacities.append(0.6)
                
                # Highlight do vencedor com outline
                if is_winner:
                    marker_colors.append(base_color)
                    marker_line_widths.append(3)
                    marker_line_colors.append(COLORS["text_primary"])
                else:
                    marker_colors.append(base_color)
                    marker_line_widths.append(0)
                    marker_line_colors.append("rgba(0,0,0,0)")
            
            # Tooltips avançados
            hover_texts = []
            for i, m in enumerate(metrics):
                other_value = model_values[other_model][m]
                diff = model_values[model][m] - other_value
                diff_text = f"+{diff:.1%}" if diff > 0 else f"{diff:.1%}"
                
                tooltip = f"<b>{metric_labels[i]}</b><br>"
                tooltip += f"Model: {MODEL_NAMES.get(model, model)}<br>"
                tooltip += f"Value: {values[i]:.2%}<br>"
                tooltip += f"<span style='color:{COLORS['accent']}'>{diff_text} vs {MODEL_NAMES.get(other_model, other_model)}</span><br>"
                tooltip += f"<br><i style='color:{COLORS['text_muted']}'>{METRIC_EXPLANATIONS.get(m, '')}</i>"
                
                hover_texts.append(tooltip)
            
            fig.add_trace(go.Bar(
                name=MODEL_NAMES.get(model, model),
                x=metric_labels,
                y=display_values if display_mode == "absolute" else [abs(d) for d in diffs],
                marker=dict(
                    color=marker_colors,
                    line=dict(
                        width=marker_line_widths,
                        color=marker_line_colors
                    ),
                    opacity=opacities
                ),
                text=bar_texts,
                textposition="outside",
                textfont={"size": 11, "color": COLORS["text_secondary"]},
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts
            ))
        
        # Título para modo global
        title_suffix = ""
        if display_mode == "relative":
            title_suffix = " - Relative Difference"
        title_text = f"Metric Comparison by Model{title_suffix}"
    
    # Título com subtitle integrado de forma clean
    full_title = f"{title_text}<span style='font-size:11px; color:{COLORS['text_muted']}'> — {mode_label}</span>"
    
    fig.update_layout(
        title=dict(
            text=full_title,
            font=dict(size=16)
        ),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        yaxis_range=[0, 1.15] if display_mode == "absolute" else [-0.15, 0.15],
        yaxis_tickformat=".0%",
        showlegend=True,
        legend=CHART_LEGEND_CONFIG,
        height=400,
        margin=dict(b=100),  # Espaço extra para a legenda no fundo
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig


def get_fairness_warnings(df: pd.DataFrame, threshold: float = 0.5) -> list:
    """
    Retorna lista de warnings de fairness para exibir na UI.
    
    Args:
        df: DataFrame de avaliação
        threshold: Limiar de decisão
        
    Returns:
        Lista de dicionários com alertas
    """
    df_thresh = recompute_with_threshold(df, threshold)
    alerts = detect_fairness_disparity(df_thresh)
    
    warnings = []
    for metric, data in alerts.items():
        warnings.append({
            "metric": metric,
            "gap": data["gap"],
            "groups": data["groups"],
            "message": f"Potential disparity detected in {metric.title()}: {data['gap']:.1%} gap"
        })
    
    return warnings


def create_roc_curves(
    df: pd.DataFrame, 
    analysis_focus: str = "global",
    model_focus: str = "both"
) -> go.Figure:
    """
    Curvas ROC com suporte a análise por subgrupo (fairness audit).
    
    Args:
        df: DataFrame de avaliação
        analysis_focus: "global", "sex", ou "race" - define agrupamento
        model_focus: "logreg", "rf", ou "both" - modelos a mostrar
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    warnings = []
    
    # Determinar modelos ativos
    if model_focus == "logreg":
        active_models = ["logreg"]
    elif model_focus == "rf":
        active_models = ["rf"]
    else:
        active_models = ["logreg", "rf"]
    
    # Filtrar para modelos existentes no dataframe
    active_models = [m for m in active_models if m in df["model"].unique()]
    
    # Configurar subgrupos baseado em analysis_focus
    if analysis_focus == "sex":
        subgroups = [
            ("Male", df["sex"] == "Male", "solid"),
            ("Female", df["sex"] == "Female", "dash")
        ]
        title_text = "ROC Curves by Sex"
        subtitle_text = "Discriminative performance (TPR vs FPR) across sex groups"
    elif analysis_focus == "race":
        subgroups = [
            ("White", df["race"] == "White", "solid"),
            ("Non-White", df["race"] != "White", "dash")
        ]
        title_text = "ROC Curves by Race"
        subtitle_text = "Discriminative performance (TPR vs FPR) across race groups"
    else:  # global
        subgroups = [("Global", pd.Series([True] * len(df), index=df.index), "solid")]
        title_text = "ROC Curves"
        subtitle_text = "Receiver Operating Characteristic — threshold-independent"
    
    # Linha diagonal (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line={"dash": "dash", "color": COLORS["text_muted"], "width": 1},
        name="Random (AUC = 0.5)",
        hoverinfo="skip"
    ))
    
    # Anotação na diagonal
    fig.add_annotation(
        x=0.65, y=0.35,
        text="Random Classifier<br>(AUC = 0.5)",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["text_muted"],
        ax=-30, ay=-30,
        font=dict(size=9, color=COLORS["text_muted"])
    )
    
    # Plotar curvas para cada modelo e subgrupo
    for model in active_models:
        model_df = df[df["model"] == model]
        model_name = MODEL_NAMES.get(model, model)
        model_color = MODEL_COLORS.get(model, COLORS["primary"])
        
        for subgroup_name, subgroup_mask, line_dash in subgroups:
            # Aplicar máscara de subgrupo ao modelo
            subgroup_df = model_df[subgroup_mask[model_df.index]]
            
            if len(subgroup_df) < 10:
                warnings.append(f"ROC for {model_name} ({subgroup_name}) unavailable (insufficient data)")
                continue
            
            y_true = subgroup_df["y_true"].values
            y_proba = subgroup_df["y_proba"].values
            
            # Verificar se há ambas as classes
            n_pos = np.sum(y_true)
            n_neg = len(y_true) - n_pos
            
            if n_pos == 0 or n_neg == 0:
                warnings.append(f"ROC for {model_name} ({subgroup_name}) unavailable (insufficient class variety)")
                continue
            
            # Calcular ROC
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Criar hover texts
            hover_texts = []
            for i, (f, t, thresh) in enumerate(zip(fpr, tpr, thresholds)):
                y_pred = (y_proba >= thresh).astype(int)
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                tooltip = f"<b>{model_name}</b>"
                if subgroup_name != "Global":
                    tooltip += f" ({subgroup_name})"
                tooltip += f"<br>Threshold: {thresh:.3f}<br>"
                tooltip += f"TPR (Recall): {t:.3f}<br>"
                tooltip += f"FPR: {f:.3f}<br>"
                tooltip += f"Precision: {precision:.3f}<br>"
                tooltip += f"AUC: {roc_auc:.3f}"
                
                hover_texts.append(tooltip)
            
            # Construir nome para legenda
            if subgroup_name == "Global":
                legend_name = f"{model_name} (AUC = {roc_auc:.3f})"
            else:
                legend_name = f"{model_name} ({subgroup_name}) — AUC: {roc_auc:.3f}"
            
            # Determinar se usar fill (apenas em modo global)
            use_fill = analysis_focus == "global"
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=legend_name,
                line={
                    "color": model_color, 
                    "width": 2.5,
                    "dash": line_dash if analysis_focus != "global" else "solid"
                },
                fill="tozeroy" if use_fill else None,
                fillcolor=hex_to_rgba(model_color, 0.12) if use_fill else None,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts
            ))
    
    # Configurar layout
    annotations = []
    
    # Adicionar warnings se existirem
    if warnings:
        warning_text = "<br>".join(warnings)
        annotations.append(dict(
            x=0.02, y=0.02,
            xref="paper", yref="paper",
            text=f"<span style='color:{COLORS['warning']}'>⚠ {warning_text}</span>",
            showarrow=False,
            font=dict(size=8, color=COLORS["warning"]),
            xanchor="left", yanchor="bottom",
            bgcolor="rgba(0,0,0,0.5)"
        ))
    
    # Título com subtitle integrado de forma clean
    full_title = f"{title_text}<span style='font-size:11px; color:{COLORS['text_muted']}'> — {subtitle_text}</span>"
    
    fig.update_layout(
        title=dict(text=full_title, font=dict(size=16)),
        xaxis=dict(title=dict(text="False Positive Rate", standoff=6)),
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
        height=400,
        legend=CHART_LEGEND_CONFIG,
        margin=dict(b=60),  # Espaço extra para a legenda no fundo
        transition=dict(duration=500, easing="cubic-in-out"),
        annotations=annotations
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
        legend=CHART_LEGEND_CONFIG,
        margin=dict(b=100),  # Espaço extra para a legenda no fundo
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION PLOT AVANÇADO
# ═══════════════════════════════════════════════════════════════════════════════

CALIBRATION_DECISION_CONFIG = {
    "balanced": {
        "range": [0, 1],
        "description": "Análise equilibrada de toda a gama de probabilidades"
    },
    "precision": {
        "range": [0.6, 1.0],
        "description": "Foco em alta probabilidade (minimizar FP)"
    },
    "recall": {
        "range": [0.3, 0.7],
        "description": "Foco em probabilidades médias (capturar mais positivos)"
    }
}


def compute_calibration_bins(
    df: pd.DataFrame,
    n_bins: int = 10,
    subgroup: str = "global",
    subgroup_value: str = None
) -> dict:
    """
    Calcula bins de calibração com métricas detalhadas.
    
    Args:
        df: DataFrame de avaliação
        n_bins: Número de bins (5, 10, ou 20)
        subgroup: Modo de subgrupo ("global", "sex", "race")
        subgroup_value: Valor específico do subgrupo (e.g., "Male", "Female")
        
    Returns:
        Dicionário com bins por modelo
    """
    # Filtrar por subgrupo
    if subgroup == "global" or subgroup_value is None:
        filtered_df = df
    elif subgroup == "sex":
        filtered_df = df[df["sex"] == subgroup_value]
    elif subgroup == "race":
        if subgroup_value == "White":
            filtered_df = df[df["race"] == "White"]
        else:
            filtered_df = df[df["race"] != "White"]
    else:
        filtered_df = df
    
    results = {}
    
    for model in filtered_df["model"].unique():
        model_df = filtered_df[filtered_df["model"] == model]
        
        if len(model_df) < n_bins:
            continue
            
        y_true = model_df["y_true"].values
        y_proba = model_df["y_proba"].values
        
        # Criar bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bin_edges[1:-1])
        
        bins_data = []
        for i in range(n_bins):
            mask = bin_indices == i
            bin_count = np.sum(mask)
            
            if bin_count > 0:
                bin_center = np.mean(y_proba[mask])
                frac_positives = np.mean(y_true[mask])
                calibration_error = frac_positives - bin_center
                abs_error = abs(calibration_error)
                
                bins_data.append({
                    "bin_center": bin_center,
                    "frac_positives": frac_positives,
                    "bin_count": int(bin_count),
                    "calibration_error": calibration_error,
                    "abs_error": abs_error,
                    "low_support": bin_count < 30
                })
        
        # Calcular Brier Score
        brier = brier_score_loss(y_true, y_proba)
        
        results[model] = {
            "bins": bins_data,
            "brier_score": brier,
            "n_samples": len(model_df),
            "subgroup": subgroup_value if subgroup != "global" else "Global"
        }
    
    return results


def generate_calibration_insight(
    calibration_data: dict,
    threshold: float,
    subgroup: str,
    error_threshold: float
) -> str:
    """
    Gera insight automático sobre a calibração.
    
    Args:
        calibration_data: Dados de calibração por modelo
        threshold: Threshold atual
        subgroup: Modo de subgrupo
        error_threshold: Limiar de erro
        
    Returns:
        String com insight
    """
    insights = []
    
    for model, data in calibration_data.items():
        model_name = MODEL_NAMES.get(model, model)
        bins = data["bins"]
        brier = data["brier_score"]
        
        if not bins:
            continue
        
        # Encontrar bins com maior erro
        high_error_bins = [b for b in bins if b["abs_error"] > error_threshold]
        
        # Classificar calibração geral
        avg_abs_error = np.mean([b["abs_error"] for b in bins])
        
        if avg_abs_error < 0.03:
            calibration_quality = "bem calibrado"
        elif avg_abs_error < 0.07:
            calibration_quality = "razoavelmente calibrado"
        else:
            calibration_quality = "com problemas de calibração"
        
        insight = f"{model_name} está {calibration_quality} (Brier={brier:.3f})"
        
        # Identificar regiões problemáticas
        if high_error_bins:
            # Determinar se overconfident ou underconfident
            overconfident_bins = [b for b in high_error_bins if b["calibration_error"] < 0]
            underconfident_bins = [b for b in high_error_bins if b["calibration_error"] > 0]
            
            if len(overconfident_bins) > len(underconfident_bins):
                region = "alta" if np.mean([b["bin_center"] for b in overconfident_bins]) > 0.5 else "baixa"
                insight += f"; overconfident na região de {region} probabilidade"
            elif underconfident_bins:
                region = "alta" if np.mean([b["bin_center"] for b in underconfident_bins]) > 0.5 else "baixa"
                insight += f"; underconfident na região de {region} probabilidade"
        
        insights.append(insight)
    
    # Adicionar contexto de subgrupo
    if subgroup != "global":
        insights.append(f"(Análise para subgrupo específico - interpretar com cautela)")
    
    return ". ".join(insights) + "."


def create_advanced_calibration_plot(
    df: pd.DataFrame,
    threshold: float = 0.5,
    n_bins: int = 10,
    subgroup: str = "global",
    subgroup_value: str = None,
    decision_mode: str = "balanced",
    error_threshold: float = 0.07,
    selected_models: list = None
) -> tuple:
    """
    Calibration Plot avançado com todas as funcionalidades.
    
    Args:
        df: DataFrame de avaliação
        threshold: Limiar de decisão (para marcador vertical)
        n_bins: Número de bins (5, 10, ou 20)
        subgroup: Modo de subgrupo ("global", "sex", "race")
        subgroup_value: Valor específico do subgrupo
        decision_mode: Modo de decisão ("balanced", "precision", "recall")
        error_threshold: Limiar para destacar erros de calibração
        selected_models: Lista de modelos selecionados
        
    Returns:
        Tuple (figura Plotly, texto de insight)
    """
    if selected_models is None:
        selected_models = df["model"].unique().tolist()
    
    # Calcular dados de calibração
    calibration_data = compute_calibration_bins(df, n_bins, subgroup, subgroup_value)
    
    # Gerar insight
    insight_text = generate_calibration_insight(
        calibration_data, threshold, subgroup, error_threshold
    )
    
    fig = go.Figure()
    
    # Configuração do modo de decisão
    decision_config = CALIBRATION_DECISION_CONFIG.get(decision_mode, CALIBRATION_DECISION_CONFIG["balanced"])
    emphasis_range = decision_config["range"]
    
    # Adicionar banda de ênfase do decision mode
    if decision_mode != "balanced":
        fig.add_vrect(
            x0=emphasis_range[0], x1=emphasis_range[1],
            fillcolor=COLORS["primary"],
            opacity=0.08,
            line_width=0,
            annotation_text=decision_config["description"],
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color=COLORS["text_muted"]
        )
    
    # Linha diagonal (perfect calibration)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color=COLORS["text_muted"], width=1.5),
        name="Calibração Perfeita",
        hoverinfo="skip"
    ))
    
    # Plotar cada modelo
    for model in selected_models:
        if model not in calibration_data:
            continue
            
        data = calibration_data[model]
        bins = data["bins"]
        
        if not bins:
            continue
        
        x_values = [b["bin_center"] for b in bins]
        y_values = [b["frac_positives"] for b in bins]
        
        # Configurar marcadores baseado em erros
        marker_sizes = []
        marker_symbols = []
        marker_line_widths = []
        marker_line_colors = []
        opacities = []
        
        for b in bins:
            # Tamanho baseado em erro
            if b["abs_error"] > error_threshold:
                marker_sizes.append(14)
                marker_symbols.append("diamond")
                marker_line_widths.append(2)
                marker_line_colors.append(COLORS["warning"])
            else:
                marker_sizes.append(9)
                marker_symbols.append("circle")
                marker_line_widths.append(0)
                marker_line_colors.append("rgba(0,0,0,0)")
            
            # Opacidade baseada no decision mode
            if decision_mode == "balanced":
                opacities.append(1.0 if not b["low_support"] else 0.5)
            else:
                in_range = emphasis_range[0] <= b["bin_center"] <= emphasis_range[1]
                if in_range:
                    opacities.append(1.0 if not b["low_support"] else 0.6)
                else:
                    opacities.append(0.4 if not b["low_support"] else 0.25)
        
        # Criar hover texts detalhados
        hover_texts = []
        for b in bins:
            error_direction = "Overconfident" if b["calibration_error"] < 0 else "Underconfident"
            support_warning = " (Suporte baixo)" if b["low_support"] else ""
            
            tooltip = f"<b>{MODEL_NAMES.get(model, model)}</b>{support_warning}<br>"
            if subgroup != "global" and subgroup_value:
                tooltip += f"Grupo: {subgroup_value}<br>"
            tooltip += f"Prob. Prevista (média): {b['bin_center']:.2f}<br>"
            tooltip += f"Fração Observada: {b['frac_positives']:.2f}<br>"
            tooltip += f"Erro (obs - pred): {b['calibration_error']:+.3f}<br>"
            tooltip += f"|Erro|: {b['abs_error']:.3f}<br>"
            tooltip += f"Direção: {error_direction}<br>"
            tooltip += f"Suporte: {b['bin_count']} amostras"
            
            hover_texts.append(tooltip)
        
        # Linha de calibração
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values,
            mode="lines+markers",
            name=f"{MODEL_NAMES.get(model, model)} (Brier={data['brier_score']:.3f})",
            line=dict(color=MODEL_COLORS.get(model, COLORS["primary"]), width=2.5),
            marker=dict(
                size=marker_sizes,
                symbol=marker_symbols,
                color=MODEL_COLORS.get(model, COLORS["primary"]),
                line=dict(width=marker_line_widths, color=marker_line_colors),
                opacity=opacities
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts
        ))
    
    # Adicionar marcador vertical de threshold (LINKED VIEW)
    fig.add_vline(
        x=threshold,
        line_width=2,
        line_dash="dot",
        line_color=COLORS["accent"],
        annotation_text=f"Threshold: {threshold:.2f}",
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color=COLORS["accent"]
    )
    
    # Título dinâmico
    title_suffix = ""
    if subgroup != "global" and subgroup_value:
        title_suffix = f" ({subgroup_value})"
    
    # Nota de rodapé integrada no título

    full_title = f"Calibration Plot (Reliability Diagram){title_suffix}<span style='font-size:11px; color:{COLORS['text_muted']}'></span>"
    
    fig.update_layout(
        title=dict(
            text=full_title,
            font=dict(size=16)
        ),
        xaxis_title="Probabilidade Prevista (média do bin)",
        yaxis_title="Fração de Positivos Observados",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.05],
        height=420,
        legend=CHART_LEGEND_CONFIG,
        margin=dict(b=100),  # Espaço extra para a legenda no fundo
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    return fig, insight_text


def create_calibration_subgroup_comparison(
    df: pd.DataFrame,
    threshold: float = 0.5,
    n_bins: int = 10,
    subgroup_type: str = "sex",
    decision_mode: str = "balanced",
    error_threshold: float = 0.07,
    selected_models: list = None
) -> tuple:
    """
    Cria comparação de calibração entre subgrupos (overlay).
    
    Args:
        df: DataFrame de avaliação
        threshold: Limiar de decisão
        n_bins: Número de bins
        subgroup_type: "sex" ou "race"
        decision_mode: Modo de decisão
        error_threshold: Limiar de erro
        selected_models: Lista de modelos
        
    Returns:
        Tuple (figura Plotly, texto de insight)
    """
    if selected_models is None:
        selected_models = df["model"].unique().tolist()[:2]  # Limitar a 2 modelos
    
    # Definir grupos
    if subgroup_type == "sex":
        groups = ["Male", "Female"]
        group_styles = {"Male": "solid", "Female": "dash"}
    else:
        groups = ["White", "Non-White"]
        group_styles = {"White": "solid", "Non-White": "dash"}
    
    fig = go.Figure()
    
    # Configuração do modo de decisão
    decision_config = CALIBRATION_DECISION_CONFIG.get(decision_mode, CALIBRATION_DECISION_CONFIG["balanced"])
    emphasis_range = decision_config["range"]
    
    # Adicionar banda de ênfase
    if decision_mode != "balanced":
        fig.add_vrect(
            x0=emphasis_range[0], x1=emphasis_range[1],
            fillcolor=COLORS["primary"],
            opacity=0.08,
            line_width=0
        )
    
    # Linha diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color=COLORS["text_muted"], width=1.5),
        name="Calibração Perfeita",
        hoverinfo="skip"
    ))
    
    insights = []
    
    for model in selected_models:
        for group in groups:
            # Calcular calibração para este grupo
            if subgroup_type == "sex":
                group_df = df[(df["model"] == model) & (df["sex"] == group)]
            else:
                if group == "White":
                    group_df = df[(df["model"] == model) & (df["race"] == "White")]
                else:
                    group_df = df[(df["model"] == model) & (df["race"] != "White")]
            
            if len(group_df) < n_bins:
                continue
            
            y_true = group_df["y_true"].values
            y_proba = group_df["y_proba"].values
            
            # Calcular bins
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_proba, bin_edges[1:-1])
            
            x_values = []
            y_values = []
            hover_texts = []
            
            for i in range(n_bins):
                mask = bin_indices == i
                bin_count = np.sum(mask)
                
                if bin_count > 0:
                    bin_center = np.mean(y_proba[mask])
                    frac_positives = np.mean(y_true[mask])
                    error = frac_positives - bin_center
                    
                    x_values.append(bin_center)
                    y_values.append(frac_positives)
                    
                    tooltip = f"<b>{MODEL_NAMES.get(model, model)} ({group})</b><br>"
                    tooltip += f"Prob. Prevista: {bin_center:.2f}<br>"
                    tooltip += f"Fração Observada: {frac_positives:.2f}<br>"
                    tooltip += f"Erro: {error:+.3f}<br>"
                    tooltip += f"Suporte: {bin_count}"
                    hover_texts.append(tooltip)
            
            brier = brier_score_loss(y_true, y_proba)
            
            fig.add_trace(go.Scatter(
                x=x_values, y=y_values,
                mode="lines+markers",
                name=f"{MODEL_NAMES.get(model, model)} ({group})",
                line=dict(
                    color=MODEL_COLORS.get(model, COLORS["primary"]),
                    width=2.5,
                    dash=group_styles[group]
                ),
                marker=dict(size=8),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts
            ))
            
            # Calcular erro médio para insight
            if x_values:
                avg_error = np.mean([abs(y - x) for x, y in zip(x_values, y_values)])
                insights.append(f"{MODEL_NAMES.get(model, model)} ({group}): erro médio {avg_error:.3f}")
    
    # Marcador de threshold
    fig.add_vline(
        x=threshold,
        line_width=2,
        line_dash="dot",
        line_color=COLORS["accent"],
        annotation_text=f"Threshold: {threshold:.2f}",
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color=COLORS["accent"]
    )
    
    fig.update_layout(
        title=dict(
            text=f"Calibração por Subgrupo ({subgroup_type.title()})",
            font=dict(size=16)
        ),
        xaxis_title="Probabilidade Prevista (média do bin)",
        yaxis_title="Fração de Positivos Observados",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.05],
        height=420,
        legend=CHART_LEGEND_CONFIG,
        margin=dict(b=100),  # Espaço extra para a legenda no fundo
        transition=dict(duration=500, easing="cubic-in-out")
    )
    
    insight_text = "Comparação entre subgrupos: " + "; ".join(insights) if insights else "Dados insuficientes para comparação."
    
    return fig, insight_text

