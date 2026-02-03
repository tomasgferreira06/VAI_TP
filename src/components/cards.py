"""
Componentes de UI reutilizáveis para o Dashboard.
"""
from dash import html
import dash_bootstrap_components as dbc

from src.config.settings import COLORS, MODEL_NAMES


def create_metric_card(value, label: str, color: str = None, icon: str = None):
    """
    Cria um card de métrica estilizado.
    
    Args:
        value: Valor da métrica (float ou string)
        label: Label da métrica
        color: Cor do valor (opcional)
        icon: Ícone (não implementado)
        
    Returns:
        html.Div com o card
    """
    color = color or COLORS["primary"]
    return html.Div([
        html.Div([
            html.Span(
                f"{value:.1%}" if isinstance(value, float) else str(value), 
                className="metric-value",
                style={"color": color}
            ),
            html.Div(label, className="metric-label")
        ])
    ], className="metric-card")


def create_comparison_metric_card(metric_name: str, logreg_value: float, rf_value: float, label: str):
    """
    Card de métrica com comparação lado a lado de 2 modelos.
    Design premium com indicador de qual modelo é melhor.
    
    Args:
        metric_name: Nome da métrica
        logreg_value: Valor do Logistic Regression
        rf_value: Valor do Random Forest
        label: Label da métrica
        
    Returns:
        html.Div com o card de comparação
    """
    diff = logreg_value - rf_value
    winner = "logreg" if diff > 0.001 else ("rf" if diff < -0.001 else "tie")
    
    return html.Div([
        # Header com nome da métrica
        html.Div(label, style={
            "fontSize": "0.7rem",
            "color": COLORS["text_muted"],
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "marginBottom": "0.75rem",
            "textAlign": "center"
        }),
        
        # Container dos dois valores
        html.Div([
            # Logistic Regression
            html.Div([
                html.Div([
                    html.Span("●", style={
                        "color": COLORS["logreg"],
                        "fontSize": "0.5rem",
                        "marginRight": "0.35rem"
                    }),
                    html.Span("LR", style={
                        "fontSize": "0.65rem",
                        "color": COLORS["text_muted"]
                    })
                ], style={"marginBottom": "0.25rem"}),
                html.Div(f"{logreg_value:.1%}", style={
                    "fontSize": "1.4rem",
                    "fontWeight": "700",
                    "color": COLORS["logreg"] if winner == "logreg" else COLORS["text_secondary"],
                    "lineHeight": "1"
                })
            ], style={"textAlign": "center", "flex": "1"}),
            
            # Separador vertical
            html.Div(style={
                "width": "1px",
                "background": f"linear-gradient(180deg, transparent, {COLORS['border']}44, transparent)",
                "margin": "0 0.75rem"
            }),
            
            # Random Forest
            html.Div([
                html.Div([
                    html.Span("●", style={
                        "color": COLORS["rf"],
                        "fontSize": "0.5rem",
                        "marginRight": "0.35rem"
                    }),
                    html.Span("RF", style={
                        "fontSize": "0.65rem",
                        "color": COLORS["text_muted"]
                    })
                ], style={"marginBottom": "0.25rem"}),
                html.Div(f"{rf_value:.1%}", style={
                    "fontSize": "1.4rem",
                    "fontWeight": "700",
                    "color": COLORS["rf"] if winner == "rf" else COLORS["text_secondary"],
                    "lineHeight": "1"
                })
            ], style={"textAlign": "center", "flex": "1"})
        ], style={"display": "flex", "alignItems": "center", "justifyContent": "center"}),
        
        # Indicador de diferença
        html.Div([
            html.Span(
                f"Δ {abs(diff):.1%}" if winner != "tie" else "=",
                style={
                    "fontSize": "0.7rem",
                    "color": COLORS["logreg"] if winner == "logreg" else (COLORS["rf"] if winner == "rf" else COLORS["text_muted"]),
                    "fontWeight": "500"
                }
            )
        ], style={"textAlign": "center", "marginTop": "0.5rem"})
        
    ], className="metric-card", style={"minHeight": "120px"})


def create_section_header(title: str, subtitle: str = None):
    """
    Cria um cabeçalho de secção.
    
    Args:
        title: Título da secção
        subtitle: Subtítulo opcional
        
    Returns:
        html.Div com o header
    """
    children = [html.H3(title, className="section-title")]
    if subtitle:
        children.append(html.P(subtitle, style={
            "color": COLORS["text_muted"],
            "fontSize": "0.85rem",
            "marginTop": "-0.5rem",
            "marginBottom": "1rem"
        }))
    return html.Div(children)


def create_model_badge(model_name: str):
    """
    Cria badge colorido para identificar modelo.
    
    Args:
        model_name: Nome do modelo (logreg ou rf)
        
    Returns:
        html.Span com o badge
    """
    display_name = MODEL_NAMES.get(model_name, model_name)
    badge_class = f"badge-model badge-{model_name}"
    return html.Span([
        html.Span("●", style={"fontSize": "0.5rem"}),
        display_name
    ], className=badge_class)
