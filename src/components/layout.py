"""
Header e Sidebar do Dashboard.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc

from src.config.settings import COLORS, MODEL_NAMES
from src.components.cards import create_model_badge


def create_header():
    """
    Cria o header premium da aplicação.
    
    Returns:
        html.Div com o header completo
    """
    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.H1("Model Evaluation Dashboard", className="main-title", style={"display": "inline"}),
                            html.I(
                                className="bi bi-info-circle",
                                id="dashboard-info-icon",
                                style={
                                    "fontSize": "1rem",
                                    "color": COLORS["primary_light"],
                                    "cursor": "pointer",
                                    "marginLeft": "0.5rem",
                                    "verticalAlign": "middle"
                                }
                            ),
                            dbc.Tooltip(
                                "Adult Income Dataset (UCI) — Binary classification task (>50K vs ≤50K) based on "
                                "sociodemographic and professional attributes. The dataset is imbalanced (majority ≤50K), "
                                "so it is important to analyze metrics beyond accuracy and explore the impact of the "
                                "decision threshold on errors and fairness.",
                                target="dashboard-info-icon",
                                placement="bottom"
                            )
                        ], style={"display": "flex", "alignItems": "center"}),
                        html.P(
                            "A Comparative Study of Classification Models Beyond Overall Accuracy",
                            className="subtitle"
                        )
                    ])
                ], md=8),
                dbc.Col([
                    html.Div([
                        html.Div([
                            create_model_badge("logreg"),
                            html.Span(" vs ", style={"color": COLORS["text_muted"], "margin": "0 0.5rem"}),
                            create_model_badge("rf")
                        ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"})
                    ], style={"height": "100%", "display": "flex", "alignItems": "center", "justifyContent": "flex-end"})
                ], md=4)
            ], align="center")
        ], fluid=True)
    ], style={
        "background": f"linear-gradient(135deg, {COLORS['bg_card']} 0%, {COLORS['bg_hover']}66 100%)",
        "borderBottom": f"1px solid {COLORS['border']}33",
        "padding": "1.5rem 1.5rem",
        "marginBottom": "1.5rem"
    })


def create_controls_sidebar(test_samples: int, positive_rate: float):
    """
    Cria a sidebar com controlos globais, botão reset e download.
    
    Args:
        test_samples: Número de amostras de teste
        positive_rate: Taxa de positivos no dataset
        
    Returns:
        html.Div com a sidebar completa
    """
    return html.Div([
        html.Div([
            # Logo/Icon
            html.Div([
                html.Div([
                    html.I(className="bi bi-sliders", style={"fontSize": "1.25rem"})
                ], style={"marginBottom": "0.25rem"}),
                html.Span("Controls", style={
                    "fontSize": "0.75rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.1em"
                })
            ], style={"textAlign": "center", "marginBottom": "1.5rem"}),
            
            html.Div(className="divider"),
            
            # Threshold Slider
            html.Div([
                html.Label("Decision Threshold", className="control-label"),
                html.Div([
                    dcc.Slider(
                        id="threshold-slider",
                        min=0.1,
                        max=0.9,
                        step=0.05,
                        value=0.5,
                        marks={i/10: {"label": f"{i/10:.1f}", "style": {"color": COLORS["text_muted"]}} 
                               for i in range(1, 10, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={"marginTop": "0.5rem"})
            ], style={"marginBottom": "2rem"}),
            
            # Model Selector
            html.Div([
                html.Label("Model Focus", className="control-label"),
                dcc.Dropdown(
                    id="model-selector",
                    options=[
                        {"label": "Logistic Regression", "value": "logreg"},
                        {"label": "Random Forest", "value": "rf"},
                        {"label": "Both", "value": "both"}
                    ],
                    value="both",
                    clearable=False,
                    style={"marginTop": "0.5rem"}
                )
            ], style={"marginBottom": "2rem"}),
            
            # Analysis Focus (unified control for all views)
            html.Div([
                html.Label("Analysis Focus", className="control-label"),
                html.P(
                    "Global or by demographic group",
                    style={"fontSize": "0.7rem", "color": COLORS["text_muted"], "margin": "0.25rem 0 0.5rem 0"}
                ),
                dcc.Dropdown(
                    id="sensitive-selector",
                    options=[
                        {"label": "Global (All Data)", "value": "global"},
                        {"label": "Sex", "value": "sex"},
                        {"label": "Race", "value": "race"}
                    ],
                    value="global",
                    clearable=False,
                    style={"marginTop": "0.5rem"}
                )
            ], style={"marginBottom": "2rem"}),
            
            html.Div(className="divider"),
            
            # Global Decision Mode Control
            html.Div([
                html.Label("Decision Mode", className="control-label"),
                html.P(
                    "Affects metric emphasis across views",
                    style={"fontSize": "0.7rem", "color": COLORS["text_muted"], "margin": "0.25rem 0 0.5rem 0"}
                ),
                html.Div([
                    dbc.Button(
                        [html.I(className="bi bi-sliders", style={"marginRight": "0.35rem"}), "Balanced"],
                        id="btn-global-mode-balanced",
                        color="primary",
                        outline=False,
                        size="sm",
                        style={"width": "100%", "marginBottom": "0.35rem", "textAlign": "left"}
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-bullseye", style={"marginRight": "0.35rem"}), "Precision"],
                        id="btn-global-mode-precision",
                        color="primary",
                        outline=True,
                        size="sm",
                        style={"width": "100%", "marginBottom": "0.35rem", "textAlign": "left"}
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-search", style={"marginRight": "0.35rem"}), "Recall"],
                        id="btn-global-mode-recall",
                        color="primary",
                        outline=True,
                        size="sm",
                        style={"width": "100%", "textAlign": "left"}
                    ),
                ]),
                dcc.Store(id="global-decision-mode-store", data="balanced"),
            ], style={"marginBottom": "2rem"}),
            
            html.Div(className="divider"),
            
            # Botões de Ação: Reset e Download
            html.Div([
                html.Label("Actions", className="control-label"),
                html.Div([
                    # Botão Reset
                    dbc.Button(
                        [html.I(className="bi bi-arrow-counterclockwise", style={"marginRight": "0.5rem"}), "Reset"],
                        id="reset-button",
                        color="secondary",
                        outline=True,
                        size="sm",
                        style={"width": "100%", "marginBottom": "0.5rem"}
                    ),
                    # Botão Download
                    dbc.Button(
                        [html.I(className="bi bi-download", style={"marginRight": "0.5rem"}), "Export Data"],
                        id="download-button",
                        color="primary",
                        outline=True,
                        size="sm",
                        style={"width": "100%"}
                    ),
                    dcc.Download(id="download-data")
                ], style={"marginTop": "0.75rem"})
            ], style={"marginBottom": "2rem"}),
            
            html.Div(className="divider"),
            
            # Quick Stats
            html.Div([
                html.Label("Dataset Info", className="control-label"),
                html.Div([
                    html.Div([
                        html.Span("Test Samples", style={"color": COLORS["text_muted"], "fontSize": "0.8rem"}),
                        html.Div(f"{test_samples:,}", style={"fontWeight": "600", "fontSize": "1.1rem"})
                    ], style={"marginBottom": "0.75rem"}),
                    html.Div([
                        html.Span("Positive Rate", style={"color": COLORS["text_muted"], "fontSize": "0.8rem"}),
                        html.Div(f"{positive_rate:.1%}", style={"fontWeight": "600", "fontSize": "1.1rem"})
                    ]),
                    html.Div([
                        html.Span("Models", style={"color": COLORS["text_muted"], "fontSize": "0.8rem"}),
                        html.Div("LR vs RF", style={"fontWeight": "600", "fontSize": "1.1rem"})
                    ], style={"marginTop": "0.75rem"})
                ], style={"marginTop": "0.75rem"})
            ])
        ], className="dashboard-card", style={"height": "100%"})
    ], style={"position": "sticky", "top": "1rem"})
