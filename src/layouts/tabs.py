"""
Layouts das tabs do dashboard.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc

from src.config.settings import COLORS
from src.components.cards import create_section_header


def create_tab_global() -> html.Div:
    """Tab de comparação global de modelos."""
    return html.Div([
        # Metrics Cards Row
        html.Div(id="metrics-cards-row", style={"marginBottom": "1.5rem"}),
        
        # Charts Row 1: Métricas + ROC
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id="metrics-comparison-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=6),
            dbc.Col([
                html.Div([
                    dcc.Graph(id="roc-curves-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=6)
        ], style={"marginBottom": "1.5rem"}),
        
        # Charts Row 2: Feature Importance + Calibration Plot
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id="feature-importance-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=6),
            dbc.Col([
                html.Div([
                    dcc.Graph(id="calibration-plot-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=6)
        ])
    ])


def create_tab_tradeoffs() -> html.Div:
    """Tab de análise de trade-offs (precision vs recall)."""
    return html.Div([
        # Explanation Card
        html.Div([
            html.Div([
                html.I(className="bi bi-info-circle", style={"fontSize": "1.1rem", "marginRight": "0.75rem", "color": COLORS["primary_light"]}),
                html.Span(
                    "O threshold de decisão afeta diretamente o trade-off entre Precision e Recall. "
                    "Um threshold mais alto aumenta a Precision (menos FP) mas reduz o Recall (mais FN).",
                    style={"color": COLORS["text_secondary"], "fontSize": "0.9rem"}
                )
            ], style={"display": "flex", "alignItems": "center"})
        ], style={
            "background": f"{COLORS['primary']}11",
            "border": f"1px solid {COLORS['primary']}33",
            "borderRadius": "12px",
            "padding": "1rem 1.25rem",
            "marginBottom": "1.5rem"
        }),
        
        # Row 1: PR Curve + Threshold Analysis
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id="pr-curve-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=6),
            dbc.Col([
                html.Div([
                    dcc.Graph(id="threshold-analysis-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=6)
        ], style={"marginBottom": "1.5rem"}),
        
        # Row 2: Evolução de FP/FN
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id="fp-fn-evolution-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=12)
        ], style={"marginBottom": "1.5rem"}),
        
        # Row 3: Threshold Impact
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id="threshold-impact-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=12)
        ])
    ])


def create_tab_errors() -> html.Div:
    """Tab de análise de erros."""
    return html.Div([
        dbc.Row([
            # Confusion Matrix
            dbc.Col([
                html.Div([
                    dcc.Graph(id="confusion-matrix-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=5),
            
            # Error Rates Comparison
            dbc.Col([
                html.Div([
                    dcc.Graph(id="error-rates-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=7)
        ], style={"marginBottom": "1.5rem"}),
        
        # Error by Feature
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header(
                        "Distribuição de Erros por Atributo",
                        "Identifica onde o modelo comete mais erros"
                    ),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="error-by-sex-chart", config={"displayModeBar": False})
                        ], md=6),
                        dbc.Col([
                            dcc.Graph(id="error-by-race-chart", config={"displayModeBar": False})
                        ], md=6)
                    ])
                ], className="dashboard-card")
            ], lg=12)
        ])
    ])


def create_tab_fairness() -> html.Div:
    """Tab de análise de fairness."""
    return html.Div([
        # Fairness Explanation
        html.Div([
            html.Div([
                html.I(className="bi bi-balance-scale", style={"fontSize": "1.25rem", "marginRight": "0.75rem", "color": COLORS["accent"]}),
                html.Div([
                    html.Span("Fairness Analysis", style={
                        "fontWeight": "600", 
                        "color": COLORS["text_primary"],
                        "display": "block",
                        "marginBottom": "0.25rem"
                    }),
                    html.Span(
                        "Avalia se o modelo trata de forma equitativa diferentes grupos demográficos. "
                        "Disparidades significativas podem indicar bias no modelo.",
                        style={"color": COLORS["text_secondary"], "fontSize": "0.85rem"}
                    )
                ])
            ], style={"display": "flex", "alignItems": "flex-start"})
        ], style={
            "background": f"{COLORS['accent']}11",
            "border": f"1px solid {COLORS['accent']}33",
            "borderRadius": "12px",
            "padding": "1rem 1.25rem",
            "marginBottom": "1.5rem"
        }),
        
        # Main Fairness Charts
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id="fairness-accuracy-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=6),
            dbc.Col([
                html.Div([
                    dcc.Graph(id="fairness-disparity-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=6)
        ], style={"marginBottom": "1.5rem"}),
        
        # Error Rates by Group
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id="fairness-rates-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=12)
        ])
    ])


def create_tab_advanced() -> html.Div:
    """Tab de visualizações avançadas (Parallel Coordinates, Radar, Sunburst)."""
    return html.Div([
        # Explanation Card
        html.Div([
            html.Div([
                html.I(className="bi bi-stars", style={"fontSize": "1.25rem", "marginRight": "0.75rem", "color": COLORS["primary_light"]}),
                html.Div([
                    html.Span("Advanced Visualizations", style={
                        "fontWeight": "600", 
                        "color": COLORS["text_primary"],
                        "display": "block",
                        "marginBottom": "0.25rem"
                    }),
                    html.Span(
                        "Tecnicas avancadas de visualizacao multidimensional para comparacao holistica de modelos. "
                        "Parallel Coordinates para multi-metricas, Radar Chart para perfil visual, e Sunburst para hierarquia de erros.",
                        style={"color": COLORS["text_secondary"], "fontSize": "0.85rem"}
                    )
                ])
            ], style={"display": "flex", "alignItems": "flex-start"})
        ], style={
            "background": f"{COLORS['primary']}11",
            "border": f"1px solid {COLORS['primary']}33",
            "borderRadius": "12px",
            "padding": "1rem 1.25rem",
            "marginBottom": "1.5rem"
        }),
        
        # Row 1: Parallel Coordinates + Radar Chart
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id="parallel-coords-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=6),
            dbc.Col([
                html.Div([
                    dcc.Graph(id="radar-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=6)
        ], style={"marginBottom": "1.5rem"}),
        
        # Row 2: Sunburst
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id="sunburst-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=12)
        ])
    ])
