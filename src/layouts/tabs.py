"""
Layouts das tabs do dashboard.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc

from src.config.settings import COLORS
from src.components.cards import create_section_header


def create_metrics_controls() -> html.Div:
    """Cria os controlos para o gráfico de métricas."""
    return html.Div([
        dbc.Row([
            # Toggle: Absolute vs Relative
            dbc.Col([
                html.Label("Display Mode", style={
                    "fontSize": "0.7rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "0.25rem",
                    "display": "block"
                }),
                dbc.ButtonGroup([
                    dbc.Button(
                        "Absolute",
                        id="btn-display-absolute",
                        color="primary",
                        outline=False,
                        size="sm",
                        className="active"
                    ),
                    dbc.Button(
                        "Relative",
                        id="btn-display-relative",
                        color="primary",
                        outline=True,
                        size="sm"
                    ),
                ], size="sm")
            ], width="auto"),
            
            # Subgroup selector
            dbc.Col([
                html.Label("Subgroup", style={
                    "fontSize": "0.7rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "0.25rem",
                    "display": "block"
                }),
                dcc.Dropdown(
                    id="subgroup-selector",
                    options=[
                        {"label": "Global", "value": "global"},
                        {"label": "Male", "value": "Male"},
                        {"label": "Female", "value": "Female"},
                        {"label": "White", "value": "White"},
                        {"label": "Non-White", "value": "Non-White"},
                    ],
                    value="global",
                    clearable=False,
                    style={"minWidth": "120px", "fontSize": "0.85rem"}
                )
            ], width="auto"),
            
            # Decision Mode
            dbc.Col([
                html.Label("Decision Mode", style={
                    "fontSize": "0.7rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "0.25rem",
                    "display": "block"
                }),
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="bi bi-balance-scale", style={"marginRight": "0.25rem"}), "Balanced"],
                        id="btn-mode-balanced",
                        color="primary",
                        outline=False,
                        size="sm",
                        className="active"
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-bullseye", style={"marginRight": "0.25rem"}), "Precision"],
                        id="btn-mode-precision",
                        color="primary",
                        outline=True,
                        size="sm"
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-search", style={"marginRight": "0.25rem"}), "Recall"],
                        id="btn-mode-recall",
                        color="primary",
                        outline=True,
                        size="sm"
                    ),
                ], size="sm")
            ], width="auto"),
        ], className="g-3 align-items-end", justify="start"),
        
        # Stores para os estados
        dcc.Store(id="display-mode-store", data="absolute"),
        dcc.Store(id="decision-mode-store", data="balanced"),
        
    ], style={
        "background": f"{COLORS['bg_hover']}44",
        "borderRadius": "8px",
        "padding": "0.75rem 1rem",
        "marginBottom": "1rem"
    })


def create_calibration_controls() -> html.Div:
    """Cria os controlos avançados para o Calibration Plot."""
    return html.Div([
        dbc.Row([
            # Bin Granularity
            dbc.Col([
                html.Label("Bins", style={
                    "fontSize": "0.7rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "0.25rem",
                    "display": "block"
                }),
                dbc.ButtonGroup([
                    dbc.Button("5", id="btn-bins-5", color="primary", outline=True, size="sm"),
                    dbc.Button("10", id="btn-bins-10", color="primary", outline=False, size="sm"),
                    dbc.Button("20", id="btn-bins-20", color="primary", outline=True, size="sm"),
                ], size="sm")
            ], width="auto"),
            
            # Subgroup Mode
            dbc.Col([
                html.Label("Subgroup", style={
                    "fontSize": "0.7rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "0.25rem",
                    "display": "block"
                }),
                dcc.Dropdown(
                    id="calib-subgroup-selector",
                    options=[
                        {"label": "Global", "value": "global"},
                        {"label": "By Sex", "value": "sex"},
                        {"label": "By Race", "value": "race"},
                    ],
                    value="global",
                    clearable=False,
                    style={"minWidth": "110px", "fontSize": "0.85rem"}
                )
            ], width="auto"),
            
            # Decision Mode for Calibration
            dbc.Col([
                html.Label("Focus", style={
                    "fontSize": "0.7rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "0.25rem",
                    "display": "block"
                }),
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="bi bi-distribute-horizontal", style={"marginRight": "0.25rem"}), "All"],
                        id="btn-calib-balanced",
                        color="primary",
                        outline=False,
                        size="sm"
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-bullseye", style={"marginRight": "0.25rem"}), "Prec"],
                        id="btn-calib-precision",
                        color="primary",
                        outline=True,
                        size="sm"
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-search", style={"marginRight": "0.25rem"}), "Rec"],
                        id="btn-calib-recall",
                        color="primary",
                        outline=True,
                        size="sm"
                    ),
                ], size="sm")
            ], width="auto"),
            
            # Error Threshold
            dbc.Col([
                html.Label("Error Threshold", style={
                    "fontSize": "0.7rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "0.25rem",
                    "display": "block"
                }),
                dcc.Slider(
                    id="calib-error-threshold",
                    min=0.03,
                    max=0.15,
                    step=0.01,
                    value=0.07,
                    marks={0.03: "3%", 0.07: "7%", 0.10: "10%", 0.15: "15%"},
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            ], width=3),
        ], className="g-3 align-items-end", justify="start"),
        
        # Stores para os estados do Calibration
        dcc.Store(id="calib-bins-store", data=10),
        dcc.Store(id="calib-decision-mode-store", data="balanced"),
        
    ], style={
        "background": f"{COLORS['bg_hover']}44",
        "borderRadius": "8px",
        "padding": "0.75rem 1rem",
        "marginBottom": "0.75rem"
    })


def create_tab_global() -> html.Div:
    """Tab de comparação global de modelos."""
    return html.Div([
        # Metrics Cards Row
        html.Div(id="metrics-cards-row", style={"marginBottom": "1.5rem"}),
        
        # Charts Row 1: Métricas + ROC
        dbc.Row([
            dbc.Col([
                html.Div([
                    # Controlos do gráfico
                    create_metrics_controls(),
                    # Gráfico
                    dcc.Graph(id="metrics-comparison-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=6),
            dbc.Col([
                html.Div([
                    dcc.Graph(id="roc-curves-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=6)
        ], style={"marginBottom": "1.5rem"}),
        
        # Charts Row 2: Calibration Plot with Advanced Controls
        dbc.Row([
            dbc.Col([
                html.Div([
                    # Controlos avançados do Calibration Plot
                    create_calibration_controls(),
                    # Gráfico
                    dcc.Graph(id="calibration-plot-chart", config={"displayModeBar": False}),
                    # Insight automático
                    html.Div(
                        id="calibration-insight-text",
                        style={
                            "padding": "0.75rem 1rem",
                            "background": f"{COLORS['bg_hover']}66",
                            "borderRadius": "6px",
                            "marginTop": "0.5rem",
                            "fontSize": "0.85rem",
                            "color": COLORS["text_secondary"],
                            "fontStyle": "italic"
                        }
                    ),
                    # Caption
                    html.P(
                        "Calibration avalia se as probabilidades previstas correspondem às frequências observadas.",
                        style={
                            "fontSize": "0.75rem",
                            "color": COLORS["text_muted"],
                            "textAlign": "center",
                            "marginTop": "0.5rem",
                            "marginBottom": "0"
                        }
                    )
                ], className="dashboard-card")
            ], lg=12)
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
