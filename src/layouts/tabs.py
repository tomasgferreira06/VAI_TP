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
        ], className="g-3 align-items-end", justify="start"),
        
        # Stores para os estados
        dcc.Store(id="display-mode-store", data="absolute"),
        
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
        
        # Store para bins do Calibration
        dcc.Store(id="calib-bins-store", data=10),
        
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
        
        # Row 1: Metric Comparison
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header(
                        "Metric Comparison by Model",
                        "Performance metrics across models — highlight winners and decision mode emphasis"
                    ),
                    # Controlos do gráfico
                    create_metrics_controls(),
                    # Gráfico
                    dcc.Graph(id="metrics-comparison-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=12)
        ], style={"marginBottom": "1.5rem"}),
        
        # Row 2: ROC Curves
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header(
                        "ROC Curves",
                        "Receiver Operating Characteristic — trade-off between TPR and FPR"
                    ),
                    dcc.Graph(id="roc-curves-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=12)
        ], style={"marginBottom": "1.5rem"}),
        
        # Row 3: Calibration Plot with Advanced Controls
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header(
                        "Calibration Plot",
                        "Reliability diagram — predicted probabilities vs observed frequencies"
                    ),
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


def create_pr_curve_controls() -> html.Div:
    """Cria os controlos para o PR curve avançado."""
    return html.Div([
        dbc.Row([
            # Toggle: Show/Hide Area Under Curve
            dbc.Col([
                html.Label("Show Area", style={
                    "fontSize": "0.7rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "0.25rem",
                    "display": "block"
                }),
                dbc.Checklist(
                    options=[{"label": " Fill under curve", "value": "show_area"}],
                    value=[],
                    id="pr-show-area-toggle",
                    switch=True,
                    style={"fontSize": "0.85rem"}
                )
            ], width="auto"),
            
            # Info icon with tooltip about AP
            dbc.Col([
                html.Div([
                    html.I(
                        className="bi bi-info-circle",
                        id="pr-info-icon",
                        style={
                            "fontSize": "1rem",
                            "color": COLORS["primary_light"],
                            "cursor": "pointer",
                            "marginTop": "1.5rem"
                        }
                    ),
                    dbc.Tooltip(
                        "Average Precision (AP) summarizes the PR curve across all thresholds. "
                        "Higher AP indicates better overall precision-recall trade-off.",
                        target="pr-info-icon",
                        placement="right"
                    )
                ])
            ], width="auto"),
        ], className="g-3 align-items-end", justify="start"),
        
        # Store for PR curve settings
        dcc.Store(id="pr-settings-store", data={"show_area": False}),
        
    ], style={
        "background": f"{COLORS['bg_hover']}44",
        "borderRadius": "8px",
        "padding": "0.75rem 1rem",
        "marginBottom": "0.75rem"
    })


def create_threshold_analysis_controls() -> html.Div:
    """Cria os controlos para o gráfico Metrics vs Threshold avançado."""
    return html.Div([], style={"display": "none"})


def create_fp_fn_controls() -> html.Div:
    """Cria os controlos para o gráfico Errors vs Threshold."""
    return html.Div([
        dbc.Row([
            # Toggle: Counts vs Rates
            dbc.Col([
                html.Label("Display Mode", style={
                    "fontSize": "0.7rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "0.25rem",
                    "display": "block"
                }),
                dbc.RadioItems(
                    options=[
                        {"label": " Counts (FP, FN)", "value": "counts"},
                        {"label": " Rates (FPR, FNR)", "value": "rates"},
                    ],
                    value="counts",
                    id="fp-fn-display-toggle",
                    inline=True,
                    style={"fontSize": "0.8rem"}
                )
            ], width="auto"),
            
            # Info icon with tooltip
            dbc.Col([
                html.Div([
                    html.I(
                        className="bi bi-info-circle",
                        id="fp-fn-info-icon",
                        style={
                            "fontSize": "1rem",
                            "color": COLORS["primary_light"],
                            "cursor": "pointer",
                            "marginTop": "1.5rem"
                        }
                    ),
                    dbc.Tooltip(
                        "This plot shows how False Positives and False Negatives evolve with threshold changes. "
                        "Switch between absolute counts and normalized rates. "
                        "The optimal point changes based on the global Decision Mode.",
                        target="fp-fn-info-icon",
                        placement="right"
                    )
                ])
            ], width="auto"),
        ], className="g-3 align-items-end", justify="start"),
        
    ], style={
        "background": f"{COLORS['bg_hover']}44",
        "borderRadius": "8px",
        "padding": "0.75rem 1rem",
        "marginBottom": "0.75rem"
    })



def create_pcp_controls() -> html.Div:
    """Cria os controlos para o Parallel Coordinates Plot."""
    return html.Div([
        dbc.Row([
            # Distinguish by (formerly Color By)
            dbc.Col([
                html.Label("Distinguish by", style={
                    "fontSize": "0.7rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "0.25rem",
                    "display": "block"
                }),
                dbc.RadioItems(
                    options=[
                        {"label": " Model", "value": "model"},
                        {"label": " Feature", "value": "subgroup"},
                    ],
                    value="model",
                    id="pcp-color-by",
                    inline=True,
                    style={"fontSize": "0.8rem"}
                )
            ], width="auto"),
            
            # Info icon with tooltip
            dbc.Col([
                html.Div([
                    html.I(
                        className="bi bi-info-circle",
                        id="pcp-info-icon",
                        style={
                            "fontSize": "1rem",
                            "color": COLORS["primary_light"],
                            "cursor": "pointer",
                            "marginTop": "1.5rem"
                        }
                    ),
                    dbc.Tooltip(
                        "Each polyline represents an operating point (model + threshold + subgroup). "
                        "Brush axes to filter operating points. The recommended point is highlighted "
                        "based on the global Decision Mode. Fairness Gap shows disparity between groups.",
                        target="pcp-info-icon",
                        placement="right"
                    )
                ])
            ], width="auto"),
        ], className="g-3 align-items-end", justify="start"),
        
        # Store for selected operating points from brushing
        dcc.Store(id="pcp-selected-ops-store", data=[]),
        
    ], style={
        "background": f"{COLORS['bg_hover']}44",
        "borderRadius": "8px",
        "padding": "0.75rem 1rem",
        "marginBottom": "0.75rem"
    })


def create_tab_tradeoffs() -> html.Div:
    """Tab de análise de trade-offs (precision vs recall)."""
    return html.Div([
        # Explanation Card
        html.Div([
            html.Div([
                html.I(className="bi bi-info-circle", style={"fontSize": "1.1rem", "marginRight": "0.75rem", "color": COLORS["primary_light"]}),
                html.Span(
                    "O threshold de decisão afeta diretamente o trade-off entre Precision e Recall. "
                    "Um threshold mais alto aumenta a Precision (menos FP) mas reduz o Recall (mais FN). "
                    "Use o Decision Mode nos controlos globais para enfatizar diferentes regiões operacionais.",
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
        
        # Row 1: PR Curve
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header(
                        "Precision-Recall Curve",
                        "Trade-off between precision and recall across thresholds"
                    ),
                    # PR Curve Controls
                    create_pr_curve_controls(),
                    # PR Curve Chart
                    dcc.Graph(id="pr-curve-chart", config={"displayModeBar": False}),
                    # Delta AP annotation
                    html.Div(
                        id="pr-delta-ap-text",
                        style={
                            "padding": "0.5rem 0.75rem",
                            "background": f"{COLORS['bg_hover']}66",
                            "borderRadius": "6px",
                            "marginTop": "0.5rem",
                            "fontSize": "0.8rem",
                            "color": COLORS["text_secondary"],
                            "textAlign": "center"
                        }
                    )
                ], className="dashboard-card")
            ], lg=12)
        ], style={"marginBottom": "1.5rem"}),
        
        # Row 2: Threshold Analysis
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header(
                        "Metrics vs Threshold",
                        "How precision, recall and F1 evolve as threshold changes"
                    ),
                    # Threshold Analysis Controls
                    create_threshold_analysis_controls(),
                    # Threshold Analysis Chart
                    dcc.Graph(id="threshold-analysis-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=12)
        ], style={"marginBottom": "1.5rem"}),
        
        # Row 3: Evolução de FP/FN
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header(
                        "Error Evolution",
                        "False positives and false negatives as threshold changes"
                    ),
                    # FP/FN Controls
                    create_fp_fn_controls(),
                    # FP/FN Chart
                    dcc.Graph(id="fp-fn-evolution-chart", config={"displayModeBar": False})
                ], className="dashboard-card")
            ], lg=12)
        ], style={"marginBottom": "1.5rem"}),
                
        # Row 4: Parallel Coordinates Operating Points (Advanced Visualization)
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header(
                        "Operating Points Analysis",
                        "Multi-criteria comparison across models, thresholds, and subgroups — brush axes to filter"
                    ),
                    
                    # PCP Controls
                    create_pcp_controls(),
                    
                    # Parallel Coordinates Chart
                    dcc.Graph(
                        id="pcp-operating-points-chart", 
                        config={"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
                    ),
                    
                    # Caption
                    html.Div(
                        "Each polyline is an operating point (model + threshold + subgroup). "
                        "Brush axes to filter operating points and see linked updates in other views.",
                        style={
                            "fontSize": "0.75rem",
                            "color": COLORS["text_muted"],
                            "fontStyle": "italic",
                            "textAlign": "center",
                            "marginTop": "0.5rem"
                        }
                    ),
                    
                    # Divider
                    html.Hr(style={
                        "borderColor": f"{COLORS['border']}33",
                        "margin": "1.25rem 0"
                    }),
                    
                    # Selected Operating Points Table Header
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-table", style={
                                "fontSize": "0.9rem", 
                                "marginRight": "0.5rem", 
                                "color": COLORS["primary_light"]
                            }),
                            html.Span("Selected Operating Points", style={
                                "fontWeight": "500",
                                "color": COLORS["text_primary"],
                                "fontSize": "0.875rem"
                            }),
                            html.Span(
                                " — Top points near current threshold",
                                style={
                                    "color": COLORS["text_muted"],
                                    "fontSize": "0.75rem"
                                }
                            )
                        ], style={"display": "flex", "alignItems": "center"})
                    ], style={"marginBottom": "0.75rem"}),
                    
                    # Table container
                    html.Div(id="pcp-selected-table-container", className="dark-table-container")
                ], className="dashboard-card")
            ], lg=12)
        ])
    ])


def create_tab_errors() -> html.Div:
    """Tab de análise de erros."""
    return html.Div([
        # Confusion Matrix Section with Controls
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header(
                        "Confusion Matrix",
                        "Explore error patterns across thresholds and models — hover cells for diagnostics"
                    ),
                    
                    # Controls Row
                    html.Div([
                        dbc.Row([
                            # Normalization Mode
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
                                    dbc.Button("Counts", id="btn-cm-counts", color="primary", outline=False, size="sm"),
                                    dbc.Button("% Total", id="btn-cm-pct-total", color="primary", outline=True, size="sm"),
                                    dbc.Button("% Row", id="btn-cm-pct-row", color="primary", outline=True, size="sm"),
                                    dbc.Button("% Col", id="btn-cm-pct-col", color="primary", outline=True, size="sm"),
                                ], size="sm")
                            ], width="auto"),
                            
                            # Info Tooltip
                            dbc.Col([
                                html.Div([
                                    html.I(
                                        className="bi bi-info-circle",
                                        id="cm-info-icon",
                                        style={
                                            "fontSize": "1rem",
                                            "color": COLORS["primary_light"],
                                            "cursor": "pointer"
                                        }
                                    ),
                                    dbc.Tooltip(
                                        [
                                            html.B("Display Modes:"), html.Br(),
                                            "• Counts: Raw TP/TN/FP/FN", html.Br(),
                                            "• % Total: Cell as % of dataset", html.Br(),
                                            "• % Row: TPR/FPR/TNR/FNR rates", html.Br(),
                                            "• % Col: Precision/NPV rates", html.Br(), html.Br(),
                                            html.B("Model Focus:"), html.Br(),
                                            "• Single model: Shows one matrix", html.Br(),
                                            "• Both: Side-by-side comparison"
                                        ],
                                        target="cm-info-icon",
                                        placement="right",
                                        style={"fontSize": "0.8rem"}
                                    )
                                ])
                            ], width="auto"),
                        ], className="g-3 align-items-end", justify="start"),
                        
                        # Store for confusion matrix normalization state
                        dcc.Store(id="cm-norm-mode-store", data="counts"),
                        
                    ], style={
                        "background": f"{COLORS['bg_hover']}44",
                        "borderRadius": "8px",
                        "padding": "0.75rem 1rem",
                        "marginBottom": "1rem"
                    }),
                    
                    # Confusion Matrix Chart
                    dcc.Graph(id="confusion-matrix-chart", config={"displayModeBar": False}),
                    
                    # Mode description caption
                    html.Div(
                        id="cm-mode-caption",
                        children="Hover over any cell to see detailed diagnostics including all normalization views.",
                        style={
                            "fontSize": "0.75rem",
                            "color": COLORS["text_muted"],
                            "fontStyle": "italic",
                            "textAlign": "center",
                            "marginTop": "0.25rem",
                            "paddingBottom": "0.25rem"
                        }
                    )
                ], className="dashboard-card")
            ], lg=12)
        ], style={"marginBottom": "1.5rem"}),
        
        # Error Trade-off Scatter
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header(
                        "Connected Bubble Scatter Plot - Error Trade-off Trajectories",
                        "FPR vs FNR as threshold changes — dynamic behavior as static trajectories"
                    ),
                    
                    # Error tradeoff scatter chart
                    dcc.Graph(id="error-tradeoff-chart", config={"displayModeBar": False}),
                    
                    # Caption explaining the visualization
                    html.Div(
                        "Each curve shows how a model moves in error space (FPR vs FNR) as the decision threshold changes. "
                        "The highlighted point corresponds to the current threshold. Lowering the threshold moves toward "
                        "lower FNR but higher FPR.",
                        style={
                            "fontSize": "0.75rem",
                            "color": COLORS["text_muted"],
                            "fontStyle": "italic",
                            "textAlign": "center",
                            "marginTop": "0.25rem",
                            "paddingBottom": "0.25rem"
                        }
                    )
                ], className="dashboard-card")
            ], lg=12)
        ], style={"marginBottom": "1.5rem"}),
    ])


def create_horizon_controls() -> html.Div:
    """Controls for the Horizon Graph fairness visualization."""
    return html.Div([
        dbc.Row([
            # Metric selector
            dbc.Col([
                html.Label("Fairness Metric", style={
                    "fontSize": "0.7rem",
                    "color": COLORS["text_muted"],
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": "0.25rem",
                    "display": "block"
                }),
                dcc.Dropdown(
                    id="horizon-metric-selector",
                    options=[
                        {"label": "False Negative Rate (FNR)", "value": "FNR"},
                        {"label": "False Positive Rate (FPR)", "value": "FPR"},
                        {"label": "Recall (TPR)", "value": "Recall"},
                    ],
                    value="FNR",
                    clearable=False,
                    style={"minWidth": "210px", "fontSize": "0.85rem"}
                )
            ], width="auto"),

            # Info icon
            dbc.Col([
                html.Div([
                    html.I(
                        className="bi bi-info-circle",
                        id="horizon-info-icon",
                        style={
                            "fontSize": "1rem",
                            "color": COLORS["primary_light"],
                            "cursor": "pointer",
                            "marginTop": "1.5rem"
                        }
                    ),
                    dbc.Tooltip(
                        "Horizon Graph: a compact visualization where the value range is split "
                        "into bands of increasing colour intensity. Darker bands indicate higher "
                        "error rates. The bottom row shows the absolute gap between groups — "
                        "a quick way to spot fairness issues across all thresholds.",
                        target="horizon-info-icon",
                        placement="right"
                    )
                ])
            ], width="auto"),
        ], className="g-3 align-items-end", justify="start"),
    ], style={
        "background": f"{COLORS['bg_hover']}44",
        "borderRadius": "8px",
        "padding": "0.75rem 1rem",
        "marginBottom": "0.75rem"
    })


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
        
        # ═══════════════════════════════════════════════════════════════════════
        # HORIZON GRAPH — Advanced Fairness Visualization
        # ═══════════════════════════════════════════════════════════════════════
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header(
                        "Horizon Graph — Fairness Across Thresholds",
                        "Error rates by demographic group as threshold changes — darker bands indicate higher error"
                    ),

                    # Controls
                    create_horizon_controls(),

                    # Chart
                    dcc.Graph(
                        id="horizon-fairness-chart",
                        config={"displayModeBar": False}
                    ),

                    # Dynamic band legend
                    html.Div(id="horizon-band-legend", style={"marginTop": "0.25rem"}),

                    # Explanatory text
                    html.Div([
                        html.P(
                            "This Horizon Graph shows how error rates for different "
                            "demographic groups evolve as the decision threshold changes. "
                            "Darker / more intense bands indicate higher error. "
                            "The bottom row displays the absolute disparity (gap) between "
                            "groups — regions above the 5 % line signal potential bias.",
                            style={
                                "fontSize": "0.8rem",
                                "color": COLORS["text_secondary"],
                                "marginBottom": "0.35rem",
                                "lineHeight": "1.5"
                            }
                        ),
                        html.P(
                            "💡 Groups with consistently darker bands at the same "
                            "thresholds are systematically disadvantaged.",
                            style={
                                "fontSize": "0.78rem",
                                "color": COLORS["text_muted"],
                                "fontStyle": "italic",
                                "marginBottom": "0"
                            }
                        ),
                    ], style={
                        "background": f"{COLORS['bg_hover']}44",
                        "borderRadius": "6px",
                        "padding": "0.75rem 1rem",
                        "marginTop": "0.5rem"
                    })
                ], className="dashboard-card")
            ], lg=12)
        ], style={"marginBottom": "1.5rem"}),
        
        # Sunburst - Hierarchical Error Distribution
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header(
                        "Hierarchical Error Distribution",
                        "Interactive breakdown of predictions by error type and demographic group"
                    ),
                    dcc.Graph(id="fairness-sunburst-chart", config={"displayModeBar": False}),
                    
                    html.Div([
                        html.P(
                            "Click on segments to zoom in and explore the distribution. "
                            "The size of each segment represents the count of predictions in that category.",
                            style={
                                "fontSize": "0.8rem",
                                "color": COLORS["text_secondary"],
                                "marginBottom": "0",
                                "lineHeight": "1.5",
                                "fontStyle": "italic"
                            }
                        )
                    ], style={
                        "background": f"{COLORS['bg_hover']}44",
                        "borderRadius": "6px",
                        "padding": "0.75rem 1rem",
                        "marginTop": "0.5rem"
                    })
                ], className="dashboard-card")
            ], lg=12)
        ], style={"marginTop": "1.5rem"})
    ])



