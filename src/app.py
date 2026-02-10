from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.graph_objects as go

from src.config.settings import COLORS
from src.config.styles import get_custom_css
from src.utils.helpers import hex_to_rgba
from src.components.layout import create_header, create_controls_sidebar
from src.layouts.tabs import (
    create_tab_global,
    create_tab_tradeoffs,
    create_tab_errors,
    create_tab_fairness
)


def create_plotly_template() -> dict:
    return {
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {
                "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                "color": COLORS["text_secondary"],
                "size": 12
            },
            "title": {
                "font": {
                    "size": 16,
                    "color": COLORS["text_primary"],
                    "family": "Inter, sans-serif"
                },
                "x": 0,
                "xanchor": "left"
            },
            "xaxis": {
                "gridcolor": hex_to_rgba(COLORS["border"], 0.13),
                "linecolor": hex_to_rgba(COLORS["border"], 0.27),
                "tickcolor": COLORS["border"],
                "title_font": {"color": COLORS["text_secondary"], "size": 11},
                "tickfont": {"color": COLORS["text_muted"], "size": 10},
                "zeroline": False
            },
            "yaxis": {
                "gridcolor": hex_to_rgba(COLORS["border"], 0.13),
                "linecolor": hex_to_rgba(COLORS["border"], 0.27),
                "tickcolor": COLORS["border"],
                "title_font": {"color": COLORS["text_secondary"], "size": 11},
                "tickfont": {"color": COLORS["text_muted"], "size": 10},
                "zeroline": False
            },
            "legend": {
                "bgcolor": "rgba(0,0,0,0)",
                "font": {"color": COLORS["text_secondary"], "size": 11},
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1
            },
            "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
            "hoverlabel": {
                "bgcolor": COLORS["bg_card"],
                "bordercolor": COLORS["border"],
                "font": {"color": COLORS["text_primary"], "size": 12}
            },
            "colorway": [COLORS["primary"], COLORS["accent"], COLORS["secondary"], 
                         COLORS["warning"], COLORS["success"]]
        }
    }


def create_app(test_samples: int, positive_rate: float) -> Dash:
    """
    Cria e configura a aplicação Dash.
    
    Args:
        test_samples: Número de amostras de teste (para sidebar)
        positive_rate: Taxa de positivos no dataset (para sidebar)
        
    Returns:
        Aplicação Dash configurada
    """
    # Inicializar app
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Configurar template Plotly
    pio.templates["custom_dark"] = go.layout.Template(create_plotly_template())
    pio.templates.default = "custom_dark"
    
    # Injetar CSS customizado
    custom_css = get_custom_css()
    app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>Model Evaluation Dashboard</title>
        {{%favicon%}}
        {{%css%}}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">
        <style>{custom_css}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''
    
    # Layout principal
    app.layout = html.Div([
        # Store para estado de seleção (linked brushing)
        dcc.Store(id="selection-store", data={"model": None, "metric": None}),
        
        # Header
        create_header(),
        
        # Main Content
        dbc.Container([
            dbc.Row([
                # Sidebar Controls
                dbc.Col([
                    create_controls_sidebar(test_samples, positive_rate)
                ], lg=2, md=3, style={"marginBottom": "1.5rem"}),
                
                # Main Content Area
                dbc.Col([
                    # Tabs
                    dbc.Tabs([
                        dbc.Tab(
                            create_tab_global(),
                            label="Global Comparison",
                            tab_id="tab-global",
                            label_style={"fontWeight": "500", "color": COLORS["text_secondary"]},
                            active_label_style={"color": COLORS["bg_dark"], "fontWeight": "600"}
                        ),
                        dbc.Tab(
                            create_tab_tradeoffs(),
                            label="Trade-offs",
                            tab_id="tab-tradeoffs",
                            label_style={"fontWeight": "500", "color": COLORS["text_secondary"]},
                            active_label_style={"color": COLORS["bg_dark"], "fontWeight": "600"}
                        ),
                        dbc.Tab(
                            create_tab_errors(),
                            label="Error Analysis",
                            tab_id="tab-errors",
                            label_style={"fontWeight": "500", "color": COLORS["text_secondary"]},
                            active_label_style={"color": COLORS["bg_dark"], "fontWeight": "600"}
                        ),
                        dbc.Tab(
                            create_tab_fairness(),
                            label="Fairness",
                            tab_id="tab-fairness",
                            label_style={"fontWeight": "500", "color": COLORS["text_secondary"]},
                            active_label_style={"color": COLORS["bg_dark"], "fontWeight": "600"}
                        ),
                    ], id="main-tabs", active_tab="tab-global", className="nav-pills")
                ], lg=10, md=9)
            ])
        ], fluid=True, style={"padding": "0 2rem", "maxWidth": "1800px"})
    ], style={"minHeight": "100vh", "background": f"linear-gradient(135deg, {COLORS['bg_dark']} 0%, #1a1f35 100%)"})
    
    return app
