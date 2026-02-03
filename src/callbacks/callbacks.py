"""
Callbacks do Dash para o dashboard.
"""
import dash
from dash import Input, Output, State, dcc
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from src.config.settings import COLORS, MODEL_NAMES
from src.models.training import global_metrics, recompute_with_threshold
from src.components.cards import create_metric_card
from src.charts import (
    create_metrics_comparison_chart,
    create_roc_curves,
    create_feature_importance_chart,
    create_calibration_plot,
    create_precision_recall_curve,
    create_threshold_analysis,
    create_fp_fn_evolution_chart,
    create_threshold_impact_bars,
    create_confusion_matrix_heatmap,
    create_error_distribution_by_feature,
    create_error_rates_comparison,
    create_fairness_accuracy_chart,
    create_fairness_rates_chart,
    create_fairness_disparity_chart,
    create_parallel_coordinates,
    create_radar_chart,
    create_sunburst_errors
)


def register_callbacks(app, eval_df: pd.DataFrame, pipelines: dict, cat_cols: list, num_cols: list):
    """
    Regista todos os callbacks da aplicação.
    
    Args:
        app: Aplicação Dash
        eval_df: DataFrame de avaliação
        pipelines: Dicionário de pipelines treinados
        cat_cols: Lista de colunas categóricas
        num_cols: Lista de colunas numéricas
    """
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # VIEW 1: Comparação Global
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("metrics-cards-row", "children"),
        [Input("threshold-slider", "value"),
         Input("model-selector", "value")]
    )
    def update_metrics_cards(threshold, selected_model):
        """Atualiza os cards de métricas."""
        df = recompute_with_threshold(eval_df, threshold)
        model_df = df[df["model"] == selected_model]
        metrics = global_metrics(model_df)
        
        cards = [
            ("accuracy", "Accuracy", COLORS["primary"]),
            ("precision", "Precision", COLORS["accent"]),
            ("recall", "Recall", COLORS["secondary"]),
            ("f1", "F1-Score", COLORS["warning"])
        ]
        
        return dbc.Row([
            dbc.Col([
                create_metric_card(metrics[metric], label, color)
            ], md=3, sm=6, style={"marginBottom": "1rem"})
            for metric, label, color in cards
        ])

    @app.callback(
        Output("metrics-comparison-chart", "figure"),
        Input("threshold-slider", "value")
    )
    def update_metrics_comparison(threshold):
        return create_metrics_comparison_chart(eval_df, threshold)

    @app.callback(
        Output("roc-curves-chart", "figure"),
        Input("threshold-slider", "value")
    )
    def update_roc_curves(_):
        return create_roc_curves(eval_df)

    @app.callback(
        Output("feature-importance-chart", "figure"),
        Input("threshold-slider", "value")
    )
    def update_feature_importance(_):
        return create_feature_importance_chart(pipelines, cat_cols, num_cols)

    @app.callback(
        Output("calibration-plot-chart", "figure"),
        Input("threshold-slider", "value")
    )
    def update_calibration_plot(_):
        return create_calibration_plot(eval_df)

    # ═══════════════════════════════════════════════════════════════════════════════
    # VIEW 2: Trade-offs
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("pr-curve-chart", "figure"),
        Input("threshold-slider", "value")
    )
    def update_pr_curve(_):
        return create_precision_recall_curve(eval_df)

    @app.callback(
        Output("threshold-analysis-chart", "figure"),
        Input("model-selector", "value")
    )
    def update_threshold_analysis(selected_model):
        return create_threshold_analysis(eval_df, selected_model)

    @app.callback(
        Output("fp-fn-evolution-chart", "figure"),
        Input("model-selector", "value")
    )
    def update_fp_fn_evolution(selected_model):
        return create_fp_fn_evolution_chart(eval_df, selected_model)

    @app.callback(
        Output("threshold-impact-chart", "figure"),
        Input("threshold-slider", "value")
    )
    def update_threshold_impact(threshold):
        return create_threshold_impact_bars(eval_df, threshold)

    # ═══════════════════════════════════════════════════════════════════════════════
    # VIEW 3: Análise de Erros
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("confusion-matrix-chart", "figure"),
        [Input("model-selector", "value"),
         Input("threshold-slider", "value")]
    )
    def update_confusion_matrix(selected_model, threshold):
        return create_confusion_matrix_heatmap(eval_df, selected_model, threshold)

    @app.callback(
        Output("error-rates-chart", "figure"),
        Input("threshold-slider", "value")
    )
    def update_error_rates(threshold):
        return create_error_rates_comparison(eval_df, threshold)

    @app.callback(
        Output("error-by-sex-chart", "figure"),
        [Input("model-selector", "value"),
         Input("threshold-slider", "value")]
    )
    def update_error_by_sex(selected_model, threshold):
        return create_error_distribution_by_feature(eval_df, selected_model, "sex", threshold)

    @app.callback(
        Output("error-by-race-chart", "figure"),
        [Input("model-selector", "value"),
         Input("threshold-slider", "value")]
    )
    def update_error_by_race(selected_model, threshold):
        return create_error_distribution_by_feature(eval_df, selected_model, "race", threshold)

    # ═══════════════════════════════════════════════════════════════════════════════
    # VIEW 4: Fairness
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("fairness-accuracy-chart", "figure"),
        [Input("sensitive-selector", "value"),
         Input("threshold-slider", "value")]
    )
    def update_fairness_accuracy(sensitive_col, threshold):
        return create_fairness_accuracy_chart(eval_df, sensitive_col, threshold)

    @app.callback(
        Output("fairness-disparity-chart", "figure"),
        [Input("sensitive-selector", "value"),
         Input("threshold-slider", "value")]
    )
    def update_fairness_disparity(sensitive_col, threshold):
        return create_fairness_disparity_chart(eval_df, sensitive_col, threshold)

    @app.callback(
        Output("fairness-rates-chart", "figure"),
        [Input("sensitive-selector", "value"),
         Input("threshold-slider", "value")]
    )
    def update_fairness_rates(sensitive_col, threshold):
        return create_fairness_rates_chart(eval_df, sensitive_col, threshold)

    # ═══════════════════════════════════════════════════════════════════════════════
    # VIEW 5: Visualizações Avançadas
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("parallel-coords-chart", "figure"),
        Input("threshold-slider", "value")
    )
    def update_parallel_coords(threshold):
        return create_parallel_coordinates(eval_df, threshold)

    @app.callback(
        Output("radar-chart", "figure"),
        Input("threshold-slider", "value")
    )
    def update_radar_chart(threshold):
        return create_radar_chart(eval_df, threshold)

    @app.callback(
        Output("sunburst-chart", "figure"),
        [Input("threshold-slider", "value"),
         Input("model-selector", "value")]
    )
    def update_sunburst_chart(threshold, model):
        return create_sunburst_errors(eval_df, threshold, model)

    # ═══════════════════════════════════════════════════════════════════════════════
    # BOTÃO RESET
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        [Output("threshold-slider", "value"),
         Output("model-selector", "value"),
         Output("sensitive-selector", "value")],
        Input("reset-button", "n_clicks"),
        prevent_initial_call=True
    )
    def reset_controls(n_clicks):
        """Reset todos os controlos para valores default."""
        if n_clicks:
            return 0.5, "logreg", "sex"
        return dash.no_update, dash.no_update, dash.no_update

    # ═══════════════════════════════════════════════════════════════════════════════
    # BOTÃO DOWNLOAD/EXPORT
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("download-data", "data"),
        Input("download-button", "n_clicks"),
        [State("threshold-slider", "value"),
         State("model-selector", "value")],
        prevent_initial_call=True
    )
    def download_metrics(n_clicks, threshold, model):
        """Exporta métricas atuais para CSV."""
        if n_clicks:
            # Calcular métricas
            export_data = []
            for model_name in ["logreg", "rf"]:
                model_df = eval_df[eval_df["model"] == model_name].copy()
                model_df["y_pred"] = (model_df["y_proba"] >= threshold).astype(int)
                
                export_data.append({
                    "Model": MODEL_NAMES.get(model_name, model_name),
                    "Threshold": threshold,
                    "Accuracy": accuracy_score(model_df["y_true"], model_df["y_pred"]),
                    "Precision": precision_score(model_df["y_true"], model_df["y_pred"], zero_division=0),
                    "Recall": recall_score(model_df["y_true"], model_df["y_pred"], zero_division=0),
                    "F1-Score": f1_score(model_df["y_true"], model_df["y_pred"], zero_division=0),
                    "ROC-AUC": roc_auc_score(model_df["y_true"], model_df["y_proba"])
                })
            
            export_df = pd.DataFrame(export_data)
            return dcc.send_data_frame(export_df.to_csv, f"model_metrics_threshold_{threshold}.csv", index=False)
        return dash.no_update

    # ═══════════════════════════════════════════════════════════════════════════════
    # LINKED BRUSHING
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("selection-store", "data"),
        [Input("metrics-comparison-chart", "clickData"),
         Input("roc-curves-chart", "clickData")],
        prevent_initial_call=True
    )
    def update_selection_store(click_metrics, click_roc):
        """Armazena informação de seleção para linked brushing."""
        ctx = dash.callback_context
        
        if not ctx.triggered:
            return {"model": None, "metric": None}
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trigger_id == "metrics-comparison-chart" and click_metrics:
            point = click_metrics["points"][0]
            return {
                "model": point.get("customdata", [None])[0] if "customdata" in point else None,
                "metric": point.get("x", None)
            }
        elif trigger_id == "roc-curves-chart" and click_roc:
            point = click_roc["points"][0]
            return {
                "model": point.get("curveNumber", 0),
                "metric": "ROC"
            }
        
        return {"model": None, "metric": None}

    @app.callback(
        Output("selection-info", "children"),
        Input("selection-store", "data")
    )
    def update_selection_display(selection):
        """Mostra informação da seleção atual na sidebar."""
        if selection and selection.get("model"):
            model_info = selection.get("model", "Unknown")
            metric_info = selection.get("metric", "")
            return html.Div([
                html.Span(f"Selected: {metric_info}", style={"color": COLORS["primary_light"], "fontSize": "0.85rem", "display": "block"}),
                html.Span("Click to clear", style={"color": COLORS["text_muted"], "fontSize": "0.75rem"})
            ])
        return html.Span("Click on charts to select", style={"color": COLORS["text_muted"], "fontSize": "0.8rem"})
