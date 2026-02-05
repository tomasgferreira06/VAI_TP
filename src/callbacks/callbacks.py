"""
Callbacks do Dash para o dashboard.
"""
import dash
from dash import Input, Output, State, dcc, ctx
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
    create_calibration_plot,
    create_advanced_calibration_plot,
    create_calibration_subgroup_comparison,
    create_precision_recall_curve,
    create_precision_recall_curve_enhanced,
    create_threshold_analysis,
    create_threshold_analysis_enhanced,
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
         Input("model-selector", "value"),
         Input("subgroup-selector", "value")]
    )
    def update_metrics_cards(threshold, selected_model, subgroup):
        """Atualiza os cards de métricas."""
        df = recompute_with_threshold(eval_df, threshold)
        
        # Filtrar por subgrupo
        if subgroup == "global":
            filtered_df = df
        elif subgroup in ["Male", "Female"]:
            filtered_df = df[df["sex"] == subgroup]
        elif subgroup == "White":
            filtered_df = df[df["race"] == "White"]
        elif subgroup == "Non-White":
            filtered_df = df[df["race"] != "White"]
        else:
            filtered_df = df
            
        model_df = filtered_df[filtered_df["model"] == selected_model]
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

    # ═══════════════════════════════════════════════════════════════════════════════
    # VIEW 1: Controlos do Gráfico de Métricas (7 Funcionalidades)
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("display-mode-store", "data"),
        [Input("btn-display-absolute", "n_clicks"),
         Input("btn-display-relative", "n_clicks")],
        prevent_initial_call=True
    )
    def update_display_mode(n_abs, n_rel):
        """Alterna entre modo absoluto e relativo."""
        triggered = ctx.triggered_id
        if triggered == "btn-display-absolute":
            return "absolute"
        elif triggered == "btn-display-relative":
            return "relative"
        return "absolute"

    @app.callback(
        [Output("btn-display-absolute", "outline"),
         Output("btn-display-relative", "outline"),
         Output("btn-display-absolute", "style"),
         Output("btn-display-relative", "style")],
        Input("display-mode-store", "data")
    )
    def update_display_buttons(mode):
        """Atualiza estilo dos botões de display."""
        active_style = {}
        inactive_style = {"backgroundColor": "transparent", "color": COLORS["primary"]}
        
        if mode == "absolute":
            return False, True, active_style, inactive_style
        return True, False, inactive_style, active_style

    # ═══════════════════════════════════════════════════════════════════════════════
    # GLOBAL DECISION MODE (in sidebar controls)
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("global-decision-mode-store", "data"),
        [Input("btn-global-mode-balanced", "n_clicks"),
         Input("btn-global-mode-precision", "n_clicks"),
         Input("btn-global-mode-recall", "n_clicks")],
        prevent_initial_call=True
    )
    def update_global_decision_mode(n_bal, n_prec, n_rec):
        """Alterna entre modos de decisão globais."""
        triggered = ctx.triggered_id
        if triggered == "btn-global-mode-balanced":
            return "balanced"
        elif triggered == "btn-global-mode-precision":
            return "precision"
        elif triggered == "btn-global-mode-recall":
            return "recall"
        return "balanced"

    @app.callback(
        [Output("btn-global-mode-balanced", "outline"),
         Output("btn-global-mode-precision", "outline"),
         Output("btn-global-mode-recall", "outline"),
         Output("btn-global-mode-balanced", "style"),
         Output("btn-global-mode-precision", "style"),
         Output("btn-global-mode-recall", "style")],
        Input("global-decision-mode-store", "data")
    )
    def update_global_decision_buttons(mode):
        """Atualiza estilo dos botões de decision mode global."""
        active_style = {"width": "100%", "marginBottom": "0.35rem", "textAlign": "left"}
        inactive_style = {"backgroundColor": "transparent", "color": COLORS["primary"], 
                          "width": "100%", "marginBottom": "0.35rem", "textAlign": "left"}
        # Last button doesn't need marginBottom
        active_style_last = {"width": "100%", "textAlign": "left"}
        inactive_style_last = {"backgroundColor": "transparent", "color": COLORS["primary"], 
                               "width": "100%", "textAlign": "left"}
        
        return (
            mode != "balanced",
            mode != "precision",
            mode != "recall",
            active_style if mode == "balanced" else inactive_style,
            active_style if mode == "precision" else inactive_style,
            active_style_last if mode == "recall" else inactive_style_last
        )

    @app.callback(
        Output("metrics-comparison-chart", "figure"),
        [Input("threshold-slider", "value"),
         Input("display-mode-store", "data"),
         Input("subgroup-selector", "value")],
        State("global-decision-mode-store", "data")
    )
    def update_metrics_comparison(threshold, display_mode, subgroup, decision_mode):
        """Atualiza o gráfico de comparação de métricas."""
        return create_metrics_comparison_chart(
            eval_df, 
            threshold, 
            display_mode=display_mode,
            subgroup=subgroup,
            decision_mode=decision_mode or "balanced"
        )

    @app.callback(
        Output("roc-curves-chart", "figure"),
        Input("threshold-slider", "value")
    )
    def update_roc_curves(_):
        return create_roc_curves(eval_df)

    # ═══════════════════════════════════════════════════════════════════════════════
    # CALIBRATION PLOT AVANÇADO
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("calib-bins-store", "data"),
        [Input("btn-bins-5", "n_clicks"),
         Input("btn-bins-10", "n_clicks"),
         Input("btn-bins-20", "n_clicks")],
        prevent_initial_call=True
    )
    def update_calib_bins(n5, n10, n20):
        """Atualiza o número de bins do calibration plot."""
        triggered = ctx.triggered_id
        if triggered == "btn-bins-5":
            return 5
        elif triggered == "btn-bins-10":
            return 10
        elif triggered == "btn-bins-20":
            return 20
        return 10

    @app.callback(
        [Output("btn-bins-5", "outline"),
         Output("btn-bins-10", "outline"),
         Output("btn-bins-20", "outline"),
         Output("btn-bins-5", "style"),
         Output("btn-bins-10", "style"),
         Output("btn-bins-20", "style")],
        Input("calib-bins-store", "data")
    )
    def update_bins_buttons(bins):
        """Atualiza estilo dos botões de bins."""
        active_style = {}
        inactive_style = {"backgroundColor": "transparent", "color": COLORS["primary"]}
        return (
            bins != 5,
            bins != 10,
            bins != 20,
            active_style if bins == 5 else inactive_style,
            active_style if bins == 10 else inactive_style,
            active_style if bins == 20 else inactive_style
        )

    @app.callback(
        Output("calib-decision-mode-store", "data"),
        [Input("btn-calib-balanced", "n_clicks"),
         Input("btn-calib-precision", "n_clicks"),
         Input("btn-calib-recall", "n_clicks")],
        prevent_initial_call=True
    )
    def update_calib_decision_mode(n_bal, n_prec, n_rec):
        """Atualiza o modo de decisão do calibration plot."""
        triggered = ctx.triggered_id
        if triggered == "btn-calib-balanced":
            return "balanced"
        elif triggered == "btn-calib-precision":
            return "precision"
        elif triggered == "btn-calib-recall":
            return "recall"
        return "balanced"

    @app.callback(
        [Output("btn-calib-balanced", "outline"),
         Output("btn-calib-precision", "outline"),
         Output("btn-calib-recall", "outline"),
         Output("btn-calib-balanced", "style"),
         Output("btn-calib-precision", "style"),
         Output("btn-calib-recall", "style")],
        Input("calib-decision-mode-store", "data")
    )
    def update_calib_decision_buttons(mode):
        """Atualiza estilo dos botões de decision mode do calibration."""
        active_style = {}
        inactive_style = {"backgroundColor": "transparent", "color": COLORS["primary"]}
        return (
            mode != "balanced",
            mode != "precision",
            mode != "recall",
            active_style if mode == "balanced" else inactive_style,
            active_style if mode == "precision" else inactive_style,
            active_style if mode == "recall" else inactive_style
        )

    @app.callback(
        [Output("calibration-plot-chart", "figure"),
         Output("calibration-insight-text", "children")],
        [Input("threshold-slider", "value"),
         Input("calib-bins-store", "data"),
         Input("calib-subgroup-selector", "value"),
         Input("calib-decision-mode-store", "data"),
         Input("calib-error-threshold", "value")]
    )
    def update_calibration_plot(threshold, n_bins, subgroup_mode, decision_mode, error_threshold):
        """Atualiza o calibration plot avançado."""
        if subgroup_mode == "global":
            fig, insight = create_advanced_calibration_plot(
                eval_df,
                threshold=threshold,
                n_bins=n_bins,
                subgroup="global",
                subgroup_value=None,
                decision_mode=decision_mode,
                error_threshold=error_threshold
            )
        else:
            # Mostrar comparação entre subgrupos
            fig, insight = create_calibration_subgroup_comparison(
                eval_df,
                threshold=threshold,
                n_bins=n_bins,
                subgroup_type=subgroup_mode,
                decision_mode=decision_mode,
                error_threshold=error_threshold
            )
        
        return fig, insight

    # ═══════════════════════════════════════════════════════════════════════════════
    # VIEW 2: Trade-offs
    # ═══════════════════════════════════════════════════════════════════════════════

    # PR Curve Settings Store Update
    @app.callback(
        Output("pr-settings-store", "data"),
        Input("pr-show-area-toggle", "value"),
        prevent_initial_call=True
    )
    def update_pr_settings(show_area_values):
        """Update PR curve settings based on toggle."""
        show_area = "show_area" in show_area_values if show_area_values else False
        return {"show_area": show_area}

    @app.callback(
        [Output("pr-curve-chart", "figure"),
         Output("pr-delta-ap-text", "children")],
        [Input("threshold-slider", "value"),
         Input("global-decision-mode-store", "data"),
         Input("pr-settings-store", "data")]
    )
    def update_pr_curve(threshold, decision_mode, pr_settings):
        """Atualiza a curva PR com todas as funcionalidades enhanced."""
        show_area = pr_settings.get("show_area", False) if pr_settings else False
        
        fig, delta_ap_text = create_precision_recall_curve_enhanced(
            eval_df,
            threshold=threshold,
            decision_mode=decision_mode or "balanced",
            show_area=show_area
        )
        
        return fig, delta_ap_text

    @app.callback(
        Output("threshold-analysis-chart", "figure"),
        [Input("model-selector", "value"),
         Input("threshold-slider", "value"),
         Input("global-decision-mode-store", "data"),
         Input("threshold-metrics-toggle", "value"),
         Input("threshold-overlay-toggle", "value")]
    )
    def update_threshold_analysis(selected_model, threshold, decision_mode, metrics_toggle, overlay_toggle):
        """Atualiza o gráfico Metrics vs Threshold com funcionalidades enhanced."""
        # Parse toggles
        show_precision = "precision" in (metrics_toggle or [])
        show_recall = "recall" in (metrics_toggle or [])
        show_f1 = "f1" in (metrics_toggle or [])
        overlay_models = "overlay" in (overlay_toggle or [])
        
        return create_threshold_analysis_enhanced(
            eval_df,
            selected_model=selected_model,
            threshold=threshold,
            decision_mode=decision_mode or "balanced",
            show_precision=show_precision,
            show_recall=show_recall,
            show_f1=show_f1,
            overlay_models=overlay_models
        )

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
         Output("sensitive-selector", "value"),
         Output("global-decision-mode-store", "data", allow_duplicate=True)],
        Input("reset-button", "n_clicks"),
        prevent_initial_call=True
    )
    def reset_controls(n_clicks):
        """Reset todos os controlos para valores default."""
        if n_clicks:
            return 0.5, "logreg", "sex", "balanced"
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

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
