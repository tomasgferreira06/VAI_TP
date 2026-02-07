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
    create_fp_fn_evolution_enhanced,
    create_threshold_impact_bars,
    create_prediction_distribution_enhanced,
    build_operating_points_df,
    create_parallel_coordinates_operating_points,
    create_selected_operating_points_table,
    create_confusion_matrix_heatmap,
    create_advanced_confusion_matrix,
    create_error_rates_comparison,
    create_error_tradeoff_scatter,
    create_fairness_accuracy_chart,
    create_fairness_rates_chart,
    create_fairness_disparity_chart,
    create_fairness_horizon_chart,
    HORIZON_METRIC_CONFIG,
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
         Input("subgroup-selector", "value"),
         Input("global-decision-mode-store", "data")]
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
        [Input("model-selector", "value"),
         Input("threshold-slider", "value"),
         Input("global-decision-mode-store", "data"),
         Input("fp-fn-display-toggle", "value")]
    )
    def update_fp_fn_evolution(selected_model, threshold, decision_mode, display_mode):
        """Atualiza o gráfico Errors vs Threshold com funcionalidades enhanced."""
        show_counts = display_mode == "counts"
        
        return create_fp_fn_evolution_enhanced(
            eval_df,
            selected_model=selected_model,
            threshold=threshold,
            decision_mode=decision_mode or "balanced",
            show_counts=show_counts
        )


    # ═══════════════════════════════════════════════════════════════════════════════
    # VIEW 2: Parallel Coordinates Operating Points (Advanced Visualization)
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("pcp-operating-points-chart", "figure"),
        [Input("threshold-slider", "value"),
         Input("global-decision-mode-store", "data"),
         Input("pcp-subgroup-mode", "value"),
         Input("pcp-color-by", "value")]
    )
    def update_pcp_operating_points(threshold, decision_mode, subgroup_mode, color_by):
        """Atualiza o Parallel Coordinates Plot de operating points."""
        # Build operating points dataframe
        subgroup_attr = "sex" if subgroup_mode == "Sex" else "race" if subgroup_mode == "Race" else "sex"
        
        ops_df = build_operating_points_df(
            eval_df,
            models_selected=None,  # All models
            thresholds=None,  # Default range
            subgroup_mode=subgroup_mode,
            subgroup_attr=subgroup_attr,
            subgroup_pairs={
                "sex": ("Male", "Female"),
                "race": ("White", "Non-White")
            }
        )
        
        return create_parallel_coordinates_operating_points(
            ops_df,
            current_threshold=threshold,
            decision_mode=decision_mode or "balanced",
            color_by=color_by
        )

    @app.callback(
        Output("pcp-selected-table-container", "children"),
        [Input("threshold-slider", "value"),
         Input("global-decision-mode-store", "data"),
         Input("pcp-subgroup-mode", "value"),
         Input("pcp-operating-points-chart", "restyleData")]
    )
    def update_pcp_selected_table(threshold, decision_mode, subgroup_mode, restyle_data):
        """Atualiza a tabela de operating points selecionados."""
        import dash_bootstrap_components as dbc
        
        # Build operating points dataframe
        subgroup_attr = "sex" if subgroup_mode == "Sex" else "race" if subgroup_mode == "Race" else "sex"
        
        ops_df = build_operating_points_df(
            eval_df,
            models_selected=None,
            thresholds=None,
            subgroup_mode=subgroup_mode,
            subgroup_attr=subgroup_attr,
            subgroup_pairs={
                "sex": ("Male", "Female"),
                "race": ("White", "Non-White")
            }
        )
        
        # Get selected operating points table
        table_df = create_selected_operating_points_table(
            ops_df,
            selected_indices=None,  # Show top points near threshold
            current_threshold=threshold,
            decision_mode=decision_mode or "balanced",
            max_rows=6
        )
        
        if len(table_df) == 0:
            return html.Div("No operating points available", 
                           style={"color": COLORS["text_muted"], "textAlign": "center"})
        
        # Create bootstrap table
        return dbc.Table.from_dataframe(
            table_df,
            striped=True,
            bordered=False,
            hover=True,
            responsive=True,
            size="sm",
            style={"fontSize": "0.8rem"}
        )

    # ═══════════════════════════════════════════════════════════════════════════════
    # VIEW 3: Análise de Erros
    # ═══════════════════════════════════════════════════════════════════════════════

    # Confusion Matrix Normalization Mode Toggle
    @app.callback(
        [Output("cm-norm-mode-store", "data"),
         Output("btn-cm-counts", "outline"),
         Output("btn-cm-pct-total", "outline"),
         Output("btn-cm-pct-row", "outline"),
         Output("btn-cm-pct-col", "outline")],
        [Input("btn-cm-counts", "n_clicks"),
         Input("btn-cm-pct-total", "n_clicks"),
         Input("btn-cm-pct-row", "n_clicks"),
         Input("btn-cm-pct-col", "n_clicks")],
        prevent_initial_call=True
    )
    def update_cm_norm_mode(n1, n2, n3, n4):
        triggered = ctx.triggered_id
        modes = {
            "btn-cm-counts": "counts",
            "btn-cm-pct-total": "pct_total",
            "btn-cm-pct-row": "pct_row",
            "btn-cm-pct-col": "pct_col"
        }
        mode = modes.get(triggered, "counts")
        outlines = [triggered != btn for btn in modes.keys()]
        return mode, *outlines

    # Confusion Matrix Comparison Mode Toggle
    @app.callback(
        [Output("cm-comparison-mode-store", "data"),
         Output("btn-cm-single", "outline"),
         Output("btn-cm-compare", "outline"),
         Output("btn-cm-delta", "outline")],
        [Input("btn-cm-single", "n_clicks"),
         Input("btn-cm-compare", "n_clicks"),
         Input("btn-cm-delta", "n_clicks")],
        prevent_initial_call=True
    )
    def update_cm_comparison_mode(n1, n2, n3):
        triggered = ctx.triggered_id
        modes = {
            "btn-cm-single": "single",
            "btn-cm-compare": "side_by_side",
            "btn-cm-delta": "delta"
        }
        mode = modes.get(triggered, "single")
        outlines = [triggered != btn for btn in modes.keys()]
        return mode, *outlines

    @app.callback(
        Output("confusion-matrix-chart", "figure"),
        [Input("model-selector", "value"),
         Input("threshold-slider", "value"),
         Input("cm-norm-mode-store", "data"),
         Input("cm-comparison-mode-store", "data")]
    )
    def update_confusion_matrix(selected_model, threshold, norm_mode, comparison_mode):
        return create_advanced_confusion_matrix(
            eval_df, selected_model, threshold, 
            norm_mode or "counts", 
            comparison_mode or "single"
        )

    @app.callback(
        Output("cm-mode-caption", "children"),
        [Input("cm-norm-mode-store", "data"),
         Input("cm-comparison-mode-store", "data")]
    )
    def update_cm_caption(norm_mode, comparison_mode):
        norm_descriptions = {
            "counts": "Showing raw counts (TP, TN, FP, FN).",
            "pct_total": "Showing each cell as % of total dataset.",
            "pct_row": "Row-normalized: Shows TPR, FPR, TNR, FNR rates (each row sums to 100%).",
            "pct_col": "Column-normalized: Shows Precision, NPV rates (each column sums to 100%)."
        }
        comparison_descriptions = {
            "single": "Single model view.",
            "side_by_side": "Comparing both models side-by-side with same color scale.",
            "delta": "Delta view: Positive = RF has more, Negative = LR has more."
        }
        norm_desc = norm_descriptions.get(norm_mode, "")
        comp_desc = comparison_descriptions.get(comparison_mode, "")
        return f"{norm_desc} {comp_desc} Hover cells for rich diagnostics."

    @app.callback(
        Output("error-tradeoff-chart", "figure"),
        [Input("threshold-slider", "value"),
         Input("error-tradeoff-subgroup", "value")]
    )
    def update_error_tradeoff(threshold, subgroup):
        return create_error_tradeoff_scatter(eval_df, threshold, subgroup)


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
    # VIEW 4: Horizon Graph (Advanced Fairness Visualization)
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        [Output("horizon-fairness-chart", "figure"),
         Output("horizon-band-legend", "children")],
        [Input("sensitive-selector", "value"),
         Input("threshold-slider", "value"),
         Input("horizon-metric-selector", "value"),
         Input("horizon-model-focus", "value")]
    )
    def update_horizon_chart(sensitive_col, threshold, metric_name, model_focus):
        """Atualiza o Horizon Graph de fairness."""
        fig = create_fairness_horizon_chart(
            eval_df,
            sensitive_col=sensitive_col,
            current_threshold=threshold,
            metric_name=metric_name,
            model_focus=model_focus,
            n_bands=4,
        )

        # Build dynamic band legend
        cfg = HORIZON_METRIC_CONFIG.get(metric_name, HORIZON_METRIC_CONFIG["FNR"])
        r, g, b = cfg["rgb"]
        band_labels = ["Low", "Medium", "High", "Very High"]
        band_opacities = [0.18, 0.38, 0.58, 0.82]

        legend_items = []
        for lbl, opa in zip(band_labels, band_opacities):
            legend_items.append(
                html.Span([
                    html.Span("\u2588\u2588", style={
                        "color": f"rgba({r},{g},{b},{opa})",
                        "marginRight": "0.25rem",
                        "fontSize": "0.95rem",
                    }),
                    html.Span(lbl, style={
                        "color": COLORS["text_secondary"],
                        "fontSize": "0.8rem",
                        "marginRight": "0.75rem",
                    }),
                ])
            )

        legend = html.Div([
            html.Span("Band Intensity:  ", style={
                "fontWeight": "600",
                "color": COLORS["text_secondary"],
                "fontSize": "0.8rem",
                "marginRight": "0.5rem",
            }),
            *legend_items
        ], style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "flexWrap": "wrap",
            "padding": "0.5rem 0",
        })

        return fig, legend

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
