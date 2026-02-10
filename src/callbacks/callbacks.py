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

from src.config.settings import COLORS, MODEL_NAMES, MODEL_COLORS
from src.models.training import global_metrics, recompute_with_threshold
from src.components.cards import create_metric_card
from src.utils.helpers import hex_to_rgba, get_demographic_groups
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
    create_fairness_horizon_chart,
    create_fairness_sunburst,
    HORIZON_METRIC_CONFIG
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
         Input("sensitive-selector", "value")]
    )
    def update_metrics_cards(threshold, selected_model, analysis_focus):
        """Atualiza os cards de métricas com suporte a comparação por subgrupo."""
        df = recompute_with_threshold(eval_df, threshold)
        
        cards = [
            ("accuracy", "Accuracy", COLORS["warning"]),
            ("precision", "Precision", COLORS["secondary"]),
            ("recall", "Recall", COLORS["error"]),
            ("f1", "F1-Score", "#A78BFA")  # Purple
        ]
        
        # Determinar grupos a mostrar
        if analysis_focus == "global":
            groups = [("Global", df)]
        elif analysis_focus == "sex":
            groups = [
                ("Male", df[df["sex"] == "Male"]),
                ("Female", df[df["sex"] == "Female"])
            ]
        elif analysis_focus == "race":
            groups = [
                ("White", df[df["race"] == "White"]),
                ("Non-White", df[df["race"] != "White"])
            ]
        else:
            groups = [("Global", df)]
        
        # Handle "both" mode - show both models
        if selected_model == "both":
            all_sections = []
            
            for model_key, model_label in [("logreg", "Logistic Regression"), ("rf", "Random Forest")]:
                model_color = MODEL_COLORS.get(model_key, COLORS["primary"])
                
                # Se temos múltiplos grupos, mostrar comparação
                if len(groups) > 1:
                    group_rows = []
                    for group_name, group_df in groups:
                        model_df = group_df[group_df["model"] == model_key]
                        metrics = global_metrics(model_df)
                        
                        group_rows.append(
                            html.Div([
                                html.Span(group_name, style={
                                    "fontSize": "0.8rem",
                                    "color": COLORS["text_muted"],
                                    "marginBottom": "0.5rem",
                                    "display": "block"
                                }),
                                dbc.Row([
                                    dbc.Col([
                                        create_metric_card(metrics[metric], label, color)
                                    ], md=3, sm=6)
                                    for metric, label, color in cards
                                ], className="g-2")
                            ], style={"marginBottom": "1rem"})
                        )
                    
                    model_section = html.Div([
                        html.Div([
                            html.Div(style={
                                "width": "4px",
                                "height": "20px",
                                "background": model_color,
                                "borderRadius": "2px",
                                "marginRight": "0.75rem"
                            }),
                            html.Span(model_label, style={
                                "fontSize": "0.95rem",
                                "fontWeight": "600",
                                "color": COLORS["text_primary"]
                            })
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "marginBottom": "0.75rem"
                        }),
                        html.Div(group_rows)
                    ], style={
                        "marginBottom": "2rem",
                        "paddingBottom": "1.5rem",
                        "borderBottom": f"1px solid {COLORS['border']}"
                    })
                else:
                    # Modo global - layout original
                    model_df = groups[0][1][groups[0][1]["model"] == model_key]
                    metrics = global_metrics(model_df)
                    
                    model_section = html.Div([
                        html.Div([
                            html.Div(style={
                                "width": "4px",
                                "height": "20px",
                                "background": model_color,
                                "borderRadius": "2px",
                                "marginRight": "0.75rem"
                            }),
                            html.Span(model_label, style={
                                "fontSize": "0.95rem",
                                "fontWeight": "600",
                                "color": COLORS["text_primary"]
                            })
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "marginBottom": "0.75rem"
                        }),
                        dbc.Row([
                            dbc.Col([
                                create_metric_card(metrics[metric], label, color)
                            ], md=3, sm=6)
                            for metric, label, color in cards
                        ], className="g-3")
                    ], style={
                        "marginBottom": "2rem",
                        "paddingBottom": "1.5rem",
                        "borderBottom": f"1px solid {COLORS['border']}"
                    })
                
                all_sections.append(model_section)
            
            # Remove border from last section
            if all_sections:
                all_sections[-1].style["borderBottom"] = "none"
            
            return html.Div(all_sections)
        
        # Single model mode
        if len(groups) > 1:
            group_rows = []
            for group_name, group_df in groups:
                model_df = group_df[group_df["model"] == selected_model]
                metrics = global_metrics(model_df)
                
                group_rows.append(
                    html.Div([
                        html.Span(group_name, style={
                            "fontSize": "0.85rem",
                            "fontWeight": "600",
                            "color": COLORS["text_secondary"],
                            "marginBottom": "0.5rem",
                            "display": "block"
                        }),
                        dbc.Row([
                            dbc.Col([
                                create_metric_card(metrics[metric], label, color)
                            ], md=3, sm=6)
                            for metric, label, color in cards
                        ], className="g-2")
                    ], style={"marginBottom": "1.5rem"})
                )
            return html.Div(group_rows)
        else:
            model_df = groups[0][1][groups[0][1]["model"] == selected_model]
            metrics = global_metrics(model_df)
            
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
         Input("sensitive-selector", "value"),
         Input("global-decision-mode-store", "data")]
    )
    def update_metrics_comparison(threshold, display_mode, analysis_focus, decision_mode):
        """Atualiza o gráfico de comparação de métricas."""
        print(f"[DEBUG] update_metrics_comparison called: threshold={threshold}, display_mode={display_mode}, analysis_focus={analysis_focus}, decision_mode={decision_mode}")
        # Para o gráfico de métricas, usamos "global" quando analysis_focus é global
        # Quando é sex/race, a função precisa mostrar comparação entre subgrupos
        result = create_metrics_comparison_chart(
            eval_df, 
            threshold, 
            display_mode=display_mode,
            subgroup=analysis_focus,  # Passa "global", "sex" ou "race"
            decision_mode=decision_mode or "balanced"
        )
        print(f"[DEBUG] metrics chart: {len(result.data)} traces, title={result.layout.title.text[:50] if result.layout.title.text else 'None'}")
        return result

    @app.callback(
        Output("roc-curves-chart", "figure"),
        [Input("sensitive-selector", "value"),
         Input("model-selector", "value")]
    )
    def update_roc_curves(analysis_focus, model_focus):
        """Atualiza as curvas ROC com suporte a subgrupos."""
        return create_roc_curves(eval_df, analysis_focus, model_focus)

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
        [Output("calibration-plot-chart", "figure"),
         Output("calibration-insight-text", "children")],
        [Input("threshold-slider", "value"),
         Input("calib-bins-store", "data"),
         Input("sensitive-selector", "value"),
         Input("global-decision-mode-store", "data"),
         Input("calib-error-threshold", "value")]
    )
    def update_calibration_plot(threshold, n_bins, analysis_focus, decision_mode, error_threshold):
        """Atualiza o calibration plot - global ou comparação entre subgrupos."""
        if analysis_focus == "global":
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
            # Mostrar comparação entre subgrupos (Male vs Female ou White vs Non-White)
            fig, insight = create_calibration_subgroup_comparison(
                eval_df,
                threshold=threshold,
                n_bins=n_bins,
                subgroup_type=analysis_focus,  # "sex" ou "race"
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
         Input("pr-settings-store", "data"),
         Input("sensitive-selector", "value")]
    )
    def update_pr_curve(threshold, decision_mode, pr_settings, analysis_focus):
        """Atualiza a curva PR com todas as funcionalidades enhanced."""
        show_area = pr_settings.get("show_area", False) if pr_settings else False
        
        fig, delta_ap_text = create_precision_recall_curve_enhanced(
            eval_df,
            threshold=threshold,
            decision_mode=decision_mode or "balanced",
            show_area=show_area,
            analysis_focus=analysis_focus or "global"
        )
        
        return fig, delta_ap_text

    @app.callback(
        Output("threshold-analysis-chart", "figure"),
        [Input("model-selector", "value"),
         Input("threshold-slider", "value"),
         Input("global-decision-mode-store", "data"),
         Input("sensitive-selector", "value")]
    )
    def update_threshold_analysis(selected_model, threshold, decision_mode, analysis_focus):
        """Atualiza o gráfico Metrics vs Threshold com funcionalidades enhanced."""
        # Always show all metrics
        show_precision = True
        show_recall = True
        show_f1 = True
        
        # Overlay when "both" is selected
        overlay_models = (selected_model == "both")
        
        # Default to logreg for single model view when needed
        effective_model = "logreg" if selected_model == "both" else selected_model
        
        return create_threshold_analysis_enhanced(
            eval_df,
            selected_model=effective_model,
            threshold=threshold,
            decision_mode=decision_mode or "balanced",
            show_precision=show_precision,
            show_recall=show_recall,
            show_f1=show_f1,
            overlay_models=overlay_models,
            analysis_focus=analysis_focus or "global"
        )

    @app.callback(
        Output("fp-fn-evolution-chart", "figure"),
        [Input("model-selector", "value"),
         Input("threshold-slider", "value"),
         Input("global-decision-mode-store", "data"),
         Input("fp-fn-display-toggle", "value"),
         Input("sensitive-selector", "value")]
    )
    def update_fp_fn_evolution(selected_model, threshold, decision_mode, display_mode, analysis_focus):
        """Atualiza o gráfico Errors vs Threshold com funcionalidades enhanced."""
        show_counts = display_mode == "counts"
        # Default to logreg when "both" is selected (this chart shows single model)
        effective_model = "logreg" if selected_model == "both" else selected_model
        
        return create_fp_fn_evolution_enhanced(
            eval_df,
            selected_model=effective_model,
            threshold=threshold,
            decision_mode=decision_mode or "balanced",
            show_counts=show_counts,
            analysis_focus=analysis_focus or "global"
        )


    # ═══════════════════════════════════════════════════════════════════════════════
    # VIEW 2: Parallel Coordinates Operating Points (Advanced Visualization)
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("pcp-operating-points-chart", "figure"),
        [Input("threshold-slider", "value"),
         Input("global-decision-mode-store", "data"),
         Input("sensitive-selector", "value"),
         Input("pcp-color-by", "value")]
    )
    def update_pcp_operating_points(threshold, decision_mode, analysis_focus, color_by):
        """Atualiza o Parallel Coordinates Plot de operating points."""
        # Map analysis_focus to subgroup_mode format
        if analysis_focus == "sex":
            subgroup_mode = "Sex"
            subgroup_attr = "sex"
        elif analysis_focus == "race":
            subgroup_mode = "Race"
            subgroup_attr = "race"
        else:
            subgroup_mode = "Global"
            subgroup_attr = "sex"  # Default, not used in Global mode
        
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
         Input("sensitive-selector", "value"),
         Input("pcp-operating-points-chart", "restyleData")]
    )
    def update_pcp_selected_table(threshold, decision_mode, analysis_focus, restyle_data):
        """Atualiza a tabela de operating points selecionados."""
        import dash_bootstrap_components as dbc
        
        # Map analysis_focus to subgroup_mode format
        if analysis_focus == "sex":
            subgroup_mode = "Sex"
            subgroup_attr = "sex"
        elif analysis_focus == "race":
            subgroup_mode = "Race"
            subgroup_attr = "race"
        else:
            subgroup_mode = "Global"
            subgroup_attr = "sex"  # Default, not used in Global mode
        
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

    @app.callback(
        Output("confusion-matrix-chart", "figure"),
        [Input("model-selector", "value"),
         Input("threshold-slider", "value"),
         Input("cm-norm-mode-store", "data")]
    )
    def update_confusion_matrix(selected_model, threshold, norm_mode):
        # Derive comparison mode from model-selector
        if selected_model == "both":
            effective_comparison = "side_by_side"
            effective_model = "logreg"  # not used for side_by_side
        else:
            effective_comparison = "single"
            effective_model = selected_model
        
        return create_advanced_confusion_matrix(
            eval_df, effective_model, threshold, 
            norm_mode or "counts", 
            effective_comparison
        )

    @app.callback(
        Output("cm-mode-caption", "children"),
        [Input("cm-norm-mode-store", "data"),
         Input("model-selector", "value")]
    )
    def update_cm_caption(norm_mode, selected_model):
        norm_descriptions = {
            "counts": "Showing raw counts (TP, TN, FP, FN).",
            "pct_total": "Showing each cell as % of total dataset.",
            "pct_row": "Row-normalized: Shows TPR, FPR, TNR, FNR rates (each row sums to 100%).",
            "pct_col": "Column-normalized: Shows Precision, NPV rates (each column sums to 100%)."
        }
        norm_desc = norm_descriptions.get(norm_mode, "")
        if selected_model == "both":
            comp_desc = "Comparing both models side-by-side."
        else:
            comp_desc = f"Showing {MODEL_NAMES.get(selected_model, selected_model)}."
        return f"{norm_desc} {comp_desc} Hover cells for rich diagnostics."

    @app.callback(
        Output("error-tradeoff-chart", "figure"),
        [Input("threshold-slider", "value"),
         Input("sensitive-selector", "value")]
    )
    def update_error_tradeoff(threshold, analysis_focus):
        return create_error_tradeoff_scatter(eval_df, threshold, analysis_focus or "global")


    # ═══════════════════════════════════════════════════════════════════════════════
    # VIEW 4: Fairness
    # ═══════════════════════════════════════════════════════════════════════════════
    
    @app.callback(
        Output("fairness-sunburst-chart", "figure"),
        [Input("sensitive-selector", "value"),
         Input("threshold-slider", "value"),
         Input("model-selector", "value")]
    )
    def update_fairness_sunburst(sensitive_col, threshold, model_focus):
        if sensitive_col == "global":
            sensitive_col = "sex"
        return create_fairness_sunburst(eval_df, sensitive_col, threshold, model_focus)

    # ═══════════════════════════════════════════════════════════════════════════════
    # VIEW 4: Horizon Graph (Advanced Fairness Visualization)
    # ═══════════════════════════════════════════════════════════════════════════════

    @app.callback(
        [Output("horizon-fairness-chart", "figure"),
         Output("horizon-band-legend", "children")],
        [Input("sensitive-selector", "value"),
         Input("threshold-slider", "value"),
         Input("horizon-metric-selector", "value"),
         Input("model-selector", "value")]
    )
    def update_horizon_chart(sensitive_col, threshold, metric_name, model_focus):
        """Atualiza o Horizon Graph de fairness."""
        # Para visualizações de fairness, "global" usa "sex" como default
        if sensitive_col == "global":
            sensitive_col = "sex"
        
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
            return 0.5, "both", "global", "balanced"
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
