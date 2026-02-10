"""
Módulo de gráficos.
"""
from src.charts.view1_global import (
    create_metrics_comparison_chart,
    create_roc_curves,
    create_calibration_plot,
    create_advanced_calibration_plot,
    create_calibration_subgroup_comparison,
    compute_calibration_bins,
    generate_calibration_insight,
    get_fairness_warnings,
    compute_subgroup_metrics,
    METRIC_EXPLANATIONS,
    DECISION_MODE_CONFIG,
    CALIBRATION_DECISION_CONFIG
)
from src.charts.view2_tradeoffs import (
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
    get_operating_point_details,
    PR_DECISION_MODE_CONFIG,
    THRESHOLD_DECISION_MODE_CONFIG,
    FP_FN_DECISION_MODE_CONFIG,
    PREDICTION_DIST_MODE_CONFIG,
    PCP_DECISION_MODE_CONFIG
)
from src.charts.view3_errors import (
    create_confusion_matrix_heatmap,
    create_advanced_confusion_matrix,
    compute_confusion_matrix_data,
    create_error_rates_comparison,
    create_error_tradeoff_scatter,
    compute_error_tradeoff_data
)
from src.charts.view4_fairness import (
    create_fairness_horizon_chart,
    create_fairness_sunburst,
    compute_fairness_metrics_grid,
    HORIZON_METRIC_CONFIG
)


__all__ = [
    # View 1
    "create_metrics_comparison_chart",
    "create_roc_curves",
    "create_calibration_plot",
    "create_advanced_calibration_plot",
    "create_calibration_subgroup_comparison",
    "compute_calibration_bins",
    "generate_calibration_insight",
    "get_fairness_warnings",
    "compute_subgroup_metrics",
    "METRIC_EXPLANATIONS",
    "DECISION_MODE_CONFIG",
    "CALIBRATION_DECISION_CONFIG",
    # View 2
    "create_precision_recall_curve",
    "create_precision_recall_curve_enhanced",
    "create_threshold_analysis",
    "create_threshold_analysis_enhanced",
    "create_fp_fn_evolution_chart",
    "create_fp_fn_evolution_enhanced",
    "create_threshold_impact_bars",
    "create_prediction_distribution_enhanced",
    "build_operating_points_df",
    "create_parallel_coordinates_operating_points",
    "create_selected_operating_points_table",
    "get_operating_point_details",
    "PR_DECISION_MODE_CONFIG",
    "THRESHOLD_DECISION_MODE_CONFIG",
    "FP_FN_DECISION_MODE_CONFIG",
    "PREDICTION_DIST_MODE_CONFIG",
    "PCP_DECISION_MODE_CONFIG",
    # View 3
    "create_confusion_matrix_heatmap",
    "create_advanced_confusion_matrix",
    "compute_confusion_matrix_data",
    "create_error_rates_comparison",
    "create_error_tradeoff_scatter",
    "compute_error_tradeoff_data",
    # View 4
    "create_fairness_horizon_chart",
    "create_fairness_sunburst",
    "compute_fairness_metrics_grid",
    "HORIZON_METRIC_CONFIG"
]
