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
    create_threshold_analysis,
    create_fp_fn_evolution_chart,
    create_threshold_impact_bars
)
from src.charts.view3_errors import (
    create_confusion_matrix_heatmap,
    create_error_distribution_by_feature,
    create_error_rates_comparison
)
from src.charts.view4_fairness import (
    create_fairness_accuracy_chart,
    create_fairness_rates_chart,
    create_fairness_disparity_chart
)
from src.charts.advanced import (
    create_parallel_coordinates,
    create_radar_chart,
    create_sunburst_errors
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
    "create_threshold_analysis",
    "create_fp_fn_evolution_chart",
    "create_threshold_impact_bars",
    # View 3
    "create_confusion_matrix_heatmap",
    "create_error_distribution_by_feature",
    "create_error_rates_comparison",
    # View 4
    "create_fairness_accuracy_chart",
    "create_fairness_rates_chart",
    "create_fairness_disparity_chart",
    # Advanced
    "create_parallel_coordinates",
    "create_radar_chart",
    "create_sunburst_errors"
]
