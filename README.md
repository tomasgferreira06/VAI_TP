# Model Evaluation Dashboard: Classification Beyond Accuracy

Interactive visual analytics dashboard for comparing classification models across metrics, decision thresholds, error types, and demographic subgroups. Built with Plotly/Dash, the system uses coordinated views to expose trade-offs that aggregate metrics hide.

> Developed as part of the **Visualisation for Artificial Intelligence** course at the University of Coimbra, 2025/2026.

---

## Problem

Classification models are often evaluated through a single metric (accuracy, F1). This obscures how performance shifts when the decision threshold changes, which error types dominate, and whether the model behaves differently across demographic groups. The dashboard addresses this by providing interactive, linked views that let the user explore these dimensions simultaneously.

---

## Data

The system uses the [Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult) (UCI). Binary classification task: predict whether annual income exceeds 50K based on sociodemographic and professional attributes. The dataset has imbalanced classes (~75% negative), which makes threshold-sensitive analysis and fairness auditing relevant.

- **Train:** 32,561 samples, 14 features (6 numerical, 8 categorical)
- **Test:** 16,281 samples

---

## Models

Two scikit-learn pipelines are trained and compared:

- **Logistic Regression** (with StandardScaler + OneHotEncoder preprocessing)
- **Random Forest** (same preprocessing pipeline)

Both produce probability outputs, allowing threshold-based analysis across the full [0, 1] range.

---

## Dashboard Views

The interface is organised in four tabs, all linked by a global control panel (threshold slider, model selector, subgroup selector).

**Global Comparison.** Side-by-side metric cards (accuracy, precision, recall, F1), bar charts with absolute/relative toggle, ROC curves, and calibration plots with per-subgroup breakdowns.

**Trade-offs.** Precision-Recall curves, threshold impact analysis showing how metrics shift continuously, FP/FN count evolution, prediction distribution histograms, and a Parallel Coordinates plot for operating point analysis with brushing interaction.

**Error Analysis.** Confusion matrices, error rate comparison between models, and a Connected Bubble Scatter Plot that traces each model's trajectory in FPR × FNR space as the threshold varies.

**Fairness.** Horizon Graph showing error rates (FPR, FNR) per demographic group across thresholds, with a disparity band that reveals where and how much groups diverge. Sunburst chart for hierarchical subgroup decomposition.

---

## Visualisation Techniques

| Technique | Purpose |
|-----------|---------|
| Horizon Graph | Fairness metrics across thresholds, compact multi-group comparison |
| Connected Bubble Scatter | Error trade-off trajectories in FPR × FNR space |
| Parallel Coordinates | Operating point analysis with brushing |
| Coordinated Multiple Views | Global controls propagate to all tabs |

---

## How to Run

```bash
git clone https://github.com/tomasgferreira06/VAI_TP.git
cd VAI_TP

pip install -r requirements.txt

python run.py
```

On first run, models are trained and cached. Subsequent launches load from cache. To retrain, delete the `.cache/` directory.

---

## Authors

- **Tomás Ferreira** · 2025168427
- **Eduardo Pereira** · 2021233890

University of Coimbra · Visualisation for Artificial Intelligence, 2025/2026