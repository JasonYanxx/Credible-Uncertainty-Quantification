# Reproducibility Guide

This guide follows the paper presentation order as closely as possible.

## Environment

```bash
pip install -r requirements.txt
```

## Paper Order

### Fig. 1

```bash
MPLBACKEND=Agg python experiments/simulation/es_scale_study.py --output-path results/es_scale.png
MPLBACKEND=Agg python experiments/simulation/es_bias_study.py --output-path results/es_bias.png
```

### Table I

Not a script-generated item.

### Fig. 2

Not a script-generated item.

### Table II

```bash
MPLBACKEND=Agg python experiments/real_data/algorithm_comparison.py --output-dir results
```

### Table III

```bash
MPLBACKEND=Agg python experiments/simulation/experiment_accuracy.py --output-dir results
```

### Fig. 3

```bash
MPLBACKEND=Agg python experiments/simulation/sensitivity_analysis.py --save results/sensitivity_analysis.png --save-csv results/sensitivity_analysis_results.csv
```

### Fig. 4

```bash
MPLBACKEND=Agg python experiments/simulation/scalability_analysis.py --dimensions 2 5 10 20 50 100 --output-dir results
MPLBACKEND=Agg python experiments/simulation/visualize_scalability.py --input results/scalability_detailed_results.csv --output results/scalability_accuracy_plot.png
```

## Real-Data Preparation

Before running the remaining items, place these files under `data/external/`:

- `starloc_data_grid_s3_uwb.csv`
- `uwb_markers_v2.csv`

### Fig. 5

```bash
MPLBACKEND=Agg python experiments/real_data/static_period_extraction.py
MPLBACKEND=Agg python experiments/real_data/uwb_static_positioning.py
```

### Fig. 6

Produced by:

```bash
MPLBACKEND=Agg python experiments/real_data/uwb_static_positioning.py
```

### Fig. 7

Produced by:

```bash
MPLBACKEND=Agg python experiments/real_data/uwb_static_positioning.py
```

### Table IV

```bash
MPLBACKEND=Agg python experiments/real_data/uwb_credibility_classification.py
```

### Fig. 8

```bash
MPLBACKEND=Agg python experiments/simulation/elt_B_analysis.py --output-dir results
```

### Fig. 9

```bash
MPLBACKEND=Agg python experiments/simulation/es_computioin_analysis.py --output-dir results
```

## Main Output Files

Typical outputs include:

- `results/algorithm_comparison_summary.csv`
- `results/accuracy_results.csv`
- `results/confusion_matrix.csv`
- `results/sensitivity_analysis.png`
- `results/scalability_accuracy_plot.png`
- `results/static_periods.csv`
- `results/uwb_trajectory_with_covariance.csv`
- `results/uwb_credibility_classification_results.csv`

Use [docs/paper-traceability.md](docs/paper-traceability.md) to map each paper item to its script and outputs.
