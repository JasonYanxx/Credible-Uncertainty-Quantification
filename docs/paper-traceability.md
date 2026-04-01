# Paper Traceability

This note maps the paper's figures and tables to the scripts in this public package.

## Method Modules

- core unified method: `src/nci_nll_es_algorithm.py`
- baseline methods: `src/baseline_classifiers.py`
- synthetic data generation: `src/synthetic_scenarios.py`
- UWB period-data loading: `src/uwb_period_data.py`

## Figures And Tables


| Paper Item | Script(s)                                                                                              | Main Output(s)                                                                                                                                            |
| ---------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fig. 1     | `experiments/simulation/es_scale_study.py`, `experiments/simulation/es_bias_study.py`                  | `results/es_scale.png`, `results/es_bias.png`                                                                                                             |
| Table I    | not script-generated                                                                                   | not applicable                                                                                                                                            |
| Fig. 2     | not script-generated                                                                                   | not applicable                                                                                                                                            |
| Table II   | `experiments/real_data/algorithm_comparison.py`                                                        | `results/algorithm_comparison_summary.csv`                                                                                                                |
| Table III  | `experiments/simulation/experiment_accuracy.py`                                                        | `results/accuracy_results.csv`, `results/confusion_matrix.csv`                                                                                            |
| Fig. 3     | `experiments/simulation/sensitivity_analysis.py`                                                       | `results/sensitivity_analysis.png`, `results/sensitivity_analysis_results.csv`                                                                            |
| Fig. 4     | `experiments/simulation/scalability_analysis.py`, `experiments/simulation/visualize_scalability.py`    | `results/scalability_trends.csv`, `results/scalability_accuracy_plot.png`                                                                                 |
| Fig. 5     | `experiments/real_data/static_period_extraction.py`, `experiments/real_data/uwb_static_positioning.py` | `results/static_period_xyz.png`, `results/uwb_positioning_analysis_3d.png`                                                                                |
| Fig. 6     | `experiments/real_data/uwb_static_positioning.py`                                                      | `results/uwb_bias_calib_boxplots_by_period.png`                                                                                                           |
| Fig. 7     | `experiments/real_data/uwb_static_positioning.py`                                                      | `results/uwb_std_boxplots_with_bias_calib.png`                                                                                                            |
| Table IV   | `experiments/real_data/uwb_credibility_classification.py`                                              | `results/uwb_credibility_classification_results.csv`                                                                                                      |
| Fig. 8     | `experiments/simulation/elt_B_analysis.py`                                                             | `results/elt_B_impact_testing_power_time_time_vs_B.png`, `results/elt_B_impact_testing_power_miss_vs_B.png`, `results/elt_B_impact_stability_heatmap.png` |
| Fig. 9     | `experiments/simulation/es_computioin_analysis.py`                                                     | `results/es_analysis_time.png`, `results/es_analysis_difference.png`                                                                                      |


## Real-Data Dependency Chain

1. `static_period_extraction.py` produces `results/static_periods.csv`
2. `uwb_static_positioning.py` consumes `results/static_periods.csv` and produces `results/uwb_trajectory_with_covariance.csv`
3. `uwb_credibility_classification.py` consumes `results/uwb_trajectory_with_covariance.csv`
