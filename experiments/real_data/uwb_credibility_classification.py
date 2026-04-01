#!/usr/bin/env python3
"""
UWB Credibility Classification using Multiple Algorithms
Applies NCI_NLL_ES, Pure_NLL, Pure_ES, NEES_Chi2, Pure_NCI, and Energy_Distance
algorithms to classify the credibility of UWB positioning results per period.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Add parent directory to path to import modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Import existing classifiers from reusable modules
from baseline_classifiers import (
    PureNLLClassifier,
    PureESClassifier, 
    NEESChiSquaredClassifier,
    PureNCIClassifier,
)
# Import the NCI_NLL_ES algorithm
from nci_nll_es_algorithm import NCI_NLL_ES_Algorithm
from uwb_period_data import load_uwb_period_data as load_uwb_data


def classify_uwb_periods(periods_data: Dict[int, Dict], state_dim: int = 2, 
                        n_samples_es: int = 800) -> pd.DataFrame:
    """Apply all classification algorithms to UWB periods"""
    
    # Initialize all classifiers
    classifiers = {
        # 'Pure_NLL': PureNLLClassifier(state_dim=state_dim),
        # 'Pure_ES': PureESClassifier(state_dim=state_dim, n_samples=n_samples_es),
        'NEES_Chi2': NEESChiSquaredClassifier(state_dim=state_dim),
        'Pure_NCI': PureNCIClassifier(state_dim=state_dim),
    }
    
    # Add NCI_NLL_ES if available
    if NCI_NLL_ES_Algorithm is not None:
        classifiers['NCI_NLL_ES'] = NCI_NLL_ES_Algorithm(state_dim=state_dim, n_samples=n_samples_es)
    
    results = []
    
    print(f"\nClassifying {len(periods_data)} periods using {len(classifiers)} algorithms...")
    
    for period_id, measurements in periods_data.items():
        print(f"Processing period {period_id}...")
        
        # Convert measurements to arrays
        N = len(measurements)
        true_states = np.array([m['true_pos'] for m in measurements])
        estimated_states = np.array([m['estimated_pos'] for m in measurements])
        claimed_covariances = np.array([m['claimed_cov'] for m in measurements])
        
        period_result = {'period_id': period_id, 'num_measurements': N}
        
        # Apply each classifier
        for alg_name, classifier in classifiers.items():
            try:
                if alg_name == 'NCI_NLL_ES':
                    result = classifier.run_algorithm(true_states, estimated_states, claimed_covariances)
                    classification = result.classification
                else:
                    classification = classifier.classify(true_states, estimated_states, claimed_covariances)
                
                period_result[alg_name] = classification
                
            except Exception as e:
                print(f"  Error in {alg_name}: {e}")
                period_result[alg_name] = 'Error'
        
        # # Ground truth-based classification
        # estimation_errors = estimated_states - true_states  # Shape: (N, 2)
        
        # # Calculate empirical covariance of estimation errors
        # error_mean = np.mean(estimation_errors, axis=0)
        # error_cov = np.cov(estimation_errors.T)
        # error_trace = np.trace(error_cov)
        
        # # Calculate average trace of claimed covariances
        # claimed_traces = [np.trace(cov) for cov in claimed_covariances]
        # mean_claimed_trace = np.mean(claimed_traces)
        
        # # Bias detection using the E|L|T statistic from NCI_NLL_ES_Algorithm
        # # This is a more robust test for bias than a simple threshold on error mean
        # try:
        #     elt_stat = NCI_NLL_ES_Algorithm(state_dim=2, n_samples=800).compute_elt(
        #         true_states, estimated_states, claimed_covariances
        #     )
        #     # Heuristic: if E|L|T > 2.5, consider as biased (threshold can be tuned)
        #     is_biased = elt_stat !=0
        # except Exception as e:
        #     print(f"  Warning: Failed to compute E|L|T for bias detection: {e}")
        #     # Fallback to old method if E|L|T fails
        #     error_std = np.sqrt(np.diag(error_cov))
        #     bias_threshold = 1.0  # 1-sigma threshold
        #     is_biased = np.any(np.abs(error_mean) > bias_threshold * error_std)
        
        # # Determine ground truth classification
        # if is_biased:
        #     # Trace comparison for optimistic/pessimistic classification
        #     if error_trace < 0.8 * mean_claimed_trace:
        #         gt_classification = "Pessimistic + Bias"
        #     elif error_trace > 1.2 * mean_claimed_trace:
        #         gt_classification = "Optimistic + Bias"
        #     else:
        #         gt_classification = "Bias"
        # else:
        #     # Trace comparison for optimistic/pessimistic classification
        #     if error_trace < 0.8 * mean_claimed_trace:
        #         gt_classification = "Pessimistic"
        #     elif error_trace > 1.2 * mean_claimed_trace:
        #         gt_classification = "Optimistic"
        #     else:
        #         gt_classification = "Calibrated"
        
        # period_result['Ground_Truth'] = gt_classification
        
        results.append(period_result)

    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame):
    """Analyze and summarize classification results"""
    print("\n" + "=" * 80)
    print("UWB CREDIBILITY CLASSIFICATION RESULTS")
    print("=" * 80)
    
    # Display results table
    print("\nClassification Results by Period:")
    print(results_df.to_string(index=False))


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run UWB credibility classification for the paper workflow.")
    parser.add_argument(
        "--input-csv",
        default=os.path.join(PROJECT_ROOT, "results", "uwb_trajectory_with_covariance.csv"),
        help="Path to the trajectory-with-covariance CSV generated by the positioning workflow",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(PROJECT_ROOT, "results", "uwb_credibility_classification_results.csv"),
        help="Path to save the classification table",
    )
    parser.add_argument(
        "--detail-output",
        default=os.path.join(PROJECT_ROOT, "results", "uwb_credibility_detailed_analysis.txt"),
        help="Path to save the detailed text summary",
    )
    parser.add_argument("--state-dim", type=int, default=3, help="State dimension of the positioning data")
    parser.add_argument("--es-samples", type=int, default=800, help="Monte Carlo samples for ES evaluations")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Load UWB data
    periods_data = load_uwb_data(args.input_csv)
    
    if not periods_data:
        print("No data loaded. Please check the CSV file path and format.")
        return
    
    # Classify periods
    results_df = classify_uwb_periods(periods_data, args.state_dim, args.es_samples)
    
    # Analyze results
    analyze_results(results_df)
    
    # Save results
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    
    # Create detailed output with statistics
    with open(args.detail_output, 'w') as f:
        f.write("UWB Credibility Classification - Detailed Analysis\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Classification Results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
    
    print(f"Detailed analysis saved to {args.detail_output}")


if __name__ == "__main__":
    main()
