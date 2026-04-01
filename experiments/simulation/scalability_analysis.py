import sys
import os
import numpy as np
import pandas as pd
import time
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# Add src directory to path to import project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from nci_nll_es_algorithm import (
    NCI_NLL_ES_Algorithm,
)
from synthetic_scenarios import EXPECTED_LABEL, SCENARIOS, generate_synthetic_data


def evaluate_scalability(
    dimensions: List[int] = [2, 5, 10, 20],
    num_trials: int = 50,
    N_per_trial: int = 100,
    n_samples_es: int = 800,
    base_seed: int = 1234,
) -> pd.DataFrame:
    """
    Evaluate algorithm scalability across different state dimensions.
    
    Measures:
    - Classification accuracy per scenario
    - Average computation time per trial
    - Total computation time per dimension
    
    Args:
        dimensions: List of state dimensions to test
        num_trials: Number of trials per scenario per dimension
        N_per_trial: Number of samples per trial
        n_samples_es: Number of ES Monte Carlo samples
        base_seed: Base random seed
        
    Returns:
        DataFrame with scalability results
    """
    rng = np.random.RandomState(base_seed)
    
    results = []
    
    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"Evaluating dimension: {dim}D")
        print(f"{'='*60}")
        
        # Initialize algorithm for this dimension
        algorithm = NCI_NLL_ES_Algorithm(state_dim=dim, n_samples=n_samples_es)
        
        # Track metrics per dimension
        dim_start_time = time.time()
        
        for scen in SCENARIOS:
            print(f"  Processing scenario: {scen}")
            
            correct_count = 0
            total_count = 0
            trial_times = []
            
            for trial in range(num_trials):
                # Use different seed per trial for independence
                seed = int(rng.randint(0, 2**31 - 1))
                scen_rng = np.random.RandomState(seed)
                
                # Generate synthetic data
                true_states, estimated_states, claimed_covariances, bias_vec = generate_synthetic_data(
                    scen, N=N_per_trial, d=dim, rng=scen_rng
                )
                
                # Measure computation time
                trial_start = time.time()
                result = algorithm.run_algorithm(true_states, estimated_states, claimed_covariances)
                trial_end = time.time()
                trial_time = trial_end - trial_start
                trial_times.append(trial_time)
                
                # Check accuracy
                predicted = result.classification
                expected = EXPECTED_LABEL[scen]
                
                total_count += 1
                if predicted == expected:
                    correct_count += 1
            
            dim_end_time = time.time()
            total_dim_time = dim_end_time - dim_start_time
            
            # Calculate statistics
            accuracy = correct_count / max(1, total_count)
            avg_trial_time = np.mean(trial_times)
            std_trial_time = np.std(trial_times)
            min_trial_time = np.min(trial_times)
            max_trial_time = np.max(trial_times)
            
            results.append({
                'Dimension': dim,
                'Scenario': scen,
                'Trials': total_count,
                'Correct': correct_count,
                'Accuracy': accuracy,
                'Avg_Time_per_Trial_s': avg_trial_time,
                'Std_Time_per_Trial_s': std_trial_time,
                'Min_Time_per_Trial_s': min_trial_time,
                'Max_Time_per_Trial_s': max_trial_time,
                'Total_Time_s': total_dim_time,
            })
            
            print(f"    Accuracy: {accuracy:.3f} ({correct_count}/{total_count})")
            print(f"    Avg time per trial: {avg_trial_time:.4f}s ± {std_trial_time:.4f}s")
    
    return pd.DataFrame(results)


def analyze_scalability_trends(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze scalability trends across dimensions.
    
    Returns:
        DataFrame with aggregated metrics per dimension
    """
    aggregated = []
    
    for dim in results_df['Dimension'].unique():
        dim_data = results_df[results_df['Dimension'] == dim]
        
        aggregated.append({
            'Dimension': dim,
            'Mean_Accuracy': dim_data['Accuracy'].mean(),
            'Std_Accuracy': dim_data['Accuracy'].std(),
            'Mean_Time_per_Trial_s': dim_data['Avg_Time_per_Trial_s'].mean(),
            'Std_Time_per_Trial_s': dim_data['Avg_Time_per_Trial_s'].std(),
            'Total_Trials': dim_data['Trials'].sum(),
            'Total_Time_s': dim_data['Total_Time_s'].iloc[0],  # Should be same for all scenarios
        })
    
    return pd.DataFrame(aggregated)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluate algorithm scalability across different state dimensions.'
    )
    parser.add_argument(
        '--dimensions', 
        type=int, 
        nargs='+', 
        default=[2, 5, 10, 20],
        help='State dimensions to test (default: 5 10 20)'
    )
    parser.add_argument(
        '--trials', 
        type=int, 
        default=50, 
        help='Number of trials per scenario per dimension (default: 30)'
    )
    parser.add_argument(
        '--N', 
        type=int, 
        default=100, 
        help='Number of samples per trial (default: 100)'
    )
    parser.add_argument(
        '--es', 
        type=int, 
        default=800, 
        help='ES Monte Carlo samples per evaluation (default: 800)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=22, 
        help='Base random seed (default: 22)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run scalability evaluation
    print("Starting scalability evaluation...")
    print(f"Dimensions: {args.dimensions}")
    print(f"Trials per scenario: {args.trials}")
    print(f"Samples per trial: {args.N}")
    print(f"ES samples: {args.es}")
    
    results_df = evaluate_scalability(
        dimensions=args.dimensions,
        num_trials=args.trials,
        N_per_trial=args.N,
        n_samples_es=args.es,
        base_seed=args.seed,
    )
    
    # Analyze trends
    trends_df = analyze_scalability_trends(results_df)
    
    # Print detailed results
    print('\n' + '='*80)
    print('DETAILED RESULTS (per dimension and scenario)')
    print('='*80)
    print(results_df.to_string(index=False))
    
    # Print aggregated trends
    print('\n' + '='*80)
    print('SCALABILITY TRENDS (aggregated per dimension)')
    print('='*80)
    print(trends_df.to_string(index=False))
    
    # Save results
    detailed_path = os.path.join(args.output_dir, 'scalability_detailed_results.csv')
    trends_path = os.path.join(args.output_dir, 'scalability_trends.csv')
    
    results_df.to_csv(detailed_path, index=False)
    trends_df.to_csv(trends_path, index=False)
    
    print(f'\nResults saved to:')
    print(f'  - {detailed_path}')
    print(f'  - {trends_path}')
    
    # Print summary statistics
    print('\n' + '='*80)
    print('SUMMARY STATISTICS')
    print('='*80)
    for dim in args.dimensions:
        dim_data = results_df[results_df['Dimension'] == dim]
        print(f"\n{dim}D:")
        print(f"  Overall Accuracy: {dim_data['Accuracy'].mean():.3f} ± {dim_data['Accuracy'].std():.3f}")
        print(f"  Avg Time per Trial: {dim_data['Avg_Time_per_Trial_s'].mean():.4f}s ± {dim_data['Avg_Time_per_Trial_s'].std():.4f}s")
    
    # Calculate scaling factor
    if len(args.dimensions) >= 2:
        print('\n' + '='*80)
        print('SCALING ANALYSIS')
        print('='*80)
        base_dim = args.dimensions[0]
        base_time = trends_df[trends_df['Dimension'] == base_dim]['Mean_Time_per_Trial_s'].values[0]
        
        for dim in args.dimensions[1:]:
            dim_time = trends_df[trends_df['Dimension'] == dim]['Mean_Time_per_Trial_s'].values[0]
            scaling_factor = dim_time / base_time
            dim_ratio = dim / base_dim
            print(f"{dim}D vs {base_dim}D:")
            print(f"  Dimension ratio: {dim_ratio:.1f}x")
            print(f"  Time ratio: {scaling_factor:.2f}x")
            print(f"  Scaling efficiency: {dim_ratio/scaling_factor:.2f}x (ideal = 1.0x)")


if __name__ == '__main__':
    main()
