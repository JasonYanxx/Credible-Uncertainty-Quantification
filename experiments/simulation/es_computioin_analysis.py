"""
Test computation time and accuracy of Energy Score (ES) against n_samples (M).

This script measures:
1. Computation time of single ES vs n_samples
2. Absolute mean difference vs reference (n_samples=5000) vs n_samples
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add src directory to path to import project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from nci_nll_es_algorithm import NCI_NLL_ES_Algorithm


def test_single_es_time(
    n_samples_list: List[int],
    state_dim: int = 2,
    num_runs: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Test computation time of single compute_energy_score call vs n_samples.
    
    Args:
        n_samples_list: List of n_samples (M) values to test
        state_dim: State dimension (d)
        num_runs: Number of runs per n_samples for averaging
        seed: Random seed
        
    Returns:
        DataFrame with columns: n_samples, mean_time_sec, std_time_sec
    """
    rng = np.random.RandomState(seed)
    
    rows = []
    for M in n_samples_list:
        algorithm = NCI_NLL_ES_Algorithm(state_dim=state_dim, n_samples=M)
        
        times = []
        for run in range(num_runs):
            # Generate random test data for each run
            true_state = rng.normal(size=state_dim)
            estimated_state = rng.normal(size=state_dim)
            covariance = np.eye(state_dim)
            
            t0 = time.perf_counter()
            _ = algorithm.compute_energy_score(true_state, estimated_state, covariance)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        rows.append({
            'n_samples': M,
            'mean_time_sec': mean_time,
            'std_time_sec': std_time,
        })
        print(f"  n_samples={M:5d}: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
    
    return pd.DataFrame(rows)


def plot_computation_time_and_difference(
    df_time: pd.DataFrame,
    df_diff: pd.DataFrame,
    save_path: str = None,
):
    """Plot computation time and absolute mean difference vs n_samples as two separate figures."""
    # Same figure size and resolution as es_scale_study.py
    target_pixel_width = 428
    target_pixel_height = 322
    base_dpi = 100
    fig_width_in = target_pixel_width / base_dpi
    fig_height_in = target_pixel_height / base_dpi
    resolution_dpi = 300
    label_fontsize = 10
    title_fontsize = 11
    legend_fontsize = 9

    # Figure 1: Computation time vs n_samples
    fig1, ax1 = plt.subplots(figsize=(fig_width_in, fig_height_in))
    ax1.plot(df_time['n_samples'], df_time['mean_time_sec'] * 1000,
             '-o', linewidth=2, markersize=6, color='blue', label='Mean time')
    ax1.set_xlabel('M', fontsize=label_fontsize)
    ax1.set_ylabel('Mean computation time (ms)', fontsize=label_fontsize)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=legend_fontsize)
    fig1.tight_layout()
    if save_path:
        base, ext = os.path.splitext(save_path)
        path_time = base + '_time' + (ext or '.png')
        os.makedirs(os.path.dirname(path_time) or '.', exist_ok=True)
        fig1.savefig(path_time, dpi=resolution_dpi, bbox_inches='tight')
        print(f"Figure saved to {path_time}")
    plt.show()

    # Figure 2: Absolute mean difference vs n_samples
    fig2, ax2 = plt.subplots(figsize=(fig_width_in, fig_height_in))
    ax2.plot(df_diff['n_samples'], df_diff['mean_abs_diff'],
             '-s', linewidth=2, markersize=6, color='green', label='Mean abs. diff.')
    ax2.set_xlabel('M', fontsize=label_fontsize)
    ax2.set_ylabel('Mean absolute difference', fontsize=label_fontsize)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=legend_fontsize)
    fig2.tight_layout()
    if save_path:
        base, ext = os.path.splitext(save_path)
        path_diff = base + '_difference' + (ext or '.png')
        os.makedirs(os.path.dirname(path_diff) or '.', exist_ok=True)
        fig2.savefig(path_diff, dpi=resolution_dpi, bbox_inches='tight')
        print(f"Figure saved to {path_diff}")
    plt.show()


def compute_es_difference(
    n_samples_list: List[int],
    reference_n_samples: int = 5000,
    state_dim: int = 2,
    num_runs: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute ES difference between different n_samples and reference (n_samples=5000).
    
    For each n_samples, compute ES values multiple times and compare with reference.
    Calculate mean absolute difference and ±3σ bounds.
    
    Args:
        n_samples_list: List of n_samples values to test
        reference_n_samples: Reference n_samples value (default: 5000)
        state_dim: State dimension
        num_runs: Number of runs for each n_samples
        seed: Random seed
        
    Returns:
        DataFrame with columns: n_samples, mean_abs_diff, mean_diff, std_diff, 
                                lower_3sigma, upper_3sigma
    """
    rng = np.random.RandomState(seed)
    
    # Compute reference ES values (n_samples=5000)
    print(f"\nComputing reference ES values (n_samples={reference_n_samples})...")
    ref_algorithm = NCI_NLL_ES_Algorithm(state_dim=state_dim, n_samples=reference_n_samples)
    ref_es_values = []
    
    for run in range(num_runs):
        true_state = rng.normal(size=state_dim)
        estimated_state = rng.normal(size=state_dim)
        covariance = np.eye(state_dim)
        es_val = ref_algorithm.compute_energy_score(true_state, estimated_state, covariance)
        ref_es_values.append(es_val)
    
    ref_es_values = np.array(ref_es_values)
    print(f"  Reference ES: mean={np.mean(ref_es_values):.6f}, std={np.std(ref_es_values):.6f}")
    
    rows = []
    for M in n_samples_list:
        print(f"\nComputing ES for n_samples={M}...")
        algorithm = NCI_NLL_ES_Algorithm(state_dim=state_dim, n_samples=M)
        
        # Use same random seeds for fair comparison - generate data once
        rng_data = np.random.RandomState(seed)
        es_values = []
        differences = []
        
        for run in range(num_runs):
            # Generate same input data for both algorithms
            true_state = rng_data.normal(size=state_dim)
            estimated_state = rng_data.normal(size=state_dim)
            covariance = np.eye(state_dim)
            
            # Compute ES with current n_samples
            es_val = algorithm.compute_energy_score(true_state, estimated_state, covariance)
            es_values.append(es_val)
            
            # Compute ES with reference n_samples using same input data
            # Need to reset RNG for reference to use same random numbers for sampling
            ref_es_val = ref_algorithm.compute_energy_score(true_state, estimated_state, covariance)
            diff = es_val - ref_es_val
            differences.append(diff)
        
        es_values = np.array(es_values)
        differences = np.array(differences)
        
        mean_abs_diff = np.mean(np.abs(differences))
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        lower_3sigma = mean_diff - 3 * std_diff
        upper_3sigma = mean_diff + 3 * std_diff
        
        rows.append({
            'n_samples': M,
            'mean_abs_diff': mean_abs_diff,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'lower_3sigma': lower_3sigma,
            'upper_3sigma': upper_3sigma,
        })
        
        print(f"  ES: mean={np.mean(es_values):.6f}, std={np.std(es_values):.6f}")
        print(f"  Difference vs reference: mean={mean_diff:.6f}, std={std_diff:.6f}")
        print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"  ±3σ range: [{lower_3sigma:.6f}, {upper_3sigma:.6f}]")
    
    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Test ES computation time vs n_samples and N'
    )
    parser.add_argument('--n-samples', type=int, nargs='+',
                        default=[50, 100, 200, 500, 800, 1000, 2000],
                        help='n_samples (M) values to test')
    parser.add_argument('--dim', type=int, default=2,
                        help='State dimension')
    parser.add_argument('--runs-time', type=int, default=1000,
                        help='Number of runs for computation time test')
    parser.add_argument('--runs-diff', type=int, default=500,
                        help='Number of runs for difference test')
    parser.add_argument('--reference-n-samples', type=int, default=5000,
                        help='Reference n_samples for difference test (default: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save figure (default: results/es_analysis.png)')
    parser.add_argument('--save-csv', type=str, default=None,
                        help='Path prefix to save CSV (default: results/es_analysis)')
    args = parser.parse_args()
    
    results_dir = 'results'
    save_path = args.save or os.path.join(results_dir, 'es_analysis.png')
    csv_prefix = args.save_csv or os.path.join(results_dir, 'es_analysis')
    
    print("=" * 70)
    print("ES Computation Time and Accuracy Analysis")
    print("=" * 70)
    print(f"State dimension: {args.dim}")
    print(f"n_samples values: {args.n_samples}")
    print(f"Reference n_samples: {args.reference_n_samples}")
    print("=" * 70)
    
    # Test 1: Computation time vs n_samples
    print("\n--- Test 1: Computation time vs n_samples ---")
    df_time = test_single_es_time(
        n_samples_list=args.n_samples,
        state_dim=args.dim,
        num_runs=args.runs_time,
        seed=args.seed,
    )
    csv_time = csv_prefix + '_time.csv'
    os.makedirs(os.path.dirname(csv_time) or '.', exist_ok=True)
    df_time.to_csv(csv_time, index=False)
    print(f"Results saved to {csv_time}")
    
    # Test 2: ES difference vs reference
    print("\n--- Test 2: ES difference vs reference ---")
    df_diff = compute_es_difference(
        n_samples_list=args.n_samples,
        reference_n_samples=args.reference_n_samples,
        state_dim=args.dim,
        num_runs=args.runs_diff,
        seed=args.seed,
    )
    csv_diff = csv_prefix + '_difference.csv'
    os.makedirs(os.path.dirname(csv_diff) or '.', exist_ok=True)
    df_diff.to_csv(csv_diff, index=False)
    print(f"Results saved to {csv_diff}")
    
    # Plot combined figure
    print("\n--- Generating combined plot ---")
    plot_computation_time_and_difference(df_time, df_diff, save_path=save_path)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
