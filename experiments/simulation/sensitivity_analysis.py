import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict

# Add src directory to path to import project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from nci_nll_es_algorithm import NCI_NLL_ES_Algorithm
from synthetic_scenarios import EXPECTED_LABEL, SCENARIOS, generate_synthetic_data


def evaluate_accuracy_for_param_value(
    param_name: str,
    param_value: float,
    num_trials: int,
    N_per_trial: int,
    state_dim: int,
    n_samples_es: int,
    base_seed: int,
    default_tau_nci: float = 0.5,
    default_alpha_sig: float = 0.05,
    default_prop_scale: float = 2.0,
) -> float:
    """
    Evaluate average accuracy for a given parameter value.
    
    Returns:
        Average accuracy across all scenarios and trials
    """
    rng = np.random.RandomState(base_seed)
    
    # Create algorithm instance with default parameters
    algorithm = NCI_NLL_ES_Algorithm(state_dim=state_dim, n_samples=n_samples_es)
    algorithm.tau_nci = default_tau_nci
    algorithm.alpha_sig = default_alpha_sig
    algorithm.prop_scale = default_prop_scale
    
    # Override the parameter being tested
    if param_name == 'tau_nci':
        algorithm.tau_nci = param_value
    elif param_name == 'alpha_sig':
        algorithm.alpha_sig = param_value
    elif param_name == 'prop_scale':
        algorithm.prop_scale = param_value
    else:
        raise ValueError(f"Unknown parameter: {param_name}")
    
    correct_counts = 0
    total_counts = 0
    
    for trial in range(num_trials):
        for scen in SCENARIOS:
            seed = int(rng.randint(0, 2**31 - 1))
            scen_rng = np.random.RandomState(seed)
            true_states, estimated_states, claimed_covariances, bias_vec = generate_synthetic_data(
                scen, N=N_per_trial, d=state_dim, rng=scen_rng
            )
            
            result = algorithm.run_algorithm(true_states, estimated_states, claimed_covariances)
            
            predicted = result.classification
            expected = EXPECTED_LABEL[scen]
            
            total_counts += 1
            if predicted == expected:
                correct_counts += 1
    
    return correct_counts / max(1, total_counts)


# Scenarios with no bias (ELT should be 0); bias-related scenarios (ELT should be 1)
NO_BIAS_SCENARIOS = ['calibrated', 'optimistic', 'pessimistic']
BIAS_SCENARIOS = ['bias', 'mixed_o_b', 'mixed_p_b']
BIAS_RELATED_LABELS = {'Bias', 'Optimistic + Bias', 'Pessimistic + Bias'}


def analyze_elt_B_impact(
    B_values: List[int],
    num_trials: int = 50,
    N_per_trial: int = 5000,
    state_dim: int = 2,
    n_samples_es: int = 800,
    base_seed: int = 1234,
    default_tau_nci: float = 0.5,
    default_alpha_sig: float = 0.05,
    default_prop_scale: float = 2.0,
) -> pd.DataFrame:
    """
    Analyze the impact of ELT parameter B (number of sign-flip randomizations) on
    diagnosis: false alarm rate, miss detection rate, and mean computation time.

    - False alarm: true scenario has no bias (calibrated/optimistic/pessimistic) but
      the algorithm reports a bias-related class (Bias, Optimistic + Bias, Pessimistic + Bias).
    - Miss detection: true scenario has bias (bias/mixed_o_b/mixed_p_b) but the
      algorithm reports a non-bias class (Calibrated, Optimistic, Pessimistic).

    Returns:
        DataFrame with columns: B, false_alarm_rate, miss_detect_rate, mean_time_sec, accuracy
    """
    rng = np.random.RandomState(base_seed)
    algorithm = NCI_NLL_ES_Algorithm(state_dim=state_dim, n_samples=n_samples_es)
    algorithm.tau_nci = default_tau_nci
    algorithm.alpha_sig = default_alpha_sig
    algorithm.prop_scale = default_prop_scale

    rows = []
    for B in B_values:
        algorithm.elt_B = B
        false_alarms = 0
        no_bias_total = 0
        miss_detects = 0
        bias_total = 0
        correct_counts = 0
        total_counts = 0
        time_list = []

        for trial in range(num_trials):
            for scen in SCENARIOS:
                seed = int(rng.randint(0, 2**31 - 1))
                scen_rng = np.random.RandomState(seed)
                true_states, estimated_states, claimed_covariances, bias_vec = generate_synthetic_data(
                    scen, N=N_per_trial, d=state_dim, rng=scen_rng
                )
                t0 = time.perf_counter()
                result = algorithm.run_algorithm(true_states, estimated_states, claimed_covariances)
                t1 = time.perf_counter()
                time_list.append(t1 - t0)

                pred = result.classification
                expected = EXPECTED_LABEL[scen]
                total_counts += 1
                if pred == expected:
                    correct_counts += 1

                if scen in NO_BIAS_SCENARIOS:
                    no_bias_total += 1
                    if pred in BIAS_RELATED_LABELS:
                        false_alarms += 1
                else:
                    assert scen in BIAS_SCENARIOS
                    bias_total += 1
                    if pred not in BIAS_RELATED_LABELS:
                        miss_detects += 1

        false_alarm_rate = false_alarms / max(1, no_bias_total)
        miss_detect_rate = miss_detects / max(1, bias_total)
        mean_time_sec = np.mean(time_list)
        accuracy = correct_counts / max(1, total_counts)

        rows.append({
            'B': B,
            'false_alarm_rate': false_alarm_rate,
            'miss_detect_rate': miss_detect_rate,
            'mean_time_sec': mean_time_sec,
            'accuracy': accuracy,
        })
        print(f"  B={B}: FA={false_alarm_rate:.4f}, MD={miss_detect_rate:.4f}, "
              f"time={mean_time_sec:.4f}s, acc={accuracy:.4f}")

    return pd.DataFrame(rows)


def save_elt_B_results_to_csv(
    df: pd.DataFrame,
    csv_path: str = 'results/elt_B_impact_results.csv',
) -> None:
    """Save ELT B impact analysis results to CSV."""
    df.to_csv(csv_path, index=False)
    print(f"ELT B impact results saved to {csv_path}")


def plot_elt_B_impact(
    df: pd.DataFrame,
    save_path: str = None,
) -> None:
    """Plot ELT B impact: false alarm rate, miss detection rate, and mean time vs B."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    B_vals = df['B'].values

    axes[0].plot(B_vals, df['false_alarm_rate'], 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel(r'$B$ (ELT sign-flip samples)', fontsize=11)
    axes[0].set_ylabel('False alarm rate', fontsize=11)
    axes[0].set_title('(a) False alarm rate vs. $B$', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    axes[1].plot(B_vals, df['miss_detect_rate'], 'r-s', linewidth=2, markersize=6)
    axes[1].set_xlabel(r'$B$ (ELT sign-flip samples)', fontsize=11)
    axes[1].set_ylabel('Miss detection rate', fontsize=11)
    axes[1].set_title('(b) Miss detection rate vs. $B$', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    axes[2].plot(B_vals, df['mean_time_sec'], 'g-^', linewidth=2, markersize=6)
    axes[2].set_xlabel(r'$B$ (ELT sign-flip samples)', fontsize=11)
    axes[2].set_ylabel('Mean time (s)', fontsize=11)
    axes[2].set_title('(c) Computation time vs. $B$', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ELT B impact figure saved to {save_path}")
    plt.show()


def sensitivity_analysis(
    num_trials: int = 50,
    N_per_trial: int = 5000,
    state_dim: int = 2,
    n_samples_es: int = 800,
    base_seed: int = 1234,
    tau_nci_range: Tuple[float, float] = (0.1, 2.0),
    prop_scale_range: Tuple[float, float] = (1.0, 5.0),
    n_points: int = 10,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Perform sensitivity analysis for tau_nci, alpha_sig, and prop_scale.
    
    Returns:
        Dictionary with keys 'tau_nci', 'alpha_sig', 'prop_scale',
        each containing (param_values, accuracy_values) tuple
    """
    results = {}
    
    # Sensitivity analysis for tau_nci
    # Use fixed values: step 0.2, max 2.0, includes default 0.5
    print("Analyzing sensitivity to tau_nci...")
    tau_nci_values = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    tau_nci_accuracies = []
    for i, tau_val in enumerate(tau_nci_values):
        print(f"  tau_nci = {tau_val:.3f} ({i+1}/{len(tau_nci_values)})")
        acc = evaluate_accuracy_for_param_value(
            'tau_nci', tau_val, num_trials, N_per_trial, state_dim, n_samples_es, base_seed
        )
        tau_nci_accuracies.append(acc)
    results['tau_nci'] = (tau_nci_values, np.array(tau_nci_accuracies))
    
    # Sensitivity analysis for alpha_sig
    # Use fixed list of statistically meaningful values
    print("\nAnalyzing sensitivity to alpha_sig...")
    alpha_sig_values = np.array([0.001, 0.01, 0.05])
    alpha_sig_accuracies = []
    for i, alpha_val in enumerate(alpha_sig_values):
        print(f"  alpha_sig = {alpha_val:.4f} ({i+1}/{len(alpha_sig_values)})")
        acc = evaluate_accuracy_for_param_value(
            'alpha_sig', alpha_val, num_trials, N_per_trial, state_dim, n_samples_es, base_seed
        )
        alpha_sig_accuracies.append(acc)
    results['alpha_sig'] = (alpha_sig_values, np.array(alpha_sig_accuracies))
    
    # Sensitivity analysis for prop_scale
    # Use fixed values: step 0.5, max 5.0, includes default 2.0
    print("\nAnalyzing sensitivity to prop_scale...")
    prop_scale_values = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    prop_scale_accuracies = []
    for i, prop_val in enumerate(prop_scale_values):
        print(f"  prop_scale = {prop_val:.3f} ({i+1}/{len(prop_scale_values)})")
        acc = evaluate_accuracy_for_param_value(
            'prop_scale', prop_val, num_trials, N_per_trial, state_dim, n_samples_es, base_seed
        )
        prop_scale_accuracies.append(acc)
    results['prop_scale'] = (prop_scale_values, np.array(prop_scale_accuracies))
    
    return results


def save_sensitivity_results_to_csv(results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                     csv_path: str = 'results/sensitivity_analysis_results.csv'):
    """
    Save sensitivity analysis results to CSV file.
    
    Args:
        results: Dictionary from sensitivity_analysis()
        csv_path: Path to save the CSV file
    """
    rows = []
    for param_name, (param_vals, acc_vals) in results.items():
        for param_val, acc_val in zip(param_vals, acc_vals):
            rows.append({
                'parameter': param_name,
                'parameter_value': param_val,
                'accuracy': acc_val
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Sensitivity analysis results saved to {csv_path}")


def plot_sensitivity_analysis(results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                              save_path: str = None):
    """
    Plot sensitivity analysis results in three subplots.
    
    Args:
        results: Dictionary from sensitivity_analysis()
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Subplot (a): Accuracy vs. tau_nci
    ax_a = axes[0]
    param_vals, acc_vals = results['tau_nci']
    ax_a.plot(param_vals, acc_vals, 'b-', linewidth=2, marker='o', markersize=4)
    ax_a.axvline(x=0.5, color='r', linestyle='--', linewidth=1.5, label='Default (0.5)')
    ax_a.set_xlabel(r'$\tau_{\text{NCI}}$', fontsize=12)
    ax_a.set_ylabel('Average Diagnosis Accuracy', fontsize=11)
    ax_a.set_title('(a) Accuracy vs. $\\tau_{\\text{NCI}}$', fontsize=12, fontweight='bold')
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=9)
    ax_a.set_ylim([0, 1])
    
    # Subplot (b): Accuracy vs. alpha_sig
    ax_b = axes[1]
    param_vals, acc_vals = results['alpha_sig']
    ax_b.plot(param_vals, acc_vals, 'g-', linewidth=2, marker='s', markersize=8)
    ax_b.axvline(x=0.05, color='r', linestyle='--', linewidth=1.5, label='Default (0.05)')
    ax_b.set_xlabel(r'$\alpha_{\text{sig}}$', fontsize=12)
    ax_b.set_ylabel('Average Diagnosis Accuracy', fontsize=11)
    ax_b.set_title('(b) Accuracy vs. $\\alpha_{\\text{sig}}$', fontsize=12, fontweight='bold')
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(fontsize=9)
    ax_b.set_ylim([0, 1])
    # Use log scale for alpha_sig since values are on different orders of magnitude
    ax_b.set_xscale('log')
    
    # Subplot (c): Accuracy vs. prop_scale (c)
    ax_c = axes[2]
    param_vals, acc_vals = results['prop_scale']
    ax_c.plot(param_vals, acc_vals, 'm-', linewidth=2, marker='^', markersize=4)
    ax_c.axvline(x=2.0, color='r', linestyle='--', linewidth=1.5, label='Default (2.0)')
    ax_c.set_xlabel(r'$c$', fontsize=12)
    ax_c.set_ylabel('Average Diagnosis Accuracy', fontsize=11)
    ax_c.set_title('(c) Accuracy vs. $c$', fontsize=12, fontweight='bold')
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=9)
    ax_c.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")
    
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Sensitivity analysis for NCI-NLL-ES algorithm parameters.'
    )
    parser.add_argument('--trials', type=int, default=50, 
                       help='Number of trials per parameter value (default: 50)')
    parser.add_argument('--N', type=int, default=100, 
                       help='Number of samples per trial (default: 100)')
    parser.add_argument('--dim', type=int, default=2, 
                       help='State dimension (default: 2)')
    parser.add_argument('--es', type=int, default=800, 
                       help='ES Monte Carlo samples (default: 800)')
    parser.add_argument('--seed', type=int, default=22, 
                       help='Base random seed (default: 22)')
    parser.add_argument('--n-points', type=int, default=10, 
                       help='Number of parameter values to test (default: 10)')
    parser.add_argument('--tau-min', type=float, default=0.1, 
                       help='Minimum tau_nci value (default: 0.1)')
    parser.add_argument('--tau-max', type=float, default=2.0, 
                       help='Maximum tau_nci value (default: 2.0)')
    parser.add_argument('--prop-min', type=float, default=1.0, 
                       help='Minimum prop_scale value (default: 1.0)')
    parser.add_argument('--prop-max', type=float, default=5.0, 
                       help='Maximum prop_scale value (default: 5.0)')
    parser.add_argument('--save', type=str, default=None, 
                       help='Path to save the figure (default: None, shows plot)')
    parser.add_argument('--save-csv', type=str, default=None, 
                       help='Path to save the CSV results (default: results/sensitivity_analysis_results.csv)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sensitivity Analysis for NCI-NLL-ES Algorithm")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Trials per parameter value: {args.trials}")
    print(f"  Samples per trial: {args.N}")
    print(f"  State dimension: {args.dim}")
    print(f"  ES samples: {args.es}")
    print(f"  Parameter points: {args.n_points}")
    print(f"  tau_nci range: [{args.tau_min}, {args.tau_max}]")
    print(f"  alpha_sig values: [0.001, 0.01, 0.05] (fixed, statistically meaningful)")
    print(f"  prop_scale range: [{args.prop_min}, {args.prop_max}]")
    print("=" * 60)
    
    results = sensitivity_analysis(
        num_trials=args.trials,
        N_per_trial=args.N,
        state_dim=args.dim,
        n_samples_es=args.es,
        base_seed=args.seed,
        tau_nci_range=(args.tau_min, args.tau_max),
        prop_scale_range=(args.prop_min, args.prop_max),
        n_points=args.n_points,
    )
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for param_name, (param_vals, acc_vals) in results.items():
        max_idx = np.argmax(acc_vals)
        print(f"\n{param_name}:")
        print(f"  Max accuracy: {acc_vals[max_idx]:.4f} at {param_name} = {param_vals[max_idx]:.4f}")
        print(f"  Mean accuracy: {np.mean(acc_vals):.4f}")
        print(f"  Std accuracy: {np.std(acc_vals):.4f}")
    
    # Save results to CSV
    csv_path = args.save_csv if args.save_csv else 'results/sensitivity_analysis_results.csv'
    save_sensitivity_results_to_csv(results, csv_path)
    
    # Generate plot
    save_path = args.save if args.save else 'results/sensitivity_analysis.png'
    plot_sensitivity_analysis(results, save_path=save_path)


if __name__ == '__main__':
    main()
