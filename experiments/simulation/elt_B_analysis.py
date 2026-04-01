"""
ELT test: effect of B (sign-flip samples) and bias magnitude on testing power.

Only the 'bias' scenario is tested. For each (B, bias_magnitude) we estimate:
- Miss detection rate (Type I error in the sense of the test: failing to detect
  bias when bias is present, i.e. ELT=0 when truth is bias).
- Testing power = 1 - miss_detect_rate (probability of correctly detecting bias).
"""
import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add src directory to path to import project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from nci_nll_es_algorithm import NCI_NLL_ES_Algorithm
from synthetic_scenarios import generate_fixed_bias_data


def analyze_elt_B_and_bias_magnitude_testing_power(
    B_values: List[int],
    bias_magnitudes: List[float],
    num_trials: int = 50,
    N_per_trial: int = 5000,
    state_dim: int = 2,
    base_seed: int = 1234,
    default_alpha_sig: float = 0.05,
) -> pd.DataFrame:
    """
    Analyze ELT test under the 'bias' scenario only. Vary B and bias magnitude.

    - Miss detection: true scenario is bias but ELT=0 (fail to detect).
    - Power = 1 - miss_detect_rate.

    Returns:
        DataFrame with columns: B, bias_magnitude, miss_detect_rate, power, mean_time_sec
    """
    rng = np.random.RandomState(base_seed)
    algorithm = NCI_NLL_ES_Algorithm(state_dim=state_dim, n_samples=800)
    algorithm.alpha_sig = default_alpha_sig

    rows = []
    for B in B_values:
        for bias_mag in bias_magnitudes:
            miss_detects = 0
            time_list = []

            for trial in range(num_trials):
                seed = int(rng.randint(0, 2**31 - 1)) # testing avergae power
                scen_rng = np.random.RandomState(seed)
                true_states, estimated_states, claimed_covariances = generate_fixed_bias_data(
                    N=N_per_trial, d=state_dim, rng=scen_rng, bias_magnitude=bias_mag
                )
                t0 = time.perf_counter()
                elt, _ = algorithm.compute_elt(
                    true_states, estimated_states, claimed_covariances, B=B, rng=scen_rng)
                t1 = time.perf_counter()
                time_list.append(t1 - t0)
                if elt == 0:
                    miss_detects += 1

            miss_detect_rate = miss_detects / num_trials
            power = 1.0 - miss_detect_rate
            mean_time_sec = np.mean(time_list)
            rows.append({
                'B': B,
                'bias_magnitude': bias_mag,
                'miss_detect_rate': miss_detect_rate,
                'power': power,
                'mean_time_sec': mean_time_sec,
            })
            print(f"  B={B}, bias_mag={bias_mag}: miss_detect={miss_detect_rate:.4f}, "
                  f"power={power:.4f}, time={mean_time_sec:.4f}s")

    return pd.DataFrame(rows)


def analyze_elt_B_and_bias_magnitude_stability(
    B_values: List[int],
    bias_magnitudes: List[float],
    num_trials: int = 50,
    N_per_trial: int = 5000,
    state_dim: int = 2,
    base_seed: int = 1234,
    default_alpha_sig: float = 0.05,
) -> pd.DataFrame:
    """
    Analyze ELT test under the 'bias' scenario only. Vary B and bias magnitude.

    - Miss detection: true scenario is bias but ELT=0 (fail to detect).
    - Power = 1 - miss_detect_rate.

    Returns:
        DataFrame with columns: B, bias_magnitude, miss_detect_rate, power, mean_time_sec
    """
    rng = np.random.RandomState(base_seed)
    algorithm = NCI_NLL_ES_Algorithm(state_dim=state_dim, n_samples=800)
    algorithm.alpha_sig = default_alpha_sig

    rows = []
    for B in B_values:
        for bias_mag in bias_magnitudes:
            miss_detects = 0
            time_list = []
            scen_rng = np.random.RandomState(1234)
            true_states, estimated_states, claimed_covariances = generate_fixed_bias_data(
                    N=N_per_trial, d=state_dim, rng=scen_rng, bias_magnitude=bias_mag
            )
            for trial in range(num_trials):
                seed = int(rng.randint(0, 2**31 - 1)) # testing avergae power
                test_rng = np.random.RandomState(seed)
                t0 = time.perf_counter()
                elt, _ = algorithm.compute_elt(
                    true_states, estimated_states, claimed_covariances, B=B, rng=test_rng
                )
                t1 = time.perf_counter()
                time_list.append(t1 - t0)
                if elt == 0:
                    miss_detects += 1

            miss_detect_rate = miss_detects / num_trials
            power = 1.0 - miss_detect_rate
            mean_time_sec = np.mean(time_list)
            rows.append({
                'B': B,
                'bias_magnitude': bias_mag,
                'miss_detect_rate': miss_detect_rate,
                'power': power,
                'mean_time_sec': mean_time_sec,
            })
            print(f"  B={B}, bias_mag={bias_mag}: miss_detect={miss_detect_rate:.4f}, "
                  f"power={power:.4f}, time={mean_time_sec:.4f}s")

    return pd.DataFrame(rows)

def save_results_to_csv(
    df: pd.DataFrame,
    csv_path: str = 'results/elt_B_impact_results.csv',
) -> None:
    """Save results to CSV."""
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


def plot_elt_B_and_bias_impact(
    df: pd.DataFrame,
    save_path: str = None,
    analysis_label: str = '',
) -> None:
    """Plot effects of B and bias magnitude on miss detection rate (separate figures)."""
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

    B_vals = sorted(df['B'].unique())
    bias_vals = sorted(df['bias_magnitude'].unique())
    title_suffix = f' — {analysis_label}' if analysis_label else ''

    # (1) Miss detection rate vs B, one line per bias magnitude
    fig1, ax1 = plt.subplots(figsize=(fig_width_in, fig_height_in))
    for bm in bias_vals:
        sub = df[df['bias_magnitude'] == bm].sort_values('B')
        ax1.plot(sub['B'], 1-sub['miss_detect_rate'], '-o', label=f'bias mag = {bm}', linewidth=2, markersize=6)
    ax1.set_xlabel(r'$B$ (ELT sign-flip samples)', fontsize=label_fontsize)
    ax1.set_ylabel('Average testing power', fontsize=label_fontsize)
    # ax1.set_title('(a) Miss detection rate vs. $B$' + title_suffix, fontsize=title_fontsize, fontweight='bold')
    ax1.legend(loc='best', fontsize=legend_fontsize)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    fig1.tight_layout()
    if save_path:
        base, ext = os.path.splitext(save_path)
        path1 = base + '_miss_vs_B' + (ext or '.png')
        os.makedirs(os.path.dirname(path1) or '.', exist_ok=True)
        fig1.savefig(path1, dpi=resolution_dpi, bbox_inches='tight')
        print(f"Figure saved to {path1}")
    plt.show()

    # (2) Miss detection rate vs bias magnitude, one line per B
    fig2, ax2 = plt.subplots(figsize=(fig_width_in, fig_height_in))
    for B in B_vals:
        sub = df[df['B'] == B].sort_values('bias_magnitude')
        ax2.plot(sub['bias_magnitude'], sub['miss_detect_rate'], '-s', label=f'B = {B}', linewidth=2, markersize=6)
    ax2.set_xlabel('Bias magnitude (dimensionless)', fontsize=label_fontsize)
    ax2.set_ylabel('Miss detection rate (Type I error)', fontsize=label_fontsize)
    # ax2.set_title('(b) Miss detection rate vs. bias magnitude' + title_suffix, fontsize=title_fontsize, fontweight='bold')
    ax2.legend(loc='best', fontsize=legend_fontsize)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    fig2.tight_layout()
    if save_path:
        base, ext = os.path.splitext(save_path)
        path2 = base + '_miss_vs_bias' + (ext or '.png')
        os.makedirs(os.path.dirname(path2) or '.', exist_ok=True)
        fig2.savefig(path2, dpi=resolution_dpi, bbox_inches='tight')
        print(f"Figure saved to {path2}")
    plt.show()

    # (3) Heatmap: miss detection rate as function of (B, bias_magnitude)
    pivot_miss = df.pivot(index='bias_magnitude', columns='B', values='miss_detect_rate')
    pivot_miss = pivot_miss.reindex(index=sorted(pivot_miss.index), columns=sorted(pivot_miss.columns))

    fig3, ax3 = plt.subplots(figsize=(fig_width_in, fig_height_in))
    im = ax3.imshow(pivot_miss.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
    ax3.set_xticks(np.arange(len(pivot_miss.columns)))
    ax3.set_yticks(np.arange(len(pivot_miss.index)))
    ax3.set_xticklabels(pivot_miss.columns)
    ax3.set_yticklabels(pivot_miss.index)
    ax3.set_xlabel(r'$B$ (ELT sign-flip samples)', fontsize=label_fontsize)
    ax3.set_ylabel('Bias magnitude (dimensionless)', fontsize=label_fontsize)
    # ax3.set_title('Miss detection rate vs. $B$ and bias magnitude' + title_suffix, fontsize=title_fontsize, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Miss detection rate')
    fig3.tight_layout()
    if save_path:
        base, ext = os.path.splitext(save_path)
        heatmap_path = base + '_heatmap' + (ext or '.png')
    else:
        heatmap_path = 'results/elt_B_impact_heatmap.png'
    os.makedirs(os.path.dirname(heatmap_path) or '.', exist_ok=True)
    fig3.savefig(heatmap_path, dpi=resolution_dpi, bbox_inches='tight')
    print(f"Heatmap saved to {heatmap_path}")
    plt.show()


def plot_elt_computation_time(
    df: pd.DataFrame,
    save_path: str = None,
    analysis_label: str = '',
) -> None:
    """Plot ELT average computation time vs B and bias magnitude (separate figures)."""
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

    B_vals = sorted(df['B'].unique())
    bias_vals = sorted(df['bias_magnitude'].unique())
    title_suffix = f' — {analysis_label}' if analysis_label else ''

    # (1) Computation time vs B, one line per bias magnitude
    fig1, ax1 = plt.subplots(figsize=(fig_width_in, fig_height_in))
    for bm in bias_vals:
        sub = df[df['bias_magnitude'] == bm].sort_values('B')
        ax1.plot(sub['B'], sub['mean_time_sec'], '-o', label=f'bias mag = {bm}', linewidth=2, markersize=6)
    ax1.set_xlabel(r'$B$ (ELT sign-flip samples)', fontsize=label_fontsize)
    ax1.set_ylabel('Mean computation time (s)', fontsize=label_fontsize)
    # ax1.set_title('(a) Computation time vs. $B$' + title_suffix, fontsize=title_fontsize, fontweight='bold')
    # ax1.legend(loc='best', fontsize=legend_fontsize)
    ax1.grid(True, alpha=0.3)
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')

    fig1.tight_layout()
    if save_path:
        base, ext = os.path.splitext(save_path)
        path1 = base + '_time_vs_B' + (ext or '.png')
        os.makedirs(os.path.dirname(path1) or '.', exist_ok=True)
        fig1.savefig(path1, dpi=resolution_dpi, bbox_inches='tight')
        print(f"Figure saved to {path1}")
    plt.show()




def main():
    parser = argparse.ArgumentParser(
        description='ELT test: effect of B and bias magnitude on testing power (bias scenario only).'
    )
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of trials per (B, bias_magnitude)')
    parser.add_argument('--N', type=int, default=100,
                        help='Number of samples per trial')
    parser.add_argument('--dim', type=int, default=2,
                        help='State dimension')
    parser.add_argument('--seed', type=int, default=11,
                        help='Base random seed')
    parser.add_argument('--B-values', type=float, nargs='+',
                        default=[10, 20, 50, 100, 200, 500, 1000],
                        help='B values to test')
    parser.add_argument('--bias-magnitudes', type=float, nargs='+',
                        # default=[0.2,0.5,0.8,1.0],
                        # default=[0.2,0.25,0.3,0.35,0.4,0.45,0.5],
                        default=[0.1,0.3,0.32,0.34,0.36,0.38,0.4,0.5,1.0],
                        # default=[0.5],
                        help='Bias magnitudes to test')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save figures (default: results/elt_B_impact.png)')
    parser.add_argument('--save-csv', type=str, default=None,
                        help='Path to save CSV (default: results/elt_B_impact_results.csv)')
    args = parser.parse_args()

    # Ensure B_values are integers
    B_values = [int(b) for b in args.B_values]

    print("=" * 60)
    print("ELT test: bias scenario only — B and bias magnitude vs. testing power")
    print("=" * 60)
    print(f"B values: {B_values}")
    print(f"Bias magnitudes: {args.bias_magnitudes}")
    print(f"Trials per (B, bias_magnitude): {args.trials}, N per trial: {args.N}")
    print("=" * 60)

    results_dir = 'results'
    csv_base = (args.save_csv or os.path.join(results_dir, 'elt_B_impact')).replace('.csv', '').rstrip('_')
    plot_base = (args.save or os.path.join(results_dir, 'elt_B_impact')).replace('.png', '').rstrip('_')

    # 1) Testing power: random data per trial (average power over data realizations)
    print("\n--- Analysis 1: Testing power (random data per trial) ---")
    df_power = analyze_elt_B_and_bias_magnitude_testing_power(
        B_values=B_values,
        bias_magnitudes=args.bias_magnitudes,
        num_trials=args.trials,
        N_per_trial=args.N,
        state_dim=args.dim,
        base_seed=args.seed,
    )
    csv_power = csv_base + '_testing_power_results.csv'
    save_results_to_csv(df_power, csv_power)
    plot_power = plot_base + '_testing_power.png'
    plot_elt_B_and_bias_impact(df_power, save_path=plot_power, analysis_label='Testing power (random data per trial)')
    plot_time_power = plot_base + '_testing_power_time.png'
    plot_elt_computation_time(df_power, save_path=plot_time_power, analysis_label='Testing power (random data per trial)')

    # 2) Stability: fixed data, random ELT signs per trial
    print("\n--- Analysis 2: Stability (fixed data, random ELT sign-flips) ---")
    df_stability = analyze_elt_B_and_bias_magnitude_stability(
        B_values=B_values,
        bias_magnitudes=args.bias_magnitudes,
        num_trials=args.trials,
        N_per_trial=args.N,
        state_dim=args.dim,
        base_seed=args.seed,
    )
    csv_stability = csv_base + '_stability_results.csv'
    save_results_to_csv(df_stability, csv_stability)
    plot_stability = plot_base + '_stability.png'
    plot_elt_B_and_bias_impact(df_stability, save_path=plot_stability, analysis_label='Stability (fixed data, random ELT signs)')

if __name__ == '__main__':
    main()
