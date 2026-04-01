import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import argparse
import os

def energy_score(F_mean, F_cov, x):
    """
    Calculate the energy score ES(F,x) = E[||Y-x||_2] - 0.5*E[||Y-Y'||_2]
    where Y, Y' ~ F = N(F_mean, F_cov)
    
    Args:
        F_mean: mean of distribution F
        F_cov: covariance matrix of distribution F
        x: observation point
    
    Returns:
        energy score value
    """
    # There is no analytic solution for E[||Y-x||_2] or E[||Y-Y'||_2] (L2-norm, not squared).
    # Use Monte Carlo simulation to estimate both terms.
    n_samples = 5000
    Y_samples = np.random.multivariate_normal(F_mean, F_cov, n_samples)
    # First term: E[||Y-x||_2]
    first_term = np.mean(np.linalg.norm(Y_samples - x, axis=1))
    # Second term: 0.5 * E[||Y-Y'||_2]
    # To estimate the second term 0.5 * E[||Y - Y'||_2], we need to compute the expected L2 distance
    # between two independent samples Y and Y' drawn from the same distribution F.
    # Since we have already generated a large set of samples Y_samples, we can approximate this expectation
    # by randomly pairing up samples and computing the mean of their pairwise distances.
    # We do this by selecting two sets of random indices (idx1 and idx2), each of size n_samples // 2,
    # and then compute the L2 norm between the corresponding pairs.
    idx1 = np.random.choice(n_samples, n_samples // 2, replace=False)
    idx2 = np.random.choice(n_samples, n_samples // 2, replace=False)
    # Compute the mean L2 distance between the paired samples, then multiply by 0.5 as per the energy score definition.
    second_term = 0.5 * np.mean(np.linalg.norm(Y_samples[idx1] - Y_samples[idx2], axis=1))
    return first_term - second_term

def calculate_expected_energy_score(rho, x, Sigma, n_samples=10000):
    """
    Calculate E[ES(F,x)] where F = N(x_hat, rho*Sigma) and x_hat ~ N(x, Sigma)
    
    Args:
        rho: scaling factor for covariance in F
        x: true observation point
        Sigma: covariance matrix for x_hat distribution
        n_samples: number of Monte Carlo samples
    
    Returns:
        expected energy score
    """
    # Sample x_hat from N(x, Sigma)
    x_hat_samples = np.random.multivariate_normal(x, Sigma, n_samples)
    
    # Calculate ES(F,x) for each x_hat sample
    energy_scores = []
    F_cov = rho * Sigma  # tilde_Sigma = rho * Sigma
    
    for x_hat in x_hat_samples:
        es = energy_score(x_hat, F_cov, x)
        energy_scores.append(es)
    
    return np.mean(energy_scores), np.std(energy_scores)

def main():
    parser = argparse.ArgumentParser(description="Generate the ES scale figure used in the paper.")
    parser.add_argument(
        "--output-path",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "es_scale.png"),
        help="Path to save the output figure",
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Parameters
    x = np.array([0.0, 0.0])  # x = 0 (2D)
    Sigma = np.eye(2)         # Sigma = I_2
    rho_values = [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    
    print("Energy Score Calculation")
    print("=" * 50)
    print(f"x = {x}")
    print(f"Sigma = I_2")
    print(f"F = N(x_hat, rho * Sigma)")
    print(f"x_hat ~ N(x, Sigma)")
    print()
    
    results = []
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for rho in rho_values:
        print(f"Calculating for rho = {rho}...")
        
        # Numerical calculation
        expected_es_numerical, std_es = calculate_expected_energy_score(rho, x, Sigma, n_samples=1000)

        
        results.append({
            'rho': rho,
            'numerical': expected_es_numerical,
            'numerical_std': std_es,
        })
        
        print(f"  Numerical E[ES(F,x)]: {expected_es_numerical:.6f} ± {std_es:.6f}")
        print()
    
    # Plot results
    target_pixel_width = 428
    target_pixel_height = 322
    base_dpi = 100
    fig_width_in = target_pixel_width / base_dpi
    fig_height_in = target_pixel_height / base_dpi
    resolution_dpi = 300

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    # Plot the expected_es_numerical as a curve with shaded confidence interval
    # Use log scale for rho on the x-axis
    rho_array = np.array(rho_values)
    numerical_means = np.array([result['numerical'] for result in results])
    numerical_stds = np.array([result['numerical_std'] for result in results])

    ax.plot(rho_array, numerical_means, label='Mean', color='blue', marker='o', markersize=6)
    ax.fill_between(rho_array, numerical_means - 1.96*numerical_stds, numerical_means + 1.96*numerical_stds, 
                     color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.xscale('log')  # Set x-axis to log scale for rho
    plt.xlabel('ρ (log scale)')
    plt.ylabel('Energy Score')
    # ax.title('Energy Score vs log10(rho)')
    plt.legend()

    fig.tight_layout() 
    fig.savefig(args.output_path, dpi=resolution_dpi)
    
    plt.show()

    
   

if __name__ == "__main__":
    main()
