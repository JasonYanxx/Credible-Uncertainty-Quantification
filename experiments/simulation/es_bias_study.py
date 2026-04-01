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
    # Use Monte Carlo simulation to estimate both terms
    n_samples = 5000
    Y_samples = np.random.multivariate_normal(F_mean, F_cov, n_samples)
    
    # First term: E[||Y-x||_2]
    first_term = np.mean(np.linalg.norm(Y_samples - x, axis=1))
    
    # Second term: 0.5 * E[||Y-Y'||_2]
    # Randomly pair up samples to estimate E[||Y-Y'||_2]
    idx1 = np.random.choice(n_samples, n_samples // 2, replace=False)
    idx2 = np.random.choice(n_samples, n_samples // 2, replace=False)
    second_term = 0.5 * np.mean(np.linalg.norm(Y_samples[idx1] - Y_samples[idx2], axis=1))
    
    return first_term - second_term

def calculate_expected_energy_score(gamma, x, Sigma, N=10000):
    """
    Calculate E[ES(F,x)] where F = N(x_hat, Sigma) and x_hat ~ N(x + mu, Sigma)
    with mu = gamma * [1, 1]^T
    
    Args:
        gamma: bias parameter (mu = gamma * [1, 1]^T)
        x: true observation point
        Sigma: covariance matrix
        N: number of Monte Carlo samples
    
    Returns:
        expected energy score
    """
    # Bias vector
    mu = gamma * np.array([1.0, 1.0])
    
    # Sample x_hat from N(x + mu, Sigma)
    x_hat_samples = np.random.multivariate_normal(x + mu, Sigma, N)
    
    # Calculate ES(F,x) for each x_hat sample
    energy_scores = []
    F_cov = Sigma  # tilde_Sigma = Sigma
    
    for x_hat in x_hat_samples:
        es = energy_score(x_hat, F_cov, x)
        energy_scores.append(es)
    
    return np.mean(energy_scores), np.std(energy_scores)



def main():
    parser = argparse.ArgumentParser(description="Generate the ES bias figure used in the paper.")
    parser.add_argument(
        "--output-path",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "es_bias.png"),
        help="Path to save the output figure",
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Parameters
    x = np.array([0.0, 0.0])  # x = 0 (2D)
    Sigma = np.eye(2)         # Sigma = I_2
    gamma_values = [0.1, 0.5, 1, 5, 10, 20, 50, 100]
    
    print("Energy Score Misspecification Study")
    print("=" * 50)
    print(f"x = {x}")
    print(f"Sigma = I_2")
    print(f"F = N(x_hat, Sigma)")
    print(f"x_hat ~ N(x + mu, Sigma) where mu = gamma * [1, 1]^T")
    print()
    
    results = []
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for gamma in gamma_values:
        print(f"Calculating for gamma = {gamma}...")
        
        # Numerical calculation
        expected_es_numerical, std_es = calculate_expected_energy_score(gamma, x, Sigma, N=5000)
        
        results.append({
            'gamma': gamma,
            'numerical': expected_es_numerical,
            'numerical_std': std_es,
            'bias_norm': gamma * np.sqrt(2)  # ||mu||_2 = gamma * sqrt(2)
        })
        
        print(f"  E[ES(F,x)]: {expected_es_numerical:.6f} ± {std_es:.6f}")
        print(f"  Bias ||mu||_2: {gamma * np.sqrt(2):.6f}")
        print()
    
    # Summary table
    print("Summary Table:")
    print("-" * 50)
    print(f"{'gamma':<8} {'||mu||_2':<12} {'E[ES(F,x)]':<15} {'Std Error':<12}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['gamma']:<8} {result['bias_norm']:<12.6f} {result['numerical']:<15.6f} "
              f"{result['numerical_std']:<12.6f}")
    
    # # Plot results
    # plt.figure(figsize=(15, 10))
    
    # # Extract data for plotting
    # gamma_array = np.array(gamma_values)
    # bias_norms = np.array([result['bias_norm'] for result in results])
    # numerical_values = np.array([result['numerical'] for result in results])
    # numerical_stds = np.array([result['numerical_std'] for result in results])
    
    # # Plot 1: Energy Score vs gamma
    # plt.plot(gamma_array, numerical_values, 'bo-', label='Mean', markersize=8)
    # plt.fill_between(gamma_array, numerical_values - 1.96*numerical_stds, 
    #                  numerical_values + 1.96*numerical_stds, 
    #                  color='blue', alpha=0.2, label='95% Confidence Interval')
    # plt.xlabel('γ (log scale)')
    # plt.ylabel('Energy Score')
    # # plt.title('Energy Score vs Bias Parameter γ')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # plt.show()

    # Plot results
    target_pixel_width = 428
    target_pixel_height = 322
    base_dpi = 100
    fig_width_in = target_pixel_width / base_dpi
    fig_height_in = target_pixel_height / base_dpi
    resolution_dpi = 300

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    # Extract data for plotting
    gamma_array = np.array(gamma_values)
    bias_norms = np.array([result['bias_norm'] for result in results])
    numerical_values = np.array([result['numerical'] for result in results])
    numerical_stds = np.array([result['numerical_std'] for result in results])
    
    # Plot 1: Energy Score vs gamma
    ax.plot(gamma_array, numerical_values, 'bo-', label='Mean', markersize=8)
    ax.fill_between(gamma_array, numerical_values - 1.96*numerical_stds, 
                     numerical_values + 1.96*numerical_stds, 
                     color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('γ')
    plt.ylabel('Energy Score')
    plt.legend()

    fig.tight_layout() 
    fig.savefig(args.output_path, dpi=resolution_dpi)
    
    plt.show()

    

if __name__ == "__main__":
    main()
