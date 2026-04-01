import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
from typing import List, Tuple, Dict, Optional, Union
import pandas as pd
from dataclasses import dataclass
import warnings
from scipy.stats import chisquare, chi2, kstest

@dataclass
class AlgorithmResult:
    """Container for algorithm results"""
    nci: float
    nll_values: Dict[float, float]  # k -> NLL(k)
    es_values: Dict[float, float]   # k -> ES(k)
    delta_nll_minus: float
    delta_nll_plus: float
    delta_es_minus: float
    delta_es_plus: float
    elt: int
    p_elt: float
    classification: str

class NCI_NLL_ES_Algorithm:
    """
    Implementation of the NCI, NLL, and ES algorithm for uncertainty quantification evaluation.
    
    This algorithm combines:
    1. NCI (Normalized Covariance Index) for scale signal
    2. NLL (Negative Log-Likelihood) for optimism-sided sensitivity  
    3. ES (Energy Score) for pessimism-sided sensitivity
    
    Based on the document: "Algorithm Development and Evaluation Using NCI, NLL, and ES"
    """
    
    def __init__(self, state_dim: int = 2, n_samples: int = 1000):
        """
        Initialize the algorithm.
        
        Args:
            state_dim: Dimension of the state vector
            n_samples: Number of Monte Carlo samples for ES calculation
        """
        self.state_dim = state_dim
        self.n_samples = n_samples
        
        # Default thresholds (can be calibrated via bootstrap)
        self.tau_nci = 0.5  # dB - used for NCI classification
        self.tau_nll = 0.1  # Not used in SRD approach, kept for compatibility
        self.tau_es = 0.1   # Not used in SRD approach, kept for compatibility
        self.nees_chi2_alpha = 0.01 
        self.alpha_sig = 0.05
        self.prop_scale = 2
        self.elt_B = 500  # number of sign-flip randomizations for ELT
        
    def compute_nees_nci(self, true_states: np.ndarray, estimated_states: np.ndarray, 
                   claimed_covariances: np.ndarray) -> float:
        """
        Compute Normalized Covariance Index (NCI).
        
        NCI = 10 * (mean(log10(epsilon)) - mean(log10(epsilon_star)))
        where epsilon = e^T * Sigma_tilde^(-1) * e
        and epsilon_star = e^T * M^(-1) * e, with M = Sigma + mu*mu^T
        
        Args:
            true_states: True state vectors (N x d)
            estimated_states: Estimated state vectors (N x d) 
            claimed_covariances: Claimed covariance matrices (N x d x d)
            
        Returns:
            NCI value in dB
        """
        N = len(true_states)
        errors = true_states - estimated_states
        
        # Compute ness values
        nees_values = []
        for k in range(N):
            e_k = errors[k]
            Sigma_tilde_k = claimed_covariances[k]
            
            # Add regularization to avoid numerical issues
            Sigma_tilde_k_reg = Sigma_tilde_k + np.eye(self.state_dim) * 1e-8
            
            try:
                epsilon_k = e_k.T @ np.linalg.inv(Sigma_tilde_k_reg) @ e_k
                nees_values.append(epsilon_k)
            except np.linalg.LinAlgError:
                nees_values.append(1000.0)  # Large value for singular matrix
        

        # Use the Kolmogorov-Smirnov test for goodness-of-fit to chi2
        eps_arr = np.array(nees_values)
        ks_stat, nees_ks_pvalue = stats.kstest(eps_arr, lambda x: chi2.cdf(x, df=self.state_dim))

        # Estimate M = Sigma + mu*mu^T from samples
        error_mean = np.mean(errors, axis=0)
        error_cov = np.cov(errors.T)
        M = error_cov + np.outer(error_mean, error_mean)
        
        # Add regularization
        M_reg = M + np.eye(self.state_dim) * 1e-8
        
        # Compute epsilon_star values
        epsilon_star_values = []
        for k in range(N):
            e_k = errors[k]
            try:
                epsilon_star_k = e_k.T @ np.linalg.inv(M_reg) @ e_k
                epsilon_star_values.append(epsilon_star_k)
            except np.linalg.LinAlgError:
                epsilon_star_values.append(1000.0)
        
        # Compute NCI
        log_epsilon_mean = np.mean(np.log10(nees_values))
        log_epsilon_star_mean = np.mean(np.log10(epsilon_star_values))
        
        nci = 10 * (log_epsilon_mean - log_epsilon_star_mean)
        return nees_ks_pvalue,nci
    
    def compute_nll(self, true_state: np.ndarray, estimated_state: np.ndarray, 
                   covariance: np.ndarray) -> float:
        """
        Compute Negative Log-Likelihood.
        
        NLL(F_k, x_k) = 0.5 * (x_k - x_hat_k)^T * Sigma^(-1) * (x_k - x_hat_k) 
                       + 0.5 * log|Sigma_k| + d/2 * log(2*pi)
        
        Args:
            true_state: True state vector
            estimated_state: Estimated state vector
            covariance: Covariance matrix
            
        Returns:
            NLL value
        """
        error = true_state - estimated_state
        
        # Add regularization
        cov_reg = covariance + np.eye(self.state_dim) * 1e-8
        
        try:
            # Quadratic term
            quad_term = 0.5 * error.T @ np.linalg.inv(cov_reg) @ error
            
            # Log determinant term
            log_det_term = 0.5 * np.log(np.linalg.det(cov_reg))
            
            # Constant term
            const_term = 0.5 * self.state_dim * np.log(2 * np.pi)
            
            nll = quad_term + log_det_term + const_term
            return float(nll)
            
        except np.linalg.LinAlgError:
            return 1000.0  # Large value for singular matrix
    
    def _matrix_inverse_sqrt(self, covariance: np.ndarray) -> np.ndarray:
        """Compute covariance^(-1/2) with regularization for stability."""
        cov_reg = covariance + np.eye(self.state_dim) * 1e-8
        try:
            # Use eigen-decomposition for symmetric PSD matrices
            eigenvalues, eigenvectors = np.linalg.eigh(cov_reg)
            inv_sqrt_eigenvalues = 1.0 / np.sqrt(np.clip(eigenvalues, 1e-12, None))
            inv_sqrt = (eigenvectors * inv_sqrt_eigenvalues) @ eigenvectors.T
            return inv_sqrt
        except np.linalg.LinAlgError:
            # Fallback to identity scaling
            return np.eye(self.state_dim)
    
    def compute_elt(self, true_states: np.ndarray, estimated_states: np.ndarray,
                    claimed_covariances: np.ndarray,
                    B: int = 2000, max_pairs: int = 50000,
                    rng: Optional[Union[np.random.RandomState, np.random.Generator]] = None) -> Tuple[int, float]:
        """
        Energy Location Test (ELT) via sign-flip randomization.
        Returns (ELT_indicator, p_value).
        If rng is provided, all randomness uses it for reproducibility.
        """
        rng = rng if rng is not None else np.random.default_rng()
        N = len(true_states)
        errors = true_states - estimated_states
        # Per-sample whitening: s_k = Sigma_tilde^{-1/2} e_k
        s_list = []
        for k in range(N):
            inv_sqrt = self._matrix_inverse_sqrt(claimed_covariances[k])
            s_list.append(inv_sqrt @ errors[k])
        s = np.vstack(s_list)
        
        # Helper to compute T-hat from a set of vectors (possibly sign-flipped)
        def compute_T(value_vectors: np.ndarray) -> float:
            # Subsample pairs for efficiency if needed
            if N < 2:
                return 0.0
            total_pairs = N * (N - 1) // 2
            if total_pairs <= max_pairs:
                # Compute all pairs
                # Sample all pairs indices
                idx_i, idx_j = np.triu_indices(N, k=1)
            else:
                # Randomly sample pairs
                rng_i = rng.integers(0, N, size=max_pairs) if hasattr(rng, 'integers') else rng.randint(0, N, size=max_pairs)
                rng_j = rng.integers(0, N, size=max_pairs) if hasattr(rng, 'integers') else rng.randint(0, N, size=max_pairs)
                mask = rng_i != rng_j
                idx_i = rng_i[mask]
                idx_j = rng_j[mask]
            v_i = value_vectors[idx_i]
            v_j = value_vectors[idx_j]
            plus = np.linalg.norm(v_i + v_j, axis=1)
            minus = np.linalg.norm(v_i - v_j, axis=1)
            T_hat = (2.0 / (N * (N - 1))) * np.sum(plus - minus)
            return float(T_hat)
        
        T_obs = compute_T(s)
        # Randomization with Rademacher signs
        count_ge = 0
        for _ in range(B):
            signs = rng.choice([-1.0, 1.0], size=N)
            s_flip = s * signs[:, None]
            T_b = compute_T(s_flip)
            if T_b >= T_obs:
                count_ge += 1
        p_val = (1 + count_ge) / (B + 1)
        ELT = 1 if p_val < self.alpha_sig else 0
        return ELT, float(p_val)
    
    def compute_energy_score(self, true_state: np.ndarray, estimated_state: np.ndarray,
                           covariance: np.ndarray) -> float:
        """
        Compute Energy Score.
        
        ES(F_k, x_k) = E[||Y - x_k||_2] - 0.5 * E[||Y - Y'||_2]
        where Y, Y' ~ F_k = N(x_hat_k, Sigma_k)
        
        Computational Complexity Analysis:
        ==================================
        Let:
        - d = state_dim (dimension of state vector)
        - M = n_samples (number of Monte Carlo samples)
        - n_pairs = min(200, M // 2) (number of sample pairs for second term)
        
        Step-by-step complexity:
        1. Covariance regularization: O(d^2) - matrix addition
        2. Sample generation (multivariate_normal): O(M * d^2) 
           - Cholesky decomposition: O(d^3) once
           - Generate M samples: O(M * d^2)
           - Total: O(d^3 + M * d^2) ≈ O(M * d^2) when M >> d
        3. First term (distances to true state): O(M * d)
           - Compute M distances, each O(d)
        4. Second term (pairwise distances):
           - Random index selection: O(n_pairs) ≈ O(min(200, M))
           - Compute pairwise distances: O(n_pairs * d)
           - Total: O(n_pairs * d) = O(min(200, M) * d)
        
        Overall complexity: O(M * d^2 + M * d + min(200, M) * d)
                          = O(M * d^2)  (dominant term)
        
        When M is large (e.g., M = 1000), the complexity is dominated by:
        - Sample generation: O(M * d^2)
        - The pairwise term is capped at O(200 * d) = O(d), which is negligible
        
        Space complexity: O(M * d) for storing samples
        
        Args:
            true_state: True state vector
            estimated_state: Estimated state vector
            covariance: Covariance matrix
            
        Returns:
            Energy Score value
        """
        # Add regularization: O(d^2)
        cov_reg = covariance + np.eye(self.state_dim) * 1e-8
        
        try:
            # Generate samples from estimated distribution: O(M * d^2)
            # Cholesky decomposition: O(d^3), then M samples: O(M * d^2)
            samples = np.random.multivariate_normal(estimated_state, cov_reg, self.n_samples)
            
            # First term: E[||Y - x_k||_2] - O(M * d)
            # Compute M distances, each requiring O(d) operations
            distances_to_true = np.linalg.norm(samples - true_state, axis=1)
            term1 = np.mean(distances_to_true)
            
            # Second term: 0.5 * E[||Y - Y'||_2] - O(min(200, M) * d)
            # Optimization: Instead of computing all M*(M-1)/2 pairs (O(M^2 * d)),
            # we sample n_pairs pairs, capping at 200 for efficiency
            n_pairs = min(200, self.n_samples // 2)  # Capped at 200 pairs
            idx1 = np.random.choice(self.n_samples, n_pairs, replace=False)  # O(n_pairs)
            idx2 = np.random.choice(self.n_samples, n_pairs, replace=False)  # O(n_pairs)
            
            # Compute pairwise distances: O(n_pairs * d)
            pairwise_distances = np.linalg.norm(samples[idx1] - samples[idx2], axis=1)
            term2 = 0.5 * np.mean(pairwise_distances)
            
            return term1 - term2
            
        except np.linalg.LinAlgError:
            return 1000.0  # Large value for singular matrix
    
    def compute_nll_es_for_k(self, true_states: np.ndarray, estimated_states: np.ndarray,
                                 claimed_covariances: np.ndarray, k: float) -> Tuple[float, float]:
        """
        Compute mean NLL and ES for a given scale factor k.

        Computational Complexity:
        =========================
        Let:
        - N = number of samples
        - d = state_dim
        - M = n_samples (for ES Monte Carlo)
        
        For each sample i in [1, N]:
        - compute_nll: O(d^3) (matrix inversion)
        - compute_energy_score: O(M * d^2) (see compute_energy_score docstring)
        
        Total: O(N * (d^3 + M * d^2)) = O(N * M * d^2) when M >> d
        
        Args:
            true_states: True state vectors (N x d)
            estimated_states: Estimated state vectors (N x d)
            claimed_covariances: Claimed covariance matrices (N x d x d)
            k: scale factor

        Returns:
            Tuple of (mean_nll, mean_es)
        """
        N = len(true_states)
        nll_list = []
        es_list = []
        for i in range(N):
            scaled_cov = k * claimed_covariances[i]
            nll_k = self.compute_nll(true_states[i], estimated_states[i], scaled_cov)
            es_k = self.compute_energy_score(true_states[i], estimated_states[i], scaled_cov)
            nll_list.append(nll_k)
            es_list.append(es_k)
        mean_nll = np.mean(nll_list)
        mean_es = np.mean(es_list)
        return mean_nll, mean_es
        
    def compute_directional_probes(self, true_states: np.ndarray, estimated_states: np.ndarray,
                                 claimed_covariances: np.ndarray) -> Tuple[Dict, Dict, Dict]:
        """
        Compute directional probes using NLL and ES at different scales.
        
        Computational Complexity:
        =========================
        Calls compute_nll_es_for_k three times (k=1, prop_scale, 1/prop_scale).
        Each call: O(N * M * d^2) where N=samples, M=n_samples, d=state_dim.
        Total: O(3 * N * M * d^2) = O(N * M * d^2)
        
        Args:
            true_states: True state vectors (N x d)
            estimated_states: Estimated state vectors (N x d)
            claimed_covariances: Claimed covariance matrices (N x d x d)
            
        Returns:
            Tuple of (nll_values, es_values, deltas)
        """
        N = len(true_states)

        # Compute directional differences
        # These are used to compute Slope Relative Differences (SRD) in classification:
        # SRD_{NLL} = abs((2×|Δ⁻_NLL| - |Δ⁺_NLL|) / |Δ⁺_NLL|)
        # SRD_{ES} = abs((2×|Δ⁻_ES| - |Δ⁺_ES|) / |Δ⁺_ES|)
        # For calibrated: Δ⁻_NLL > 0, Δ⁺_NLL > 0, Δ⁻_ES > 0, Δ⁺_ES > 0
        # Complexity: Each call is O(N * M * d^2), total O(3 * N * M * d^2)
        nll_mean, es_mean = self.compute_nll_es_for_k(true_states, estimated_states, claimed_covariances, 1)
        nll_mean_k, es_mean_k = self.compute_nll_es_for_k(true_states, estimated_states, claimed_covariances, self.prop_scale)
        nll_mean_rk, es_mean_rk = self.compute_nll_es_for_k(true_states, estimated_states, claimed_covariances, 1/self.prop_scale)
        deltas = {
            'delta_nll_minus': nll_mean_rk - nll_mean,
            'delta_nll_plus': nll_mean_k - nll_mean,
            'delta_es_minus': es_mean_rk - es_mean,
            'delta_es_plus': es_mean_k - es_mean
        }

        return nll_mean, es_mean, deltas

    
    def classify_uncertainty(self, elt: int, nees_ks_pvalue: float, nci: float, deltas: Dict) -> Tuple[str, bool]:
        """
        Classify uncertainty using ELT, NCI, and SRD per the document.
        
        Args:
            nci: NCI value
            deltas: Dictionary of directional differences
            
        Returns:
            Tuple of (classification, end_of_judgement)
        """
        delta_nll_minus = deltas['delta_nll_minus']
        delta_nll_plus = deltas['delta_nll_plus']
        delta_es_minus = deltas['delta_es_minus']
        delta_es_plus = deltas['delta_es_plus']
        
        # Compute Slope Relative Differences (SRD)
        srd_nll = float('inf')
        srd_es = float('inf')
        
        if abs(delta_nll_plus) > 1e-10:
            srd_nll = abs((self.prop_scale * abs(delta_nll_minus) - abs(delta_nll_plus)) / abs(delta_nll_plus))
        
        if abs(delta_es_plus) > 1e-10:
            srd_es = abs((self.prop_scale * abs(delta_es_minus) - abs(delta_es_plus)) / abs(delta_es_plus))
        
        # Branch on ELT (bias test)
        if elt == 0:
            # No bias
            if nees_ks_pvalue > self.nees_chi2_alpha or abs(nci) <=self.tau_nci:
                return "Calibrated", True
            else:
                if nci < -self.tau_nci:
                    return "Pessimistic", True
                elif nci > self.tau_nci:
                    return "Optimistic", True
        else:
            return "Further Justification", False


                
        # # Branch on ELT (bias test)
        # if elt == 0:
        #     # No bias
        #     if nees_ks_pvalue > self.nees_chi2_alpha:
        #         if (delta_nll_minus > 0 and delta_nll_plus > 0 and 
        #             delta_es_minus > 0 and delta_es_plus > 0):
        #             return "Calibrated", True
        #         else:
        #             return "Bias", True
        #     else:
        #         if nci < -self.tau_nci:
        #             return "Pessimistic", True
        #         elif nci > self.tau_nci:
        #             return "Optimistic", True
        #         else:
        #             return "Calibrated", True
        # else:
        #     return "Further Justification", False
    
    def classify_uncertainty_remove_bias(self, nci: float, deltas: Dict) -> Tuple[str, bool]:
        """
        Classify uncertainty using ELT, NCI, and SRD per the document.
        
        Args:
            nci: NCI value
            deltas: Dictionary of directional differences
            
        Returns:
            Tuple of (classification, end_of_judgement)
        """
        delta_nll_minus = deltas['delta_nll_minus']
        delta_nll_plus = deltas['delta_nll_plus']
        delta_es_minus = deltas['delta_es_minus']
        delta_es_plus = deltas['delta_es_plus']
        
        # Compute Slope Relative Differences (SRD)
        srd_nll = float('inf')
        srd_es = float('inf')
        
        if abs(delta_nll_plus) > 1e-10:
            srd_nll = abs((self.prop_scale * abs(delta_nll_minus) - abs(delta_nll_plus)) / abs(delta_nll_plus))
        
        if abs(delta_es_plus) > 1e-10:
            srd_es = abs((self.prop_scale * abs(delta_es_minus) - abs(delta_es_plus)) / abs(delta_es_plus))
        

        # bias
        if nci < -self.tau_nci:
            return "Pessimistic + Bias", True
        elif nci > self.tau_nci:
            # Maybe also optimistic
            if (delta_nll_minus > 0 and delta_nll_plus > 0 and 
                delta_es_minus > 0 and delta_es_plus > 0):
                return "Bias", True
            elif srd_nll > srd_es:
                return "Optimistic + Bias", True
            else:
                return "Bias", True
        else:
            return "Bias", True 


        # # bias
        # if nci < -self.tau_nci:
        #     return "Pessimistic + Bias", True
        # elif nci > self.tau_nci:
        #     # Maybe also optimistic or small pessimistic
        #     if (delta_nll_minus > 0 and delta_nll_plus > 0 and 
        #         delta_es_minus > 0 and delta_es_plus > 0):
        #         return "Bias", True
        #     elif srd_nll > srd_es:
        #         return "Optimistic + Bias", True
        #     elif srd_es > srd_nll:
        #         return "Pessimistic + Bias", True
        #     else:
        #         return "Bias", True
        # else:
        #     return "Bias", True 

    

    def run_algorithm(self, true_states: np.ndarray, estimated_states: np.ndarray,
                     claimed_covariances: np.ndarray) -> AlgorithmResult:
        """
        Run the complete NCI, NLL, and ES algorithm.
        
        Args:
            true_states: True state vectors (N x d)
            estimated_states: Estimated state vectors (N x d)
            claimed_covariances: Claimed covariance matrices (N x d x d)
            
        Returns:
            AlgorithmResult object containing all results
        """
        # Step 1: Compute ELT and NCI
        B = getattr(self, 'elt_B', 500)
        elt, p_elt = self.compute_elt(true_states, estimated_states, claimed_covariances, B=B)
        nees_ks_pvalue, nci = self.compute_nees_nci(true_states, estimated_states, claimed_covariances)
        
        # Step 2: Compute directional probes
        nll_values, es_values, deltas = self.compute_directional_probes(
            true_states, estimated_states, claimed_covariances
        )
        
        # Step 3: Classify uncertainty
        classification, is_end = self.classify_uncertainty(elt, nees_ks_pvalue,nci, deltas)
        if is_end == False:
            # remove bias effect
            bias_estiamte = np.mean(estimated_states-true_states,axis=0)
            # Compute NCI
            nees_ks_pvalue, nci = self.compute_nees_nci(true_states, estimated_states-bias_estiamte, claimed_covariances)
            # Compute directional probes
            nll_values, es_values, deltas = self.compute_directional_probes(
                true_states, estimated_states-bias_estiamte, claimed_covariances
            )
            classification, is_end = self.classify_uncertainty_remove_bias(nci, deltas)
            
        
        return AlgorithmResult(
            nci=nci,
            nll_values=nll_values,
            es_values=es_values,
            delta_nll_minus=deltas['delta_nll_minus'],
            delta_nll_plus=deltas['delta_nll_plus'],
            delta_es_minus=deltas['delta_es_minus'],
            delta_es_plus=deltas['delta_es_plus'],
            elt=elt,
            p_elt=p_elt,
            classification=classification,
        )

class NCI_ELT_Algorithm:
    """
    Implementation of the NCI, NLL, and ES algorithm for uncertainty quantification evaluation.
    
    This algorithm combines:
    1. NCI (Normalized Covariance Index) for scale signal
    2. NLL (Negative Log-Likelihood) for optimism-sided sensitivity  
    3. ES (Energy Score) for pessimism-sided sensitivity
    
    Based on the document: "Algorithm Development and Evaluation Using NCI, NLL, and ES"
    """
    
    def __init__(self, state_dim: int = 2, n_samples: int = 1000):
        """
        Initialize the algorithm.
        
        Args:
            state_dim: Dimension of the state vector
            n_samples: Number of Monte Carlo samples for ES calculation
        """
        self.state_dim = state_dim
        self.n_samples = n_samples
        
        # Default thresholds (can be calibrated via bootstrap)
        self.tau_nci = 0.5  # dB - used for NCI classificationy
        self.nees_chi2_alpha = 0.01 
        self.alpha_sig = 0.05
        
    def compute_nees_nci(self, true_states: np.ndarray, estimated_states: np.ndarray, 
                   claimed_covariances: np.ndarray) -> float:
        """
        Compute Normalized Covariance Index (NCI).
        
        NCI = 10 * (mean(log10(epsilon)) - mean(log10(epsilon_star)))
        where epsilon = e^T * Sigma_tilde^(-1) * e
        and epsilon_star = e^T * M^(-1) * e, with M = Sigma + mu*mu^T
        
        Args:
            true_states: True state vectors (N x d)
            estimated_states: Estimated state vectors (N x d) 
            claimed_covariances: Claimed covariance matrices (N x d x d)
            
        Returns:
            NCI value in dB
        """
        N = len(true_states)
        errors = true_states - estimated_states
        
        # Compute ness values
        nees_values = []
        for k in range(N):
            e_k = errors[k]
            Sigma_tilde_k = claimed_covariances[k]
            
            # Add regularization to avoid numerical issues
            Sigma_tilde_k_reg = Sigma_tilde_k + np.eye(self.state_dim) * 1e-8
            
            try:
                epsilon_k = e_k.T @ np.linalg.inv(Sigma_tilde_k_reg) @ e_k
                nees_values.append(epsilon_k)
            except np.linalg.LinAlgError:
                nees_values.append(1000.0)  # Large value for singular matrix
        

        # Use the Kolmogorov-Smirnov test for goodness-of-fit to chi2
        eps_arr = np.array(nees_values)
        ks_stat, nees_ks_pvalue = stats.kstest(eps_arr, lambda x: chi2.cdf(x, df=self.state_dim))

        # Estimate M = Sigma + mu*mu^T from samples
        error_mean = np.mean(errors, axis=0)
        error_cov = np.cov(errors.T)
        M = error_cov + np.outer(error_mean, error_mean)
        
        # Add regularization
        M_reg = M + np.eye(self.state_dim) * 1e-8
        
        # Compute epsilon_star values
        epsilon_star_values = []
        for k in range(N):
            e_k = errors[k]
            try:
                epsilon_star_k = e_k.T @ np.linalg.inv(M_reg) @ e_k
                epsilon_star_values.append(epsilon_star_k)
            except np.linalg.LinAlgError:
                epsilon_star_values.append(1000.0)
        
        # Compute NCI
        log_epsilon_mean = np.mean(np.log10(nees_values))
        log_epsilon_star_mean = np.mean(np.log10(epsilon_star_values))
        
        nci = 10 * (log_epsilon_mean - log_epsilon_star_mean)
        return nees_ks_pvalue,nci
    

    def _matrix_inverse_sqrt(self, covariance: np.ndarray) -> np.ndarray:
        """Compute covariance^(-1/2) with regularization for stability."""
        cov_reg = covariance + np.eye(self.state_dim) * 1e-8
        try:
            # Use eigen-decomposition for symmetric PSD matrices
            eigenvalues, eigenvectors = np.linalg.eigh(cov_reg)
            inv_sqrt_eigenvalues = 1.0 / np.sqrt(np.clip(eigenvalues, 1e-12, None))
            inv_sqrt = (eigenvectors * inv_sqrt_eigenvalues) @ eigenvectors.T
            return inv_sqrt
        except np.linalg.LinAlgError:
            # Fallback to identity scaling
            return np.eye(self.state_dim)
    
    def compute_elt(self, true_states: np.ndarray, estimated_states: np.ndarray,
                    claimed_covariances: np.ndarray,
                    B: int = 2000, max_pairs: int = 50000,
                    rng: Optional[Union[np.random.RandomState, np.random.Generator]] = None) -> Tuple[int, float]:
        """
        Energy Location Test (ELT) via sign-flip randomization.
        Returns (ELT_indicator, p_value).
        If rng is provided, all randomness uses it for reproducibility.
        """
        rng = rng if rng is not None else np.random.default_rng()
        N = len(true_states)
        errors = true_states - estimated_states
        # Per-sample whitening: s_k = Sigma_tilde^{-1/2} e_k
        s_list = []
        for k in range(N):
            inv_sqrt = self._matrix_inverse_sqrt(claimed_covariances[k])
            s_list.append(inv_sqrt @ errors[k])
        s = np.vstack(s_list)
        
        # Helper to compute T-hat from a set of vectors (possibly sign-flipped)
        def compute_T(value_vectors: np.ndarray) -> float:
            # Subsample pairs for efficiency if needed
            if N < 2:
                return 0.0
            total_pairs = N * (N - 1) // 2
            if total_pairs <= max_pairs:
                # Compute all pairs
                # Sample all pairs indices
                idx_i, idx_j = np.triu_indices(N, k=1)
            else:
                # Randomly sample pairs
                rng_i = rng.integers(0, N, size=max_pairs) if hasattr(rng, 'integers') else rng.randint(0, N, size=max_pairs)
                rng_j = rng.integers(0, N, size=max_pairs) if hasattr(rng, 'integers') else rng.randint(0, N, size=max_pairs)
                mask = rng_i != rng_j
                idx_i = rng_i[mask]
                idx_j = rng_j[mask]
            v_i = value_vectors[idx_i]
            v_j = value_vectors[idx_j]
            plus = np.linalg.norm(v_i + v_j, axis=1)
            minus = np.linalg.norm(v_i - v_j, axis=1)
            T_hat = (2.0 / (N * (N - 1))) * np.sum(plus - minus)
            return float(T_hat)
        
        T_obs = compute_T(s)
        # Randomization with Rademacher signs
        count_ge = 0
        for _ in range(B):
            signs = rng.choice([-1.0, 1.0], size=N)
            s_flip = s * signs[:, None]
            T_b = compute_T(s_flip)
            if T_b >= T_obs:
                count_ge += 1
        p_val = (1 + count_ge) / (B + 1)
        ELT = 1 if p_val < self.alpha_sig else 0
        return ELT, float(p_val)

    def classify(self, true_states: np.ndarray, estimated_states: np.ndarray,
                     claimed_covariances: np.ndarray) -> AlgorithmResult:
        """
        Run the complete NCI, NLL, and ES algorithm.
        
        Args:
            true_states: True state vectors (N x d)
            estimated_states: Estimated state vectors (N x d)
            claimed_covariances: Claimed covariance matrices (N x d x d)
            
        Returns:
            AlgorithmResult object containing all results
        """
        # Step 1: Compute ELT and NCI
        elt, p_elt = self.compute_elt(true_states, estimated_states, claimed_covariances)
        nees_ks_pvalue, nci = self.compute_nees_nci(true_states, estimated_states, claimed_covariances)
        
        # Step 2: Classify uncertainty
        if elt == 0:
            if nci < -self.tau_nci:
                return "Pessimistic"
            elif nci > self.tau_nci:
                return "Optimistic"
            else:
                return "Calibrated"
        else:
            # remove bias effect
            bias_estimate = np.mean(estimated_states-true_states,axis=0)
            # Compute NCI
            nees_ks_pvalue, nci = self.compute_nees_nci(true_states, estimated_states-bias_estimate, claimed_covariances)

            if nci < -self.tau_nci:
                return "Pessimistic + Bias"
            elif nci > self.tau_nci:
                return "Optimistic + Bias"
            else:
                return "Bias"
            



def generate_synthetic_data(scenario: str, N: int = 5000, d: int = 2, 
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for different scenarios as described in the document.
    
    Args:
        scenario: One of ['calibrated', 'optimistic', 'pessimistic', 'bias', 'mixed_o_b', 'mixed_p_b']
        N: Number of Monte Carlo runs
        d: State dimension
        seed: Random seed
        
    Returns:
        Tuple of (true_states, estimated_states, claimed_covariances)
    """
    np.random.seed(seed)
    
    if scenario == 'calibrated':
        # Truth: x ~ N(0, I_2), Estimates: x_hat ~ N(x, I), Covariance: I
        true_states = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)
        estimated_states = np.array([np.random.multivariate_normal(x, np.eye(d)) for x in true_states])
        claimed_covariances = np.tile(np.eye(d), (N, 1, 1))
        
    elif scenario == 'optimistic':
        # Truth: x ~ N(0, I_2), Estimates: x_hat ~ N(x, I), Covariance: 0.5*I
        true_states = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)
        estimated_states = np.array([np.random.multivariate_normal(x, np.eye(d)) for x in true_states])
        claimed_covariances = np.tile(0.5 * np.eye(d), (N, 1, 1))
        
    elif scenario == 'pessimistic':
        # Truth: x ~ N(0, I_2), Estimates: x_hat ~ N(x, I), Covariance: 2*I
        true_states = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)
        estimated_states = np.array([np.random.multivariate_normal(x, np.eye(d)) for x in true_states])
        claimed_covariances = np.tile(2.0 * np.eye(d), (N, 1, 1))
        
    elif scenario == 'bias':
        # Truth: x ~ N(0, I_2), Estimates: x_hat ~ N(x + [0.5, 0.5], I), Covariance: I
        true_states = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)
        bias = np.array([0.5, 0.5])
        estimated_states = np.array([np.random.multivariate_normal(x + bias, np.eye(d)) for x in true_states])
        claimed_covariances = np.tile(np.eye(d), (N, 1, 1))
        
    elif scenario == 'mixed_o_b':
        # Truth: x ~ N(0, I_2), Estimates: x_hat ~ N(x + [0.5, 0.5], I), Covariance: 0.5*I
        true_states = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)
        bias = np.array([0.5, 0.5])
        estimated_states = np.array([np.random.multivariate_normal(x + bias, np.eye(d)) for x in true_states])
        claimed_covariances = np.tile(0.5 * np.eye(d), (N, 1, 1))
        
    elif scenario == 'mixed_p_b':
        # Truth: x ~ N(0, I_2), Estimates: x_hat ~ N(x + [0.5, 0.5], I), Covariance: 2*I
        true_states = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)
        bias = np.array([0.5, 0.5])
        estimated_states = np.array([np.random.multivariate_normal(x + bias, np.eye(d)) for x in true_states])
        claimed_covariances = np.tile(2.0 * np.eye(d), (N, 1, 1))
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return true_states, estimated_states, claimed_covariances

def run_experiment():
    """
    Run the complete experiment as described in the document.
    """
    print("NCI, NLL, and ES Algorithm Experiment")
    print("=" * 50)
    
    # Initialize algorithm
    algorithm = NCI_NLL_ES_Algorithm(state_dim=2, n_samples=1000)
    
    # Test scenarios
    scenarios = ['calibrated', 'optimistic', 'pessimistic', 'bias', 'mixed_o_b', 'mixed_p_b']
    results = {}
    
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario}")
        print("-" * 30)
        
        # Generate data
        true_states, estimated_states, claimed_covariances = generate_synthetic_data(
            scenario, N=5000, d=2, seed=42
        )
        
        # Run algorithm
        result = algorithm.run_algorithm(true_states, estimated_states, claimed_covariances)
        results[scenario] = result
        
        # Print results
        print(f"NCI: {result.nci:.3f} dB")
        print(f"NLL values: {result.nll_values}")
        print(f"ES values: {result.es_values}")
        print(f"Δ⁻ NLL: {result.delta_nll_minus:.6f}")
        print(f"Δ⁺ NLL: {result.delta_nll_plus:.6f}")
        print(f"Δ⁻ ES: {result.delta_es_minus:.6f}")
        print(f"Δ⁺ ES: {result.delta_es_plus:.6f}")
        print(f"Classification: {result.classification}")
    
    # Create summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    summary_data = []
    for scenario in scenarios:
        result = results[scenario]
        summary_data.append({
            'Scenario': scenario,
            'NCI (dB)': f"{result.nci:.3f}",
            'Δ⁻ NLL': f"{result.delta_nll_minus:.6f}",
            'Δ⁺ NLL': f"{result.delta_nll_plus:.6f}",
            'Δ⁻ ES': f"{result.delta_es_minus:.6f}",
            'Δ⁺ ES': f"{result.delta_es_plus:.6f}",
            'Classification': result.classification,
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    return results

# if __name__ == "__main__":
#     # Run the experiment
#     results = run_experiment()
