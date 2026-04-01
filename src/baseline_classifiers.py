import numpy as np
from scipy.stats import chi2, multivariate_normal


class PureNLLClassifier:
    """Pure Negative Log-Likelihood based classifier."""

    def __init__(self, state_dim: int = 2):
        self.state_dim = state_dim

    def classify(self, true_states: np.ndarray, estimated_states: np.ndarray, claimed_covariances: np.ndarray) -> str:
        nll_values = []
        for k in range(len(true_states)):
            e_k = true_states[k] - estimated_states[k]
            sigma_k = claimed_covariances[k] + np.eye(self.state_dim) * 1e-8
            try:
                nll = -multivariate_normal.logpdf(e_k, mean=np.zeros(self.state_dim), cov=sigma_k)
                nll_values.append(nll)
            except Exception:
                nll_values.append(1000.0)

        mean_nll = np.mean(nll_values)
        expected_nll = self.state_dim / 2
        if mean_nll < expected_nll * 0.8:
            return 'Optimistic'
        if mean_nll > expected_nll * 1.2:
            return 'Pessimistic'
        return 'Calibrated'


class PureESClassifier:
    """Pure Energy Score based classifier."""

    def __init__(self, state_dim: int = 2, n_samples: int = 1000):
        self.state_dim = state_dim
        self.n_samples = n_samples

    def calculate_energy_score(self, true_state: np.ndarray, estimated_state: np.ndarray, covariance: np.ndarray) -> float:
        try:
            samples = np.random.multivariate_normal(estimated_state, covariance, self.n_samples)
        except Exception:
            samples = np.random.multivariate_normal(estimated_state, np.eye(len(estimated_state)), self.n_samples)

        distances_to_true = np.linalg.norm(samples - true_state, axis=1)
        term1 = np.mean(distances_to_true)

        n_pairs = min(200, self.n_samples // 2)
        idx1 = np.random.choice(self.n_samples, n_pairs, replace=False)
        idx2 = np.random.choice(self.n_samples, n_pairs, replace=False)
        pairwise_distances = np.linalg.norm(samples[idx1] - samples[idx2], axis=1)
        term2 = 0.5 * np.mean(pairwise_distances)
        return term1 - term2

    def classify(self, true_states: np.ndarray, estimated_states: np.ndarray, claimed_covariances: np.ndarray) -> str:
        es_values = []
        for k in range(len(true_states)):
            es = self.calculate_energy_score(true_states[k], estimated_states[k], claimed_covariances[k])
            es_values.append(es)

        mean_es = np.mean(es_values)
        if mean_es < 0.5:
            return 'Optimistic'
        if mean_es > 2.0:
            return 'Pessimistic'
        return 'Calibrated'


class NEESChiSquaredClassifier:
    """NEES Chi-squared test based classifier."""

    def __init__(self, state_dim: int = 2, alpha: float = 0.05):
        self.state_dim = state_dim
        self.alpha = alpha

    def classify(self, true_states: np.ndarray, estimated_states: np.ndarray, claimed_covariances: np.ndarray) -> str:
        nees_values = []
        for k in range(len(true_states)):
            e_k = true_states[k] - estimated_states[k]
            sigma_k = claimed_covariances[k] + np.eye(self.state_dim) * 1e-8
            try:
                nees = e_k.T @ np.linalg.inv(sigma_k) @ e_k
                nees_values.append(nees)
            except Exception:
                nees_values.append(1000.0)

        nees_array = np.array(nees_values)
        expected_nees = self.state_dim
        chi2_stat = np.sum((nees_array - expected_nees) ** 2 / expected_nees)
        p_value = 1 - chi2.cdf(chi2_stat, df=len(true_states) - 1)

        if p_value < self.alpha:
            return 'Optimistic' if np.mean(nees_array) > expected_nees else 'Pessimistic'
        return 'Calibrated'


class PureNCIClassifier:
    """Pure NCI based classifier."""

    def __init__(self, state_dim: int = 2):
        self.state_dim = state_dim

    def classify(self, true_states: np.ndarray, estimated_states: np.ndarray, claimed_covariances: np.ndarray) -> str:
        nees_values = []
        for k in range(len(true_states)):
            e_k = true_states[k] - estimated_states[k]
            sigma_k = claimed_covariances[k] + np.eye(self.state_dim) * 1e-8
            try:
                nees = e_k.T @ np.linalg.inv(sigma_k) @ e_k
                nees_values.append(nees)
            except Exception:
                nees_values.append(1000.0)

        mean_log_nees = np.mean(np.log10(nees_values))
        expected_log_nees = np.log10(self.state_dim)
        nci = 10 * (mean_log_nees - expected_log_nees)
        if nci > 0.5:
            return 'Optimistic'
        if nci < -0.5:
            return 'Pessimistic'
        return 'Calibrated'
