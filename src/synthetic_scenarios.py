import numpy as np
from typing import Dict, Optional, Tuple


SCENARIOS = ['calibrated', 'optimistic', 'pessimistic', 'bias', 'mixed_o_b', 'mixed_p_b']


EXPECTED_LABEL: Dict[str, str] = {
    'calibrated': 'Calibrated',
    'optimistic': 'Optimistic',
    'pessimistic': 'Pessimistic',
    'bias': 'Bias',
    'mixed_o_b': 'Optimistic + Bias',
    'mixed_p_b': 'Pessimistic + Bias',
}


def generate_synthetic_data(
    scenario: str,
    N: int,
    d: int,
    rng: np.random.RandomState,
    bias_magnitude: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one synthetic trial for the shared six-scenario setup.

    Returns:
        true_states, estimated_states, claimed_covariances, bias_vec
    """

    def random_bias(mag_normal: float, mag_jitter: float = 0.2) -> np.ndarray:
        low = (1 - mag_jitter) * mag_normal
        high = (1 + mag_jitter) * mag_normal
        mag = float(rng.uniform(low, high))
        direction = rng.normal(size=d)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        return direction * mag

    def random_posdef_cov(rng_obj: np.random.RandomState, dim: int, eigval_range=(0.5, 2.0)) -> np.ndarray:
        q, _ = np.linalg.qr(rng_obj.normal(size=(dim, dim)))
        eigvals = rng_obj.uniform(eigval_range[0], eigval_range[1], size=dim)
        return q @ np.diag(eigvals) @ q.T

    true_cov = random_posdef_cov(rng, d)
    true_states = rng.multivariate_normal(np.zeros(d), true_cov, N)

    mag = bias_magnitude if bias_magnitude is not None else 2
    if scenario == 'calibrated':
        cov_scale = 1
        bias_vec = np.zeros(d)
    elif scenario == 'optimistic':
        cov_scale = rng.uniform(0.1, 0.8)
        bias_vec = np.zeros(d)
    elif scenario == 'pessimistic':
        cov_scale = rng.uniform(1.25, 10)
        bias_vec = np.zeros(d)
    elif scenario == 'bias':
        cov_scale = 1
        bias_vec = random_bias(mag, mag_jitter=0.1 if bias_magnitude is not None else 0.2)
    elif scenario == 'mixed_o_b':
        cov_scale = rng.uniform(0.1, 0.8)
        bias_vec = random_bias(mag)
    elif scenario == 'mixed_p_b':
        cov_scale = rng.uniform(1.25, 10)
        bias_vec = random_bias(mag)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    estimated_states = np.array([
        rng.multivariate_normal(x + bias_vec, true_cov) for x in true_states
    ])
    claimed_covariances = np.tile(cov_scale * true_cov, (N, 1, 1))
    return true_states, estimated_states, claimed_covariances, bias_vec


def generate_fixed_bias_data(
    N: int,
    d: int,
    rng: np.random.RandomState,
    bias_magnitude: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the bias-only setup used by ELT B analyses.
    """
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eigvals = np.ones(d)
    true_cov = q @ np.diag(eigvals) @ q.T
    true_states = rng.multivariate_normal(np.zeros(d), true_cov, N)

    direction = rng.normal(size=d)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    bias_vec = direction * bias_magnitude

    estimated_states = np.array([
        rng.multivariate_normal(x + bias_vec, true_cov) for x in true_states
    ])
    claimed_covariances = np.tile(true_cov, (N, 1, 1))
    return true_states, estimated_states, claimed_covariances
