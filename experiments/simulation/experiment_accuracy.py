import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# Add src directory to path to import project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from nci_nll_es_algorithm import (
    NCI_NLL_ES_Algorithm,
)
from synthetic_scenarios import EXPECTED_LABEL, SCENARIOS, generate_synthetic_data


def evaluate_accuracy(
    num_trials: int = 50,
    N_per_trial: int = 5000,
    state_dim: int = 2,
    n_samples_es: int = 800,
    base_seed: int = 1234,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run many trials to estimate recognition accuracy for each scenario.

    Returns: (accuracy_df, confusion_df)
    """
    rng = np.random.RandomState(base_seed)
    algorithm = NCI_NLL_ES_Algorithm(state_dim=state_dim, n_samples=n_samples_es)

    # Records
    correct_counts: Dict[str, int] = {s: 0 for s in SCENARIOS}
    total_counts: Dict[str, int] = {s: 0 for s in SCENARIOS}
    confusion: Dict[str, Counter] = {s: Counter() for s in SCENARIOS}

    for trial in range(num_trials):
        # Use different seed per scenario for independence, derived from base
        print(trial)
        for scen in SCENARIOS:
            seed = int(rng.randint(0, 2**31 - 1))
            scen_rng = np.random.RandomState(seed)
            true_states, estimated_states, claimed_covariances,bias_vec = generate_synthetic_data(
                scen, N=N_per_trial, d=state_dim, rng=scen_rng
            )

            result = algorithm.run_algorithm(true_states, estimated_states, claimed_covariances)

            predicted = result.classification
            expected = EXPECTED_LABEL[scen]

            total_counts[scen] += 1
            if predicted == expected:
                correct_counts[scen] += 1

            confusion[scen][predicted] += 1

    # Build accuracy dataframe
    rows = []
    for scen in SCENARIOS:
        acc = correct_counts[scen] / max(1, total_counts[scen])
        rows.append({
            'Scenario': scen,
            'Trials': total_counts[scen],
            'Correct': correct_counts[scen],
            'Accuracy': f"{acc:.3f}",
        })
    accuracy_df = pd.DataFrame(rows)

    # Build confusion matrix dataframe
    all_pred_labels: List[str] = sorted({lbl for cnt in confusion.values() for lbl in cnt.keys()} | set(EXPECTED_LABEL.values()))
    conf_rows = []
    for scen in SCENARIOS:
        row = {'Scenario': scen}
        total = sum(confusion[scen].values())
        for lbl in all_pred_labels:
            row[lbl] = confusion[scen][lbl]
        row['Total'] = total
        conf_rows.append(row)
    confusion_df = pd.DataFrame(conf_rows).fillna(0).astype({lbl: int for lbl in all_pred_labels} | {'Total': int})

    return accuracy_df, confusion_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate algorithm recognition accuracy across scenarios.')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials per scenario')
    parser.add_argument('--N', type=int, default=100, help='Number of MC runs per trial')
    parser.add_argument('--dim', type=int, default=2, help='State dimension')
    parser.add_argument('--es', type=int, default=800, help='ES Monte Carlo samples per evaluation')
    parser.add_argument('--seed', type=int, default=22, help='Base random seed')
    parser.add_argument(
        '--output-dir',
        default=os.path.join(PROJECT_ROOT, 'results'),
        help='Directory to save accuracy and confusion CSV outputs',
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    accuracy_df, confusion_df = evaluate_accuracy(
        num_trials=args.trials,
        N_per_trial=args.N,
        state_dim=args.dim,
        n_samples_es=args.es,
        base_seed=args.seed,
    )

    print('\n=== Per-scenario Accuracy ===')
    print(accuracy_df.to_string(index=False))

    print('\n=== Confusion Matrix (counts over trials) ===')
    print(confusion_df.to_string(index=False))

    accuracy_df.to_csv(os.path.join(args.output_dir, 'accuracy_results.csv'), index=False)
    confusion_df.to_csv(os.path.join(args.output_dir, 'confusion_matrix.csv'), index=False)


if __name__ == '__main__':
    main()


