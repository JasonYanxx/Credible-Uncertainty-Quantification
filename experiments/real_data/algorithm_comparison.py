import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# Add src directory to path to import project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from nci_nll_es_algorithm import NCI_NLL_ES_Algorithm, NCI_ELT_Algorithm
from baseline_classifiers import (
    NEESChiSquaredClassifier,
    PureESClassifier,
    PureNCIClassifier,
    PureNLLClassifier,
)
from synthetic_scenarios import EXPECTED_LABEL, SCENARIOS, generate_synthetic_data


def evaluate_algorithm_performance(
    num_trials: int = 50,
    N_per_trial: int = 5000,
    state_dim: int = 2,
    n_samples_es: int = 800,
    base_seed: int = 1234,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run many trials to estimate recognition accuracy for each algorithm across scenarios.
    
    Returns: (accuracy_df, confusion_df)
    """
    rng = np.random.RandomState(base_seed)
    
    # Initialize all classifiers
    classifiers = {
        'NCI_NLL_ES': NCI_NLL_ES_Algorithm(state_dim=state_dim, n_samples=n_samples_es),
        'NCI_ELT': NCI_ELT_Algorithm(state_dim=state_dim, n_samples=n_samples_es),
        'Pure_NLL': PureNLLClassifier(state_dim=state_dim),
        'Pure_ES': PureESClassifier(state_dim=state_dim, n_samples=n_samples_es),
        'NEES_Chi2': NEESChiSquaredClassifier(state_dim=state_dim),
        'Pure_NCI': PureNCIClassifier(state_dim=state_dim),
    }
    
    # Records for each algorithm
    correct_counts: Dict[str, Dict[str, int]] = {alg: {s: 0 for s in SCENARIOS} for alg in classifiers.keys()}
    total_counts: Dict[str, Dict[str, int]] = {alg: {s: 0 for s in SCENARIOS} for alg in classifiers.keys()}
    confusion: Dict[str, Dict[str, Counter]] = {alg: {s: Counter() for s in SCENARIOS} for alg in classifiers.keys()}

    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")
        
        for scen in SCENARIOS:
            seed = int(rng.randint(0, 2**31 - 1))
            scen_rng = np.random.RandomState(seed)
            
            true_states, estimated_states, claimed_covariances, bias_vec = generate_synthetic_data(
                scen, N=N_per_trial, d=state_dim, rng=scen_rng
            )

            # Test each classifier
            for alg_name, classifier in classifiers.items():
                try:
                    if alg_name == 'NCI_NLL_ES':
                        result = classifier.run_algorithm(true_states, estimated_states, claimed_covariances)
                        predicted = result.classification
                    else:
                        predicted = classifier.classify(true_states, estimated_states, claimed_covariances)
                    
                    expected = EXPECTED_LABEL[scen]
                    
                    total_counts[alg_name][scen] += 1
                    if predicted == expected:
                        correct_counts[alg_name][scen] += 1
                    
                    confusion[alg_name][scen][predicted] += 1
                    
                except Exception as e:
                    print(f"Error in {alg_name} for scenario {scen}: {e}")
                    # Count as incorrect
                    total_counts[alg_name][scen] += 1
                    confusion[alg_name][scen]['Error'] += 1

    # Build accuracy dataframe
    rows = []
    for alg_name in classifiers.keys():
        for scen in SCENARIOS:
            acc = correct_counts[alg_name][scen] / max(1, total_counts[alg_name][scen])
            rows.append({
                'Algorithm': alg_name,
                'Scenario': scen,
                'Trials': total_counts[alg_name][scen],
                'Correct': correct_counts[alg_name][scen],
                'Accuracy': f"{acc:.3f}",
            })
    accuracy_df = pd.DataFrame(rows)

    # Build confusion matrix dataframe
    all_pred_labels = sorted({lbl for alg_conf in confusion.values() for cnt in alg_conf.values() for lbl in cnt.keys()})
    conf_rows = []
    
    for alg_name in classifiers.keys():
        for scen in SCENARIOS:
            row = {'Algorithm': alg_name, 'Scenario': scen}
            total = sum(confusion[alg_name][scen].values())
            for lbl in all_pred_labels:
                row[lbl] = confusion[alg_name][scen][lbl]
            row['Total'] = total
            conf_rows.append(row)
    
    confusion_df = pd.DataFrame(conf_rows).fillna(0)
    
    # Convert numeric columns to int
    for lbl in all_pred_labels:
        if lbl in confusion_df.columns:
            confusion_df[lbl] = confusion_df[lbl].astype(int)
    confusion_df['Total'] = confusion_df['Total'].astype(int)

    return accuracy_df, confusion_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare performance of multiple UQ algorithms across scenarios.')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials per scenario')
    parser.add_argument('--N', type=int, default=100, help='Number of MC runs per trial')
    parser.add_argument('--dim', type=int, default=2, help='State dimension')
    parser.add_argument('--es', type=int, default=800, help='ES Monte Carlo samples per evaluation')
    parser.add_argument('--seed', type=int, default=22, help='Base random seed')
    parser.add_argument(
        '--output-dir',
        default=os.path.join(PROJECT_ROOT, 'results'),
        help='Directory to save summary CSV outputs',
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Running algorithm comparison...")
    accuracy_df, confusion_df = evaluate_algorithm_performance(
        num_trials=args.trials,
        N_per_trial=args.N,
        state_dim=args.dim,
        n_samples_es=args.es,
        base_seed=args.seed,
    )

    print('\n=== Algorithm Performance Comparison ===')
    print(accuracy_df.to_string(index=False))

    print('\n=== Confusion Matrix (counts over trials) ===')
    print(confusion_df.to_string(index=False))

    # Create summary table
    summary_df = accuracy_df.pivot(index='Algorithm', columns='Scenario', values='Accuracy')
    print('\n=== Summary Table (Accuracy by Algorithm and Scenario) ===')
    print(summary_df.to_string())
    accuracy_df.to_csv(os.path.join(args.output_dir, 'algorithm_comparison_accuracy.csv'), index=False)
    confusion_df.to_csv(os.path.join(args.output_dir, 'algorithm_comparison_confusion.csv'), index=False)
    summary_df.to_csv(os.path.join(args.output_dir, 'algorithm_comparison_summary.csv'))


if __name__ == '__main__':
    main()
