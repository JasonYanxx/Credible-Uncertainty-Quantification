import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src directory to path to import project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))


def plot_scalability_accuracy(csv_path: str, output_path: str = None):
    """
    Visualize accuracy vs dimension for each scenario.
    
    Args:
        csv_path: Path to the scalability_detailed_results.csv file
        output_path: Optional path to save the figure
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get unique dimensions and scenarios
    dimensions = sorted(df['Dimension'].unique())
    scenarios = df['Scenario'].unique()
    
    # Define colors and markers for each scenario
    scenario_styles = {
        'calibrated': {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'label': 'Credible'},
        'optimistic': {'color': 'green', 'marker': 's', 'linestyle': '--', 'label': 'Optimism'},
        'pessimistic': {'color': 'red', 'marker': '^', 'linestyle': '-.', 'label': 'Pessimism'},
        'bias': {'color': 'orange', 'marker': 'D', 'linestyle': ':', 'label': 'SMM'},
        'mixed_o_b': {'color': 'purple', 'marker': 'v', 'linestyle': '-', 'label': 'Optimism+ SMM'},
        'mixed_p_b': {'color': 'brown', 'marker': 'p', 'linestyle': '--', 'label': 'Pessimism + SMM'},
    }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each scenario
    for scenario in scenarios:
        if scenario not in scenario_styles:
            continue
            
        # Filter data for this scenario
        scenario_data = df[df['Scenario'] == scenario].sort_values('Dimension')
        
        # Extract dimensions and accuracies
        dims = scenario_data['Dimension'].values
        accuracies = scenario_data['Accuracy'].values
        
        # Get style for this scenario
        style = scenario_styles[scenario]
        
        # Plot the curve
        ax.plot(
            dims, 
            accuracies,
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            label=style['label'],
            linewidth=2,
            markersize=8,
            alpha=0.8
        )
    
    # Customize the plot
    ax.set_xlabel('State Dimension', fontsize=16, fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontsize=16, fontweight='bold')
    # ax.set_title('Algorithm Scalability: Accuracy vs Dimension', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(
        loc='best', 
        frameon=True, 
        fancybox=True, 
        shadow=True, 
        fontsize=14, 
        ncol=2
    )
    # Set x-axis to show all dimensions
    ax.set_xticks(dimensions)
    ax.set_xlim(min(dimensions) - 1, max(dimensions) + 1)
    
    # Set y-axis limits to show accuracy range clearly
    ax.set_ylim(0, 1.05)
    # ax.set_yticks(np.arange(0.7, 1.05, 0.05))

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # Add horizontal line at accuracy = 1.0 for reference
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()
    
    return fig, ax


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Visualize scalability results: accuracy vs dimension for each scenario.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/scalability_detailed_results.csv',
        help='Path to scalability_detailed_results.csv (default: results/scalability_detailed_results.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/scalability_accuracy_plot.png',
        help='Path to save the output figure (default: results/scalability_accuracy_plot.png)'
    )
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate the plot
    plot_scalability_accuracy(args.input, args.output)


if __name__ == '__main__':
    main()
