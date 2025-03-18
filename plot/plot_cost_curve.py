import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d
import argparse
import os

def plot_comparison(algorithms, metric, scenario_num=50, output_file=None):
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set("notebook", "darkgrid")
    seeds = [1, 2, 3]
    
    for algo in algorithms:
        # List to store data for each seed
        seed_dfs = []
        
        for seed in seeds:
            try:
                data = pd.read_csv(f"./logs/{algo}/scenarios-{scenario_num}-seed-{seed}.txt", sep='\t')
                data = data[data["Steps"] < 2_000_000]
                seed_dfs.append(data)
            except FileNotFoundError:
                print(f"Warning: File for {algo} scenarios-{scenario_num}-seed-{seed} not found")
        
        if not seed_dfs:
            print(f"No data found for {algo} scenarios-{scenario_num}")
            continue
        
        # Find common step range
        max_min_step = max(df["Steps"].min() for df in seed_dfs)
        min_max_step = min(df["Steps"].max() for df in seed_dfs)
        
        if max_min_step >= min_max_step:
            print(f"Error: No common step range for {algo} scenarios-{scenario_num}")
            continue
        
        # Create common step points with fixed intervals
        desired_steps = np.arange(20000, 2000000 + 1, 10000)
        common_steps = desired_steps[(desired_steps >= max_min_step) & (desired_steps <= min_max_step)]
        
        if len(common_steps) == 0:
            print(f"Error: No common steps in range for {algo} scenarios-{scenario_num}")
            continue
        
        # Interpolate values for each seed
        interpolated_values = []
        for df in seed_dfs:
            f = interp1d(df["Steps"], df[metric])
            interpolated_values.append(f(common_steps))
        
        # Calculate mean and standard deviation across seeds
        mean_values = np.mean(interpolated_values, axis=0)
        std_values = np.std(interpolated_values, axis=0)
        
        # Create dataframe for plotting
        plot_data = pd.DataFrame({
            "Steps": common_steps,
            metric: mean_values,
            "std": std_values
        })
        
        # Plot the averaged data with standard deviation
        ax = sns.lineplot(
            data=plot_data,
            x="Steps",
            y=metric,
            label=f"{algo.upper()}"
        )
        
        # Add shaded area for standard deviation
        plt.fill_between(
            common_steps,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.2  # Transparency of the shaded area
        )

    metric_name = metric.replace("test/", "").replace("train/", "").replace("_", " ").title()
    ax.set_title(f"PPO-Lag vs SAC-Lag in SafeMetaDrive (50 Scenarios)")
    ax.set_ylabel(f"{metric_name}")
    ax.set_xlabel("Sampled Steps")
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x')

    plt.legend()
    if not output_file:
        output_file = f'comparison-{"-".join(algorithms)}-{metric.replace("/", "-")}.png'
    plt.savefig(output_file, format='png', dpi=300, bbox_inches="tight")
    print(f"Figure saved to {output_file}")

def plot_train_test_comparison(algorithms, train_metric, test_metric, scenario_num=50, output_file=None):
    """
    Plot training and testing curves for multiple algorithms on the same figure.
    """
    # Increase font size for all text elements
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.title_fontsize': 14
    })
    
    plt.figure(figsize=(12, 8), dpi=300)  # Slightly larger figure
    sns.set("notebook", "darkgrid")
    seeds = [1, 2, 3]
    
    linestyles = {'train': '-', 'test': '--'}  # Solid for train, dashed for test
    
    # Get default matplotlib colors for each algorithm
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    algo_colors = {algo: colors[i % len(colors)] for i, algo in enumerate(algorithms)}
    
    for algo in algorithms:
        # List to store data for each seed
        seed_dfs = []
        
        for seed in seeds:
            try:
                data = pd.read_csv(f"./logs/{algo}/scenarios-{scenario_num}-seed-{seed}.txt", sep='\t')
                data = data[data["Steps"] < 2_000_000]
                seed_dfs.append(data)
            except FileNotFoundError:
                print(f"Warning: File for {algo} scenarios-{scenario_num}-seed-{seed} not found")
        
        if not seed_dfs:
            print(f"No data found for {algo} scenarios-{scenario_num}")
            continue
        
        # Find common step range
        max_min_step = max(df["Steps"].min() for df in seed_dfs)
        min_max_step = min(df["Steps"].max() for df in seed_dfs)
        
        if max_min_step >= min_max_step:
            print(f"Error: No common step range for {algo} scenarios-{scenario_num}")
            continue
        
        # Create common step points with fixed intervals
        desired_steps = np.arange(20000, 2000000 + 1, 10000)
        common_steps = desired_steps[(desired_steps >= max_min_step) & (desired_steps <= min_max_step)]
        
        if len(common_steps) == 0:
            print(f"Error: No common steps in range for {algo} scenarios-{scenario_num}")
            continue
        
        # Process both train and test metrics
        for metric_type, metric in [('train', train_metric), ('test', test_metric)]:
            try:
                # Interpolate values for each seed
                interpolated_values = []
                for df in seed_dfs:
                    if metric in df.columns:
                        f = interp1d(df["Steps"], df[metric])
                        interpolated_values.append(f(common_steps))
                    else:
                        print(f"Warning: Metric {metric} not found in data for {algo}")
                        break
                
                if not interpolated_values:
                    continue
                
                # Calculate mean and standard deviation across seeds
                mean_values = np.mean(interpolated_values, axis=0)
                std_values = np.std(interpolated_values, axis=0)
                
                # Create dataframe for plotting
                plot_data = pd.DataFrame({
                    "Steps": common_steps,
                    metric: mean_values,
                    "std": std_values
                })
                
                # Plot with same color for each algorithm but different line styles for train/test
                ax = sns.lineplot(
                    data=plot_data,
                    x="Steps",
                    y=metric,
                    label=f"{algo.upper()} ({metric_type})",
                    linestyle=linestyles[metric_type],
                    color=algo_colors[algo]
                )
                
                # Add shaded area for standard deviation with the same color
                plt.fill_between(
                    common_steps,
                    mean_values - std_values,
                    mean_values + std_values,
                    alpha=0.1,
                    color=algo_colors[algo]
                )
            except Exception as e:
                print(f"Error plotting {metric_type} metric for {algo}: {e}")

    # Get base metric name without prefixes for axis label
    base_metric = train_metric.split('/')[-1] if '/' in train_metric else train_metric
    base_metric = base_metric.replace("_", " ").title()

    if base_metric == "Complete":
        base_metric = "Route Completion"
    
    ax.set_title(f"PPO-Lag vs SAC-Lag (50 Scenarios)", fontsize=20, pad=15)
    ax.set_ylabel(f"{base_metric}", fontsize=16)
    ax.set_xlabel("Sampled Steps", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x')

    # Improved legend with larger font size and better positioning
    plt.legend(fontsize=14, frameon=True, fancybox=True, framealpha=0.9, 
               loc='best', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()  # Adjust layout to make room for larger text

    # Create output directory if it doesn't exist
    os.makedirs("./pics", exist_ok=True)
    
    if not output_file:
        output_file = f'./pics/{base_metric}-comparison-{"-".join(algorithms)}.png'
    plt.savefig(output_file, format='png', dpi=300, bbox_inches="tight")
    print(f"Figure saved to {output_file}")

def plot_cost_comparison(algorithms, costs, metric, scenario_num=50, output_file=None):
    """
    Plot comparison of algorithms with different cost values on the same figure.
    
    Args:
        algorithms (list): List of algorithm names (e.g., ['ppol', 'sacl'])
        costs (list): List of cost values to compare (e.g., [1, 10])
        metric (str): Metric name to plot
        scenario_num (int): Number of scenarios
        output_file (str): Output file path
    """
    # Increase font size for all text elements
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.title_fontsize': 14
    })
    
    plt.figure(figsize=(12, 8), dpi=300)
    sns.set("notebook", "darkgrid")
    
    # Define different linestyles for different costs
    cost_linestyles = {1: '-', 10: '--'}
    
    # Get default matplotlib colors for each algorithm
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    algo_colors = {algo: colors[i % len(colors)] for i, algo in enumerate(algorithms)}
    
    for algo in algorithms:
        for cost in costs:
            try:
                # Load data from the specified directory structure
                data_path = f"./logs/cost/{algo}-scenarios-{scenario_num}-cost-{cost}"
                data = pd.read_csv(f"{data_path}.txt", sep='\t')
                data = data[data["Steps"] < 2_000_000]
                
                # Filter steps for consistent intervals
                desired_steps = np.arange(20000, 2000000 + 1, 10000)
                min_step = data["Steps"].min()
                max_step = data["Steps"].max()
                valid_steps = desired_steps[(desired_steps >= min_step) & (desired_steps <= max_step)]
                
                if len(valid_steps) == 0:
                    print(f"Error: No valid steps in range for {algo} with cost {cost}")
                    continue
                
                # Interpolate values to get consistent step intervals
                if metric in data.columns:
                    f = interp1d(data["Steps"], data[metric])
                    interpolated_values = f(valid_steps)
                    
                    # Create dataframe for plotting
                    plot_data = pd.DataFrame({
                        "Steps": valid_steps,
                        metric: interpolated_values
                    })
                    
                    # Plot with consistent color per algorithm but different line style per cost
                    ax = sns.lineplot(
                        data=plot_data,
                        x="Steps",
                        y=metric,
                        label=f"{algo.upper()} (Cost={cost})",
                        linestyle=cost_linestyles[cost],
                        color=algo_colors[algo]
                    )
                else:
                    print(f"Warning: Metric {metric} not found in data for {algo} with cost {cost}")
            
            except FileNotFoundError:
                print(f"Warning: File for {algo} with cost {cost} not found")
                print(f"Attempted path: ./logs/cost/{algo}-scenarios-{scenario_num}-cost-{cost}.txt")
    
    # Get base metric name without prefixes for axis label
    base_metric = metric.split('/')[-1] if '/' in metric else metric
    base_metric = base_metric.replace("_", " ").title()
    
    if base_metric == "Complete":
        base_metric = "Route Completion"
    elif base_metric == "Cost":
        base_metric = "Safety Violations"
        # Set y-axis limit for cost metrics to range from 0 to 10
        ax.set_ylim(0, 10)
    
    ax.set_title(f"PPO-Lag vs SAC-Lag with Different Cost Values ({scenario_num} Scenarios)", fontsize=20, pad=15)
    ax.set_ylabel(f"{base_metric}", fontsize=16)
    ax.set_xlabel("Sampled Steps", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
    
    # Improved legend
    plt.legend(fontsize=14, frameon=True, fancybox=True, framealpha=0.9, 
               loc='best', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs("./pics", exist_ok=True)
    
    if not output_file:
        cost_str = "-".join([f"cost{c}" for c in costs])
        output_file = f'./pics/{base_metric}-{"-".join(algorithms)}-{cost_str}.png'
    
    plt.savefig(output_file, format='png', dpi=300, bbox_inches="tight")
    print(f"Figure saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot comparison of algorithms')
    parser.add_argument('--metrics', type=str, default='cost', 
                        help='Base metric name to plot (e.g., reward, complete, cost)')
    parser.add_argument('--train-prefix', type=str, default='train/', 
                        help='Prefix for training metrics')
    parser.add_argument('--test-prefix', type=str, default='test/', 
                        help='Prefix for testing metrics')
    parser.add_argument('--output', type=str, default=None, help='Output file name')
    parser.add_argument('--comparison-type', type=str, default='train-test',
                        choices=['standard', 'train-test', 'cost'],
                        help='Type of comparison to make')
    parser.add_argument('--costs', type=str, default='1,10',
                        help='Comma-separated list of cost values to compare')
    
    args = parser.parse_args()
    
    # Construct full metric names
    train_metric = f"{args.train_prefix}{args.metrics}"
    test_metric = f"{args.test_prefix}{args.metrics}"
    
    if args.comparison_type == 'standard':
        plot_comparison(
            ['ppol', 'sacl'], 
            train_metric, 
            scenario_num=50, 
            output_file=args.output
        )
    elif args.comparison_type == 'train-test':
        plot_train_test_comparison(
            ['ppol', 'sacl'], 
            train_metric, 
            test_metric, 
            scenario_num=50, 
            output_file=args.output
        )
    elif args.comparison_type == 'cost':
        # Parse costs into a list
        costs = [int(c) for c in args.costs.split(',')]
        
        # Use the cost comparison function
        plot_cost_comparison(
            ['ppol', 'sacl'],
            costs,
            test_metric,  # Using test metric for cost comparison
            scenario_num=50,
            output_file=args.output
        )
