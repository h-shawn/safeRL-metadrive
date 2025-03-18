import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

algo = {
    "ppol": "PPO-Lag",
    "sacl": "SAC-Lag",
}

def extract_metrics_at_step(algo, num_scenarios, seed, step=2_000_000):
    """Extract metrics at specified step from log file"""
    try:
        file_path = f"./logs/{algo}/scenarios-{num_scenarios}-seed-{seed}.txt"
        data = pd.read_csv(file_path, sep='\t')
        
        # Find row closest to 2M steps
        closest_idx = (data["Steps"] - step).abs().idxmin()
        metrics = data.iloc[closest_idx].to_dict()
        
        return metrics
    except FileNotFoundError:
        print(f"Warning: File for {algo} scenarios-{num_scenarios}-seed-{seed} not found")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def plot_metrics_side_by_side(algorithm):
    """Plot reward and route completion side by side for one algorithm"""
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    sns.set_theme("notebook", "darkgrid")
    
    metrics = ["reward", "complete"]
    metric_labels = ["Reward", "Route Completion"]
    scenario_counts = [1, 3, 10, 50]
    seeds = [1, 2, 3]
    
    # Loop through the two metrics (reward and completion)
    for i, (metric_name, metric_label, ax) in enumerate(zip(metrics, metric_labels, [ax1, ax2])):
        train_metric = f"train/{metric_name}"
        test_metric = f"test/{metric_name}"
        
        train_values = []
        test_values = []
        train_stds = []
        test_stds = []
        scenario_nums = []
        
        for num in scenario_counts:
            # Collect metrics from all seeds
            seed_metrics = [extract_metrics_at_step(algorithm, num, seed) for seed in seeds]
            seed_metrics = [m for m in seed_metrics if m is not None]
            
            if not seed_metrics:
                continue
            
            # Get individual values for calculating std
            train_metrics = [m[train_metric] for m in seed_metrics if train_metric in m]
            test_metrics = [m[test_metric] for m in seed_metrics if test_metric in m]
            
            if not train_metrics or not test_metrics:
                continue
                
            # Calculate mean and std metrics across seeds
            mean_train = np.mean(train_metrics)
            mean_test = np.mean(test_metrics)
            std_train = np.std(train_metrics)
            std_test = np.std(test_metrics)
            
            train_values.append(mean_train)
            test_values.append(mean_test)
            train_stds.append(std_train)
            test_stds.append(std_test)
            scenario_nums.append(num)
        
        if train_values and test_values:
            # Use evenly spaced indices for x-axis positions
            x_indices = np.arange(len(scenario_nums))
            
            # Plot training data with std shading on the current subplot
            ax.plot(x_indices, train_values, 'o-', label="Training")
            ax.fill_between(
                x_indices, 
                np.array(train_values) - np.array(train_stds),
                np.array(train_values) + np.array(train_stds),
                alpha=0.2,
            )
            
            # Plot testing data with std shading on the current subplot
            ax.plot(x_indices, test_values, 'o--', label="Validation")
            ax.fill_between(
                x_indices, 
                np.array(test_values) - np.array(test_stds),
                np.array(test_values) + np.array(test_stds),
                alpha=0.2,
            )
        
        ax.set_xlabel("Number of Training Maps")
        ax.set_ylabel(f"{metric_label}")
        # ax.set_title(f"{algo[algorithm]} {metric_label}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Set x-ticks to be evenly spaced but labeled with actual scenario counts
        ax.set_xticks(np.arange(len(scenario_counts)))
        ax.set_xticklabels(scenario_counts)
    
    plt.suptitle(f"{algo[algorithm]} in SafeMetaDrive", fontsize=16)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs("./pics", exist_ok=True)
    
    plt.savefig(f"./pics/{algorithm}_combined.png", format='png', dpi=300, bbox_inches="tight")
    plt.close()

def main():
    # Create combined plots for each algorithm
    plot_metrics_side_by_side("sacl")
    plot_metrics_side_by_side("ppol")
    plot_metrics_side_by_side("sacl")
    
    print("Combined plots saved to ./pics directory")

if __name__ == "__main__":
    main()