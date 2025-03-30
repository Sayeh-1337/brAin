"""
Evaluation Metrics Module

Provides functions for calculating and visualizing various performance metrics
for the agent's behavior and learning progress.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

def calculate_performance_metrics(rewards, steps, success=None):
    """
    Calculate basic performance metrics from episode rewards and steps
    
    Args:
        rewards: List of episode rewards
        steps: List of episode steps/lengths
        success: Optional list of boolean success indicators
        
    Returns:
        Dictionary of performance metrics
    """
    if not rewards:
        return {}
        
    metrics = {
        "mean_reward": np.mean(rewards),
        "median_reward": np.median(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_steps": np.mean(steps),
        "median_steps": np.median(steps),
    }
    
    # Calculate success rate if success indicators provided
    if success is not None:
        metrics["success_rate"] = np.mean(success)
        
    return metrics

def calculate_action_metrics(actions, rewards):
    """
    Calculate metrics related to action selection
    
    Args:
        actions: List of actions taken
        rewards: Corresponding rewards received
        
    Returns:
        Dictionary of action-related metrics
    """
    unique_actions = np.unique(actions)
    
    # Calculate action frequency
    action_counts = {}
    for action in unique_actions:
        action_counts[int(action)] = np.sum(np.array(actions) == action) / len(actions)
        
    # Calculate average reward per action
    action_rewards = {}
    for action in unique_actions:
        indices = np.where(np.array(actions) == action)[0]
        if len(indices) > 0:
            action_rewards[int(action)] = np.mean([rewards[i] for i in indices])
        else:
            action_rewards[int(action)] = 0
            
    return {
        "action_frequency": action_counts,
        "action_rewards": action_rewards
    }

def calculate_learning_metrics(rewards, window_size=100):
    """
    Calculate metrics that track learning progress
    
    Args:
        rewards: List of episode rewards
        window_size: Size of the window for moving averages
        
    Returns:
        Dictionary of learning-related metrics
    """
    if len(rewards) < 2:
        return {}
        
    # Calculate moving average
    moving_avg = []
    for i in range(len(rewards)):
        window_start = max(0, i - window_size + 1)
        window = rewards[window_start:i+1]
        moving_avg.append(np.mean(window))
        
    # Calculate improvement metrics
    initial_performance = np.mean(rewards[:min(window_size, len(rewards)//5)])
    final_performance = np.mean(rewards[-min(window_size, len(rewards)//5):])
    improvement = final_performance - initial_performance
    
    # Calculate convergence (when moving average stabilizes)
    convergence_threshold = 0.1  # 10% of range
    reward_range = max(rewards) - min(rewards)
    threshold = convergence_threshold * reward_range
    
    converged_at = None
    for i in range(window_size, len(moving_avg)):
        window = moving_avg[i-window_size:i]
        if np.std(window) < threshold:
            converged_at = i
            break
            
    return {
        "moving_average": moving_avg,
        "improvement": improvement,
        "improvement_percentage": (improvement / abs(initial_performance)) * 100 if initial_performance != 0 else 0,
        "initial_performance": initial_performance,
        "final_performance": final_performance,
        "convergence_episode": converged_at
    }

def calculate_statistical_significance(rewards_a, rewards_b, alpha=0.05):
    """
    Calculate statistical significance between two reward distributions
    
    Args:
        rewards_a: First list of rewards
        rewards_b: Second list of rewards
        alpha: Significance level
        
    Returns:
        Dictionary of statistical test results
    """
    if len(rewards_a) < 2 or len(rewards_b) < 2:
        return {"significant": False, "p_value": 1.0}
        
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(rewards_a, rewards_b, equal_var=False)
    
    # Perform Mann-Whitney U test (non-parametric)
    u_stat, u_p_value = stats.mannwhitneyu(rewards_a, rewards_b, alternative='two-sided')
    
    return {
        "t_test_p_value": p_value,
        "u_test_p_value": u_p_value,
        "significant_t_test": p_value < alpha,
        "significant_u_test": u_p_value < alpha,
        "effect_size": (np.mean(rewards_b) - np.mean(rewards_a)) / np.sqrt((np.std(rewards_a)**2 + np.std(rewards_b)**2) / 2)
    }

def plot_reward_distribution(rewards, title="Reward Distribution", figsize=(10, 6)):
    """
    Plot the distribution of rewards
    
    Args:
        rewards: List of rewards
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create histogram with KDE
    sns.histplot(rewards, kde=True)
    
    # Add vertical lines for statistics
    plt.axvline(np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    plt.axvline(np.median(rewards), color='g', linestyle='-.', label=f'Median: {np.median(rewards):.2f}')
    
    plt.title(title)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_learning_curve(rewards, window_size=100, figsize=(12, 6)):
    """
    Plot the learning curve with moving average
    
    Args:
        rewards: List of episode rewards
        window_size: Size of the moving average window
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, color='blue', label="Episode Rewards")
    
    # Calculate and plot moving average
    moving_avg = []
    for i in range(len(rewards)):
        window_start = max(0, i - window_size + 1)
        window = rewards[window_start:i+1]
        moving_avg.append(np.mean(window))
        
    plt.plot(moving_avg, linewidth=2, color='red', label=f"{window_size}-Episode Moving Average")
    
    # Add trend line
    if len(rewards) > 1:
        x = np.arange(len(rewards))
        z = np.polyfit(x, rewards, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), linestyle='--', color='green', label=f"Trend: {z[0]:.4f}x + {z[1]:.2f}")
    
    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_action_distribution(actions, action_names=None, figsize=(10, 6)):
    """
    Plot the distribution of actions taken
    
    Args:
        actions: List of actions taken
        action_names: Optional list of action names
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Count occurrences of each action
    unique_actions = np.unique(actions)
    action_counts = [np.sum(np.array(actions) == action) for action in unique_actions]
    
    # Set action names if provided
    if action_names is None:
        action_names = [str(int(a)) for a in unique_actions]
    else:
        action_names = [action_names[int(a)] for a in unique_actions]
    
    # Create bar chart
    bars = plt.bar(action_names, action_counts)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')
    
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3, axis='y')
    
    return plt.gcf()

def create_summary_report(metrics, fig_learning_curve=None, fig_reward_dist=None, fig_action_dist=None):
    """
    Create a comprehensive summary report of all metrics
    
    Args:
        metrics: Dictionary of all calculated metrics
        fig_learning_curve: Optional learning curve figure
        fig_reward_dist: Optional reward distribution figure
        fig_action_dist: Optional action distribution figure
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 50)
    report.append("AGENT PERFORMANCE SUMMARY REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Performance metrics
    report.append("PERFORMANCE METRICS")
    report.append("-" * 50)
    if "mean_reward" in metrics:
        report.append(f"Mean Reward: {metrics['mean_reward']:.2f}")
        report.append(f"Median Reward: {metrics['median_reward']:.2f}")
        report.append(f"Reward Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
        report.append(f"Reward StdDev: {metrics['std_reward']:.2f}")
    if "success_rate" in metrics:
        report.append(f"Success Rate: {metrics['success_rate']*100:.2f}%")
    if "mean_steps" in metrics:
        report.append(f"Mean Episode Length: {metrics['mean_steps']:.2f} steps")
    report.append("")
    
    # Learning metrics
    if "improvement" in metrics:
        report.append("LEARNING METRICS")
        report.append("-" * 50)
        report.append(f"Initial Performance: {metrics['initial_performance']:.2f}")
        report.append(f"Final Performance: {metrics['final_performance']:.2f}")
        report.append(f"Improvement: {metrics['improvement']:.2f} ({metrics['improvement_percentage']:.2f}%)")
        if metrics['convergence_episode'] is not None:
            report.append(f"Converged at Episode: {metrics['convergence_episode']}")
        report.append("")
    
    # Action metrics
    if "action_frequency" in metrics:
        report.append("ACTION METRICS")
        report.append("-" * 50)
        report.append("Action Frequencies:")
        for action, freq in metrics["action_frequency"].items():
            report.append(f"  Action {action}: {freq*100:.2f}%")
        report.append("Average Reward per Action:")
        for action, reward in metrics["action_rewards"].items():
            report.append(f"  Action {action}: {reward:.2f}")
        report.append("")
    
    # Statistical significance
    if "t_test_p_value" in metrics:
        report.append("STATISTICAL SIGNIFICANCE")
        report.append("-" * 50)
        report.append(f"T-test p-value: {metrics['t_test_p_value']:.6f} " + 
                     ("(Significant)" if metrics['significant_t_test'] else "(Not Significant)"))
        report.append(f"Mann-Whitney U test p-value: {metrics['u_test_p_value']:.6f} " + 
                     ("(Significant)" if metrics['significant_u_test'] else "(Not Significant)"))
        report.append(f"Effect Size: {metrics['effect_size']:.2f}")
        report.append("")
    
    report.append("=" * 50)
    
    return "\n".join(report)

def evaluate_generalization(scenario_results):
    """
    Evaluate how well the agent generalizes across different scenarios
    
    Args:
        scenario_results: Dictionary mapping scenario names to results
        
    Returns:
        Dictionary of generalization metrics
    """
    # Extract success rates for each scenario
    success_rates = {name: results["success_rate"] for name, results in scenario_results.items()}
    
    # Extract mean rewards for each scenario
    mean_rewards = {name: np.mean(results["rewards"]) for name, results in scenario_results.items()}
    
    # Calculate generalization metrics
    avg_success = np.mean(list(success_rates.values()))
    success_std = np.std(list(success_rates.values()))
    min_success = min(success_rates.values())
    max_success = max(success_rates.values())
    success_range = max_success - min_success
    
    return {
        "scenario_success_rates": success_rates,
        "scenario_mean_rewards": mean_rewards,
        "average_success_rate": avg_success,
        "success_rate_std": success_std,
        "success_rate_range": success_range,
        "min_success_scenario": min(success_rates, key=success_rates.get),
        "max_success_scenario": max(success_rates, key=success_rates.get),
        "generalization_score": avg_success * (1 - success_std)  # High mean, low variance
    }

def plot_scenario_comparison(scenario_results, metric="rewards", figsize=(12, 8)):
    """
    Create a comparative plot of results across different scenarios
    
    Args:
        scenario_results: Dictionary mapping scenario names to results
        metric: Metric to compare ('rewards' or 'steps')
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Extract scenario names and data
    scenarios = list(scenario_results.keys())
    data = [scenario_results[scenario][metric] for scenario in scenarios]
    
    # Create box plot
    plt.boxplot(data, labels=scenarios)
    
    # Add data points
    for i, d in enumerate(data):
        x = np.random.normal(i+1, 0.05, size=len(d))
        plt.scatter(x, d, alpha=0.4)
    
    plt.title(f"Scenario Comparison: {metric.capitalize()}")
    plt.ylabel(metric.capitalize())
    plt.grid(True, alpha=0.3, axis='y')
    
    return plt.gcf() 