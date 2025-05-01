"""
Main script for training and testing the HDC-SNN agent in VizDoom environments

This script provides command-line functionality to:
1. Train the agent on specific scenarios
2. Test the agent's performance
3. Run evaluation with different metrics
4. Visualize agent's internal representations
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from config.scenarios import SCENARIO_CONFIGS
import os
import json
import time
import random
import torch

from brain.agent.hdc_snn_agent import HDCSNNAgent
from brain.agent.optimized_agent import OptimizedAgent  # Import the optimized agent
from brain.agent.trainer import AgentTrainer
from environment.doom_environment import DoomEnvironment
from evaluation.metrics import *
from brain.utils.config import config, set_config  # Import config utilities


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train and test brain-inspired HDC-SNN agents in VizDoom"
    )
    
    # Main command (train, test, eval)
    parser.add_argument(
        "command", 
        choices=["train", "test", "eval", "visualize"],
        help="Main command: train, test, evaluate or visualize the agent"
    )
    
    # Scenario selection
    parser.add_argument(
        "--scenario", 
        default="basic",
        choices=list(SCENARIO_CONFIGS.keys()),
        help="VizDoom scenario to use"
    )
    
    # Training parameters
    parser.add_argument(
        "--episodes", 
        type=int,
        default=1000,
        help="Number of episodes for training"
    )
    
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Render environment during training/testing"
    )
    
    # Agent parameters
    parser.add_argument(
        "--hd-dim", 
        type=int,
        default=1000,
        help="Dimensionality of HD vectors"
    )
    
    parser.add_argument(
        "--snn-neurons", 
        type=int,
        default=500,
        help="Number of neurons in SNN"
    )
    
    parser.add_argument(
        "--ca-width", 
        type=int,
        default=30,
        help="Width of cellular automata grid"
    )
    
    parser.add_argument(
        "--ca-height", 
        type=int,
        default=20,
        help="Height of cellular automata grid"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float,
        default=0.01,
        help="Learning rate for SNN"
    )
    
    # YOLO detection options
    parser.add_argument(
        "--use-yolo", 
        action="store_true",
        help="Use YOLO object detection for enhanced perception"
    )
    
    parser.add_argument(
        "--show-yolo-detections", 
        action="store_true",
        help="Show YOLO detection visualizations (requires --use-yolo)"
    )
    
    # Model path for loading/saving
    parser.add_argument(
        "--model", 
        type=str,
        help="Path to load/save model"
    )
    
    # Output directory
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="results",
        help="Directory for saving results"
    )
    
    # Visualization options
    parser.add_argument(
        "--visualize-internals", 
        action="store_true",
        help="Visualize agent's internal representations during execution"
    )
    
    # Optimization options
    parser.add_argument(
        "--use-optimized", 
        action="store_true",
        help="Use the optimized implementation with GPU acceleration"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu, cuda, cuda:0, etc.)"
    )
    
    parser.add_argument(
        "--disable-jit",
        action="store_true",
        help="Disable JIT compilation for debugging"
    )
    
    parser.add_argument(
        "--disable-batch",
        action="store_true",
        help="Disable batch processing"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def create_agent(args, scenario_config):
    """Create and configure the agent based on arguments or scenario config"""
    # Use scenario-specific parameters if not overridden by args
    hd_dim = args.hd_dim
    learning_rate = args.learning_rate
    ca_width = args.ca_width
    ca_height = args.ca_height
    
    # Override with scenario-specific optimal parameters if available
    if "optimal_params" in scenario_config:
        optimal = scenario_config["optimal_params"]
        if not args.hd_dim and "hd_dim" in optimal:
            hd_dim = optimal["hd_dim"]
        if not args.learning_rate and "learning_rate" in optimal:
            learning_rate = optimal["learning_rate"]
        if not args.ca_width and "ca_width" in optimal:
            ca_width = optimal["ca_width"]
        if not args.ca_height and "ca_height" in optimal:
            ca_height = optimal["ca_height"]
    
    # Set up device
    device = args.device
    if device is None:
        # Auto-detect device (use GPU if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Configure optimizations
    if args.disable_jit:
        set_config(use_jit_compile=False)
    if args.disable_batch:
        set_config(batch_process=False)
        
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            
    # Update config
    set_config(hd_dimension=hd_dim, device=device)
    
    if args.use_optimized:
        # Create optimized agent
        print(f"Creating optimized agent on device: {device}")
        agent = OptimizedAgent(
            input_shape=(120, 160, 3),
            hd_dim=hd_dim,
            snn_neurons=args.snn_neurons,
            num_actions=5,  # Fixed for VizDoom
            ca_width=ca_width,
            ca_height=ca_height,
            memory_capacity=10000,
            learning_rate=learning_rate,
            use_yolo=args.use_yolo,
            device=device
        )
    else:
        # Create standard agent
        agent = HDCSNNAgent(
            input_shape=(120, 160, 3),
            hd_dim=hd_dim,
            snn_neurons=args.snn_neurons,
            num_actions=5,  # Fixed for VizDoom
            ca_width=ca_width,
            ca_height=ca_height,
            memory_capacity=10000,
            learning_rate=learning_rate,
            use_yolo=args.use_yolo
        )
    
    # Enable visualization if requested
    agent.visualize_internals = args.visualize_internals
    
    # Enable YOLO visualization if requested
    if args.show_yolo_detections:
        if not args.use_yolo:
            print("Warning: --show-yolo-detections requires --use-yolo, ignoring")
        else:
            agent.show_yolo_detections = True
    
    # Load model if specified
    if args.model and os.path.exists(args.model):
        print(f"Loading agent from {args.model}")
        agent.load(args.model)
    
    return agent


def create_environment(args):
    """Create and configure the environment based on arguments"""
    scenario = args.scenario
    
    # Create environment
    env = DoomEnvironment(
        scenario=scenario,
        frame_skip=4,
        visible=args.render
    )
    
    return env


def train(args):
    """Train the agent"""
    # Get scenario config
    scenario_config = SCENARIO_CONFIGS[args.scenario]
    print(f"Training on scenario: {args.scenario} - {scenario_config['description']}")
    print(f"Required skills: {', '.join(scenario_config['skills'])}")
    
    if args.use_yolo:
        print("Using YOLO object detection for enhanced perception")
    
    if args.use_optimized:
        print("Using optimized implementation with GPU acceleration")
        if args.device:
            print(f"Device: {args.device}")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Auto-selected device: {device}")
    
    # Create environment and agent
    env = create_environment(args)
    agent = create_agent(args, scenario_config)
    
    # Configure trainer
    trainer_config = {
        "max_episodes": args.episodes,
        "max_steps_per_episode": 1000,
        "eval_frequency": 20,
        "eval_episodes": 5,
        "save_frequency": 100,
        "replay_frequency": 4,
        "replay_batch_size": 32,
        "logging_frequency": 1,
        "render_during_training": args.render,
        "render_during_eval": args.render,
        "output_dir": args.output_dir
    }
    
    # Create trainer
    trainer = AgentTrainer(agent, env, trainer_config)
    
    # Train the agent
    start_time = time.time()
    metrics = trainer.train()
    total_time = time.time() - start_time
    
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Plot and save metrics
    trainer.plot_metrics(save_fig=True)
    
    # Save model if path specified
    if args.model:
        # Set full path for the model
        model_path = args.model
        if not os.path.isabs(model_path):
            model_path = os.path.join(args.output_dir, os.path.basename(model_path))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        print(f"Saving model to {model_path}")
        agent.save(model_path)
        
        # Save configuration used
        config_path = os.path.join(os.path.dirname(model_path), 
                                   f"{os.path.basename(model_path)}_config.json")
                                   
        config_data = {
            "scenario": args.scenario,
            "hd_dim": agent.hd_dim,
            "snn_neurons": args.snn_neurons,
            "ca_width": agent.ca.width if hasattr(agent.ca, 'width') else ca_width,
            "ca_height": agent.ca.height if hasattr(agent.ca, 'height') else ca_height,
            "learning_rate": agent.learning_rate,
            "use_yolo": agent.use_yolo,
            "use_optimized": args.use_optimized,
            "device": args.device,
            "training_episodes": args.episodes,
            "training_time_seconds": total_time,
            "jit_compiled": config.use_jit_compile,
            "batch_processing": config.batch_process
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
            
        print(f"Configuration saved to {config_path}")
    
    # Return metrics for evaluating performance
    return metrics


def test(args):
    """Test the agent"""
    # Get scenario config
    scenario_config = SCENARIO_CONFIGS[args.scenario]
    print(f"Testing on scenario: {args.scenario}")
    
    # Create environment and agent
    env = create_environment(args)
    agent = create_agent(args, scenario_config)
    
    # Check if model is specified
    if not args.model:
        print("Warning: No model specified for testing. Using untrained agent.")
    
    # Set up metrics tracking
    rewards = []
    steps = []
    completed = []
    
    # Test the agent for a specified number of episodes
    n_episodes = min(args.episodes, 100)  # Cap test episodes
    
    for episode in range(n_episodes):
        # Reset environment and agent
        observation = env.reset()
        agent.reset()
        
        total_reward = 0
        is_done = False
        step = 0
        
        while not is_done:
            # Select action
            action = agent.act(observation, deterministic=True)
            
            # Execute action
            next_observation, reward, is_done, info = env.step(action)
            
            # Update metrics
            total_reward += reward
            step += 1
            
            # Render if requested
            if args.render:
                env.render()
                
                # Visualize internal states if requested
                if args.visualize_internals:
                    vis_data = agent.visualize(observation)
                    # Visualization code here
                
            # Update observation
            observation = next_observation
            
            # Break if episode is too long
            if step >= 1000:
                break
                
        # Update metrics
        rewards.append(total_reward)
        steps.append(step)
        completed.append(info.get('completed', False))
        
        # Print episode summary
        print(f"Episode {episode+1}/{n_episodes}: Reward = {total_reward}, Steps = {step}, Completed = {info.get('completed', False)}")
    
    # Print overall performance
    print("\nTest Results:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Steps: {np.mean(steps):.2f} ± {np.std(steps):.2f}")
    print(f"Completion Rate: {np.mean(completed)*100:.2f}%")
    
    # Save results if output directory specified
    if args.output_dir:
        # Create results directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save test results
        results = {
            "scenario": args.scenario,
            "model": args.model,
            "num_episodes": n_episodes,
            "average_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "average_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "completion_rate": float(np.mean(completed)),
            "all_rewards": rewards,
            "all_steps": steps,
            "all_completed": completed,
            "use_optimized": args.use_optimized
        }
        
        # Add optimization details if using optimized agent
        if args.use_optimized:
            results.update({
                "device": args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
                "jit_compiled": config.use_jit_compile,
                "batch_processing": config.batch_process
            })
            
            if hasattr(agent, 'forward_times') and agent.forward_times:
                results["avg_forward_time"] = sum(agent.forward_times) / len(agent.forward_times)
        
        # Save to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(args.output_dir, f"test_results_{args.scenario}_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"Test results saved to {results_file}")
    
    return rewards, steps, completed


def evaluate(args):
    """Evaluate the agent with additional metrics"""
    # Get scenario config
    scenario_config = SCENARIO_CONFIGS[args.scenario]
    print(f"Evaluating on scenario: {args.scenario}")
    
    # Create environment and agent
    env = create_environment(args)
    agent = create_agent(args, scenario_config)
    
    # Check if model is specified
    if not args.model:
        print("Warning: No model specified for evaluation. Using untrained agent.")
    
    # Initialize metrics calculators
    sample_efficiency = SampleEfficiencyMetric()
    reaction_time = ReactionTimeMetric()
    exploration = ExplorationMetric(env.action_space.n)
    stability = StabilityMetric()
    
    # Test the agent for a specified number of episodes
    n_episodes = min(args.episodes, 50)  # Cap evaluation episodes
    
    for episode in range(n_episodes):
        # Reset environment, agent and metrics
        observation = env.reset()
        agent.reset()
        
        exploration.reset()
        reaction_time.reset()
        
        total_reward = 0
        is_done = False
        step = 0
        
        states, actions, rewards = [], [], []
        
        while not is_done:
            # Select action
            action = agent.act(observation)
            
            # Track for metrics
            reaction_time.record_decision_time()
            exploration.record_action(action)
            
            # Execute action
            next_observation, reward, is_done, info = env.step(action)
            
            # Store for stability metrics
            states.append(observation)
            actions.append(action)
            rewards.append(reward)
            
            # Update metrics
            total_reward += reward
            step += 1
            
            # Record sample efficiency data
            sample_efficiency.record_step(
                observation, action, reward, next_observation, is_done)
            
            # Render if requested
            if args.render:
                env.render()
                
            # Update observation
            observation = next_observation
            
            # Break if episode is too long
            if step >= 1000:
                break
                
        # Record episode completion for sample efficiency
        sample_efficiency.record_episode_result(total_reward, info.get('completed', False))
        
        # Calculate stability metrics for the episode
        stability.record_episode(states, actions, rewards)
        
        # Print episode summary
        print(f"Episode {episode+1}/{n_episodes}: Reward = {total_reward}, Steps = {step}")
    
    # Calculate final metrics
    efficiency_score = sample_efficiency.calculate()
    avg_reaction_time = reaction_time.calculate()
    exploration_score = exploration.calculate()
    stability_score = stability.calculate()
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Sample Efficiency Score: {efficiency_score:.4f}")
    print(f"Average Reaction Time: {avg_reaction_time:.4f} seconds")
    print(f"Exploration Score: {exploration_score:.4f}")
    print(f"Stability Score: {stability_score:.4f}")
    
    # Performance comparison if using optimized agent
    if args.use_optimized and hasattr(agent, 'forward_times') and agent.forward_times:
        avg_forward_time = sum(agent.forward_times) / len(agent.forward_times)
        print(f"Average Forward Pass Time: {avg_forward_time*1000:.2f} ms")
    
    # Save results if output directory specified
    if args.output_dir:
        # Create results directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save evaluation results
        results = {
            "scenario": args.scenario,
            "model": args.model,
            "num_episodes": n_episodes,
            "sample_efficiency_score": float(efficiency_score),
            "avg_reaction_time": float(avg_reaction_time),
            "exploration_score": float(exploration_score),
            "stability_score": float(stability_score),
            "use_optimized": args.use_optimized
        }
        
        # Add optimization details if using optimized agent
        if args.use_optimized:
            results.update({
                "device": args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
                "jit_compiled": config.use_jit_compile,
                "batch_processing": config.batch_process
            })
            
            if hasattr(agent, 'forward_times') and agent.forward_times:
                results["avg_forward_time"] = sum(agent.forward_times) / len(agent.forward_times)
        
        # Save to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(args.output_dir, f"eval_results_{args.scenario}_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"Evaluation results saved to {results_file}")
    
    return {
        "efficiency": efficiency_score,
        "reaction_time": avg_reaction_time,
        "exploration": exploration_score,
        "stability": stability_score
    }


def visualize(args):
    """Visualize agent's internal representations"""
    # Get scenario config
    scenario_config = SCENARIO_CONFIGS[args.scenario]
    print(f"Visualizing agent on scenario: {args.scenario}")
    
    # Create environment and agent
    env = create_environment(args)
    agent = create_agent(args, scenario_config)
    
    # Check if model is specified
    if not args.model:
        print("Warning: No model specified for visualization. Using untrained agent.")
    
    # Enable internal visualization
    agent.visualize_internals = True
    
    # Initialize visualization
    plt.ion()  # Enable interactive mode
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.tight_layout(pad=3.0)
    
    # Run for a few episodes
    n_episodes = min(args.episodes, 10)  # Cap visualization episodes
    
    for episode in range(n_episodes):
        # Reset environment and agent
        observation = env.reset()
        agent.reset()
        
        is_done = False
        step = 0
        
        while not is_done:
            # Update visualization
            plt.suptitle(f"Episode {episode+1}, Step {step+1}", fontsize=16)
            
            # Get visualization data
            vis_data = agent.visualize(observation)
            
            # Plot environment view
            axes[0, 0].clear()
            axes[0, 0].imshow(observation)
            axes[0, 0].set_title("Environment View")
            axes[0, 0].axis('off')
            
            # Plot cellular automata state
            if "ca_grid" in vis_data:
                axes[0, 1].clear()
                axes[0, 1].imshow(vis_data["ca_grid"], cmap='viridis')
                axes[0, 1].set_title("Cellular Automata State")
                axes[0, 1].axis('off')
                
            # Plot episodic memory stats
            if "episodic_memory" in vis_data:
                axes[0, 2].clear()
                mem_stats = vis_data["episodic_memory"]
                mem_data = [mem_stats["size"], mem_stats["capacity"] - mem_stats["size"]]
                axes[0, 2].pie(mem_data, labels=["Used", "Free"], autopct='%1.1f%%')
                axes[0, 2].set_title(f"Episodic Memory Usage: {mem_stats['size']}/{mem_stats['capacity']}")
                
            # Plot neuromodulator levels
            if "neuromodulators" in vis_data:
                axes[1, 0].clear()
                neuromod = vis_data["neuromodulators"]
                labels = list(neuromod.keys())
                values = list(neuromod.values())
                x = np.arange(len(labels))
                axes[1, 0].bar(x, values, width=0.6)
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(labels, rotation=45)
                axes[1, 0].set_title("Neuromodulator Levels")
                axes[1, 0].set_ylim(0, 1)
                
            # Plot timing information if available
            if "avg_forward_time" in vis_data:
                axes[1, 1].clear()
                timing_data = [vis_data.get("avg_forward_time", 0) * 1000, 
                              vis_data.get("avg_learning_time", 0) * 1000]
                axes[1, 1].bar(["Forward (ms)", "Learning (ms)"], timing_data)
                axes[1, 1].set_title("Processing Time")
                
            # Plot exploration rate
            axes[1, 2].clear()
            axes[1, 2].plot([0, 1], [vis_data.get("epsilon", 0), vis_data.get("epsilon", 0)], 'r-', linewidth=2)
            axes[1, 2].set_title(f"Exploration Rate: {vis_data.get('epsilon', 0):.3f}")
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            
            # Update the figure
            plt.draw()
            plt.pause(0.01)
            
            # Select action
            action = agent.act(observation)
            
            # Execute action
            next_observation, reward, is_done, info = env.step(action)
            
            # Render environment
            env.render()
            
            # Update observation
            observation = next_observation
            
            step += 1
            
            # Break if episode is too long
            if step >= 1000:
                break
        
        print(f"Episode {episode+1}/{n_episodes} completed in {step} steps")
    
    # Wait for user to close the window
    plt.ioff()
    plt.show()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Execute the specified command
    if args.command == "train":
        train(args)
    elif args.command == "test":
        test(args)
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "visualize":
        visualize(args)


if __name__ == "__main__":
    main() 