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

from brain.agent.hdc_snn_agent import HDCSNNAgent
from brain.agent.trainer import AgentTrainer
from environment.doom_environment import DoomEnvironment
from evaluation.metrics import *


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
    
    # Create agent
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
        agent.save(args.model)
        print(f"Saved trained model to {args.model}")


def test(args):
    """Test the agent on a scenario"""
    if not args.model:
        print("Error: Must specify a model path with --model")
        return
        
    # Get scenario config
    scenario_config = SCENARIO_CONFIGS[args.scenario]
    print(f"Testing on scenario: {args.scenario} - {scenario_config['description']}")
    
    if args.use_yolo:
        print("Using YOLO object detection for enhanced perception")
    
    # Create environment and agent
    env = create_environment(args)
    agent = create_agent(args, scenario_config)
    
    # Configure trainer for testing
    trainer_config = {
        "render_during_eval": args.render,
        "output_dir": args.output_dir
    }
    
    # Create trainer
    trainer = AgentTrainer(agent, env, trainer_config)
    
    # Test the agent
    num_test_episodes = 10
    print(f"Running {num_test_episodes} test episodes...")
    
    rewards = []
    steps = []
    successes = []
    actions_taken = []
    
    # Run test episodes
    for episode in range(num_test_episodes):
        observation = env.reset()
        agent.reset()
        
        episode_reward = 0
        episode_steps = 0
        episode_actions = []
        done = False
        
        while not done:
            if args.render:
                env.render()
                
            # Get motion information
            motion = env.get_motion_frames()
            
            # Select action deterministically
            action = agent.act(observation, motion, deterministic=True)
            episode_actions.append(action)
            
            # Take action
            observation, reward, done, info = env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_steps += 1
            
            # Visualization
            if args.visualize_internals:
                agent.visualize(observation)
                
            # Break if episode is too long
            if episode_steps >= 1000:
                break
                
        # Record results
        rewards.append(episode_reward)
        steps.append(episode_steps)
        successes.append(episode_reward > 0)  # Simple success criterion
        actions_taken.extend(episode_actions)
        
        print(f"Episode {episode+1}: Reward = {episode_reward}, Steps = {episode_steps}")
        
    # Calculate and print metrics
    performance_metrics = calculate_performance_metrics(rewards, steps, successes)
    action_metrics = calculate_action_metrics(actions_taken, rewards[:len(actions_taken)])
    
    # Combine metrics
    all_metrics = {**performance_metrics, **action_metrics}
    
    # Create report
    report = create_summary_report(all_metrics)
    print("\n" + report)
    
    # Save report
    report_path = os.path.join(args.output_dir, f"test_report_{args.scenario}.txt")
    with open(report_path, "w") as f:
        f.write(report)
        
    print(f"Saved test report to {report_path}")
    
    # Create and save plots
    fig_reward = plot_reward_distribution(rewards, title=f"Reward Distribution - {args.scenario}")
    fig_reward.savefig(os.path.join(args.output_dir, f"reward_dist_{args.scenario}.png"))
    
    fig_actions = plot_action_distribution(actions_taken, action_names=env.get_action_names())
    fig_actions.savefig(os.path.join(args.output_dir, f"action_dist_{args.scenario}.png"))
    
    plt.close("all")


def evaluate(args):
    """Run comprehensive evaluation across multiple scenarios"""
    if not args.model:
        print("Error: Must specify a model path with --model")
        return
        
    print("Running comprehensive evaluation across multiple scenarios")
    
    if args.use_yolo:
        print("Using YOLO object detection for enhanced perception")
    
    # List of scenarios to evaluate on
    scenarios = list(SCENARIO_CONFIGS.keys())
    
    # Results for each scenario
    scenario_results = {}
    
    for scenario in scenarios:
        print(f"\nEvaluating on scenario: {scenario}")
        
        # Create environment and agent for this scenario
        args.scenario = scenario
        env = create_environment(args)
        agent = create_agent(args, SCENARIO_CONFIGS[scenario])
        
        # Configure trainer
        trainer_config = {
            "render_during_eval": args.render,
            "output_dir": args.output_dir
        }
        
        # Create trainer
        trainer = AgentTrainer(agent, env, trainer_config)
        
        # Run scenario evaluation
        results = trainer.run_scenario(
            SCENARIO_CONFIGS[scenario],
            num_episodes=5,
            record=False
        )
        
        # Store results
        scenario_results[scenario] = results
        
        # Close environment
        env.close()
        
    # Calculate generalization metrics
    gen_metrics = evaluate_generalization(scenario_results)
    
    # Print summary
    print("\n" + "="*50)
    print("GENERALIZATION EVALUATION SUMMARY")
    print("="*50)
    print(f"Average Success Rate: {gen_metrics['average_success_rate']*100:.2f}%")
    print(f"Success Rate Std Dev: {gen_metrics['success_rate_std']*100:.2f}%")
    print(f"Best Scenario: {gen_metrics['max_success_scenario']} "
          f"({gen_metrics['scenario_success_rates'][gen_metrics['max_success_scenario']]*100:.2f}%)")
    print(f"Worst Scenario: {gen_metrics['min_success_scenario']} "
          f"({gen_metrics['scenario_success_rates'][gen_metrics['min_success_scenario']]*100:.2f}%)")
    print(f"Generalization Score: {gen_metrics['generalization_score']*100:.2f}%")
    
    # Create and save scenario comparison plot
    fig = plot_scenario_comparison(scenario_results, metric="rewards")
    fig.savefig(os.path.join(args.output_dir, "scenario_comparison.png"))
    plt.close(fig)
    
    # Save detailed metrics
    metrics_path = os.path.join(args.output_dir, "generalization_metrics.json")
    with open(metrics_path, "w") as f:
        # Convert numpy values to Python primitives for JSON serialization
        serializable_metrics = {
            k: float(v) if isinstance(v, np.floating) else v
            for k, v in gen_metrics.items()
        }
        json.dump(serializable_metrics, f, indent=2)
        
    print(f"Saved generalization metrics to {metrics_path}")


def visualize(args):
    """Visualize agent's internal representations"""
    if not args.model:
        print("Error: Must specify a model path with --model")
        return
    
    if args.use_yolo:
        print("Using YOLO object detection for enhanced perception")
        
    # Create environment and agent
    env = create_environment(args)
    agent = create_agent(args, SCENARIO_CONFIGS[args.scenario])
    
    # Enable visualization
    agent.visualize_internals = True
    
    print("Running visualization mode. Press Ctrl+C to exit.")
    
    try:
        # Run a few episodes for visualization
        for episode in range(3):
            observation = env.reset()
            agent.reset()
            
            done = False
            step = 0
            
            while not done and step < 500:
                # Always render in visualization mode
                env.render()
                
                # Get motion information
                motion = env.get_motion_frames()
                
                # Select action
                action = agent.act(observation, motion, deterministic=True)
                
                # Take action
                observation, reward, done, info = env.step(action)
                
                # Visualize internal state
                agent.visualize(observation)
                
                step += 1
                
                # Slow down visualization
                time.sleep(0.05)
                
    except KeyboardInterrupt:
        print("Visualization stopped by user")
        
    finally:
        env.close()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Execute the requested command
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