"""
Agent Trainer Module

Provides functionality for training and evaluating agents in environments.
Implements training loops, evaluation metrics, and logging functionality.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

class AgentTrainer:
    """
    Trainer class for reinforcement learning agents
    
    Handles training loops, evaluation, logging, and visualization
    for agent training across different scenarios.
    """
    
    def __init__(self, agent, environment, config=None):
        """
        Initialize the trainer
        
        Args:
            agent: The agent to train
            environment: The environment to train in
            config: Training configuration parameters
        """
        self.agent = agent
        self.env = environment
        
        # Default configuration
        self.config = {
            "max_episodes": 1000,
            "max_steps_per_episode": 1000,
            "eval_frequency": 10,  # Evaluate every N episodes
            "eval_episodes": 5,     # Number of episodes for evaluation
            "save_frequency": 100,  # Save model every N episodes
            "replay_frequency": 4,  # Replay experience every N steps
            "replay_batch_size": 32,
            "logging_frequency": 1, # Log metrics every N episodes
            "render_during_training": False,
            "render_during_eval": True,
            "output_dir": "results"
        }
        
        # Update with user config
        if config:
            self.config.update(config)
            
        # Initialize training metrics
        self.metrics = {
            "episode_rewards": [],
            "avg_rewards": [],
            "eval_rewards": [],
            "episode_lengths": [],
            "training_times": [],
            "epsilon_values": []
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])
            
        # Initialize timestamp for this training run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log file path
        self.log_file = os.path.join(
            self.config["output_dir"], 
            f"training_log_{self.timestamp}.json"
        )
        
    def train(self, resume=False):
        """
        Train the agent for the specified number of episodes
        
        Args:
            resume: Whether to resume training from a checkpoint
            
        Returns:
            Training metrics dictionary
        """
        print(f"Starting training for {self.config['max_episodes']} episodes")
        
        total_training_time = 0
        
        # Resume if requested
        start_episode = 0
        if resume:
            # TODO: Implement resume functionality
            pass
            
        for episode in range(start_episode, self.config["max_episodes"]):
            # Reset environment and agent for new episode
            observation = self.env.reset()
            self.agent.reset()
            
            episode_reward = 0
            episode_start_time = time.time()
            
            # Run episode
            for step in range(self.config["max_steps_per_episode"]):
                # Optional rendering
                if self.config["render_during_training"]:
                    self.env.render()
                    
                # Get motion information (frame difference)
                motion = self.env.get_motion_frames() if hasattr(self.env, "get_motion_frames") else None
                
                # Select action
                action = self.agent.act(observation, motion)
                
                # Take action in environment
                next_observation, reward, done, info = self.env.step(action)
                
                # Let agent learn from this experience
                self.agent.learn(observation, action, reward, next_observation, done)
                
                # Experience replay
                if step % self.config["replay_frequency"] == 0:
                    self.agent.replay_experience(batch_size=self.config["replay_batch_size"])
                    
                # Update metrics
                episode_reward += reward
                
                # Visualization
                self.agent.visualize(observation)
                
                # Move to next state
                observation = next_observation
                
                if done:
                    break
                    
            # Calculate episode time
            episode_time = time.time() - episode_start_time
            total_training_time += episode_time
            
            # Update metrics
            self.metrics["episode_rewards"].append(episode_reward)
            self.metrics["episode_lengths"].append(step + 1)
            self.metrics["training_times"].append(episode_time)
            self.metrics["epsilon_values"].append(self.agent.epsilon)
            
            # Calculate moving average
            window_size = min(10, len(self.metrics["episode_rewards"]))
            avg_reward = np.mean(self.metrics["episode_rewards"][-window_size:])
            self.metrics["avg_rewards"].append(avg_reward)
            
            # Logging
            if episode % self.config["logging_frequency"] == 0:
                print(f"Episode {episode+1}/{self.config['max_episodes']}, "
                      f"Reward: {episode_reward:.2f}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Steps: {step+1}, "
                      f"Epsilon: {self.agent.epsilon:.4f}, "
                      f"Time: {episode_time:.2f}s")
                self._save_log()
                
            # Evaluation
            if episode % self.config["eval_frequency"] == 0:
                eval_reward = self.evaluate()
                self.metrics["eval_rewards"].append(eval_reward)
                print(f"Evaluation after episode {episode+1}: Avg reward = {eval_reward:.2f}")
                
            # Save model
            if episode % self.config["save_frequency"] == 0:
                save_path = os.path.join(
                    self.config["output_dir"], 
                    f"agent_ep{episode+1}_{self.timestamp}"
                )
                self.agent.save(save_path)
                print(f"Saved model to {save_path}")
                
        print(f"Training complete. Total time: {total_training_time:.2f}s")
        
        # Final evaluation
        final_eval = self.evaluate(num_episodes=10)
        print(f"Final evaluation: Avg reward = {final_eval:.2f}")
        
        # Save final model
        final_save_path = os.path.join(
            self.config["output_dir"], 
            f"agent_final_{self.timestamp}"
        )
        self.agent.save(final_save_path)
        
        # Save final metrics
        self._save_log()
        
        return self.metrics
        
    def evaluate(self, num_episodes=None):
        """
        Evaluate the agent's performance
        
        Args:
            num_episodes: Number of episodes to evaluate over
                          (defaults to config value)
                          
        Returns:
            Average reward over evaluation episodes
        """
        if num_episodes is None:
            num_episodes = self.config["eval_episodes"]
            
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        eval_rewards = []
        
        for episode in range(num_episodes):
            observation = self.env.reset()
            self.agent.reset()
            
            episode_reward = 0
            done = False
            
            while not done:
                # Optional rendering
                if self.config["render_during_eval"]:
                    self.env.render()
                    
                # Get motion information
                motion = self.env.get_motion_frames() if hasattr(self.env, "get_motion_frames") else None
                
                # Select action deterministically
                action = self.agent.act(observation, motion, deterministic=True)
                
                # Take action
                observation, reward, done, info = self.env.step(action)
                
                # Update reward
                episode_reward += reward
                
                # Visualization
                self.agent.visualize(observation)
                
            eval_rewards.append(episode_reward)
            
        # Calculate statistics
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        print(f"Evaluation results: Avg reward = {avg_reward:.2f}, Std dev = {std_reward:.2f}")
        
        return avg_reward
        
    def _save_log(self):
        """Save training metrics to log file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f)
            
    def plot_metrics(self, save_fig=True):
        """
        Plot training metrics
        
        Args:
            save_fig: Whether to save the figure to file
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot episode rewards
        axes[0, 0].plot(self.metrics["episode_rewards"], label="Episode Reward")
        axes[0, 0].plot(self.metrics["avg_rewards"], label="Moving Average")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].set_title("Training Rewards")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot evaluation rewards
        eval_episodes = range(0, len(self.metrics["eval_rewards"]) * self.config["eval_frequency"], 
                              self.config["eval_frequency"])
        axes[0, 1].plot(eval_episodes, self.metrics["eval_rewards"], 'o-', label="Evaluation Reward")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].set_title("Evaluation Rewards")
        axes[0, 1].grid(True)
        
        # Plot episode lengths
        axes[1, 0].plot(self.metrics["episode_lengths"])
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Steps")
        axes[1, 0].set_title("Episode Lengths")
        axes[1, 0].grid(True)
        
        # Plot epsilon values
        axes[1, 1].plot(self.metrics["epsilon_values"])
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Epsilon")
        axes[1, 1].set_title("Exploration Rate")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = os.path.join(
                self.config["output_dir"], 
                f"training_metrics_{self.timestamp}.png"
            )
            plt.savefig(fig_path)
            print(f"Saved metrics plot to {fig_path}")
            
        plt.show()
        
    def run_scenario(self, scenario_config, num_episodes=1, record=False):
        """
        Run agent in a specific scenario without training
        
        Args:
            scenario_config: Configuration for the scenario
            num_episodes: Number of episodes to run
            record: Whether to record a video
            
        Returns:
            Dictionary of scenario results
        """
        # Configure environment for scenario
        # TODO: Implement scenario-specific environment setup
        
        results = {
            "rewards": [],
            "steps": [],
            "success_rate": 0,
            "scenario": scenario_config.get("description", "Custom scenario")
        }
        
        successes = 0
        
        for episode in range(num_episodes):
            observation = self.env.reset()
            self.agent.reset()
            
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Always render in scenario mode
                self.env.render()
                
                # Get motion information
                motion = self.env.get_motion_frames() if hasattr(self.env, "get_motion_frames") else None
                
                # Select action deterministically
                action = self.agent.act(observation, motion, deterministic=True)
                
                # Take action
                observation, reward, done, info = self.env.step(action)
                
                # Update metrics
                episode_reward += reward
                steps += 1
                
                # Break if max steps reached
                if steps >= self.config["max_steps_per_episode"]:
                    break
                    
            # Record results
            results["rewards"].append(episode_reward)
            results["steps"].append(steps)
            
            # Check for success (depends on scenario definition)
            success = episode_reward > 0  # Simple criterion, can be customized
            if success:
                successes += 1
                
        # Calculate success rate
        results["success_rate"] = successes / num_episodes
        
        return results 