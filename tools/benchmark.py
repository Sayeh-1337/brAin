#!/usr/bin/env python
"""
Benchmark script for comparing original and optimized brAin implementations.

This script measures the performance of both implementations on various tasks:
- Forward pass (inference) speed
- Memory operation speed
- Training speed
"""

import os
import sys
import time
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.agent.hdc_snn_agent import HDCSNNAgent
from brain.agent.optimized_agent import OptimizedAgent
from brain.utils.config import set_config, config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Benchmark brAin implementations')
    parser.add_argument('--runs', type=int, default=100, help='Number of benchmark runs')
    parser.add_argument('--dimensions', type=int, nargs='+', default=[1000, 5000, 10000], 
                        help='HD vector dimensions to test')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 10, 50],
                        help='Batch sizes to test')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument('--only-optimized', action='store_true', help='Only test optimized version')
    parser.add_argument('--plot', action='store_true', help='Generate plots of results')
    parser.add_argument('--output', type=str, default='benchmark_results', help='Output directory')
    return parser.parse_args()


def generate_random_observations(batch_size, shape=(120, 160, 3)):
    """Generate random observations for testing"""
    return np.random.randint(0, 256, (batch_size, *shape), dtype=np.uint8)


def benchmark_forward_pass(agent, observations, runs=100):
    """Benchmark forward pass speed"""
    forward_times = []
    
    # Warmup
    for _ in range(5):
        _ = agent.act(observations[0])
    
    # Benchmark
    for i in range(runs):
        obs_idx = i % len(observations)
        start_time = time.time()
        _ = agent.act(observations[obs_idx])
        elapsed = time.time() - start_time
        forward_times.append(elapsed)
    
    return forward_times


def benchmark_learning(agent, observations, runs=100):
    """Benchmark learning speed"""
    learning_times = []
    
    # Create random states, actions, rewards
    states = observations
    next_states = observations
    actions = np.random.randint(0, 5, runs)
    rewards = np.random.random(runs) * 2 - 1  # Random rewards between -1 and 1
    dones = np.zeros(runs, dtype=bool)
    
    # Warmup
    for _ in range(5):
        agent.learn(states[0], actions[0], rewards[0], next_states[0], dones[0])
    
    # Benchmark
    for i in range(runs):
        obs_idx = i % len(observations)
        start_time = time.time()
        agent.learn(states[obs_idx], actions[i % len(actions)], rewards[i % len(rewards)], 
                    next_states[obs_idx], dones[i % len(dones)])
        elapsed = time.time() - start_time
        learning_times.append(elapsed)
    
    return learning_times


def benchmark_memory_operations(agent, observations, runs=100):
    """Benchmark memory operations speed"""
    memory_times = []
    
    # Encode first few observations to have vectors to work with
    encoded_observations = []
    for i in range(min(20, len(observations))):
        if hasattr(agent, 'hdc_encoder'):
            encoded = agent.hdc_encoder.encode_observation(observations[i])
            encoded_observations.append(encoded)
    
    # Store observations in memory (warmup)
    for i in range(min(5, len(encoded_observations))):
        if hasattr(agent, 'episodic_memory'):
            agent.episodic_memory.store(
                encoded_observations[i], i % 5, 0.5, encoded_observations[(i+1) % len(encoded_observations)], False)
    
    # Benchmark memory retrieval
    for i in range(runs):
        if hasattr(agent, 'episodic_memory') and encoded_observations:
            query_idx = i % len(encoded_observations)
            start_time = time.time()
            _ = agent.episodic_memory.retrieve_similar(encoded_observations[query_idx], max_results=3)
            elapsed = time.time() - start_time
            memory_times.append(elapsed)
    
    return memory_times


def benchmark_replay(agent, observations, batch_size=32, runs=20):
    """Benchmark experience replay speed"""
    replay_times = []
    
    # Add some items to memory first
    encoded_observations = []
    for i in range(min(100, len(observations))):
        if hasattr(agent, 'hdc_encoder'):
            encoded = agent.hdc_encoder.encode_observation(observations[i])
            encoded_observations.append(encoded)
            
            if hasattr(agent, 'episodic_memory') and i > 0:
                agent.episodic_memory.store(
                    encoded_observations[i-1], i % 5, 0.5, encoded, i >= 95)
    
    # Warmup
    if hasattr(agent, 'replay_experience'):
        for _ in range(3):
            agent.replay_experience(batch_size=10)
    
    # Benchmark
    if hasattr(agent, 'replay_experience'):
        for _ in range(runs):
            start_time = time.time()
            agent.replay_experience(batch_size=batch_size)
            elapsed = time.time() - start_time
            replay_times.append(elapsed)
    
    return replay_times


def create_agent(optimized, hd_dim, device):
    """Create either original or optimized agent"""
    if optimized:
        agent = OptimizedAgent(
            input_shape=(120, 160, 3),
            hd_dim=hd_dim,
            snn_neurons=200,
            num_actions=5,
            ca_width=20,
            ca_height=20,
            memory_capacity=5000,
            device=device
        )
    else:
        agent = HDCSNNAgent(
            input_shape=(120, 160, 3),
            hd_dim=hd_dim,
            snn_neurons=200,
            num_actions=5,
            ca_width=20,
            ca_height=20,
            memory_capacity=5000
        )
    return agent


def run_benchmarks(args):
    """Run all benchmarks based on arguments"""
    # Determine device
    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize results dictionary
    results = {
        "dimensions": args.dimensions,
        "batch_sizes": args.batch_sizes,
        "device": device,
        "forward_times": {
            "original": {}, 
            "optimized": {}
        },
        "learning_times": {
            "original": {}, 
            "optimized": {}
        },
        "memory_times": {
            "original": {}, 
            "optimized": {}
        },
        "replay_times": {
            "original": {}, 
            "optimized": {}
        }
    }
    
    # Run benchmarks for each dimension
    for hd_dim in args.dimensions:
        print(f"\nBenchmarking with dimension: {hd_dim}")
        
        # Original agent, if requested
        if not args.only_optimized:
            print("Testing original implementation...")
            orig_agent = create_agent(False, hd_dim, device)
            observations = generate_random_observations(max(args.batch_sizes))
            
            print("  Forward pass benchmark...")
            results["forward_times"]["original"][hd_dim] = benchmark_forward_pass(
                orig_agent, observations, runs=args.runs)
            
            print("  Learning benchmark...")
            results["learning_times"]["original"][hd_dim] = benchmark_learning(
                orig_agent, observations, runs=args.runs)
            
            print("  Memory operations benchmark...")
            results["memory_times"]["original"][hd_dim] = benchmark_memory_operations(
                orig_agent, observations, runs=args.runs)
            
            # Run replay benchmarks for different batch sizes
            results["replay_times"]["original"][hd_dim] = {}
            for batch_size in args.batch_sizes:
                print(f"  Replay benchmark (batch_size={batch_size})...")
                results["replay_times"]["original"][hd_dim][batch_size] = benchmark_replay(
                    orig_agent, observations, batch_size=batch_size, runs=20)
        
        # Optimized agent
        print("Testing optimized implementation...")
        # Configure optimizations
        set_config(hd_dimension=hd_dim, device=device, use_jit_compile=True, batch_process=True)
        opt_agent = create_agent(True, hd_dim, device)
        observations = generate_random_observations(max(args.batch_sizes))
        
        print("  Forward pass benchmark...")
        results["forward_times"]["optimized"][hd_dim] = benchmark_forward_pass(
            opt_agent, observations, runs=args.runs)
        
        print("  Learning benchmark...")
        results["learning_times"]["optimized"][hd_dim] = benchmark_learning(
            opt_agent, observations, runs=args.runs)
        
        print("  Memory operations benchmark...")
        results["memory_times"]["optimized"][hd_dim] = benchmark_memory_operations(
            opt_agent, observations, runs=args.runs)
        
        # Run replay benchmarks for different batch sizes
        results["replay_times"]["optimized"][hd_dim] = {}
        for batch_size in args.batch_sizes:
            print(f"  Replay benchmark (batch_size={batch_size})...")
            results["replay_times"]["optimized"][hd_dim][batch_size] = benchmark_replay(
                opt_agent, observations, batch_size=batch_size, runs=20)
    
    return results


def print_summary(results):
    """Print a summary of the benchmark results"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    for hd_dim in results["dimensions"]:
        print(f"\nDimension: {hd_dim}")
        
        # Forward pass comparison
        if "original" in results["forward_times"] and hd_dim in results["forward_times"]["original"]:
            orig_forward = np.mean(results["forward_times"]["original"][hd_dim]) * 1000
            opt_forward = np.mean(results["forward_times"]["optimized"][hd_dim]) * 1000
            speedup = orig_forward / opt_forward if opt_forward > 0 else float('inf')
            
            print(f"  Forward Pass: Original: {orig_forward:.2f}ms, "
                  f"Optimized: {opt_forward:.2f}ms (Speedup: {speedup:.2f}x)")
        else:
            opt_forward = np.mean(results["forward_times"]["optimized"][hd_dim]) * 1000
            print(f"  Forward Pass: Optimized: {opt_forward:.2f}ms")
        
        # Learning comparison
        if "original" in results["learning_times"] and hd_dim in results["learning_times"]["original"]:
            orig_learning = np.mean(results["learning_times"]["original"][hd_dim]) * 1000
            opt_learning = np.mean(results["learning_times"]["optimized"][hd_dim]) * 1000
            speedup = orig_learning / opt_learning if opt_learning > 0 else float('inf')
            
            print(f"  Learning: Original: {orig_learning:.2f}ms, "
                  f"Optimized: {opt_learning:.2f}ms (Speedup: {speedup:.2f}x)")
        else:
            opt_learning = np.mean(results["learning_times"]["optimized"][hd_dim]) * 1000
            print(f"  Learning: Optimized: {opt_learning:.2f}ms")
        
        # Memory operations comparison
        if "original" in results["memory_times"] and hd_dim in results["memory_times"]["original"]:
            orig_memory = np.mean(results["memory_times"]["original"][hd_dim]) * 1000
            opt_memory = np.mean(results["memory_times"]["optimized"][hd_dim]) * 1000
            speedup = orig_memory / opt_memory if opt_memory > 0 else float('inf')
            
            print(f"  Memory Ops: Original: {orig_memory:.2f}ms, "
                  f"Optimized: {opt_memory:.2f}ms (Speedup: {speedup:.2f}x)")
        else:
            opt_memory = np.mean(results["memory_times"]["optimized"][hd_dim]) * 1000
            print(f"  Memory Ops: Optimized: {opt_memory:.2f}ms")
        
        # Replay comparison for largest batch size
        max_batch = max(results["batch_sizes"])
        if ("original" in results["replay_times"] and 
            hd_dim in results["replay_times"]["original"] and
            max_batch in results["replay_times"]["original"][hd_dim]):
            
            orig_replay = np.mean(results["replay_times"]["original"][hd_dim][max_batch]) * 1000
            opt_replay = np.mean(results["replay_times"]["optimized"][hd_dim][max_batch]) * 1000
            speedup = orig_replay / opt_replay if opt_replay > 0 else float('inf')
            
            print(f"  Replay (batch={max_batch}): Original: {orig_replay:.2f}ms, "
                  f"Optimized: {opt_replay:.2f}ms (Speedup: {speedup:.2f}x)")
        else:
            opt_replay = np.mean(results["replay_times"]["optimized"][hd_dim][max_batch]) * 1000
            print(f"  Replay (batch={max_batch}): Optimized: {opt_replay:.2f}ms")


def plot_results(results, output_dir):
    """Generate plots from benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot forward pass times by dimension
    plt.figure(figsize=(10, 6))
    
    x = results["dimensions"]
    if "original" in results["forward_times"] and results["forward_times"]["original"]:
        y_orig = [np.mean(results["forward_times"]["original"][dim]) * 1000 for dim in x]
        plt.plot(x, y_orig, 'o-', label='Original')
    
    y_opt = [np.mean(results["forward_times"]["optimized"][dim]) * 1000 for dim in x]
    plt.plot(x, y_opt, 'o-', label='Optimized')
    
    plt.xlabel('Dimension')
    plt.ylabel('Time (ms)')
    plt.title('Forward Pass Time by Dimension')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'forward_by_dimension.png'))
    
    # Plot learning times by dimension
    plt.figure(figsize=(10, 6))
    
    if "original" in results["learning_times"] and results["learning_times"]["original"]:
        y_orig = [np.mean(results["learning_times"]["original"][dim]) * 1000 for dim in x]
        plt.plot(x, y_orig, 'o-', label='Original')
    
    y_opt = [np.mean(results["learning_times"]["optimized"][dim]) * 1000 for dim in x]
    plt.plot(x, y_opt, 'o-', label='Optimized')
    
    plt.xlabel('Dimension')
    plt.ylabel('Time (ms)')
    plt.title('Learning Time by Dimension')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'learning_by_dimension.png'))
    
    # Plot replay times by batch size for a specific dimension
    plt.figure(figsize=(10, 6))
    
    # Use the largest dimension for this plot
    max_dim = max(results["dimensions"])
    x = results["batch_sizes"]
    
    if ("original" in results["replay_times"] and 
        results["replay_times"]["original"] and 
        max_dim in results["replay_times"]["original"]):
        
        y_orig = [np.mean(results["replay_times"]["original"][max_dim][batch]) * 1000 
                 for batch in x]
        plt.plot(x, y_orig, 'o-', label='Original')
    
    y_opt = [np.mean(results["replay_times"]["optimized"][max_dim][batch]) * 1000 
             for batch in x]
    plt.plot(x, y_opt, 'o-', label='Optimized')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Time (ms)')
    plt.title(f'Replay Time by Batch Size (Dimension = {max_dim})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'replay_by_batch.png'))
    
    # Plot speedup factor by dimension
    plt.figure(figsize=(10, 6))
    
    if ("original" in results["forward_times"] and 
        results["forward_times"]["original"] and
        "original" in results["learning_times"] and
        results["learning_times"]["original"] and
        "original" in results["memory_times"] and
        results["memory_times"]["original"]):
        
        speedups_forward = []
        speedups_learning = []
        speedups_memory = []
        
        for dim in x:
            orig_forward = np.mean(results["forward_times"]["original"][dim])
            opt_forward = np.mean(results["forward_times"]["optimized"][dim])
            speedup_forward = orig_forward / opt_forward if opt_forward > 0 else float('inf')
            speedups_forward.append(min(speedup_forward, 20))  # Cap for visualization
            
            orig_learning = np.mean(results["learning_times"]["original"][dim])
            opt_learning = np.mean(results["learning_times"]["optimized"][dim])
            speedup_learning = orig_learning / opt_learning if opt_learning > 0 else float('inf')
            speedups_learning.append(min(speedup_learning, 20))  # Cap for visualization
            
            orig_memory = np.mean(results["memory_times"]["original"][dim])
            opt_memory = np.mean(results["memory_times"]["optimized"][dim])
            speedup_memory = orig_memory / opt_memory if opt_memory > 0 else float('inf')
            speedups_memory.append(min(speedup_memory, 20))  # Cap for visualization
        
        plt.plot(x, speedups_forward, 'o-', label='Forward')
        plt.plot(x, speedups_learning, 'o-', label='Learning')
        plt.plot(x, speedups_memory, 'o-', label='Memory')
        
        plt.xlabel('Dimension')
        plt.ylabel('Speedup Factor (x)')
        plt.title('Optimization Speedup by Operation and Dimension')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'speedup_by_dimension.png'))


def main():
    """Main function"""
    args = parse_args()
    
    print("Running brAin benchmarks")
    print(f"Dimensions: {args.dimensions}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Number of runs: {args.runs}")
    
    # Run benchmarks
    results = run_benchmarks(args)
    
    # Print summary
    print_summary(results)
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        plot_results(results, args.output)
        print(f"Plots saved to {args.output}")


if __name__ == "__main__":
    main() 