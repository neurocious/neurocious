#!/usr/bin/env python3
"""
Simple runner script for the Ultimate Neurocious Financial Experiment

Usage:
    python3 run_experiment.py                    # Default 5 runs
    python3 run_experiment.py --runs 10         # Custom number of runs
    python3 run_experiment.py --single          # Single run only
"""

import argparse
import asyncio
from ultimate_realistic_experiment import run_multi_run_experiment, run_ultimate_realistic_experiment, UltimateExperimentConfig

def main():
    parser = argparse.ArgumentParser(description='Run Ultimate Neurocious Financial Experiment')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs for statistical averaging')
    parser.add_argument('--single', action='store_true', help='Run single experiment instead of multi-run')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--training-days', type=int, default=100, help='Number of training days')
    parser.add_argument('--test-days', type=int, default=50, help='Number of test days')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed base')
    
    args = parser.parse_args()
    
    config = UltimateExperimentConfig(
        training_days=args.training_days,
        test_days=args.test_days,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_branches=3,
        scenario_horizon=5,
        num_runs=args.runs,
        random_seed_base=args.seed
    )
    
    if args.single:
        print("🚀 Running Single Ultimate Experiment...")
        results = asyncio.run(run_ultimate_realistic_experiment(config))
        print("\n✅ Single experiment completed!")
        print("📁 Results saved to 'ultimate_realistic_experiment_results.json'")
    else:
        print(f"🚀 Running Multi-Run Ultimate Experiment ({args.runs} iterations)...")
        results = asyncio.run(run_multi_run_experiment(config))
        print(f"\n✅ Multi-run experiment completed!")
        print(f"🏆 Winner: {results['overall_winner']}")
        print(f"📊 Successful runs: {results['successful_runs']}/{results['total_runs']}")
        print("📁 Results saved to 'multi_run_ultimate_results.json'")

if __name__ == "__main__":
    main()