#!/usr/bin/env python3
"""
Neurocious Experiment Runner
============================

Simple wrapper to run the Neurocious belief navigation experiment.

Usage:
    python3 run_experiment.py quick    # Quick demo (5 minutes)
    python3 run_experiment.py full     # Full experiment (training + evaluation)
"""

import sys
import asyncio
from belief_nav_task import main, quick_demo
from realistic_market_simulation import RealisticMarketConfig, create_realistic_training_data
from realistic_evaluation import FinancialModelEvaluator, create_realistic_benchmark
from realistic_training_objectives import realistic_performance_targets

def print_help():
    print("🎯 Neurocious Experiment Runner")
    print("=" * 40)
    print()
    print("Available modes:")
    print("  quick      - Quick demonstration (5 minutes)")
    print("  full       - Full experiment with training + baseline comparison")
    print("  realistic  - Realistic financial experiment with proper metrics")
    print("  baselines  - Show available baseline models and metrics")
    print("  objectives - Show training objectives and what the system learns")
    print("  benchmarks - Show realistic financial performance benchmarks")
    print()
    print("Usage:")
    print("  python3 run_experiment.py quick")
    print("  python3 run_experiment.py full")
    print("  python3 run_experiment.py realistic")
    print("  python3 run_experiment.py baselines")
    print("  python3 run_experiment.py objectives")
    print("  python3 run_experiment.py benchmarks")
    print()

async def run_quick():
    """Run quick demonstration"""
    print("🏃‍♂️ Running Neurocious Quick Demo...")
    await quick_demo()

async def run_full():
    """Run full experiment"""
    print("🔬 Running Full Neurocious Experiment...")
    print("⚠️  This may take 30+ minutes depending on your hardware")
    print()
    
    try:
        results = await main()
        print("\n🎉 Experiment completed successfully!")
        print("📊 Check the generated files for results and visualizations")
        return results
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        print("💡 Try the quick demo first: python3 run_experiment.py quick")
        raise

async def run_realistic():
    """Run realistic financial experiment with proper metrics"""
    print("💰 Running Realistic Financial Experiment...")
    print("🎯 Using proper financial metrics and realistic data")
    print("⚠️  This may take 20+ minutes depending on your hardware")
    print()
    
    try:
        # Import realistic components
        from realistic_financial_experiment import run_realistic_experiment
        
        print("📊 Generating realistic market data...")
        print("🧠 Training with financial objectives...")
        print("📈 Evaluating with proper financial metrics...")
        
        results = await run_realistic_experiment()
        
        print("\n🎉 Realistic experiment completed!")
        print("📊 Results saved to 'realistic_experiment_results.json'")
        print("📈 Performance charts saved to 'realistic_performance.png'")
        
        # Show key results
        print("\n🔑 Key Results:")
        if 'performance' in results:
            perf = results['performance']
            print(f"  Annual Return: {perf.get('annualized_return', 0):.1%}")
            print(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"  Hit Rate: {perf.get('hit_rate', 0):.1%}")
            print(f"  Max Drawdown: {perf.get('maximum_drawdown', 0):.1%}")
        
        return results
        
    except ImportError:
        print("❌ Realistic experiment components not found")
        print("💡 Creating realistic experiment runner...")
        await create_realistic_experiment_runner()
        print("✅ Created! Now run: python3 run_experiment.py realistic")
    except KeyboardInterrupt:
        print("\n⚠️ Realistic experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during realistic experiment: {e}")
        print("💡 Try the quick demo first: python3 run_experiment.py quick")
        raise

def show_benchmarks():
    """Show realistic financial performance benchmarks"""
    print("💰 Realistic Financial Performance Benchmarks")
    print("=" * 50)
    print()
    
    benchmarks = create_realistic_benchmark()
    targets = realistic_performance_targets()
    
    print("📊 Industry Benchmarks:")
    for name, metrics in benchmarks.items():
        print(f"  {name}:")
        print(f"    Annual Return: {metrics['annual_return']:.1%}")
        print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"    Hit Rate: {metrics['hit_rate']:.1%}")
        print(f"    Max Drawdown: {metrics['max_drawdown']:.1%}")
        print()
    
    print("🎯 Performance Targets for AI Models:")
    for level, metrics in targets.items():
        print(f"  {level.replace('_', ' ').title()}:")
        print(f"    Return: {metrics['annual_return']:.1%}")
        print(f"    Sharpe: {metrics['sharpe_ratio']:.1f}")
        print(f"    Hit Rate: {metrics['hit_rate']:.1%}")
        print(f"    Drawdown: {metrics['max_drawdown']:.1%}")
        print()
    
    print("💡 Key Insights:")
    print("  • 98% prediction accuracy is impossible")
    print("  • 55-60% hit rate is excellent in finance")
    print("  • Sharpe ratio > 1.0 indicates good risk management")
    print("  • Risk-adjusted returns matter more than raw accuracy")
    print()

async def create_realistic_experiment_runner():
    """Create the realistic experiment runner if it doesn't exist"""
    from realistic_financial_experiment_template import create_experiment_file
    create_experiment_file()

def show_baselines():
    """Show available baseline models and metrics"""
    print("🔬 Neurocious Baseline Comparison Framework")
    print("=" * 50)
    print()
    print("📋 Available Baseline Models:")
    print("  • β-VAE           - Standard variational autoencoder")
    print("  • World Model     - Ha & Schmidhuber 2018 approach")
    print("  • Transformer     - Attention-based sequence model")  
    print("  • Neural ODE-VAE  - Continuous dynamics approach")
    print()
    print("📊 Evaluation Metrics:")
    print("  • Reconstruction Error    - How well models reconstruct inputs")
    print("  • Prediction Accuracy     - Future state prediction quality")
    print("  • Uncertainty Calibration - How well confidence estimates match reality")
    print("  • Inference Time         - Speed of processing")
    print("  • Interpretability Score - Quality of explanations provided")
    print()
    print("🎯 Neurocious Advantages:")
    print("  • Spatial belief navigation through learned probability fields")
    print("  • Causal explanation of belief formation and decision making")
    print("  • Multi-timescale processing (reflexes + planning)")
    print("  • World branching for counterfactual reasoning")
    print("  • Structured uncertainty quantification")
    print()

def show_objectives():
    """Show training objectives and capabilities"""
    from training_objectives import TrainingObjectives, what_is_system_learning, comparison_to_existing_methods
    
    objectives = TrainingObjectives()
    
    print("📚 Neurocious Training Objectives & Capabilities")
    print("=" * 55)
    print()
    print("🎯 What the System Learns:")
    for name, cap in what_is_system_learning().items():
        print(f"  • {name.replace('_', ' ').title()}")
        print(f"    {cap['description']}")
        print(f"    Example: {cap['example']}")
        print()
    
    print("🏆 Advantages vs Existing Methods:")
    for name, comp in comparison_to_existing_methods().items():
        method_name = name.replace('vs_', '').replace('_', ' ').title()
        print(f"  • vs {method_name}")
        print(f"    {comp['advantage']}")
        print(f"    Baselines: {comp['baseline']}")
        print()

def main_cli():
    if len(sys.argv) < 2:
        print_help()
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "quick":
        asyncio.run(run_quick())
    elif mode == "full":
        asyncio.run(run_full())
    elif mode == "realistic":
        asyncio.run(run_realistic())
    elif mode == "baselines":
        show_baselines()
    elif mode == "objectives":
        show_objectives()
    elif mode == "benchmarks":
        show_benchmarks()
    elif mode in ["help", "--help", "-h"]:
        print_help()
    else:
        print(f"❌ Unknown mode: {mode}")
        print_help()

if __name__ == "__main__":
    main_cli()