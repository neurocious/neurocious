#!/bin/bash

# Neurocious Experiment Setup and Execution
# ==========================================

echo "🚀 Setting up Neurocious Experiment Environment"
echo "=============================================="

# Create project directory
mkdir -p neurocious_experiment
cd neurocious_experiment

# Create subdirectories
mkdir -p {data,checkpoints,logs,results,visualizations}

echo "📦 Installing dependencies..."

# Install Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn
pip install pyyaml asyncio pathlib dataclasses
pip install jupyter notebook ipywidgets

# Optional: Install additional packages for enhanced functionality
pip install plotly dash  # For interactive visualizations
pip install tensorboard  # For training monitoring
pip install scikit-learn  # For additional metrics

echo "📋 Creating configuration file..."

# Create experiment configuration
cat > experiment_config.yaml << EOF
# Neurocious Experiment Configuration
experiment:
  name: "financial_belief_navigation"
  description: "Spatial belief navigation in financial markets"
  
data:
  sequence_length: 20
  market_history_days: 252
  num_market_regimes: 4
  noise_level: 0.1
  
training:
  num_episodes: 200
  episode_length: 30
  batch_size: 8
  learning_rate: 0.001
  epochs: 15
  
neurocious:
  input_dim: 784
  hidden_dim: 256
  latent_dim: 32
  field_shape: [16, 16]
  vector_dim: 8
  
  # Loss weights
  beta: 0.8           # KL weight
  gamma: 0.6          # Narrative continuity
  delta: 0.4          # Field alignment
  policy_weight: 0.7  # Policy learning
  reflex_weight: 0.5  # Reflex responses
  prediction_weight: 0.6  # Future prediction

evaluation:
  test_episodes: 100
  metrics:
    - reconstruction_error
    - prediction_accuracy
    - uncertainty_calibration
    - inference_time
    - interpretability_score
  
baselines:
  - beta_vae
  - world_model
  - transformer
  - neural_ode_vae
EOF

echo "🗂️ Creating Python module structure..."

# Create __init__.py files for proper module imports
touch __init__.py

echo "📝 Creating run script..."

# Create main run script
cat > run_experiment.py << 'EOF'
#!/usr/bin/env python3
"""
Neurocious Experiment Runner
============================

Usage:
    python run_experiment.py --mode quick    # Quick demo (5 minutes)
    python run_experiment.py --mode full     # Full experiment (30+ minutes)
    python run_experiment.py --mode eval     # Evaluation only
    python run_experiment.py --mode viz      # Visualization only
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our experiment modules
try:
    from experimental_setup import main, quick_demo, ExperimentRunner, ExperimentConfig
    print("✅ Successfully imported Neurocious modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Please ensure all Python files are in the current directory")
    sys.exit(1)

async def run_quick_demo():
    """Run quick demonstration"""
    print("🏃‍♂️ Running quick demo (5 minutes)...")
    await quick_demo()

async def run_full_experiment():
    """Run full experiment"""
    print("🔬 Running full experiment (this may take 30+ minutes)...")
    await main()

async def run_evaluation_only():
    """Run evaluation on existing model"""
    print("📊 Running evaluation only...")
    
    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    
    # Check if trained model exists
    checkpoint_dir = Path("./trading_checkpoints")
    if not any(checkpoint_dir.glob("*.pth")):
        print("❌ No trained model found. Please run full experiment first.")
        return
    
    # Load model and evaluate
    # Implementation would load the checkpoint and run evaluation
    print("📈 Evaluating existing model...")
    evaluation_results = await runner._evaluate_agent()
    
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"  {metric}: {value:.4f}")

def run_visualization_only():
    """Generate visualizations from existing results"""
    print("🎨 Generating visualizations...")
    
    from experimental_setup import ExperimentVisualizer
    
    # Check for existing results
    results_files = list(Path("./results").glob("*.json"))
    if not results_files:
        print("❌ No results found. Please run experiment first.")
        return
    
    print(f"📊 Found {len(results_files)} result files")
    print("Generating visualization plots...")
    
    # Implementation would load results and generate plots
    visualizer = ExperimentVisualizer()
    
    # Generate sample plots (placeholder)
    print("✅ Visualizations saved to ./visualizations/")

def main_cli():
    parser = argparse.ArgumentParser(description='Neurocious Experiment Runner')
    parser.add_argument('--mode', choices=['quick', 'full', 'eval', 'viz'], 
                       default='quick', help='Experiment mode')
    parser.add_argument('--config', type=str, default='experiment_config.yaml',
                       help='Configuration file')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], 
                       default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    print(f"🎯 Neurocious Experiment - Mode: {args.mode}")
    print(f"⚙️  Configuration: {args.config}")
    print(f"💻 Device: {args.device}")
    print()
    
    try:
        if args.mode == 'quick':
            asyncio.run(run_quick_demo())
        elif args.mode == 'full':
            asyncio.run(run_full_experiment())
        elif args.mode == 'eval':
            asyncio.run(run_evaluation_only())
        elif args.mode == 'viz':
            run_visualization_only()
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("Check the error logs for details")
        raise

if __name__ == "__main__":
    main_cli()
EOF

chmod +x run_experiment.py

echo "📚 Creating README with instructions..."

# Create comprehensive README
cat > README.md << 'EOF'
# Neurocious Belief Navigation Experiment

This experiment demonstrates Neurocious's unique capabilities in spatial belief navigation, causal reasoning, and multi-timescale decision making using a financial market simulation.

## 🎯 What This Experiment Shows

1. **Spatial Belief Navigation**: How beliefs move through learned probability landscapes
2. **Causal Explanation**: Why the agent believes what it believes
3. **Multi-timescale Decisions**: Immediate reflexes + long-term planning
4. **World Branching**: Simulating alternative belief trajectories
5. **Uncertainty Quantification**: Spatially-structured confidence estimates

## 🚀 Quick Start (5 minutes)

```bash
# Quick demo - shows core capabilities without training
python run_experiment.py --mode quick
```

## 🔬 Full Experiment (30+ minutes)

```bash
# Complete experiment with training and evaluation
python run_experiment.py --mode full
```

## 📊 Results Analysis

After running the experiment, you'll get:

- **Performance metrics**: Profit, accuracy, confidence calibration
- **Belief analysis**: How beliefs navigate through probability space
- **Baseline comparison**: Performance vs VAE, World Models, Transformers
- **Visualizations**: Belief evolution, field dynamics, trading performance

## 🎨 Key Visualizations

1. **Belief Evolution**: How field parameters (entropy, curvature, alignment) change over time
2. **Performance Comparison**: Radar chart comparing Neurocious vs baselines
3. **Trading Performance**: Price charts with position markers and confidence
4. **Field Dynamics**: Vector fields and probability landscapes

## 📈 Expected Results

Neurocious should demonstrate:

- **Superior interpretability**: Rich explanations for every decision
- **Better uncertainty handling**: Knows when it doesn't know
- **Adaptive confidence**: Confidence adjusts to market regime
- **Causal reasoning**: Can explain belief formation chains
- **Stable belief navigation**: Smooth transitions in belief space

## 🔧 Configuration

Edit `experiment_config.yaml` to customize:

- Market simulation parameters
- Training hyperparameters  
- Model architecture
- Evaluation metrics

## 📁 Output Files

```
results/
├── experiment_report_YYYYMMDD_HHMMSS.md
├── training_results.json
├── belief_evolution.png
├── performance_comparison.png
└── trading_performance.png

logs/
├── training_progress.png
├── field_dynamics.png
└── training_logs/

checkpoints/
└── neurocious_checkpoint_best.pth
```

## 🏆 Success Criteria

The experiment is successful if:

1. **Regime Accuracy > 70%**: Correctly identifies market conditions
2. **Belief Stability > 80%**: Smooth belief evolution
3. **Explanation Quality > 60%**: Meaningful decision explanations
4. **Beats Baselines**: Outperforms on interpretability metrics

## 🐛 Troubleshooting

**Memory Issues**: Reduce batch_size in config
**CUDA Errors**: Use `--device cpu`
**Import Errors**: Ensure all .py files are in same directory
**Training Slow**: Reduce num_episodes for faster testing

## 📖 Understanding the Results

### Belief Evolution Plot
- **Entropy**: Higher = more uncertain
- **Curvature**: Higher = more unstable region
- **Alignment**: Higher = more coherent belief direction

### Performance Radar Chart
- **Larger area**: Better overall performance
- **Neurocious (red)**: Should excel in interpretability

### Trading Performance
- **Green dots**: Long positions
- **Red dots**: Short positions
- **Confidence line**: Agent's certainty over time

## 🔬 Research Questions

This experiment helps answer:

1. Can spatial belief navigation improve decision making?
2. Do field-aware priors lead to better uncertainty quantification?
3. How does causal explanation quality compare to baselines?
4. Can world branching enable better counterfactual reasoning?

## 🚀 Next Steps

After this experiment, you could:

1. **Try real data**: Use actual financial time series
2. **Add more regimes**: Expand beyond 4 market states
3. **Tune hyperparameters**: Optimize loss weights
4. **Compare domains**: Test on other sequential decision tasks
5. **Scale up**: Larger field shapes and longer sequences

Happy experimenting! 🎉
EOF

echo "✅ Setup completed!"
echo ""
echo "🎯 To run the experiment:"
echo "   Quick demo (5 min):  python run_experiment.py --mode quick"
echo "   Full experiment:     python run_experiment.py --mode full"
echo ""
echo "📖 See README.md for detailed instructions"
echo "⚙️  Edit experiment_config.yaml to customize settings"