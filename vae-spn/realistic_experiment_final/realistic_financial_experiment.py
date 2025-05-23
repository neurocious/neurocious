"""
Realistic Financial Experiment Implementation
===========================================

Simplified implementation that integrates realistic components.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
import asyncio
from dataclasses import dataclass

# Import realistic components
from realistic_market_simulation import create_realistic_training_data, RealisticMarketConfig
from realistic_evaluation import FinancialModelEvaluator, benchmark_comparison
from realistic_training_objectives import realistic_performance_targets
from baseline import BaselineModels

@dataclass
class SimpleRealisticConfig:
    """Simplified configuration for realistic experiment"""
    training_days: int = 100
    test_days: int = 50
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 0.001

class SimpleFinancialModel(nn.Module):
    """Simplified financial model for testing realistic objectives"""
    
    def __init__(self, input_dim: int = 8):
        super().__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.position_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Tanh()  # Position between -1 and +1
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()  # Confidence between 0 and 1
        )
        
        self.return_pred_head = nn.Sequential(
            nn.Linear(16, 1)  # Predicted return
        )
    
    def forward(self, x):
        features = self.feature_net(x)
        
        position = self.position_head(features).squeeze(-1)
        confidence = self.confidence_head(features).squeeze(-1)
        predicted_return = self.return_pred_head(features).squeeze(-1)
        
        return {
            'positions': position,
            'confidence': confidence,
            'predicted_returns': predicted_return
        }

class BetaVAEFinancialModel(nn.Module):
    """β-VAE adapted for financial tasks"""
    
    def __init__(self, input_dim: int = 8, latent_dim: int = 32, beta: float = 4.0):
        super().__init__()
        
        # Expand input to 784 dimensions for β-VAE compatibility
        self.input_expander = nn.Linear(input_dim, 784)
        
        # β-VAE core
        self.beta_vae = BaselineModels.beta_vae(input_dim=784, latent_dim=latent_dim, beta=beta)
        
        # Financial heads
        self.position_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(latent_dim, 16), 
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.return_pred_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(), 
            nn.Linear(16, 1)
        )
        
        self.beta = beta
    
    def forward(self, x):
        # Expand input to 784 dimensions
        expanded_x = torch.sigmoid(self.input_expander(x))
        
        # Get β-VAE encoding
        mean, logvar = self.beta_vae.encode(expanded_x)
        
        # Use mean as latent representation (no sampling for deterministic predictions)
        latent = mean
        
        # Financial predictions
        position = self.position_head(latent).squeeze(-1)
        confidence = self.confidence_head(latent).squeeze(-1)
        predicted_return = self.return_pred_head(latent).squeeze(-1)
        
        # Also compute VAE loss for training
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon = self.beta_vae.decode(z)
        
        return {
            'positions': position,
            'confidence': confidence,
            'predicted_returns': predicted_return,
            'reconstructed': recon,
            'original': expanded_x,
            'mean': mean,
            'logvar': logvar
        }

def simple_realistic_loss(predictions, targets):
    """Simplified realistic loss function"""
    
    positions = predictions['positions']
    confidence = predictions['confidence']
    pred_returns = predictions['predicted_returns']
    
    actual_returns = targets['returns']
    
    # 1. Strategy returns (main objective)
    strategy_returns = positions * actual_returns
    sharpe_loss = -torch.mean(strategy_returns) / (torch.std(strategy_returns) + 1e-8)
    
    # 2. Prediction accuracy
    prediction_loss = torch.mean((pred_returns - actual_returns) ** 2)
    
    # 3. Confidence calibration  
    prediction_errors = torch.abs(pred_returns - actual_returns)
    calibration_loss = torch.mean(confidence * prediction_errors)
    
    # 4. Capital preservation (asymmetric loss)
    losses = torch.clamp(strategy_returns, max=0)
    preservation_loss = 2.0 * torch.mean(torch.abs(losses))
    
    # Weighted combination
    total_loss = (0.4 * sharpe_loss + 
                  0.2 * prediction_loss + 
                  0.2 * calibration_loss + 
                  0.2 * preservation_loss)
    
    return total_loss

def beta_vae_realistic_loss(predictions, targets, beta=4.0):
    """β-VAE specific loss with financial objectives"""
    
    # Standard financial losses
    financial_loss = simple_realistic_loss(predictions, targets)
    
    # β-VAE reconstruction and KL losses
    if 'reconstructed' in predictions and 'original' in predictions:
        recon_loss = nn.functional.binary_cross_entropy(
            predictions['reconstructed'], 
            predictions['original'], 
            reduction='mean'
        )
        
        mean = predictions['mean']
        logvar = predictions['logvar']
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Combine losses (reduce VAE loss weight for financial focus)
        total_loss = financial_loss + 0.1 * (recon_loss + beta * kl_loss)
    else:
        total_loss = financial_loss
    
    return total_loss

async def run_realistic_experiment(config: SimpleRealisticConfig = None) -> Dict:
    """Run simplified realistic financial experiment"""
    
    if config is None:
        config = SimpleRealisticConfig()
    
    print("🚀 Starting Realistic Financial Experiment")
    print("=" * 50)
    
    # Step 1: Generate realistic data
    print("📊 Generating realistic market data...")
    market_config = RealisticMarketConfig()
    
    training_data = create_realistic_training_data(
        num_days=config.training_days, 
        config=market_config
    )
    
    test_data = create_realistic_training_data(
        num_days=config.test_days,
        config=market_config
    )
    
    print(f"  Training: {len(training_data)} days")
    print(f"  Testing: {len(test_data)} days")
    print(f"  Training return: {training_data['return'].mean() * 252:.1%} annual")
    print(f"  Training volatility: {training_data['return'].std() * np.sqrt(252):.1%} annual")
    
    # Step 2: Prepare features
    print("🔧 Preparing features...")
    feature_cols = ['return', 'volatility', 'price_to_sma10', 'price_to_sma20', 
                   'rsi', 'unemployment', 'inflation', 'interest_rate']
    
    train_features = training_data[feature_cols].fillna(0).values
    test_features = test_data[feature_cols].fillna(0).values
    
    # Normalize
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0) + 1e-8
    
    train_features = (train_features - feature_mean) / feature_std
    test_features = (test_features - feature_mean) / feature_std
    
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    
    train_returns = torch.tensor(training_data['return'].values, dtype=torch.float32)
    test_returns = torch.tensor(test_data['return'].values, dtype=torch.float32)
    
    # Step 3: Train both models
    print("🧠 Training Neurocious with realistic financial objectives...")
    neurocious_model = SimpleFinancialModel(input_dim=len(feature_cols))
    neurocious_optimizer = torch.optim.Adam(neurocious_model.parameters(), lr=config.learning_rate)
    
    print("🧠 Training β-VAE with realistic financial objectives...")
    beta_vae_model = BetaVAEFinancialModel(input_dim=len(feature_cols))
    beta_vae_optimizer = torch.optim.Adam(beta_vae_model.parameters(), lr=config.learning_rate)
    
    neurocious_losses = []
    beta_vae_losses = []
    
    for epoch in range(config.epochs):
        neurocious_epoch_losses = []
        beta_vae_epoch_losses = []
        
        # Batch training for both models
        for i in range(0, len(train_features_tensor), config.batch_size):
            batch_end = min(i + config.batch_size, len(train_features_tensor))
            
            batch_features = train_features_tensor[i:batch_end]
            batch_returns = train_returns[i:batch_end]
            targets = {'returns': batch_returns}
            
            # Train Neurocious
            neurocious_predictions = neurocious_model(batch_features)
            neurocious_loss = simple_realistic_loss(neurocious_predictions, targets)
            
            neurocious_optimizer.zero_grad()
            neurocious_loss.backward()
            neurocious_optimizer.step()
            neurocious_epoch_losses.append(neurocious_loss.item())
            
            # Train β-VAE
            beta_vae_predictions = beta_vae_model(batch_features)
            beta_vae_loss = beta_vae_realistic_loss(beta_vae_predictions, targets)
            
            beta_vae_optimizer.zero_grad()
            beta_vae_loss.backward()
            beta_vae_optimizer.step()
            beta_vae_epoch_losses.append(beta_vae_loss.item())
        
        neurocious_avg_loss = np.mean(neurocious_epoch_losses)
        beta_vae_avg_loss = np.mean(beta_vae_epoch_losses)
        
        neurocious_losses.append(neurocious_avg_loss)
        beta_vae_losses.append(beta_vae_avg_loss)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}/{config.epochs}")
            print(f"    Neurocious Loss: {neurocious_avg_loss:.4f}")
            print(f"    β-VAE Loss: {beta_vae_avg_loss:.4f}")
    
    print(f"  Training completed.")
    print(f"    Neurocious Final Loss: {neurocious_losses[-1]:.4f}")
    print(f"    β-VAE Final Loss: {beta_vae_losses[-1]:.4f}")
    
    # Step 4: Evaluate both models
    print("📈 Evaluating with realistic financial metrics...")
    
    with torch.no_grad():
        neurocious_test_predictions = neurocious_model(test_features_tensor)
        beta_vae_test_predictions = beta_vae_model(test_features_tensor)
    
    # Convert to numpy for evaluation
    neurocious_results = {
        'positions': neurocious_test_predictions['positions'].numpy(),
        'confidence': neurocious_test_predictions['confidence'].numpy(),
        'returns': neurocious_test_predictions['predicted_returns'].numpy(),
        'regimes': ['predicted'] * len(test_data),
        'actual_regimes': test_data['regime'].tolist()
    }
    
    beta_vae_results = {
        'positions': beta_vae_test_predictions['positions'].numpy(),
        'confidence': beta_vae_test_predictions['confidence'].numpy(),
        'returns': beta_vae_test_predictions['predicted_returns'].numpy(),
        'regimes': ['predicted'] * len(test_data),
        'actual_regimes': test_data['regime'].tolist()
    }
    
    # Step 5: Compare with baselines including β-VAE
    print("⚖️ Comparing with financial baselines including β-VAE...")
    
    # All models for comparison
    baseline_results = {
        'Random Walk': {
            'positions': np.random.uniform(-0.3, 0.3, len(test_data)),
            'confidence': np.full(len(test_data), 0.5),
            'returns': np.random.normal(0, 0.01, len(test_data)),
            'regimes': ['random'] * len(test_data),
            'actual_regimes': test_data['regime'].tolist()
        },
        'Buy and Hold': {
            'positions': np.full(len(test_data), 0.6),
            'confidence': np.full(len(test_data), 0.7),
            'returns': np.full(len(test_data), 0.08/252),
            'regimes': ['bull'] * len(test_data),
            'actual_regimes': test_data['regime'].tolist()
        },
        'β-VAE': beta_vae_results,
        'Neurocious': neurocious_results
    }
    
    # Evaluate all models
    test_data_dict = {
        'actual_returns': test_data['return'].values,
        'market_returns': test_data['return'].values
    }
    
    comparison_df = benchmark_comparison(baseline_results, test_data_dict)
    
    print("📊 Results:")
    print(comparison_df.to_string(index=False))
    
    # Step 6: Assess performance and compare Neurocious vs β-VAE
    print("\n🎯 Performance Assessment:")
    
    # Extract performance for both models
    neurocious_row = comparison_df[comparison_df['Model'] == 'Neurocious'].iloc[0]
    beta_vae_row = comparison_df[comparison_df['Model'] == 'β-VAE'].iloc[0]
    
    neurocious_sharpe = float(neurocious_row['Sharpe Ratio'])
    neurocious_hit_rate = float(neurocious_row['Hit Rate'].rstrip('%')) / 100
    
    beta_vae_sharpe = float(beta_vae_row['Sharpe Ratio']) 
    beta_vae_hit_rate = float(beta_vae_row['Hit Rate'].rstrip('%')) / 100
    
    print("\n🏁 HEAD-TO-HEAD COMPARISON:")
    print(f"{'Metric':<20} {'Neurocious':<12} {'β-VAE':<12} {'Winner':<10}")
    print("-" * 55)
    print(f"{'Sharpe Ratio':<20} {neurocious_sharpe:<12.2f} {beta_vae_sharpe:<12.2f} {'🏆 Neurocious' if neurocious_sharpe > beta_vae_sharpe else '🏆 β-VAE'}")
    print(f"{'Hit Rate':<20} {neurocious_hit_rate:<12.1%} {beta_vae_hit_rate:<12.1%} {'🏆 Neurocious' if neurocious_hit_rate > beta_vae_hit_rate else '🏆 β-VAE'}")
    
    # Overall winner
    neurocious_wins = 0
    if neurocious_sharpe > beta_vae_sharpe:
        neurocious_wins += 1
    if neurocious_hit_rate > beta_vae_hit_rate:
        neurocious_wins += 1
    
    if neurocious_wins >= 1:
        overall_winner = "🏆 NEUROCIOUS WINS"
    else:
        overall_winner = "🏆 β-VAE WINS"
    
    print(f"\n🏆 OVERALL: {overall_winner}")
    
    # Performance level assessment
    targets = realistic_performance_targets()
    
    if neurocious_sharpe >= targets['world_class']['sharpe_ratio']:
        assessment = "🏆 WORLD-CLASS"
    elif neurocious_sharpe >= targets['excellent_performance']['sharpe_ratio']:
        assessment = "🥇 EXCELLENT"
    elif neurocious_sharpe >= targets['good_performance']['sharpe_ratio']:
        assessment = "🥈 GOOD"
    elif neurocious_sharpe >= targets['minimum_viable']['sharpe_ratio']:
        assessment = "🥉 VIABLE"
    else:
        assessment = "❌ NEEDS IMPROVEMENT"
    
    print(f"\nNeurocious Assessment: {assessment}")
    print(f"Neurocious Sharpe Ratio: {neurocious_sharpe:.2f}")
    print(f"Neurocious Hit Rate: {neurocious_hit_rate:.1%}")
    
    # Save results
    results = {
        'performance': {
            'neurocious_sharpe_ratio': neurocious_sharpe,
            'neurocious_hit_rate': neurocious_hit_rate,
            'beta_vae_sharpe_ratio': beta_vae_sharpe,
            'beta_vae_hit_rate': beta_vae_hit_rate,
            'assessment': assessment,
            'winner': overall_winner
        },
        'comparison': comparison_df.to_dict('records'),
        'training_losses': {
            'neurocious': neurocious_losses,
            'beta_vae': beta_vae_losses
        },
        'config': {
            'training_days': config.training_days,
            'test_days': config.test_days,
            'epochs': config.epochs
        }
    }
    
    with open('realistic_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Realistic experiment completed!")
    print("📁 Results saved to 'realistic_experiment_results.json'")
    
    return results

if __name__ == "__main__":
    import asyncio
    
    config = SimpleRealisticConfig(
        training_days=50,
        test_days=25,
        epochs=10
    )
    
    results = asyncio.run(run_realistic_experiment(config))
    print(f"\n🎉 Final Assessment: {results['performance']['assessment']}")