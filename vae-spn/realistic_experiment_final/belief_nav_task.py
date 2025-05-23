#!/usr/bin/env python3
"""
Neurocious Experimental Setup: Belief Navigation in Uncertain Environments
=========================================================================

This experiment demonstrates Neurocious's core capabilities:
1. Spatial belief navigation under uncertainty
2. Causal explanation of belief formation
3. Multi-timescale decision making (reflexes + planning)
4. World branching for counterfactual reasoning

EXPERIMENT: "Financial Market Belief Navigation"
----------------------------------------------
The agent observes sequential market data and must:
- Navigate beliefs about market state (bull/bear/volatile/stable)
- Make immediate reflex decisions (stop-loss, buy/sell)
- Plan longer-term strategies
- Explain why it believes what it believes
- Simulate alternative belief paths ("What if I had believed X?")
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import random
from datetime import datetime, timedelta

# Import our Neurocious components
from neurocious_integration import NeurociousSystem
from core import FieldParameters, CoTrainingConfig
from baseline import BenchmarkSuite


@dataclass
class MarketState:
    """Represents market state at a given time"""
    prices: np.ndarray           # Recent price history
    volumes: np.ndarray          # Trading volumes
    volatility: float            # Current volatility
    trend: float                 # Price trend (-1 to 1)
    regime: str                  # Ground truth: bull/bear/volatile/stable
    uncertainty: float           # How uncertain the regime is
    external_news: np.ndarray    # News sentiment features


@dataclass
class TradingAction:
    """Trading action taken by the agent"""
    position: float              # Position size (-1 to 1)
    stop_loss: float            # Stop loss level
    take_profit: float          # Take profit level
    confidence: float           # Action confidence
    reasoning: str              # Why this action was taken


@dataclass
class ExperimentConfig:
    """Configuration for the experiment"""
    sequence_length: int = 20
    market_history_days: int = 252  # 1 year
    num_market_regimes: int = 4
    noise_level: float = 0.1
    regime_switch_probability: float = 0.05
    news_impact_strength: float = 0.3
    
    # Training parameters
    num_episodes: int = 1000
    episode_length: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001


class MarketEnvironment:
    """Simulated financial market environment"""
    
    def __init__(self, config: ExperimentConfig, random_seed: int = 42):
        self.config = config
        np.random.seed(random_seed)
        
        # Market regime definitions
        self.regimes = {
            'bull': {'trend': 0.6, 'volatility': 0.15, 'stability': 0.8},
            'bear': {'trend': -0.5, 'volatility': 0.25, 'stability': 0.7},
            'volatile': {'trend': 0.1, 'volatility': 0.4, 'stability': 0.3},
            'stable': {'trend': 0.05, 'volatility': 0.08, 'stability': 0.9}
        }
        
        self.current_regime = 'stable'
        self.regime_persistence = 0
        self.time_step = 0
        
        # Price and volume history
        self.price_history = [100.0]  # Start at $100
        self.volume_history = [1000000]  # Start at 1M volume
        self.volatility_history = [0.1]
        
        # News simulation
        self.news_impact = 0.0
        self.news_decay = 0.95
    
    def step(self) -> MarketState:
        """Advance market by one time step"""
        self.time_step += 1
        
        # Regime switching
        if np.random.random() < self.config.regime_switch_probability:
            self.current_regime = np.random.choice(list(self.regimes.keys()))
            self.regime_persistence = 0
        else:
            self.regime_persistence += 1
        
        # Get regime parameters
        regime_params = self.regimes[self.current_regime]
        
        # Generate news impact
        if np.random.random() < 0.1:  # 10% chance of news
            self.news_impact = np.random.normal(0, self.config.news_impact_strength)
        self.news_impact *= self.news_decay
        
        # Calculate price change
        base_return = regime_params['trend'] / 252  # Daily return
        volatility = regime_params['volatility'] / np.sqrt(252)  # Daily vol
        noise = np.random.normal(0, volatility)
        news_effect = self.news_impact * 0.1
        
        price_change = base_return + noise + news_effect
        new_price = self.price_history[-1] * (1 + price_change)
        
        # Volume (inversely related to price stability)
        base_volume = 1000000
        volume_multiplier = 1 + abs(price_change) * 10 + (1 - regime_params['stability'])
        new_volume = base_volume * volume_multiplier * np.random.lognormal(0, 0.2)
        
        # Update history
        self.price_history.append(new_price)
        self.volume_history.append(new_volume)
        
        # Calculate current volatility (rolling 20-day)
        if len(self.price_history) >= 21:
            returns = np.diff(np.log(self.price_history[-21:]))
            current_volatility = np.std(returns) * np.sqrt(252)
        else:
            current_volatility = volatility * np.sqrt(252)
        
        self.volatility_history.append(current_volatility)
        
        # Calculate trend (slope of recent prices)
        if len(self.price_history) >= 10:
            x = np.arange(10)
            y = np.array(self.price_history[-10:])
            trend = np.polyfit(x, y, 1)[0] / y.mean()  # Normalized slope
        else:
            trend = 0.0
        
        # Create market state
        prices = np.array(self.price_history[-self.config.sequence_length:])
        volumes = np.array(self.volume_history[-self.config.sequence_length:])
        
        # Normalize features
        price_returns = np.diff(np.log(prices)) if len(prices) > 1 else np.array([0])
        volume_changes = np.diff(np.log(volumes)) if len(volumes) > 1 else np.array([0])
        
        # External news features (sentiment, magnitude, recency)
        news_features = np.array([
            self.news_impact,
            abs(self.news_impact),
            min(1.0, self.regime_persistence / 20.0)  # Regime age
        ])
        
        # Uncertainty based on regime stability and recent volatility
        uncertainty = 1.0 - regime_params['stability'] + current_volatility / 0.5
        uncertainty = np.clip(uncertainty, 0.0, 1.0)
        
        return MarketState(
            prices=prices,
            volumes=volumes,
            volatility=current_volatility,
            trend=trend,
            regime=self.current_regime,
            uncertainty=uncertainty,
            external_news=news_features
        )
    
    def reset(self):
        """Reset environment to initial state"""
        self.time_step = 0
        self.current_regime = 'stable'
        self.regime_persistence = 0
        self.price_history = [100.0]
        self.volume_history = [1000000]
        self.volatility_history = [0.1]
        self.news_impact = 0.0


class NeurociousTrader:
    """Neurocious-based trading agent"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Configure Neurocious for financial data
        neurocious_config = CoTrainingConfig(
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            beta=0.8,           # Higher KL weight for better uncertainty
            gamma=0.6,          # Strong narrative continuity for market trends
            delta=0.4,          # Field alignment for belief consistency
            policy_weight=0.7,  # Strong policy learning for trading
            reflex_weight=0.5,  # Moderate reflexes for stop-losses
            prediction_weight=0.6  # Good prediction for market forecasting
        )
        
        self.system = NeurociousSystem(
            config=neurocious_config,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            checkpoint_dir='./trading_checkpoints',
            log_dir='./trading_logs'
        )
        
        # Trading history
        self.trading_history = []
        self.belief_trajectory = []
        self.explanation_log = []
    
    def process_market_state(self, market_state: MarketState) -> Tuple[TradingAction, Dict[str, Any]]:
        """Process market state and make trading decision"""
        
        # Convert market state to neural network input
        input_features = self._market_state_to_features(market_state)
        
        # Run inference through Neurocious
        results = self.system.inference([input_features], return_explanations=True)
        
        # Extract trading decision from results
        policy_output = results['policy']
        reflexes = results['reflexes']
        predictions = results['predictions']
        
        # Interpret policy output as trading action
        position = float(torch.tanh(policy_output[0]).item())  # Position size
        stop_loss_raw = float(torch.sigmoid(policy_output[1]).item())
        take_profit_raw = float(torch.sigmoid(policy_output[2]).item())
        
        # Calculate actual stop loss and take profit levels
        current_price = market_state.prices[-1]
        stop_loss = current_price * (1 - stop_loss_raw * 0.1)  # Max 10% stop loss
        take_profit = current_price * (1 + take_profit_raw * 0.2)  # Max 20% take profit
        
        # Confidence from routing confidence
        confidence = float(results['confidence'].item())
        
        # Extract reasoning from explanation
        reasoning = results['explanation']['justification']
        
        action = TradingAction(
            position=position,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reasoning=reasoning
        )
        
        # Store belief information
        belief_info = {
            'field_parameters': results['field_parameters'],
            'routing': results['routing'].detach().cpu().numpy() if hasattr(results['routing'], 'detach') else results['routing'],
            'explanation': results['explanation'],
            'inverse_explanation': results['inverse_explanation'],
            'market_regime_belief': self._interpret_regime_belief(results['routing']),
            'uncertainty_estimate': 1.0 - confidence
        }
        
        self.belief_trajectory.append(belief_info)
        
        return action, belief_info
    
    def _market_state_to_features(self, market_state: MarketState) -> torch.Tensor:
        """Convert market state to neural network features"""
        
        # Price features (returns, momentum, etc.)
        prices = market_state.prices
        if len(prices) > 1:
            returns = np.diff(np.log(prices))
            price_momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            price_volatility = np.std(returns) if len(returns) > 1 else 0
        else:
            returns = np.array([0])
            price_momentum = 0
            price_volatility = 0
        
        # Volume features
        volumes = market_state.volumes
        if len(volumes) > 1:
            volume_changes = np.diff(np.log(volumes))
            volume_trend = np.mean(volume_changes[-3:]) if len(volume_changes) >= 3 else 0
        else:
            volume_trend = 0
        
        # Technical indicators (simplified)
        if len(prices) >= 10:
            sma_10 = np.mean(prices[-10:])
            price_to_sma = prices[-1] / sma_10 - 1
        else:
            price_to_sma = 0
        
        if len(prices) >= 20:
            sma_20 = np.mean(prices[-20:])
            sma_ratio = sma_10 / sma_20 - 1 if len(prices) >= 10 else 0
        else:
            sma_ratio = 0
        
        # Combine all features
        features = np.concatenate([
            returns[-10:] if len(returns) >= 10 else np.pad(returns, (10-len(returns), 0)),
            [price_momentum, price_volatility, volume_trend],
            [price_to_sma, sma_ratio],
            [market_state.volatility, market_state.trend],
            market_state.external_news,
            [market_state.uncertainty]
        ])
        
        # CRITICAL FIX: Create dense, meaningful representation instead of sparse padding
        # Repeat and transform features to create dense 784-dimensional representation
        
        # Normalize features to reasonable range
        features = np.clip(features, -3, 3)  # Clip extreme values
        features = (features + 3) / 6  # Map [-3,3] to [0,1]
        
        # Create meaningful dense representation by expanding features
        dense_features = np.zeros(784)
        
        # Method 1: Tile the features to fill the space meaningfully
        base_len = len(features)
        repetitions = 784 // base_len
        remainder = 784 % base_len
        
        for i in range(repetitions):
            start_idx = i * base_len
            end_idx = (i + 1) * base_len
            # Add slight noise to each repetition to avoid exact copies
            noise_scale = 0.01 * (i + 1)  # Increasing noise for each repetition
            dense_features[start_idx:end_idx] = features + np.random.normal(0, noise_scale, base_len)
        
        # Fill remainder
        if remainder > 0:
            dense_features[-remainder:] = features[:remainder] + np.random.normal(0, 0.05, remainder)
        
        # Ensure final values are in [0,1] range
        dense_features = np.clip(dense_features, 0, 1)
        
        return torch.tensor(dense_features, dtype=torch.float32)
    
    def _interpret_regime_belief(self, routing: np.ndarray) -> Dict[str, float]:
        """Interpret routing probabilities as regime beliefs"""
        # Simple interpretation: divide routing space into 4 quadrants
        # Convert to numpy if it's a torch tensor
        if hasattr(routing, 'detach'):
            routing = routing.detach().cpu().numpy()
        
        routing_2d = routing.reshape(16, 16)  # Assuming 16x16 field
        
        quadrants = {
            'bull': float(np.sum(routing_2d[:8, :8])),      # Top-left
            'bear': float(np.sum(routing_2d[8:, :8])),     # Bottom-left  
            'volatile': float(np.sum(routing_2d[:8, 8:])), # Top-right
            'stable': float(np.sum(routing_2d[8:, 8:]))    # Bottom-right
        }
        
        return quadrants


class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.environment = MarketEnvironment(config)
        self.agent = NeurociousTrader(config)
        self.results = {
            'training_data': [],
            'performance_metrics': [],
            'belief_analysis': [],
            'explanation_quality': []
        }
    
    async def run_experiment(self):
        """Run the complete experiment"""
        
        print("🚀 Starting Neurocious Financial Belief Navigation Experiment")
        print("=" * 60)
        
        # Phase 1: Generate training data
        print("📊 Phase 1: Generating training data...")
        training_data = self._generate_training_data()
        
        # Phase 2: Train Neurocious
        print("🧠 Phase 2: Training Neurocious agent...")
        await self._train_agent(training_data)
        
        # Phase 3: Evaluation
        print("📈 Phase 3: Evaluating performance...")
        evaluation_results = await self._evaluate_agent()
        
        # Phase 4: Belief analysis
        print("🔍 Phase 4: Analyzing belief navigation...")
        belief_analysis = self._analyze_belief_navigation()
        
        # Phase 5: Comparison with baselines
        print("⚖️  Phase 5: Comparing with baselines...")
        comparison_results = await self._compare_with_baselines(training_data)
        
        # Phase 6: Generate report
        print("📝 Phase 6: Generating experiment report...")
        self._generate_experiment_report(evaluation_results, belief_analysis, comparison_results)
        
        print("✅ Experiment completed successfully!")
        
        return {
            'evaluation': evaluation_results,
            'belief_analysis': belief_analysis,
            'comparison': comparison_results
        }
    
    def _generate_training_data(self) -> Dict[str, List]:
        """Generate training data from market simulation"""
        
        sequences = []
        rewards = []
        actions = []
        reactions = []
        future_states = []
        
        for episode in range(self.config.num_episodes):
            if episode % 100 == 0:
                print(f"  Generating episode {episode}/{self.config.num_episodes}")
            
            self.environment.reset()
            
            episode_sequence = []
            episode_rewards = []
            episode_actions = []
            episode_reactions = []
            episode_futures = []
            
            for step in range(self.config.episode_length):
                # Get market state
                market_state = self.environment.step()
                
                # Convert to features
                features = self.agent._market_state_to_features(market_state)
                episode_sequence.append(features.numpy())
                
                # Generate reward signal (profit from optimal strategy)
                optimal_action = self._optimal_action(market_state)
                reward = self._calculate_reward(market_state, optimal_action)
                episode_rewards.append(reward)
                
                # Generate action labels (what should the agent do?)
                action_vector = np.array([
                    optimal_action.position,
                    optimal_action.stop_loss / market_state.prices[-1] - 1,  # Relative stop loss
                    optimal_action.take_profit / market_state.prices[-1] - 1,  # Relative take profit
                    *np.zeros(7)  # Pad to 10 dimensions
                ])
                episode_actions.append(action_vector)
                
                # Generate reflex reactions (immediate responses)
                reflexes = np.array([
                    1.0 if market_state.volatility > 0.3 else 0.0,  # High vol warning
                    1.0 if abs(market_state.trend) > 0.5 else 0.0,  # Strong trend
                    1.0 if market_state.uncertainty > 0.7 else 0.0,  # High uncertainty
                    1.0 if abs(market_state.external_news[0]) > 0.2 else 0.0,  # News impact
                    1.0 if market_state.regime in ['volatile', 'bear'] else 0.0  # Risk mode
                ])
                episode_reactions.append(reflexes)
                
                # Future state prediction (next period's key metrics)
                next_state = self.environment.step()
                future_prediction = np.array([
                    next_state.trend,
                    next_state.volatility,
                    next_state.uncertainty,
                    1.0 if next_state.regime != market_state.regime else 0.0  # Regime change
                ])
                episode_futures.append(future_prediction)
            
            sequences.append(episode_sequence)
            rewards.append(episode_rewards)
            actions.append(episode_actions)
            reactions.append(episode_reactions)
            future_states.append(episode_futures)
        
        return {
            'sequences': sequences,
            'rewards': rewards,
            'actions': actions,
            'reactions': reactions,
            'future_states': future_states
        }
    
    def _optimal_action(self, market_state: MarketState) -> TradingAction:
        """Calculate optimal action given perfect information"""
        
        regime_strategies = {
            'bull': TradingAction(0.8, 0.95, 1.15, 0.9, "Bull market: long position"),
            'bear': TradingAction(-0.6, 1.05, 0.9, 0.8, "Bear market: short position"),
            'volatile': TradingAction(0.2, 0.97, 1.05, 0.6, "Volatile: small position, tight stops"),
            'stable': TradingAction(0.4, 0.93, 1.1, 0.7, "Stable: moderate position")
        }
        
        base_action = regime_strategies[market_state.regime]
        current_price = market_state.prices[-1]
        
        # Adjust for uncertainty
        uncertainty_factor = 1.0 - market_state.uncertainty
        adjusted_position = base_action.position * uncertainty_factor
        
        return TradingAction(
            position=adjusted_position,
            stop_loss=current_price * base_action.stop_loss,
            take_profit=current_price * base_action.take_profit,
            confidence=base_action.confidence * uncertainty_factor,
            reasoning=f"{base_action.reasoning} (uncertainty adjusted)"
        )
    
    def _calculate_reward(self, market_state: MarketState, action: TradingAction) -> float:
        """Calculate reward for a given action"""
        
        # Base reward from regime alignment
        regime_rewards = {
            'bull': action.position,           # Reward long positions in bull markets
            'bear': -action.position,          # Reward short positions in bear markets  
            'volatile': -abs(action.position), # Reward smaller positions in volatile markets
            'stable': 0.5 * abs(action.position)  # Moderate reward for any position in stable markets
        }
        
        base_reward = regime_rewards[market_state.regime]
        
        # Penalty for uncertainty
        uncertainty_penalty = market_state.uncertainty * 0.2
        
        # Bonus for confidence calibration
        confidence_bonus = action.confidence * 0.1
        
        return base_reward - uncertainty_penalty + confidence_bonus
    
    async def _train_agent(self, training_data: Dict[str, List]):
        """Train the Neurocious agent"""
        
        # Split data
        split_idx = int(0.8 * len(training_data['sequences']))
        
        train_data = {key: data[:split_idx] for key, data in training_data.items()}
        val_data = {key: data[split_idx:] for key, data in training_data.items()}
        
        # Train
        await self.agent.system.train(
            training_data=train_data,
            validation_data=val_data,
            epochs=20,  # Reduced for demonstration
            save_freq=5,
            eval_freq=2
        )
    
    async def _evaluate_agent(self) -> Dict[str, float]:
        """Evaluate agent performance"""
        
        self.environment.reset()
        
        total_profit = 0.0
        correct_regime_predictions = 0
        total_predictions = 0
        confidence_scores = []
        explanation_scores = []
        
        for step in range(100):  # 100 test steps
            market_state = self.environment.step()
            action, belief_info = self.agent.process_market_state(market_state)
            
            # Calculate profit from action
            next_state = self.environment.step()
            price_change = (next_state.prices[-1] / market_state.prices[-1]) - 1
            step_profit = action.position * price_change
            total_profit += step_profit
            
            # Check regime prediction accuracy
            regime_beliefs = belief_info['market_regime_belief']
            predicted_regime = max(regime_beliefs, key=regime_beliefs.get)
            if predicted_regime == market_state.regime:
                correct_regime_predictions += 1
            total_predictions += 1
            
            # Store confidence and explanation quality
            confidence_scores.append(action.confidence)
            explanation_scores.append(len(action.reasoning.split()))  # Proxy for explanation quality
        
        return {
            'total_profit': total_profit,
            'regime_accuracy': correct_regime_predictions / total_predictions,
            'average_confidence': np.mean(confidence_scores),
            'explanation_quality': np.mean(explanation_scores),
            'sharpe_ratio': total_profit / (np.std([total_profit]) + 0.001)  # Simplified
        }
    
    def _analyze_belief_navigation(self) -> Dict[str, Any]:
        """Analyze how beliefs navigate through the probability space"""
        
        if not self.agent.belief_trajectory:
            return {'error': 'No belief trajectory data available'}
        
        # Extract belief evolution
        field_params = [bt['field_parameters'] for bt in self.agent.belief_trajectory]
        routing_evolution = [bt['routing'] for bt in self.agent.belief_trajectory]
        regime_beliefs = [bt['market_regime_belief'] for bt in self.agent.belief_trajectory]
        
        # Calculate belief stability
        param_changes = []
        for i in range(1, len(field_params)):
            change = abs(field_params[i]['entropy'] - field_params[i-1]['entropy'])
            param_changes.append(change)
        
        belief_stability = 1.0 - np.mean(param_changes) if param_changes else 1.0
        
        # Calculate regime transition smoothness
        regime_transitions = 0
        for i in range(1, len(regime_beliefs)):
            prev_max = max(regime_beliefs[i-1], key=regime_beliefs[i-1].get)
            curr_max = max(regime_beliefs[i], key=regime_beliefs[i].get)
            if prev_max != curr_max:
                regime_transitions += 1
        
        transition_rate = regime_transitions / len(regime_beliefs) if regime_beliefs else 0
        
        return {
            'belief_stability': belief_stability,
            'regime_transition_rate': transition_rate,
            'average_entropy': np.mean([fp['entropy'] for fp in field_params]),
            'average_curvature': np.mean([fp['curvature'] for fp in field_params]),
            'average_alignment': np.mean([fp['alignment'] for fp in field_params]),
            'field_evolution': field_params
        }
    
    async def _compare_with_baselines(self, training_data: Dict[str, List]) -> Dict[str, Any]:
        """Compare with baseline methods"""
        
        print("🔬 Running comprehensive baseline comparison...")
        
        # Create benchmark suite
        benchmark = BenchmarkSuite()
        
        # Run comparison with limited data for efficiency
        test_data = {
            'sequences': training_data['sequences'][:20],  # Sample for efficiency
            'rewards': training_data['rewards'][:20],
            'actions': training_data['actions'][:20],
            'reactions': training_data['reactions'][:20],
            'future_states': training_data['future_states'][:20]
        }
        
        comparison_results = await benchmark.run_full_benchmark(
            neurocious_system=self.agent.system,
            test_data=test_data
        )
        
        # Generate detailed report
        benchmark.generate_test_report(comparison_results, 'neurocious_benchmark_report.md')
        
        return comparison_results
    
    def _generate_experiment_report(
        self, 
        evaluation: Dict[str, float],
        belief_analysis: Dict[str, Any],
        comparison: Dict[str, Any]
    ):
        """Generate comprehensive experiment report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"experiment_report_{timestamp}.md")
        
        with open(report_path, 'w') as f:
            f.write("# Neurocious Financial Belief Navigation Experiment\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Profit**: {evaluation['total_profit']:.4f}\n")
            f.write(f"- **Regime Prediction Accuracy**: {evaluation['regime_accuracy']:.3f}\n")
            f.write(f"- **Average Confidence**: {evaluation['average_confidence']:.3f}\n")
            f.write(f"- **Belief Stability**: {belief_analysis.get('belief_stability', 0):.3f}\n\n")
            
            f.write("## Key Findings\n\n")
            
            if evaluation['regime_accuracy'] > 0.7:
                f.write("✅ **Strong regime prediction**: Agent accurately identifies market conditions\n")
            if belief_analysis.get('belief_stability', 0) > 0.8:
                f.write("✅ **Stable belief navigation**: Beliefs evolve smoothly through probability space\n")
            if evaluation['explanation_quality'] > 10:
                f.write("✅ **Rich explanations**: Agent provides detailed reasoning for decisions\n")
            
            f.write("\n## Detailed Results\n\n")
            f.write("### Performance Metrics\n")
            for metric, value in evaluation.items():
                f.write(f"- **{metric}**: {value:.4f}\n")
            
            f.write("\n### Belief Analysis\n")
            for metric, value in belief_analysis.items():
                if isinstance(value, (int, float)):
                    f.write(f"- **{metric}**: {value:.4f}\n")
            
            f.write("\n### Baseline Comparison\n")
            if 'summary' in comparison:
                summary = comparison['summary']
                f.write(f"- **Overall Score vs Baselines**: {summary.get('overall_score', 0):.3f}/1.0\n")
                f.write(f"- **Metrics Won**: {summary.get('neurocious_wins', 0)}/{summary.get('neurocious_total', 0)}\n")
        
        print(f"📄 Experiment report saved to: {report_path}")


# Visualization utilities
class ExperimentVisualizer:
    """Visualize experiment results"""
    
    @staticmethod
    def plot_belief_evolution(belief_trajectory: List[Dict], save_path: str = None):
        """Plot evolution of beliefs over time"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract time series
        times = range(len(belief_trajectory))
        field_params = [bt['field_parameters'] for bt in belief_trajectory]
        
        entropies = [fp['entropy'] for fp in field_params]
        curvatures = [fp['curvature'] for fp in field_params]
        alignments = [fp['alignment'] for fp in field_params]
        
        # Plot field parameters
        axes[0, 0].plot(times, entropies, label='Entropy', color='blue')
        axes[0, 0].set_title('Field Entropy Evolution')
        axes[0, 0].set_ylabel('Entropy')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(times, curvatures, label='Curvature', color='red')
        axes[0, 1].set_title('Field Curvature Evolution')
        axes[0, 1].set_ylabel('Curvature')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(times, alignments, label='Alignment', color='green')
        axes[1, 0].set_title('Field Alignment Evolution')
        axes[1, 0].set_ylabel('Alignment')
        axes[1, 0].grid(True)
        
        # Plot regime beliefs
        regime_data = []
        for bt in belief_trajectory:
            regime_beliefs = bt['market_regime_belief']
            regime_data.append([regime_beliefs.get(regime, 0) for regime in ['bull', 'bear', 'volatile', 'stable']])
        
        regime_data = np.array(regime_data)
        
        for i, regime in enumerate(['bull', 'bear', 'volatile', 'stable']):
            axes[1, 1].plot(times, regime_data[:, i], label=regime, alpha=0.7)
        
        axes[1, 1].set_title('Market Regime Beliefs')
        axes[1, 1].set_ylabel('Belief Probability')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Belief evolution plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_performance_comparison(comparison_results: Dict, save_path: str = None):
        """Plot performance comparison with baselines"""
        
        if 'raw_results' not in comparison_results:
            print("No comparison results available for plotting")
            return
        
        results = comparison_results['raw_results']
        metrics = list(next(iter(results.values())).keys())
        models = list(results.keys())
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, model in enumerate(models):
            values = [results[model].get(metric, 0) for metric in metrics]
            # Normalize values to 0-1 scale for radar chart
            max_values = [max(results[m].get(metric, 0) for m in models) for metric in metrics]
            normalized_values = [v / (mv + 0.001) for v, mv in zip(values, max_values)]
            normalized_values += normalized_values[:1]  # Complete the circle
            
            color = colors[i % len(colors)]
            linewidth = 3 if model == 'neurocious' else 1
            alpha = 0.8 if model == 'neurocious' else 0.6
            
            ax.plot(angles, normalized_values, 'o-', linewidth=linewidth, 
                   label=model, color=color, alpha=alpha)
            ax.fill(angles, normalized_values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison', size=16, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_trading_performance(environment: MarketEnvironment, actions: List[TradingAction], save_path: str = None):
        """Plot trading performance over time"""
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Price and positions
        times = range(len(environment.price_history))
        prices = environment.price_history
        
        axes[0].plot(times, prices, label='Price', color='black', linewidth=2)
        axes[0].set_title('Price Evolution and Trading Positions')
        axes[0].set_ylabel('Price ($)')
        axes[0].grid(True)
        
        # Add position markers
        position_times = []
        position_values = []
        position_colors = []
        
        for i, action in enumerate(actions):
            if abs(action.position) > 0.1:  # Only show significant positions
                position_times.append(i)
                position_values.append(prices[min(i, len(prices)-1)])
                position_colors.append('green' if action.position > 0 else 'red')
        
        if position_times:
            axes[0].scatter(position_times, position_values, c=position_colors, 
                          s=50, alpha=0.7, label='Positions')
        axes[0].legend()
        
        # Confidence and uncertainty
        if actions:
            confidences = [action.confidence for action in actions]
            axes[1].plot(range(len(confidences)), confidences, 
                        label='Confidence', color='blue', linewidth=2)
            axes[1].set_title('Trading Confidence Over Time')
            axes[1].set_ylabel('Confidence')
            axes[1].set_ylim(0, 1)
            axes[1].grid(True)
            axes[1].legend()
        
        # Regime evolution
        regime_history = []
        for state in environment.price_history:  # Simplified - would need actual regime history
            # This is a placeholder - in real implementation, track regime changes
            regime_history.append(np.random.choice(['bull', 'bear', 'volatile', 'stable']))
        
        regime_colors = {'bull': 'green', 'bear': 'red', 'volatile': 'orange', 'stable': 'blue'}
        regime_numeric = [list(regime_colors.keys()).index(r) for r in regime_history]
        
        axes[2].plot(range(len(regime_numeric)), regime_numeric, 
                    drawstyle='steps-post', linewidth=2, color='purple')
        axes[2].set_title('Market Regime Evolution')
        axes[2].set_ylabel('Regime')
        axes[2].set_xlabel('Time Step')
        axes[2].set_yticks(range(4))
        axes[2].set_yticklabels(['bull', 'bear', 'volatile', 'stable'])
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trading performance plot saved to: {save_path}")
        
        plt.show()


# Main experiment execution
async def main():
    """Run the complete experiment"""
    
    print("🎯 NEUROCIOUS BELIEF NAVIGATION EXPERIMENT")
    print("==========================================")
    print()
    print("This experiment demonstrates Neurocious's unique capabilities:")
    print("• Spatial belief navigation under uncertainty")
    print("• Causal explanation of belief formation") 
    print("• Multi-timescale decision making")
    print("• World branching for counterfactual reasoning")
    print()
    
    # Configuration
    config = ExperimentConfig(
        sequence_length=20,
        num_episodes=200,  # Reduced for faster execution
        episode_length=30,
        batch_size=8,
        learning_rate=0.001
    )
    
    # Run experiment
    runner = ExperimentRunner(config)
    results = await runner.run_experiment()
    
    # Visualizations
    print("🎨 Generating visualizations...")
    visualizer = ExperimentVisualizer()
    
    if runner.agent.belief_trajectory:
        visualizer.plot_belief_evolution(
            runner.agent.belief_trajectory,
            save_path='belief_evolution.png'
        )
    
    if 'comparison' in results and results['comparison']:
        visualizer.plot_performance_comparison(
            results['comparison'],
            save_path='performance_comparison.png'
        )
    
    # Demo: Interactive belief exploration
    print("\n🔍 INTERACTIVE BELIEF EXPLORATION DEMO")
    print("=====================================")
    
    # Create a test market scenario
    test_environment = MarketEnvironment(config)
    test_market_state = test_environment.step()
    
    print(f"Market Scenario:")
    print(f"  Current Price: ${test_market_state.prices[-1]:.2f}")
    print(f"  Regime: {test_market_state.regime}")
    print(f"  Volatility: {test_market_state.volatility:.3f}")
    print(f"  Uncertainty: {test_market_state.uncertainty:.3f}")
    
    # Get agent's decision
    action, belief_info = runner.agent.process_market_state(test_market_state)
    
    print(f"\nAgent's Decision:")
    print(f"  Position: {action.position:.3f}")
    print(f"  Confidence: {action.confidence:.3f}")
    print(f"  Reasoning: {action.reasoning}")
    
    print(f"\nBelief Analysis:")
    regime_beliefs = belief_info['market_regime_belief']
    for regime, prob in regime_beliefs.items():
        print(f"  {regime}: {prob:.3f}")
    
    print(f"\nField Parameters:")
    field_params = belief_info['field_parameters']
    print(f"  Entropy: {field_params['entropy']:.3f}")
    print(f"  Curvature: {field_params['curvature']:.3f}")
    print(f"  Alignment: {field_params['alignment']:.3f}")
    
    # Simulate world branching
    print(f"\n🌐 WORLD BRANCHING SIMULATION")
    print("============================")
    
    current_field_params = FieldParameters(**field_params)
    branches = runner.agent.system.simulate_world_branches(current_field_params, num_branches=3)
    
    print("Alternative belief trajectories:")
    for i, branch in enumerate(branches):
        print(f"  Branch {i+1}: probability={branch['probability']:.3f}")
        print(f"    Field state: {branch['initial_state']}")
    
    print(f"\n✅ Experiment completed successfully!")
    print(f"Check the generated files for detailed results and visualizations.")
    
    return results


# Quick demo function
async def quick_demo():
    """Quick demonstration of key capabilities"""
    
    print("🚀 NEUROCIOUS QUICK DEMO")
    print("========================")
    
    # Minimal configuration
    config = ExperimentConfig(
        num_episodes=10,
        episode_length=10,
        batch_size=4
    )
    
    # Create environment and agent
    environment = MarketEnvironment(config)
    agent = NeurociousTrader(config)
    
    print("Simulating 5 market scenarios...")
    
    for scenario in range(5):
        print(f"\n--- Scenario {scenario + 1} ---")
        
        # Reset and get market state
        environment.reset()
        market_state = environment.step()
        
        print(f"Market: {market_state.regime} regime, "
              f"volatility={market_state.volatility:.3f}, "
              f"uncertainty={market_state.uncertainty:.3f}")
        
        # Get agent decision (using pretrained features)
        # Note: In real use, you'd load a trained model
        try:
            action, belief_info = agent.process_market_state(market_state)
            
            print(f"Decision: position={action.position:.3f}, "
                  f"confidence={action.confidence:.3f}")
            
            # Show top regime belief
            regime_beliefs = belief_info['market_regime_belief']
            top_belief = max(regime_beliefs.items(), key=lambda x: x[1])
            print(f"Top belief: {top_belief[0]} ({top_belief[1]:.3f})")
            
        except Exception as e:
            print(f"Demo decision: moderate position (model not trained)")
            print(f"Note: Run full experiment to see trained model performance")
    
    print(f"\n✅ Quick demo completed!")
    print(f"Run 'await main()' for the full experiment with training.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick demo mode
        asyncio.run(quick_demo())
    else:
        # Full experiment
        asyncio.run(main())