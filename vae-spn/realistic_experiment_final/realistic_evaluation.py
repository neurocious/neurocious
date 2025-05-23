"""
Realistic Financial Model Evaluation Framework
============================================

Replaces artificial 98% accuracy metrics with proper financial evaluation.
Focuses on what actually matters in finance: risk-adjusted returns, not prediction accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

@dataclass 
class FinancialPerformanceMetrics:
    """Comprehensive financial performance evaluation"""
    
    # Return-based metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    calmar_ratio: float
    
    # Risk metrics  
    maximum_drawdown: float
    value_at_risk_95: float
    expected_shortfall: float
    downside_deviation: float
    
    # Trading metrics
    hit_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    win_loss_ratio: float
    
    # Model-specific metrics
    prediction_mse: float
    prediction_mae: float
    regime_accuracy: float
    uncertainty_calibration: float
    
    # Market timing
    market_correlation: float
    beta: float
    alpha: float
    information_ratio: float

class FinancialModelEvaluator:
    """Evaluates financial models using realistic metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def evaluate_model(self, 
                      model_predictions: Dict,
                      actual_returns: np.ndarray,
                      market_returns: np.ndarray,
                      initial_capital: float = 100000) -> FinancialPerformanceMetrics:
        """
        Comprehensive model evaluation using realistic financial metrics
        
        Args:
            model_predictions: Dict with 'positions', 'returns', 'regimes', 'confidence'
            actual_returns: Actual market returns
            market_returns: Benchmark market returns  
            initial_capital: Starting capital
        """
        
        # Extract predictions
        positions = np.array(model_predictions.get('positions', []))
        predicted_returns = np.array(model_predictions.get('returns', []))
        predicted_regimes = model_predictions.get('regimes', [])
        confidence_scores = np.array(model_predictions.get('confidence', []))
        actual_regimes = model_predictions.get('actual_regimes', [])
        
        # Calculate strategy returns
        strategy_returns = self._calculate_strategy_returns(positions, actual_returns)
        
        # Portfolio value evolution
        portfolio_values = self._calculate_portfolio_values(strategy_returns, initial_capital)
        
        # Return-based metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = np.std(strategy_returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = self._sharpe_ratio(strategy_returns)
        calmar_ratio = self._calmar_ratio(strategy_returns, portfolio_values)
        
        # Risk metrics
        max_drawdown = self._maximum_drawdown(portfolio_values)
        var_95 = self._value_at_risk(strategy_returns, confidence=0.95)
        expected_shortfall = self._expected_shortfall(strategy_returns, confidence=0.95)
        downside_dev = self._downside_deviation(strategy_returns)
        
        # Trading performance
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        
        hit_rate = len(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        profit_factor = abs(np.sum(winning_trades) / np.sum(losing_trades)) if np.sum(losing_trades) != 0 else np.inf
        
        # Prediction accuracy (but realistic expectations)
        prediction_mse = mean_squared_error(actual_returns[:len(predicted_returns)], predicted_returns) if len(predicted_returns) > 0 else np.inf
        prediction_mae = mean_absolute_error(actual_returns[:len(predicted_returns)], predicted_returns) if len(predicted_returns) > 0 else np.inf
        
        # Regime accuracy (if available)
        regime_accuracy = self._regime_accuracy(predicted_regimes, actual_regimes) if predicted_regimes and actual_regimes else 0
        
        # Uncertainty calibration
        uncertainty_calibration = self._uncertainty_calibration(confidence_scores, strategy_returns) if len(confidence_scores) > 0 else 0
        
        # Market timing metrics
        market_corr = np.corrcoef(strategy_returns[:len(market_returns)], market_returns[:len(strategy_returns)])[0,1] if len(market_returns) > 0 else 0
        beta, alpha = self._calculate_beta_alpha(strategy_returns, market_returns)
        information_ratio = self._information_ratio(strategy_returns, market_returns)
        
        return FinancialPerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            maximum_drawdown=max_drawdown,
            value_at_risk_95=var_95,
            expected_shortfall=expected_shortfall,
            downside_deviation=downside_dev,
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            average_win=avg_win,
            average_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            prediction_mse=prediction_mse,
            prediction_mae=prediction_mae,
            regime_accuracy=regime_accuracy,
            uncertainty_calibration=uncertainty_calibration,
            market_correlation=market_corr,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio
        )
    
    def _calculate_strategy_returns(self, positions: np.ndarray, actual_returns: np.ndarray) -> np.ndarray:
        """Calculate strategy returns from positions and market returns"""
        # Ensure arrays are same length
        min_len = min(len(positions), len(actual_returns))
        positions = positions[:min_len]
        actual_returns = actual_returns[:min_len]
        
        # Apply position sizing to returns
        strategy_returns = positions * actual_returns
        
        # Add transaction costs (realistic)
        transaction_cost = 0.001  # 10 basis points per trade
        position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
        costs = position_changes * transaction_cost
        
        return strategy_returns - costs
    
    def _calculate_portfolio_values(self, returns: np.ndarray, initial_capital: float) -> np.ndarray:
        """Calculate portfolio value evolution"""
        return initial_capital * np.cumprod(1 + returns)
    
    def _sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = np.mean(returns) - self.risk_free_rate / 252
        return excess_returns / np.std(returns) * np.sqrt(252)
    
    def _calmar_ratio(self, returns: np.ndarray, portfolio_values: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        if len(returns) == 0:
            return 0.0
        annual_return = np.mean(returns) * 252
        max_dd = abs(self._maximum_drawdown(portfolio_values))
        return annual_return / max_dd if max_dd > 0 else 0.0
    
    def _maximum_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return np.min(drawdown)
    
    def _value_at_risk(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _expected_shortfall(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(returns) == 0:
            return 0.0
        var = self._value_at_risk(returns, confidence)
        return np.mean(returns[returns <= var])
    
    def _downside_deviation(self, returns: np.ndarray, target: float = 0.0) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target]
        if len(downside_returns) == 0:
            return 0.0
        return np.sqrt(np.mean((downside_returns - target) ** 2)) * np.sqrt(252)
    
    def _regime_accuracy(self, predicted: List, actual: List) -> float:
        """Calculate regime prediction accuracy"""
        if not predicted or not actual or len(predicted) != len(actual):
            return 0.0
        return np.mean([p == a for p, a in zip(predicted, actual)])
    
    def _uncertainty_calibration(self, confidence: np.ndarray, returns: np.ndarray) -> float:
        """
        Calculate uncertainty calibration
        Good models should have higher confidence when they perform better
        """
        if len(confidence) == 0 or len(returns) == 0:
            return 0.0
        
        # Sort by confidence
        sorted_indices = np.argsort(confidence)
        sorted_confidence = confidence[sorted_indices]
        sorted_returns = np.abs(returns[sorted_indices])  # Use absolute returns as performance
        
        # Calculate correlation between confidence and performance
        return np.corrcoef(sorted_confidence, sorted_returns)[0, 1] if len(sorted_confidence) > 1 else 0.0
    
    def _calculate_beta_alpha(self, strategy_returns: np.ndarray, market_returns: np.ndarray) -> Tuple[float, float]:
        """Calculate beta and alpha vs market"""
        min_len = min(len(strategy_returns), len(market_returns))
        if min_len < 2:
            return 0.0, 0.0
        
        strategy = strategy_returns[:min_len]
        market = market_returns[:min_len]
        
        # Linear regression: strategy = alpha + beta * market
        covariance = np.cov(strategy, market)[0, 1]
        market_variance = np.var(market)
        
        beta = covariance / market_variance if market_variance > 0 else 0.0
        alpha = np.mean(strategy) - beta * np.mean(market)
        
        return beta, alpha * 252  # Annualized alpha
    
    def _information_ratio(self, strategy_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate information ratio (active return / tracking error)"""
        min_len = min(len(strategy_returns), len(market_returns))
        if min_len < 2:
            return 0.0
        
        strategy = strategy_returns[:min_len]
        market = market_returns[:min_len]
        
        active_returns = strategy - market
        active_return = np.mean(active_returns) * 252
        tracking_error = np.std(active_returns) * np.sqrt(252)
        
        return active_return / tracking_error if tracking_error > 0 else 0.0

def benchmark_comparison(models: Dict, test_data: Dict) -> pd.DataFrame:
    """
    Compare multiple models using realistic financial metrics
    
    Args:
        models: Dict of {'model_name': model_predictions}
        test_data: Dict with 'actual_returns', 'market_returns', etc.
    
    Returns:
        DataFrame with comparison results
    """
    
    evaluator = FinancialModelEvaluator()
    results = []
    
    for model_name, predictions in models.items():
        metrics = evaluator.evaluate_model(
            predictions, 
            test_data['actual_returns'],
            test_data['market_returns']
        )
        
        result = {
            'Model': model_name,
            'Annual Return': f"{metrics.annualized_return:.1%}",
            'Volatility': f"{metrics.volatility:.1%}",
            'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
            'Max Drawdown': f"{metrics.maximum_drawdown:.1%}",
            'Hit Rate': f"{metrics.hit_rate:.1%}",
            'Profit Factor': f"{metrics.profit_factor:.2f}",
            'Prediction MSE': f"{metrics.prediction_mse:.4f}",
            'Regime Accuracy': f"{metrics.regime_accuracy:.1%}",
            'Information Ratio': f"{metrics.information_ratio:.2f}"
        }
        results.append(result)
    
    return pd.DataFrame(results)

def create_realistic_benchmark():
    """Create realistic baseline expectations for financial models"""
    
    benchmarks = {
        'Random Walk': {
            'annual_return': 0.08,
            'volatility': 0.16,
            'sharpe_ratio': 0.5,
            'max_drawdown': -0.15,
            'hit_rate': 0.50,  # Pure chance
            'prediction_mse': 0.025,  # High prediction error
            'regime_accuracy': 0.25   # Random guessing among 4 regimes
        },
        
        'Buy and Hold': {
            'annual_return': 0.08,
            'volatility': 0.16,
            'sharpe_ratio': 0.5,
            'max_drawdown': -0.25,
            'hit_rate': 0.55,  # Slight upward bias
            'prediction_mse': 0.020,
            'regime_accuracy': 0.30
        },
        
        'Professional Fund': {
            'annual_return': 0.12,
            'volatility': 0.14,
            'sharpe_ratio': 0.86,
            'max_drawdown': -0.12,
            'hit_rate': 0.58,
            'prediction_mse': 0.015,
            'regime_accuracy': 0.45
        },
        
        'Top Quant Fund': {
            'annual_return': 0.18,
            'volatility': 0.12,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.08,
            'hit_rate': 0.62,
            'prediction_mse': 0.012,
            'regime_accuracy': 0.55
        }
    }
    
    return benchmarks

if __name__ == "__main__":
    # Example usage
    print("=== REALISTIC FINANCIAL EVALUATION FRAMEWORK ===")
    
    # Show realistic benchmarks
    benchmarks = create_realistic_benchmark()
    
    print("\nRealistic Performance Benchmarks:")
    print("-" * 50)
    for name, metrics in benchmarks.items():
        print(f"{name}:")
        print(f"  Annual Return: {metrics['annual_return']:.1%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Hit Rate: {metrics['hit_rate']:.1%}")
        print(f"  Regime Accuracy: {metrics['regime_accuracy']:.1%}")
        print()
    
    print("Key Insights:")
    print("- 98% prediction accuracy is impossible")
    print("- 60%+ hit rate is excellent in finance")
    print("- Sharpe ratio > 1.0 is very good")
    print("- Focus on risk-adjusted returns, not raw accuracy")