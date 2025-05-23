"""
Realistic Financial Training Objectives for Neurocious
=====================================================

Updates the training objectives to focus on what actually matters in finance:
- Risk management over raw prediction accuracy
- Uncertainty quantification
- Regime-aware decision making
- Capital preservation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

class FinancialLossFunction(nn.Module):
    """
    Comprehensive loss function for financial applications
    Balances multiple objectives that matter in real trading
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Loss weights (sum to 1.0 for interpretability)
        self.weights = {
            'risk_adjusted_return': 0.3,     # Primary objective
            'uncertainty_calibration': 0.2,  # Know when you don't know
            'regime_consistency': 0.15,      # Consistent behavior in regimes
            'capital_preservation': 0.15,    # Avoid large losses
            'reconstruction': 0.1,           # Still need to learn representations
            'field_regularization': 0.05,    # Keep spatial structure stable
            'transaction_costs': 0.05        # Realistic trading costs
        }
    
    def forward(self, 
                predictions: Dict,
                targets: Dict,
                market_state: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate comprehensive financial loss
        
        Args:
            predictions: Model predictions (positions, confidence, reconstructions, etc.)
            targets: Target values (returns, regimes, etc.)
            market_state: Current market conditions
        
        Returns:
            total_loss, loss_breakdown
        """
        
        losses = {}
        
        # 1. Risk-Adjusted Return Loss (most important)
        losses['risk_adjusted_return'] = self._risk_adjusted_return_loss(
            predictions, targets, market_state
        )
        
        # 2. Uncertainty Calibration Loss
        losses['uncertainty_calibration'] = self._uncertainty_calibration_loss(
            predictions, targets
        )
        
        # 3. Regime Consistency Loss
        losses['regime_consistency'] = self._regime_consistency_loss(
            predictions, targets, market_state
        )
        
        # 4. Capital Preservation Loss
        losses['capital_preservation'] = self._capital_preservation_loss(
            predictions, targets
        )
        
        # 5. Reconstruction Loss (reduced importance)
        losses['reconstruction'] = self._reconstruction_loss(
            predictions, targets
        )
        
        # 6. Field Regularization
        losses['field_regularization'] = self._field_regularization_loss(
            predictions
        )
        
        # 7. Transaction Cost Penalty
        losses['transaction_costs'] = self._transaction_cost_loss(
            predictions
        )
        
        # Weighted combination
        total_loss = sum(self.weights[name] * loss for name, loss in losses.items())
        
        return total_loss, losses
    
    def _risk_adjusted_return_loss(self, predictions: Dict, targets: Dict, market_state: Dict) -> torch.Tensor:
        """
        Loss based on Sharpe ratio optimization
        Encourages high returns with low volatility
        """
        positions = predictions.get('positions', torch.zeros(1))
        returns = targets.get('returns', torch.zeros_like(positions))
        
        # Strategy returns
        strategy_returns = positions * returns
        
        # Sharpe ratio calculation (negative because we minimize loss)
        if len(strategy_returns) > 1:
            mean_return = torch.mean(strategy_returns)
            std_return = torch.std(strategy_returns) + 1e-8  # Avoid division by zero
            sharpe_ratio = mean_return / std_return
            
            # We want to maximize Sharpe ratio, so minimize negative Sharpe
            return -sharpe_ratio
        else:
            return torch.tensor(0.0, requires_grad=True)
    
    def _uncertainty_calibration_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """
        Penalize miscalibrated uncertainty estimates
        High confidence should correlate with low prediction error
        """
        confidence = predictions.get('confidence', torch.ones(1))
        predicted_returns = predictions.get('predicted_returns', torch.zeros_like(confidence))
        actual_returns = targets.get('returns', torch.zeros_like(predicted_returns))
        
        # Prediction errors
        prediction_errors = torch.abs(predicted_returns - actual_returns)
        
        # Well-calibrated model: high confidence → low error
        # Loss increases when high confidence but high error
        calibration_loss = confidence * prediction_errors
        
        return torch.mean(calibration_loss)
    
    def _regime_consistency_loss(self, predictions: Dict, targets: Dict, market_state: Dict) -> torch.Tensor:
        """
        Ensure behavior is consistent within market regimes
        Different regimes should trigger different strategies
        """
        positions = predictions.get('positions', torch.zeros(1))
        regime_routing = predictions.get('routing', torch.zeros(1, 4))  # 4 regimes
        actual_regime = targets.get('regime', torch.zeros(1))
        
        # Convert regime to one-hot if needed
        if actual_regime.dim() == 1:
            regime_onehot = torch.zeros_like(regime_routing)
            regime_onehot.scatter_(1, actual_regime.long().unsqueeze(1), 1)
        else:
            regime_onehot = actual_regime
        
        # Regime-specific position consistency
        # Similar regimes should produce similar positions
        regime_weighted_positions = regime_routing @ positions.unsqueeze(-1)
        target_positions = regime_onehot @ positions.unsqueeze(-1)
        
        consistency_loss = torch.mean((regime_weighted_positions - target_positions) ** 2)
        
        return consistency_loss
    
    def _capital_preservation_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """
        Heavily penalize large losses (asymmetric loss function)
        Capital preservation is crucial in finance
        """
        positions = predictions.get('positions', torch.zeros(1))
        returns = targets.get('returns', torch.zeros_like(positions))
        
        # Strategy returns
        strategy_returns = positions * returns
        
        # Asymmetric loss: penalize losses more than reward gains
        loss_penalty = 2.0  # Losses hurt 2x more than gains help
        
        # Apply asymmetric penalty
        losses = torch.clamp(strategy_returns, max=0)  # Only negative returns
        gains = torch.clamp(strategy_returns, min=0)   # Only positive returns
        
        asymmetric_loss = -torch.mean(gains) + loss_penalty * torch.mean(torch.abs(losses))
        
        return asymmetric_loss
    
    def _reconstruction_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """
        Standard reconstruction loss (reduced importance in financial context)
        """
        reconstructed = predictions.get('reconstructed', torch.zeros(1))
        original = targets.get('input_sequence', torch.zeros_like(reconstructed))
        
        return torch.mean((reconstructed - original) ** 2)
    
    def _field_regularization_loss(self, predictions: Dict) -> torch.Tensor:
        """
        Keep spatial probability fields stable and meaningful
        """
        field_params = predictions.get('field_parameters', {})
        
        total_reg = torch.tensor(0.0, requires_grad=True)
        
        # Regularize field parameters
        for param_name, param_value in field_params.items():
            if isinstance(param_value, torch.Tensor):
                # Prevent extreme values
                total_reg = total_reg + torch.mean(param_value ** 2)
        
        return total_reg
    
    def _transaction_cost_loss(self, predictions: Dict) -> torch.Tensor:
        """
        Penalize excessive trading (realistic transaction costs)
        """
        positions = predictions.get('positions', torch.zeros(1))
        
        if len(positions) > 1:
            # Position changes trigger transaction costs
            position_changes = torch.abs(positions[1:] - positions[:-1])
            transaction_cost_rate = 0.001  # 10 basis points per trade
            
            total_costs = torch.sum(position_changes) * transaction_cost_rate
            return total_costs
        else:
            return torch.tensor(0.0, requires_grad=True)

class FinancialTrainingObjectives:
    """
    Defines what the model should learn for financial applications
    """
    
    def __init__(self):
        self.objectives = {
            'primary': self._primary_objectives(),
            'secondary': self._secondary_objectives(),
            'constraints': self._constraints()
        }
    
    def _primary_objectives(self) -> Dict:
        """What the model must learn well"""
        return {
            'risk_management': {
                'description': 'Maximize risk-adjusted returns (Sharpe ratio)',
                'success_metric': 'Sharpe ratio > 1.0',
                'why_important': 'Core goal of any investment strategy'
            },
            
            'uncertainty_quantification': {
                'description': 'Know when predictions are reliable vs uncertain',
                'success_metric': 'High confidence correlates with low error',
                'why_important': 'Prevents overconfident bad decisions'
            },
            
            'capital_preservation': {
                'description': 'Avoid large losses, preserve capital',
                'success_metric': 'Maximum drawdown < 15%',
                'why_important': 'Staying alive is more important than beating benchmarks'
            }
        }
    
    def _secondary_objectives(self) -> Dict:
        """What the model should learn reasonably well"""
        return {
            'regime_adaptation': {
                'description': 'Adapt strategy to different market regimes',
                'success_metric': 'Different behavior in bull vs bear markets',
                'why_important': 'Markets change, strategies must adapt'
            },
            
            'prediction_accuracy': {
                'description': 'Predict returns with reasonable accuracy',
                'success_metric': 'MSE better than random walk',
                'why_important': 'Some predictive power needed for active management'
            },
            
            'transaction_efficiency': {
                'description': 'Minimize unnecessary trading',
                'success_metric': 'High information ratio despite transaction costs',
                'why_important': 'Trading costs eat into returns'
            }
        }
    
    def _constraints(self) -> Dict:
        """Hard constraints the model must satisfy"""
        return {
            'position_limits': {
                'description': 'Position sizes must be reasonable',
                'constraint': 'Positions between -1 and +1 (100% long/short)',
                'why_important': 'Prevent leverage disasters'
            },
            
            'stability': {
                'description': 'Model behavior must be stable over time',
                'constraint': 'No sudden strategy changes without market justification',
                'why_important': 'Erratic behavior indicates overfitting'
            },
            
            'interpretability': {
                'description': 'Decisions must be explainable',
                'constraint': 'Can trace decision back to market conditions',
                'why_important': 'Required for regulatory compliance and risk management'
            }
        }

def realistic_performance_targets():
    """
    Set realistic performance targets based on industry standards
    """
    return {
        'minimum_viable': {
            'annual_return': 0.08,      # Beat inflation
            'sharpe_ratio': 0.5,        # Reasonable risk adjustment
            'max_drawdown': 0.20,       # Manageable losses
            'hit_rate': 0.52            # Slight edge over random
        },
        
        'good_performance': {
            'annual_return': 0.12,      # Beat market
            'sharpe_ratio': 0.8,        # Good risk adjustment
            'max_drawdown': 0.15,       # Limited losses
            'hit_rate': 0.55            # Clear edge
        },
        
        'excellent_performance': {
            'annual_return': 0.18,      # Top tier returns
            'sharpe_ratio': 1.2,        # Excellent risk adjustment
            'max_drawdown': 0.10,       # Minimal losses
            'hit_rate': 0.60            # Strong edge
        },
        
        'world_class': {
            'annual_return': 0.25,      # Elite level
            'sharpe_ratio': 1.8,        # World-class risk adjustment
            'max_drawdown': 0.08,       # Exceptional risk control
            'hit_rate': 0.65            # Exceptional edge
        }
    }

if __name__ == "__main__":
    print("=== REALISTIC FINANCIAL TRAINING OBJECTIVES ===")
    
    objectives = FinancialTrainingObjectives()
    targets = realistic_performance_targets()
    
    print("\n1. PRIMARY OBJECTIVES (Must Master):")
    for name, obj in objectives.objectives['primary'].items():
        print(f"   {name}: {obj['description']}")
        print(f"      Success: {obj['success_metric']}")
    
    print("\n2. SECONDARY OBJECTIVES (Should Learn):")
    for name, obj in objectives.objectives['secondary'].items():
        print(f"   {name}: {obj['description']}")
        print(f"      Success: {obj['success_metric']}")
    
    print("\n3. REALISTIC PERFORMANCE TARGETS:")
    for level, metrics in targets.items():
        print(f"   {level.upper()}:")
        print(f"      Return: {metrics['annual_return']:.1%}")
        print(f"      Sharpe: {metrics['sharpe_ratio']:.1f}")
        print(f"      Hit Rate: {metrics['hit_rate']:.1%}")
    
    print("\n4. KEY INSIGHTS:")
    print("   - 98% accuracy is impossible and unnecessary")
    print("   - Risk management > raw returns")
    print("   - Uncertainty quantification is crucial")
    print("   - Transaction costs matter significantly")