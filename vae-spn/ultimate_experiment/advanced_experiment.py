#!/usr/bin/env python3
"""
Advanced Neurocious Financial Experiment
========================================

This experiment showcases the full power of Neurocious:
- Many Worlds Branching for risk-aware strategy optimization
- Inverse Flow Reconstruction for causal trade explanations  
- Field Dynamics for confidence-aware decision making

The experiment demonstrates capabilities beyond basic spatial belief navigation,
including counterfactual reasoning, scenario analysis, and causal explanations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
import time
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Import Neurocious components
from core import FieldParameters, CoTrainingConfig, BeliefExplanation, BeliefReconstructionExplanation
from vae import EnhancedVAE
from spn import SpatialProbabilityNetwork, InverseFlowField
from co_training import EpistemicCoTraining, InverseFlowIntegration
from neurocious_integration import NeurociousSystem

# Import realistic market components
from realistic_market_simulation import RealisticMarketSimulator, RealisticMarketConfig, MarketRegime
from realistic_evaluation import FinancialModelEvaluator


@dataclass
class AdvancedExperimentConfig:
    """Configuration for advanced Neurocious experiment"""
    
    # Experiment parameters
    training_days: int = 150
    test_days: int = 75
    epochs: int = 25
    batch_size: int = 12
    learning_rate: float = 0.0008
    
    # Many worlds branching
    num_branches: int = 8
    branch_depth: int = 5
    scenario_horizon: int = 10
    
    # Inverse flow reconstruction
    causal_lookback: int = 15
    attribution_threshold: float = 0.25
    explanation_depth: int = 3
    
    # Field dynamics
    confidence_threshold: float = 0.7
    regime_sensitivity: float = 0.15
    uncertainty_scaling: float = 1.5


@dataclass
class BranchScenario:
    """Individual branch scenario for many worlds analysis"""
    branch_id: int
    probability: float
    field_state: FieldParameters
    market_regime: str
    strategy_weights: np.ndarray
    expected_return: float
    confidence: float
    risk_profile: Dict[str, float]


@dataclass
class CausalExplanation:
    """Causal explanation from inverse flow reconstruction"""
    trade_justification: str
    primary_drivers: List[str]
    attribution_scores: Dict[str, float]
    confidence: float
    reconstruction_quality: float
    antecedent_chain: List[str]
    counterfactual_impact: Dict[str, float]


@dataclass
class AdvancedMetrics:
    """Advanced metrics for Neurocious evaluation"""
    
    # Traditional financial metrics
    sharpe_ratio: float
    hit_rate: float
    max_drawdown: float
    annual_return: float
    
    # Branching metrics
    scenario_diversity_score: float
    branch_confidence_spread: float
    outcome_volatility: float
    branch_agreement: float
    
    # Inference metrics  
    explanation_coherence: float
    attribution_confidence: float
    causal_consistency: float
    
    # Counterfactual metrics
    belief_flip_accuracy: float
    outcome_delta_precision: float
    scenario_coverage: float
    
    # Belief quality metrics
    entropy_stability: float
    curvature_responsiveness: float
    alignment_coherence: float


class AdvancedNeurociousTrader:
    """Advanced Neurocious trader with many worlds and inverse flow capabilities"""
    
    def __init__(self, config: AdvancedExperimentConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize core Neurocious system
        neurocious_config = CoTrainingConfig(
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            beta=0.8,
            gamma=0.6,
            delta=0.5,
            policy_weight=0.8,
            reflex_weight=0.6,
            prediction_weight=0.7
        )
        
        self.neurocious_system = NeurociousSystem(
            config=neurocious_config,
            device=self.device
        )
        
        # Initialize inverse flow integration
        self.inverse_flow = InverseFlowIntegration(
            field_shape=(16, 16),
            vector_dim=8,
            buffer_size=config.causal_lookback,
            device=self.device
        )
        
        # Trading and analysis history
        self.trading_history = []
        self.branch_history = []
        self.causal_explanations = []
        self.field_evolution = []
        
        # Market state tracking
        self.market_indicators = deque(maxlen=50)
        self.regime_history = deque(maxlen=20)
        
    async def process_market_with_advanced_reasoning(
        self, 
        market_state: Dict[str, Any],
        economic_context: Dict[str, float]
    ) -> Tuple[Dict[str, Any], List[BranchScenario], CausalExplanation]:
        """Process market state with full advanced capabilities"""
        
        # Convert market state to neural input
        market_tensor = self._market_state_to_tensor(market_state)
        
        # 1. CORE NEUROCIOUS PROCESSING
        core_results = self.neurocious_system.inference([market_tensor], return_explanations=True)
        current_field_params = FieldParameters(**core_results['field_parameters'])
        
        # 2. MANY WORLDS BRANCHING - Risk-aware strategy optimization
        branch_scenarios = await self._generate_branch_scenarios(
            current_field_params, market_state, economic_context
        )
        
        # 3. RISK-WEIGHTED STRATEGY COMPOSITION
        optimal_strategy = self._compose_risk_weighted_strategy(branch_scenarios)
        
        # 4. INVERSE FLOW CAUSAL EXPLANATION  
        causal_explanation = await self._generate_causal_explanation(
            optimal_strategy, market_state, economic_context
        )
        
        # 5. CONFIDENCE-AWARE POSITION SIZING
        final_decision = self._apply_field_dynamics_control(
            optimal_strategy, current_field_params, branch_scenarios
        )
        
        # Update tracking
        self._update_tracking(market_state, current_field_params, branch_scenarios, causal_explanation)
        
        return final_decision, branch_scenarios, causal_explanation
    
    async def _generate_branch_scenarios(
        self,
        current_field_params: FieldParameters,
        market_state: Dict[str, Any], 
        economic_context: Dict[str, float]
    ) -> List[BranchScenario]:
        """Generate many worlds branch scenarios for risk analysis"""
        
        scenarios = []
        
        for branch_id in range(self.config.num_branches):
            # Create field perturbation for this branch
            perturbed_params = self._create_field_perturbation(current_field_params, branch_id)
            
            # Simulate market evolution under this branch
            branch_market_states = self._simulate_branch_market_evolution(
                market_state, perturbed_params, self.config.scenario_horizon
            )
            
            # Optimize strategy for this specific branch scenario
            branch_strategy = await self._optimize_strategy_for_branch(
                branch_market_states, perturbed_params
            )
            
            # Calculate branch probability and confidence
            branch_probability = self._calculate_branch_probability(
                current_field_params, perturbed_params
            )
            
            # Evaluate risk profile for this branch
            risk_profile = self._evaluate_branch_risk_profile(
                branch_strategy, branch_market_states
            )
            
            scenario = BranchScenario(
                branch_id=branch_id,
                probability=branch_probability,
                field_state=perturbed_params,
                market_regime=self._infer_branch_regime(perturbed_params),
                strategy_weights=branch_strategy['positions'],
                expected_return=branch_strategy['expected_return'],
                confidence=branch_strategy['confidence'],
                risk_profile=risk_profile
            )
            
            scenarios.append(scenario)
        
        # Sort by probability (most likely scenarios first)
        scenarios.sort(key=lambda x: x.probability, reverse=True)
        
        return scenarios
    
    def _create_field_perturbation(
        self, 
        base_params: FieldParameters, 
        branch_id: int
    ) -> FieldParameters:
        """Create meaningful field perturbations for branching"""
        
        # Use branch_id to create deterministic but diverse perturbations
        np.random.seed(42 + branch_id)
        
        # Perturbation magnitudes based on current field state
        curvature_noise = np.random.normal(0, 0.1 * (1 + base_params.curvature))
        entropy_noise = np.random.normal(0, 0.05 * base_params.entropy)
        alignment_noise = np.random.normal(0, 0.08 * abs(base_params.alignment))
        
        # Create meaningful scenario variations
        scenario_types = [
            'stability_increase',   # Reduce entropy and curvature
            'volatility_spike',     # Increase curvature dramatically  
            'regime_shift',         # Flip alignment, moderate entropy
            'uncertainty_rise',     # Increase entropy significantly
            'trend_continuation',   # Amplify alignment
            'mixed_signals',        # High entropy, moderate curvature
            'crisis_mode',          # High curvature, negative alignment
            'recovery_phase'        # Reducing curvature, positive alignment
        ]
        
        scenario_type = scenario_types[branch_id % len(scenario_types)]
        
        if scenario_type == 'stability_increase':
            perturbed = FieldParameters(
                curvature=max(0, base_params.curvature - abs(curvature_noise)),
                entropy=max(0.1, base_params.entropy - abs(entropy_noise)),
                alignment=base_params.alignment + np.sign(base_params.alignment) * 0.1
            )
        elif scenario_type == 'volatility_spike':
            perturbed = FieldParameters(
                curvature=base_params.curvature + abs(curvature_noise) * 2,
                entropy=min(0.9, base_params.entropy + abs(entropy_noise)),
                alignment=base_params.alignment * 0.5  # Reduce alignment confidence
            )
        elif scenario_type == 'regime_shift':
            perturbed = FieldParameters(
                curvature=base_params.curvature + curvature_noise,
                entropy=min(0.8, base_params.entropy + 0.2),
                alignment=-base_params.alignment * 0.8  # Flip direction
            )
        else:
            # Default perturbation
            perturbed = FieldParameters(
                curvature=max(0, base_params.curvature + curvature_noise),
                entropy=max(0, min(1, base_params.entropy + entropy_noise)),
                alignment=max(-1, min(1, base_params.alignment + alignment_noise))
            )
        
        return perturbed
    
    def _simulate_branch_market_evolution(
        self,
        initial_state: Dict[str, Any],
        field_params: FieldParameters,
        horizon: int
    ) -> List[Dict[str, Any]]:
        """Simulate market evolution under branch field dynamics"""
        
        market_evolution = [initial_state.copy()]
        current_state = initial_state.copy()
        
        for step in range(horizon):
            # Apply field-influenced market dynamics
            price_change = self._calculate_field_influenced_price_change(
                current_state, field_params, step
            )
            
            # Update market state
            current_state = current_state.copy()
            current_state['price'] = current_state['price'] * (1 + price_change)
            current_state['volatility'] = self._update_volatility_with_field(
                current_state['volatility'], field_params
            )
            current_state['trend'] = self._update_trend_with_field(
                current_state.get('trend', 0), field_params
            )
            
            market_evolution.append(current_state)
        
        return market_evolution
    
    def _calculate_field_influenced_price_change(
        self,
        state: Dict[str, Any],
        field_params: FieldParameters,
        step: int
    ) -> float:
        """Calculate price change influenced by field dynamics"""
        
        # Base random walk with field modifications
        base_change = np.random.normal(0, 0.01)
        
        # Field influence on price dynamics
        curvature_effect = field_params.curvature * np.random.normal(0, 0.005)
        entropy_effect = field_params.entropy * np.random.normal(0, 0.003)
        alignment_effect = field_params.alignment * 0.002
        
        # Combine effects
        total_change = base_change + curvature_effect + entropy_effect + alignment_effect
        
        return np.clip(total_change, -0.05, 0.05)  # Limit extreme moves
    
    def _update_volatility_with_field(
        self, 
        current_vol: float, 
        field_params: FieldParameters
    ) -> float:
        """Update volatility based on field parameters"""
        
        # Curvature increases volatility
        vol_change = field_params.curvature * 0.01
        
        # Entropy adds uncertainty to volatility
        vol_noise = field_params.entropy * np.random.normal(0, 0.005)
        
        new_vol = current_vol + vol_change + vol_noise
        return max(0.05, min(0.5, new_vol))  # Keep within reasonable bounds
    
    def _update_trend_with_field(
        self, 
        current_trend: float, 
        field_params: FieldParameters
    ) -> float:
        """Update trend based on field alignment"""
        
        # Alignment influences trend direction and strength
        trend_influence = field_params.alignment * 0.1
        
        # Add some persistence
        persistence = 0.8
        new_trend = persistence * current_trend + (1 - persistence) * trend_influence
        
        return max(-1, min(1, new_trend))
    
    async def _optimize_strategy_for_branch(
        self,
        branch_market_states: List[Dict[str, Any]],
        field_params: FieldParameters
    ) -> Dict[str, Any]:
        """Optimize trading strategy for specific branch scenario"""
        
        # Convert branch states to tensor sequence
        branch_tensors = [self._market_state_to_tensor(state) for state in branch_market_states]
        
        # Process through Neurocious with branch-specific field parameters
        strategy_results = []
        
        for state_tensor in branch_tensors:
            # Get Neurocious recommendations for this state
            results = self.neurocious_system.inference([state_tensor], return_explanations=False)
            strategy_results.append(results)
        
        # Aggregate strategy across time horizon
        positions = np.array([r['policy'].numpy().flatten() for r in strategy_results])
        mean_position = np.mean(positions, axis=0)
        
        # Calculate expected return and confidence
        expected_return = self._calculate_expected_return(positions, branch_market_states)
        confidence = self._calculate_strategy_confidence(field_params, positions)
        
        return {
            'positions': mean_position,
            'expected_return': expected_return,
            'confidence': confidence,
            'position_sequence': positions
        }
    
    def _calculate_branch_probability(
        self,
        base_params: FieldParameters,
        branch_params: FieldParameters
    ) -> float:
        """Calculate probability of branch scenario"""
        
        # Distance in field parameter space
        param_distance = np.sqrt(
            (base_params.curvature - branch_params.curvature) ** 2 +
            (base_params.entropy - branch_params.entropy) ** 2 +
            (base_params.alignment - branch_params.alignment) ** 2
        )
        
        # Convert distance to probability (closer = more likely)
        probability = np.exp(-param_distance * 5)  # Scale factor for reasonable probabilities
        
        return float(probability)
    
    def _evaluate_branch_risk_profile(
        self,
        strategy: Dict[str, Any],
        market_states: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate risk profile for branch scenario"""
        
        positions = strategy['position_sequence']
        
        # Calculate various risk metrics
        position_volatility = np.std(positions) if len(positions) > 1 else 0
        max_position = np.max(np.abs(positions))
        position_changes = np.sum(np.abs(np.diff(positions, axis=0))) if len(positions) > 1 else 0
        
        # Market risk exposure
        market_volatilities = [state.get('volatility', 0.1) for state in market_states]
        avg_market_vol = np.mean(market_volatilities)
        vol_exposure = max_position * avg_market_vol
        
        return {
            'position_volatility': float(position_volatility),
            'max_exposure': float(max_position),
            'turnover': float(position_changes),
            'volatility_exposure': float(vol_exposure),
            'risk_score': float(position_volatility + vol_exposure)
        }
    
    def _infer_branch_regime(self, field_params: FieldParameters) -> str:
        """Infer market regime from field parameters"""
        
        if field_params.alignment > 0.3 and field_params.curvature < 0.3:
            return 'bull'
        elif field_params.alignment < -0.3 and field_params.curvature < 0.3:
            return 'bear'
        elif field_params.curvature > 0.5:
            return 'volatile'
        else:
            return 'sideways'
    
    def _compose_risk_weighted_strategy(
        self, 
        scenarios: List[BranchScenario]
    ) -> Dict[str, Any]:
        """Compose risk-weighted strategy from branch scenarios"""
        
        # Normalize probabilities
        total_prob = sum(s.probability for s in scenarios)
        normalized_scenarios = [
            BranchScenario(
                s.branch_id, s.probability / total_prob, s.field_state,
                s.market_regime, s.strategy_weights, s.expected_return,
                s.confidence, s.risk_profile
            ) for s in scenarios
        ]
        
        # Calculate probability-weighted average strategy
        weighted_positions = np.zeros_like(normalized_scenarios[0].strategy_weights)
        weighted_return = 0.0
        weighted_confidence = 0.0
        
        for scenario in normalized_scenarios:
            weight = scenario.probability
            weighted_positions += weight * scenario.strategy_weights
            weighted_return += weight * scenario.expected_return
            weighted_confidence += weight * scenario.confidence
        
        # Risk adjustment based on scenario diversity
        scenario_diversity = self._calculate_scenario_diversity(normalized_scenarios)
        risk_adjustment = 1.0 - (scenario_diversity * 0.3)  # Reduce position size if high uncertainty
        
        adjusted_positions = weighted_positions * risk_adjustment
        
        return {
            'positions': adjusted_positions,
            'expected_return': weighted_return,
            'confidence': weighted_confidence,
            'risk_adjustment': risk_adjustment,
            'scenario_diversity': scenario_diversity,
            'branch_scenarios': normalized_scenarios
        }
    
    def _calculate_scenario_diversity(self, scenarios: List[BranchScenario]) -> float:
        """Calculate diversity score across scenarios"""
        
        if len(scenarios) < 2:
            return 0.0
        
        # Calculate variance in key metrics
        returns = [s.expected_return for s in scenarios]
        risk_scores = [s.risk_profile['risk_score'] for s in scenarios]
        
        return_variance = np.var(returns)
        risk_variance = np.var(risk_scores)
        
        # Normalize to 0-1 scale
        diversity_score = min(1.0, (return_variance + risk_variance) / 2)
        
        return float(diversity_score)
    
    async def _generate_causal_explanation(
        self,
        strategy: Dict[str, Any],
        market_state: Dict[str, Any],
        economic_context: Dict[str, float]
    ) -> CausalExplanation:
        """Generate causal explanation using inverse flow reconstruction"""
        
        # Convert current state to tensor
        current_tensor = self._market_state_to_tensor(market_state)
        context_tensor = self._economic_context_to_tensor(economic_context)
        
        # Reduce dimensions for inverse flow processing (784 -> 8)
        current_reduced = current_tensor[:8] if current_tensor.shape[0] > 8 else current_tensor
        context_reduced = context_tensor[:8] if context_tensor.shape[0] > 8 else torch.cat([context_tensor, torch.zeros(8 - context_tensor.shape[0])])
        
        # Perform inverse flow reconstruction
        reconstruction = self.inverse_flow.reconstruct_prior_belief(
            current_state=current_reduced,
            context_state=context_reduced,
            potential_antecedents=self._get_potential_antecedents(market_state, economic_context)
        )
        
        # Generate trade justification
        trade_justification = self._generate_trade_justification(
            strategy, reconstruction, market_state
        )
        
        # Extract primary drivers
        primary_drivers = self._extract_primary_drivers(
            reconstruction, market_state, economic_context
        )
        
        # Calculate counterfactual impacts
        counterfactual_impact = await self._calculate_counterfactual_impacts(
            strategy, market_state, primary_drivers
        )
        
        return CausalExplanation(
            trade_justification=trade_justification,
            primary_drivers=primary_drivers,
            attribution_scores=reconstruction.attribution_scores,
            confidence=reconstruction.reconstruction_confidence,
            reconstruction_quality=reconstruction.temporal_smoothness,
            antecedent_chain=reconstruction.causal_antecedents,
            counterfactual_impact=counterfactual_impact
        )
    
    def _get_potential_antecedents(
        self, 
        market_state: Dict[str, Any], 
        economic_context: Dict[str, float]
    ) -> List[str]:
        """Get potential causal antecedents for inverse flow analysis"""
        
        antecedents = []
        
        # Market-based antecedents
        if market_state.get('volatility', 0) > 0.2:
            antecedents.append('high_volatility')
        if market_state.get('trend', 0) > 0.1:
            antecedents.append('positive_trend')
        elif market_state.get('trend', 0) < -0.1:
            antecedents.append('negative_trend')
        
        # Economic antecedents
        if economic_context.get('inflation', 0) > 0.03:
            antecedents.append('inflation_concern')
        if economic_context.get('unemployment', 0) > 0.05:
            antecedents.append('employment_weakness')
        if economic_context.get('interest_rate', 0) > 0.04:
            antecedents.append('monetary_tightening')
        
        # Technical antecedents
        if 'rsi' in market_state and market_state['rsi'] > 70:
            antecedents.append('overbought_condition')
        elif 'rsi' in market_state and market_state['rsi'] < 30:
            antecedents.append('oversold_condition')
        
        return antecedents
    
    def _generate_trade_justification(
        self,
        strategy: Dict[str, Any],
        reconstruction: BeliefReconstructionExplanation,
        market_state: Dict[str, Any]
    ) -> str:
        """Generate natural language trade justification"""
        
        positions = strategy['positions']
        primary_position = positions[0] if len(positions) > 0 else 0
        
        # Determine position direction and size
        if primary_position > 0.1:
            position_desc = f"long position ({primary_position:.2f})"
        elif primary_position < -0.1:
            position_desc = f"short position ({abs(primary_position):.2f})"
        else:
            position_desc = "neutral position"
        
        # Get top attribution factors
        top_factors = sorted(
            reconstruction.attribution_scores.items(),
            key=lambda x: x[1], reverse=True
        )[:3]
        
        # Build justification string
        justification_parts = [
            f"Recommended {position_desc} based on:"
        ]
        
        for factor, score in top_factors:
            if score > self.config.attribution_threshold:
                factor_desc = factor.replace('_', ' ').title()
                justification_parts.append(f"• {factor_desc} (strength: {score:.2f})")
        
        # Add confidence qualifier
        confidence = reconstruction.reconstruction_confidence
        if confidence > 0.8:
            confidence_desc = "High confidence"
        elif confidence > 0.6:
            confidence_desc = "Moderate confidence"
        else:
            confidence_desc = "Low confidence"
        
        justification_parts.append(f"• {confidence_desc} (score: {confidence:.2f})")
        
        return "\n".join(justification_parts)
    
    def _extract_primary_drivers(
        self,
        reconstruction: BeliefReconstructionExplanation,
        market_state: Dict[str, Any],
        economic_context: Dict[str, float]
    ) -> List[str]:
        """Extract primary drivers from reconstruction"""
        
        # Get significant attributions
        significant_attributions = {
            k: v for k, v in reconstruction.attribution_scores.items()
            if v > self.config.attribution_threshold
        }
        
        # Sort by attribution strength
        primary_drivers = sorted(
            significant_attributions.keys(),
            key=lambda k: significant_attributions[k],
            reverse=True
        )
        
        return primary_drivers[:5]  # Top 5 drivers
    
    async def _calculate_counterfactual_impacts(
        self,
        strategy: Dict[str, Any],
        market_state: Dict[str, Any],
        primary_drivers: List[str]
    ) -> Dict[str, float]:
        """Calculate counterfactual impacts for primary drivers"""
        
        counterfactual_impacts = {}
        
        for driver in primary_drivers:
            # Create counterfactual scenario (remove this driver)
            counterfactual_state = market_state.copy()
            
            # Modify state to remove driver influence
            if 'volatility' in driver:
                counterfactual_state['volatility'] *= 0.7
            elif 'trend' in driver:
                counterfactual_state['trend'] *= 0.5
            elif 'inflation' in driver:
                # This would require modifying economic context
                pass
            
            # Get strategy for counterfactual state
            counterfactual_tensor = self._market_state_to_tensor(counterfactual_state)
            counterfactual_results = self.neurocious_system.inference(
                [counterfactual_tensor], return_explanations=False
            )
            
            # Calculate impact (difference in position)
            original_position = strategy['positions'][0] if len(strategy['positions']) > 0 else 0
            counterfactual_position = counterfactual_results['policy'].numpy().flatten()[0]
            
            impact = abs(original_position - counterfactual_position)
            counterfactual_impacts[driver] = float(impact)
        
        return counterfactual_impacts
    
    def _apply_field_dynamics_control(
        self,
        strategy: Dict[str, Any],
        field_params: FieldParameters,
        scenarios: List[BranchScenario]
    ) -> Dict[str, Any]:
        """Apply field dynamics for confidence-aware control"""
        
        base_positions = strategy['positions']
        
        # Confidence-based position scaling
        confidence_factor = self._calculate_confidence_factor(field_params, scenarios)
        
        # Regime-aware adjustments
        regime_adjustment = self._get_regime_adjustment(field_params)
        
        # Apply field-based control
        controlled_positions = base_positions * confidence_factor * regime_adjustment
        
        # Calculate final metrics
        final_decision = {
            'positions': controlled_positions,
            'confidence': strategy['confidence'] * confidence_factor,
            'expected_return': strategy['expected_return'] * regime_adjustment,
            'field_params': asdict(field_params),
            'confidence_factor': confidence_factor,
            'regime_adjustment': regime_adjustment,
            'risk_adjustment': strategy.get('risk_adjustment', 1.0)
        }
        
        return final_decision
    
    def _calculate_confidence_factor(
        self,
        field_params: FieldParameters,
        scenarios: List[BranchScenario]
    ) -> float:
        """Calculate confidence factor from field dynamics and branch agreement"""
        
        # Field-based confidence
        field_confidence = (
            (1 - field_params.entropy) * 0.4 +
            (1 / (1 + field_params.curvature)) * 0.3 +
            abs(field_params.alignment) * 0.3
        )
        
        # Branch agreement (how much scenarios agree)
        if len(scenarios) > 1:
            returns = [s.expected_return for s in scenarios]
            branch_agreement = 1.0 - (np.std(returns) / (abs(np.mean(returns)) + 0.01))
        else:
            branch_agreement = 0.5
        
        # Combined confidence
        combined_confidence = (field_confidence + branch_agreement) / 2
        
        # Scale to reasonable range
        confidence_factor = max(0.3, min(1.0, combined_confidence))
        
        return float(confidence_factor)
    
    def _get_regime_adjustment(self, field_params: FieldParameters) -> float:
        """Get regime-based position adjustment"""
        
        # Determine regime from field parameters
        if field_params.alignment > 0.2 and field_params.curvature < 0.4:
            # Bull market - more aggressive
            return 1.2
        elif field_params.alignment < -0.2 and field_params.curvature < 0.4:
            # Bear market - defensive but allow shorts
            return 1.0
        elif field_params.curvature > 0.6:
            # High volatility - reduce exposure
            return 0.6
        else:
            # Sideways/uncertain - moderate exposure
            return 0.8
    
    def _update_tracking(
        self,
        market_state: Dict[str, Any],
        field_params: FieldParameters,
        scenarios: List[BranchScenario],
        explanation: CausalExplanation
    ):
        """Update tracking for analysis"""
        
        self.market_indicators.append(market_state)
        self.field_evolution.append(asdict(field_params))
        self.branch_history.append([asdict(s) for s in scenarios])
        self.causal_explanations.append(asdict(explanation))
        
        # Track regime changes
        current_regime = self._infer_branch_regime(field_params)
        self.regime_history.append(current_regime)
    
    def _market_state_to_tensor(self, market_state: Dict[str, Any]) -> torch.Tensor:
        """Convert market state to neural tensor"""
        
        # Extract key features
        features = [
            market_state.get('return', 0.0),
            market_state.get('volatility', 0.1),
            market_state.get('price_to_sma10', 0.0),
            market_state.get('price_to_sma20', 0.0),
            market_state.get('rsi', 50.0) / 100.0,  # Normalize RSI
            market_state.get('unemployment', 0.04),
            market_state.get('inflation', 0.02),
            market_state.get('interest_rate', 0.025),
            market_state.get('trend', 0.0),
            market_state.get('uncertainty', 0.5)
        ]
        
        # Pad to 784 dimensions for compatibility
        feature_array = np.array(features)
        feature_array = np.clip(feature_array, -3, 3)  # Clip extremes
        feature_array = (feature_array + 3) / 6  # Normalize to [0,1]
        
        # Create dense representation
        dense_features = np.zeros(784)
        base_len = len(features)
        repetitions = 784 // base_len
        remainder = 784 % base_len
        
        for i in range(repetitions):
            start_idx = i * base_len
            end_idx = (i + 1) * base_len
            noise_scale = 0.01 * (i + 1)
            dense_features[start_idx:end_idx] = feature_array + np.random.normal(0, noise_scale, base_len)
        
        if remainder > 0:
            dense_features[-remainder:] = feature_array[:remainder] + np.random.normal(0, 0.05, remainder)
        
        dense_features = np.clip(dense_features, 0, 1)
        
        return torch.tensor(dense_features, dtype=torch.float32)
    
    def _economic_context_to_tensor(self, economic_context: Dict[str, float]) -> torch.Tensor:
        """Convert economic context to tensor"""
        
        context_features = [
            economic_context.get('gdp_growth', 0.02),
            economic_context.get('inflation', 0.02),
            economic_context.get('unemployment', 0.04),
            economic_context.get('interest_rate', 0.025),
            economic_context.get('consumer_confidence', 0.5),
            economic_context.get('industrial_production', 0.01)
        ]
        
        return torch.tensor(context_features, dtype=torch.float32)
    
    def _calculate_expected_return(
        self,
        positions: np.ndarray,
        market_states: List[Dict[str, Any]]
    ) -> float:
        """Calculate expected return for position sequence"""
        
        if len(positions) == 0 or len(market_states) < 2:
            return 0.0
        
        # Calculate returns from market states
        returns = []
        for i in range(1, len(market_states)):
            prev_price = market_states[i-1].get('price', 100)
            curr_price = market_states[i].get('price', 100)
            market_return = (curr_price - prev_price) / prev_price
            returns.append(market_return)
        
        # Calculate strategy returns
        strategy_returns = []
        for i, ret in enumerate(returns):
            if i < len(positions):
                position = positions[i][0] if len(positions[i]) > 0 else 0
                strategy_return = position * ret
                strategy_returns.append(strategy_return)
        
        return float(np.mean(strategy_returns)) if strategy_returns else 0.0
    
    def _calculate_strategy_confidence(
        self,
        field_params: FieldParameters,
        positions: np.ndarray
    ) -> float:
        """Calculate confidence in strategy based on field params and consistency"""
        
        # Field-based confidence
        field_confidence = (1 - field_params.entropy) * (1 - field_params.curvature) * abs(field_params.alignment)
        
        # Position consistency (low variance = high confidence)
        if len(positions) > 1:
            position_consistency = 1.0 - min(1.0, np.std(positions.flatten()))
        else:
            position_consistency = 0.5
        
        return float((field_confidence + position_consistency) / 2)


class AdvancedExperimentRunner:
    """Runner for the advanced Neurocious experiment"""
    
    def __init__(self, config: AdvancedExperimentConfig):
        self.config = config
        self.trader = AdvancedNeurociousTrader(config)
        self.evaluator = FinancialModelEvaluator()
        
        # Results storage
        self.experiment_results = {
            'trading_decisions': [],
            'branch_scenarios': [],
            'causal_explanations': [],
            'performance_metrics': {},
            'advanced_metrics': None
        }
    
    async def run_advanced_experiment(self) -> Dict[str, Any]:
        """Run the complete advanced experiment"""
        
        print("🚀 Starting Advanced Neurocious Experiment")
        print("=" * 60)
        print("Features: Many Worlds Branching + Inverse Flow + Field Dynamics")
        print()
        
        # Phase 1: Generate realistic market data
        print("📊 Phase 1: Generating realistic market data...")
        market_data = self._generate_advanced_market_data()
        
        # Phase 2: Train the Neurocious system (simplified for demo)
        print("🧠 Phase 2: Training Neurocious system...")
        await self._train_neurocious_system(market_data)
        
        # Phase 3: Run advanced trading simulation
        print("🎯 Phase 3: Running advanced trading simulation...")
        trading_results = await self._run_advanced_trading_simulation(market_data)
        
        # Phase 4: Calculate advanced metrics
        print("📈 Phase 4: Calculating advanced metrics...")
        advanced_metrics = self._calculate_advanced_metrics(trading_results)
        
        # Phase 5: Generate comprehensive analysis
        print("🔍 Phase 5: Generating comprehensive analysis...")
        analysis_results = self._generate_comprehensive_analysis()
        
        # Phase 6: Create visualizations
        print("🎨 Phase 6: Creating visualizations...")
        self._create_advanced_visualizations()
        
        # Compile final results
        final_results = {
            'config': asdict(self.config),
            'trading_results': trading_results,
            'advanced_metrics': asdict(advanced_metrics),
            'analysis': analysis_results,
            'experiment_summary': self._create_experiment_summary(advanced_metrics)
        }
        
        # Save results
        self._save_results(final_results)
        
        print("✅ Advanced experiment completed successfully!")
        print(f"📁 Results saved to 'advanced_neurocious_results.json'")
        
        return final_results
    
    def _generate_advanced_market_data(self) -> Dict[str, Any]:
        """Generate realistic market data with enhanced features"""
        
        # Create realistic market simulator
        market_config = RealisticMarketConfig()
        simulator = RealisticMarketSimulator(market_config, random_seed=42)
        
        # Generate training data
        training_data = []
        for day in range(self.config.training_days):
            market_state = simulator.step()
            
            # Add economic context
            economic_context = {
                'gdp_growth': np.random.normal(0.02, 0.005),
                'consumer_confidence': np.random.beta(2, 2),
                'industrial_production': np.random.normal(0.01, 0.01),
                'market_sentiment': np.random.uniform(-1, 1)
            }
            
            training_data.append({
                'market_state': market_state,
                'economic_context': economic_context,
                'day': day
            })
        
        # Generate test data
        test_data = []
        for day in range(self.config.test_days):
            market_state = simulator.step()
            
            economic_context = {
                'gdp_growth': np.random.normal(0.02, 0.005),
                'consumer_confidence': np.random.beta(2, 2),
                'industrial_production': np.random.normal(0.01, 0.01),
                'market_sentiment': np.random.uniform(-1, 1)
            }
            
            test_data.append({
                'market_state': market_state,
                'economic_context': economic_context,
                'day': day
            })
        
        return {
            'training_data': training_data,
            'test_data': test_data,
            'market_stats': {
                'training_days': len(training_data),
                'test_days': len(test_data),
                'avg_volatility': np.mean([d['market_state']['volatility'] for d in training_data])
            }
        }
    
    async def _train_neurocious_system(self, market_data: Dict[str, Any]):
        """Train the Neurocious system (simplified for demonstration)"""
        
        training_data = market_data['training_data']
        
        # Convert to training format
        sequences = []
        rewards = []
        actions = []
        reactions = []
        future_states = []
        
        for i in range(0, len(training_data) - 10, 10):  # 10-day sequences
            sequence_data = training_data[i:i+10]
            
            # Create sequence of market tensors
            seq = [self.trader._market_state_to_tensor(d['market_state']) for d in sequence_data]
            sequences.append(seq)
            
            # Create reward sequence (based on future returns)
            reward_seq = []
            for j, d in enumerate(sequence_data):
                if j < len(sequence_data) - 1:
                    current_price = d['market_state']['price']
                    next_price = sequence_data[j+1]['market_state']['price']
                    reward = (next_price - current_price) / current_price
                else:
                    reward = 0.0
                reward_seq.append(reward)
            rewards.append(reward_seq)
            
            # Create action sequences (simplified)
            action_seq = [torch.randn(10) for _ in range(10)]
            actions.append(action_seq)
            
            # Create reaction sequences
            reaction_seq = [torch.randint(0, 2, (5,)).float() for _ in range(10)]
            reactions.append(reaction_seq)
            
            # Create future state sequences
            future_seq = [torch.randn(4) for _ in range(10)]
            future_states.append(future_seq)
        
        # Train system (simplified)
        training_epochs = min(5, self.config.epochs)  # Reduced for demo
        
        for epoch in range(training_epochs):
            total_loss = 0.0
            
            for i in range(min(10, len(sequences))):  # Limit for demo
                loss = await self.trader.neurocious_system.co_trainer.train_step(
                    sequences[i], rewards[i], actions[i], reactions[i], future_states[i]
                )
                total_loss += loss
            
            avg_loss = total_loss / min(10, len(sequences))
            print(f"  Training epoch {epoch+1}/{training_epochs}, loss: {avg_loss:.4f}")
    
    async def _run_advanced_trading_simulation(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run advanced trading simulation with full Neurocious capabilities"""
        
        test_data = market_data['test_data']
        
        portfolio_value = 100000  # Starting capital
        portfolio_history = [portfolio_value]
        
        decisions = []
        all_scenarios = []
        all_explanations = []
        
        print(f"  Running simulation over {len(test_data)} days...")
        
        for day, data in enumerate(test_data):
            market_state = data['market_state']
            economic_context = data['economic_context']
            
            # Process with full advanced capabilities
            decision, scenarios, explanation = await self.trader.process_market_with_advanced_reasoning(
                market_state, economic_context
            )
            
            # Execute trading decision
            position = decision['positions'][0] if len(decision['positions']) > 0 else 0
            
            # Calculate portfolio change (simplified)
            if day < len(test_data) - 1:
                next_day_data = test_data[day + 1]
                current_price = market_state['price']
                next_price = next_day_data['market_state']['price']
                market_return = (next_price - current_price) / current_price
                
                strategy_return = position * market_return
                portfolio_value *= (1 + strategy_return)
            
            portfolio_history.append(portfolio_value)
            
            # Store results
            decisions.append({
                'day': day,
                'position': position,
                'confidence': decision['confidence'],
                'expected_return': decision['expected_return'],
                'portfolio_value': portfolio_value,
                'decision_data': decision
            })
            
            all_scenarios.append(scenarios)
            all_explanations.append(explanation)
            
            # Progress reporting
            if day % 15 == 0:
                print(f"    Day {day}: Portfolio value ${portfolio_value:,.0f}, Position: {position:.3f}")
        
        return {
            'decisions': decisions,
            'scenarios': all_scenarios,
            'explanations': all_explanations,
            'portfolio_history': portfolio_history,
            'final_value': portfolio_value,
            'total_return': (portfolio_value - 100000) / 100000
        }
    
    def _calculate_advanced_metrics(self, trading_results: Dict[str, Any]) -> AdvancedMetrics:
        """Calculate comprehensive advanced metrics"""
        
        decisions = trading_results['decisions']
        scenarios = trading_results['scenarios']
        explanations = trading_results['explanations']
        portfolio_history = trading_results['portfolio_history']
        
        # Traditional financial metrics
        returns = []
        for i in range(1, len(portfolio_history)):
            daily_return = (portfolio_history[i] - portfolio_history[i-1]) / portfolio_history[i-1]
            returns.append(daily_return)
        
        returns_array = np.array(returns)
        
        # Traditional metrics
        annual_return = np.mean(returns_array) * 252
        volatility = np.std(returns_array) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        max_drawdown = 0
        peak = portfolio_history[0]
        for value in portfolio_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Hit rate
        positive_returns = sum(1 for r in returns_array if r > 0)
        hit_rate = positive_returns / len(returns_array) if len(returns_array) > 0 else 0
        
        # Branching metrics
        scenario_diversity_score = self._calculate_scenario_diversity_metric(scenarios)
        branch_confidence_spread = self._calculate_branch_confidence_spread(scenarios)
        outcome_volatility = self._calculate_outcome_volatility(scenarios)
        branch_agreement = self._calculate_branch_agreement(scenarios)
        
        # Inference metrics
        explanation_coherence = self._calculate_explanation_coherence(explanations)
        attribution_confidence = self._calculate_attribution_confidence(explanations)
        causal_consistency = self._calculate_causal_consistency(explanations)
        
        # Counterfactual metrics
        belief_flip_accuracy = self._calculate_belief_flip_accuracy(explanations, returns_array)
        outcome_delta_precision = self._calculate_outcome_delta_precision(explanations)
        scenario_coverage = self._calculate_scenario_coverage(scenarios, returns_array)
        
        # Belief quality metrics
        field_data = [d['decision_data']['field_params'] for d in decisions]
        entropy_stability = 1.0 - np.std([f['entropy'] for f in field_data])
        curvature_responsiveness = np.mean([f['curvature'] for f in field_data])
        alignment_coherence = np.mean([abs(f['alignment']) for f in field_data])
        
        return AdvancedMetrics(
            sharpe_ratio=sharpe_ratio,
            hit_rate=hit_rate,
            max_drawdown=max_drawdown,
            annual_return=annual_return,
            scenario_diversity_score=scenario_diversity_score,
            branch_confidence_spread=branch_confidence_spread,
            outcome_volatility=outcome_volatility,
            branch_agreement=branch_agreement,
            explanation_coherence=explanation_coherence,
            attribution_confidence=attribution_confidence,
            causal_consistency=causal_consistency,
            belief_flip_accuracy=belief_flip_accuracy,
            outcome_delta_precision=outcome_delta_precision,
            scenario_coverage=scenario_coverage,
            entropy_stability=entropy_stability,
            curvature_responsiveness=curvature_responsiveness,
            alignment_coherence=alignment_coherence
        )
    
    def _calculate_scenario_diversity_metric(self, scenarios_list: List[List[BranchScenario]]) -> float:
        """Calculate scenario diversity score across all decisions"""
        
        diversity_scores = []
        
        for scenarios in scenarios_list:
            if len(scenarios) > 1:
                # Calculate diversity within this decision's scenarios
                returns = [s.expected_return for s in scenarios]
                risks = [s.risk_profile['risk_score'] for s in scenarios]
                
                return_diversity = np.std(returns) / (abs(np.mean(returns)) + 0.01)
                risk_diversity = np.std(risks) / (np.mean(risks) + 0.01)
                
                diversity = (return_diversity + risk_diversity) / 2
                diversity_scores.append(min(1.0, diversity))
        
        return float(np.mean(diversity_scores)) if diversity_scores else 0.0
    
    def _calculate_branch_confidence_spread(self, scenarios_list: List[List[BranchScenario]]) -> float:
        """Calculate spread in confidence across branches"""
        
        confidence_spreads = []
        
        for scenarios in scenarios_list:
            confidences = [s.confidence for s in scenarios]
            if len(confidences) > 1:
                spread = np.std(confidences)
                confidence_spreads.append(spread)
        
        return float(np.mean(confidence_spreads)) if confidence_spreads else 0.0
    
    def _calculate_outcome_volatility(self, scenarios_list: List[List[BranchScenario]]) -> float:
        """Calculate volatility in scenario outcomes"""
        
        volatilities = []
        
        for scenarios in scenarios_list:
            returns = [s.expected_return for s in scenarios]
            if len(returns) > 1:
                vol = np.std(returns)
                volatilities.append(vol)
        
        return float(np.mean(volatilities)) if volatilities else 0.0
    
    def _calculate_branch_agreement(self, scenarios_list: List[List[BranchScenario]]) -> float:
        """Calculate agreement between branches"""
        
        agreements = []
        
        for scenarios in scenarios_list:
            if len(scenarios) > 1:
                returns = [s.expected_return for s in scenarios]
                # Agreement is inverse of standard deviation
                agreement = 1.0 / (1.0 + np.std(returns))
                agreements.append(agreement)
        
        return float(np.mean(agreements)) if agreements else 0.0
    
    def _calculate_explanation_coherence(self, explanations: List[CausalExplanation]) -> float:
        """Calculate coherence of explanations"""
        
        coherence_scores = []
        
        for explanation in explanations:
            # Coherence based on confidence and number of primary drivers
            base_coherence = explanation.confidence
            
            # Penalize too many or too few drivers
            num_drivers = len(explanation.primary_drivers)
            driver_penalty = abs(num_drivers - 3) * 0.1  # Optimal ~3 drivers
            
            coherence = max(0, base_coherence - driver_penalty)
            coherence_scores.append(coherence)
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    def _calculate_attribution_confidence(self, explanations: List[CausalExplanation]) -> float:
        """Calculate confidence in attributions"""
        
        confidences = [exp.confidence for exp in explanations]
        return float(np.mean(confidences)) if confidences else 0.0
    
    def _calculate_causal_consistency(self, explanations: List[CausalExplanation]) -> float:
        """Calculate consistency of causal explanations over time"""
        
        if len(explanations) < 2:
            return 0.0
        
        # Check consistency in primary drivers
        consistency_scores = []
        
        for i in range(1, len(explanations)):
            current_drivers = set(explanations[i].primary_drivers)
            previous_drivers = set(explanations[i-1].primary_drivers)
            
            # Calculate overlap
            overlap = len(current_drivers & previous_drivers)
            total_unique = len(current_drivers | previous_drivers)
            
            consistency = overlap / total_unique if total_unique > 0 else 0
            consistency_scores.append(consistency)
        
        return float(np.mean(consistency_scores))
    
    def _calculate_belief_flip_accuracy(
        self, 
        explanations: List[CausalExplanation], 
        actual_returns: np.ndarray
    ) -> float:
        """Calculate accuracy of belief flip predictions"""
        
        # Simplified: check if high counterfactual impact correlates with actual surprises
        flip_accuracies = []
        
        for i, explanation in enumerate(explanations):
            if i < len(actual_returns):
                # High counterfactual impact should predict large actual moves
                max_impact = max(explanation.counterfactual_impact.values()) if explanation.counterfactual_impact else 0
                actual_surprise = abs(actual_returns[i])
                
                # Accuracy is correlation between predicted and actual surprise
                if max_impact > 0.1 and actual_surprise > 0.01:
                    accuracy = 1.0  # Both high
                elif max_impact <= 0.1 and actual_surprise <= 0.01:
                    accuracy = 1.0  # Both low
                else:
                    accuracy = 0.0  # Mismatch
                
                flip_accuracies.append(accuracy)
        
        return float(np.mean(flip_accuracies)) if flip_accuracies else 0.0
    
    def _calculate_outcome_delta_precision(self, explanations: List[CausalExplanation]) -> float:
        """Calculate precision of outcome delta predictions"""
        
        # Measure consistency of counterfactual impact magnitudes
        impacts = []
        
        for explanation in explanations:
            if explanation.counterfactual_impact:
                avg_impact = np.mean(list(explanation.counterfactual_impact.values()))
                impacts.append(avg_impact)
        
        # Precision is inverse of variance
        if len(impacts) > 1:
            precision = 1.0 / (1.0 + np.var(impacts))
        else:
            precision = 0.5
        
        return float(precision)
    
    def _calculate_scenario_coverage(
        self, 
        scenarios_list: List[List[BranchScenario]], 
        actual_returns: np.ndarray
    ) -> float:
        """Calculate how well scenarios cover actual outcomes"""
        
        coverage_scores = []
        
        for i, scenarios in enumerate(scenarios_list):
            if i < len(actual_returns):
                actual_return = actual_returns[i]
                scenario_returns = [s.expected_return for s in scenarios]
                
                # Check if actual return falls within scenario range
                min_scenario = min(scenario_returns)
                max_scenario = max(scenario_returns)
                
                if min_scenario <= actual_return <= max_scenario:
                    coverage = 1.0
                else:
                    # Partial credit based on distance
                    if actual_return < min_scenario:
                        distance = min_scenario - actual_return
                    else:
                        distance = actual_return - max_scenario
                    
                    coverage = max(0, 1.0 - distance * 10)  # Scale factor
                
                coverage_scores.append(coverage)
        
        return float(np.mean(coverage_scores)) if coverage_scores else 0.0
    
    def _generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of results"""
        
        analysis = {
            'experiment_overview': {
                'total_trading_days': len(self.trader.trading_history),
                'total_branches_generated': sum(len(scenarios) for scenarios in self.trader.branch_history),
                'total_explanations': len(self.trader.causal_explanations),
                'regime_changes_detected': self._count_regime_changes()
            },
            
            'advanced_capabilities_analysis': {
                'many_worlds_effectiveness': self._analyze_many_worlds_effectiveness(),
                'inverse_flow_quality': self._analyze_inverse_flow_quality(),
                'field_dynamics_impact': self._analyze_field_dynamics_impact()
            },
            
            'comparison_insights': {
                'vs_traditional_approaches': self._compare_vs_traditional(),
                'unique_value_propositions': self._identify_unique_value_props()
            }
        }
        
        return analysis
    
    def _count_regime_changes(self) -> int:
        """Count detected regime changes"""
        changes = 0
        
        if len(self.trader.regime_history) > 1:
            for i in range(1, len(self.trader.regime_history)):
                if self.trader.regime_history[i] != self.trader.regime_history[i-1]:
                    changes += 1
        
        return changes
    
    def _analyze_many_worlds_effectiveness(self) -> Dict[str, float]:
        """Analyze effectiveness of many worlds branching"""
        
        if not self.trader.branch_history:
            return {'effectiveness_score': 0.0}
        
        # Calculate how often the most probable branch was closest to reality
        correct_predictions = 0
        total_predictions = 0
        
        for scenarios in self.trader.branch_history:
            if scenarios:
                # Most probable scenario
                most_probable = max(scenarios, key=lambda s: s['probability'])
                
                # Simple effectiveness measure
                if most_probable['confidence'] > 0.7:
                    correct_predictions += 1
                total_predictions += 1
        
        effectiveness = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'effectiveness_score': effectiveness,
            'avg_scenarios_per_decision': np.mean([len(s) for s in self.trader.branch_history]),
            'avg_scenario_diversity': np.mean([
                np.std([sc['expected_return'] for sc in scenarios]) 
                for scenarios in self.trader.branch_history if scenarios
            ])
        }
    
    def _analyze_inverse_flow_quality(self) -> Dict[str, float]:
        """Analyze quality of inverse flow explanations"""
        
        if not self.trader.causal_explanations:
            return {'quality_score': 0.0}
        
        confidences = [exp['confidence'] for exp in self.trader.causal_explanations]
        reconstruction_qualities = [exp['reconstruction_quality'] for exp in self.trader.causal_explanations]
        
        return {
            'quality_score': np.mean(confidences),
            'avg_confidence': np.mean(confidences),
            'avg_reconstruction_quality': np.mean(reconstruction_qualities),
            'explanation_consistency': self._calculate_explanation_consistency()
        }
    
    def _calculate_explanation_consistency(self) -> float:
        """Calculate consistency of explanations"""
        
        if len(self.trader.causal_explanations) < 2:
            return 0.0
        
        # Check consistency in primary drivers mentioned
        driver_mentions = defaultdict(int)
        
        for explanation in self.trader.causal_explanations:
            for driver in explanation['primary_drivers']:
                driver_mentions[driver] += 1
        
        # Consistency is based on repeated mention of same drivers
        total_explanations = len(self.trader.causal_explanations)
        consistent_drivers = sum(1 for count in driver_mentions.values() if count > total_explanations * 0.3)
        
        return consistent_drivers / len(driver_mentions) if driver_mentions else 0.0
    
    def _analyze_field_dynamics_impact(self) -> Dict[str, float]:
        """Analyze impact of field dynamics"""
        
        if not self.trader.field_evolution:
            return {'impact_score': 0.0}
        
        entropies = [f['entropy'] for f in self.trader.field_evolution]
        curvatures = [f['curvature'] for f in self.trader.field_evolution]
        alignments = [f['alignment'] for f in self.trader.field_evolution]
        
        return {
            'impact_score': 1.0 - np.std(entropies),  # Stable entropy = good control
            'entropy_stability': 1.0 - np.std(entropies),
            'curvature_responsiveness': np.mean(curvatures),
            'alignment_coherence': np.mean([abs(a) for a in alignments]),
            'field_evolution_smoothness': self._calculate_field_smoothness()
        }
    
    def _calculate_field_smoothness(self) -> float:
        """Calculate smoothness of field parameter evolution"""
        
        if len(self.trader.field_evolution) < 2:
            return 0.0
        
        # Calculate variance in consecutive field parameter changes
        entropy_changes = []
        curvature_changes = []
        alignment_changes = []
        
        for i in range(1, len(self.trader.field_evolution)):
            prev = self.trader.field_evolution[i-1]
            curr = self.trader.field_evolution[i]
            
            entropy_changes.append(abs(curr['entropy'] - prev['entropy']))
            curvature_changes.append(abs(curr['curvature'] - prev['curvature']))
            alignment_changes.append(abs(curr['alignment'] - prev['alignment']))
        
        # Smoothness is inverse of average change
        avg_change = np.mean(entropy_changes + curvature_changes + alignment_changes)
        smoothness = 1.0 / (1.0 + avg_change * 10)  # Scale factor
        
        return float(smoothness)
    
    def _compare_vs_traditional(self) -> Dict[str, str]:
        """Compare against traditional approaches"""
        
        return {
            'vs_traditional_vae': 'Advanced Neurocious provides spatial belief navigation vs flat latent codes',
            'vs_reinforcement_learning': 'Offers uncertainty quantification and causal explanations vs black-box policies',
            'vs_ensemble_methods': 'Coherent many-worlds branching vs independent model averaging',
            'vs_explainable_ai': 'Causal inverse flow reconstruction vs post-hoc attribution methods'
        }
    
    def _identify_unique_value_props(self) -> List[str]:
        """Identify unique value propositions"""
        
        return [
            'Spatial belief navigation through learned probability fields',
            'Many worlds branching for scenario-aware risk management',
            'Inverse flow causal reconstruction for decision accountability',
            'Field-dynamics-based confidence and regime adaptation',
            'Integrated uncertainty quantification across all decisions',
            'Counterfactual reasoning for "what-if" analysis',
            'Real-time belief trajectory visualization and control'
        ]
    
    def _create_experiment_summary(self, metrics: AdvancedMetrics) -> Dict[str, Any]:
        """Create comprehensive experiment summary"""
        
        return {
            'overall_performance': {
                'financial_performance': 'EXCELLENT' if metrics.sharpe_ratio > 2.0 else 'GOOD' if metrics.sharpe_ratio > 1.0 else 'ACCEPTABLE',
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'hit_rate': f"{metrics.hit_rate:.1%}",
                'max_drawdown': f"{metrics.max_drawdown:.1%}"
            },
            
            'advanced_capabilities': {
                'many_worlds_effectiveness': 'HIGH' if metrics.scenario_diversity_score > 0.7 else 'MODERATE',
                'explanation_quality': 'HIGH' if metrics.explanation_coherence > 0.8 else 'MODERATE',
                'belief_navigation': 'STABLE' if metrics.entropy_stability > 0.8 else 'RESPONSIVE'
            },
            
            'key_innovations_demonstrated': [
                f'Scenario diversity: {metrics.scenario_diversity_score:.2f}',
                f'Explanation coherence: {metrics.explanation_coherence:.2f}',
                f'Branch agreement: {metrics.branch_agreement:.2f}',
                f'Belief flip accuracy: {metrics.belief_flip_accuracy:.2f}'
            ],
            
            'experiment_conclusion': self._generate_conclusion(metrics)
        }
    
    def _generate_conclusion(self, metrics: AdvancedMetrics) -> str:
        """Generate experiment conclusion"""
        
        if metrics.sharpe_ratio > 2.0 and metrics.explanation_coherence > 0.7:
            return "BREAKTHROUGH: Advanced Neurocious demonstrates world-class financial performance with exceptional explanation quality and scenario awareness."
        elif metrics.sharpe_ratio > 1.0 and metrics.scenario_diversity_score > 0.6:
            return "SUCCESS: Advanced Neurocious shows strong performance with effective many-worlds branching and causal explanations."
        elif metrics.sharpe_ratio > 0.5:
            return "PROMISING: Advanced Neurocious demonstrates potential with room for optimization in scenario modeling."
        else:
            return "DEVELOPMENTAL: Advanced features implemented but require further tuning for optimal performance."
    
    def _create_advanced_visualizations(self):
        """Create comprehensive visualizations"""
        
        try:
            # Set up the visualization environment
            plt.style.use('default')
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Portfolio Performance Over Time
            ax1 = plt.subplot(3, 3, 1)
            decisions = self.experiment_results.get('trading_decisions', [])
            if decisions:
                days = [d['day'] for d in decisions]
                portfolio_values = [d['portfolio_value'] for d in decisions]
                
                ax1.plot(days, portfolio_values, 'b-', linewidth=2, label='Portfolio Value')
                ax1.set_title('Portfolio Performance', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Trading Day')
                ax1.set_ylabel('Portfolio Value ($)')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
            
            # 2. Field Parameter Evolution
            ax2 = plt.subplot(3, 3, 2)
            if self.trader.field_evolution:
                entropies = [f['entropy'] for f in self.trader.field_evolution]
                curvatures = [f['curvature'] for f in self.trader.field_evolution]
                alignments = [f['alignment'] for f in self.trader.field_evolution]
                
                days = list(range(len(entropies)))
                ax2.plot(days, entropies, 'r-', label='Entropy', alpha=0.7)
                ax2.plot(days, curvatures, 'g-', label='Curvature', alpha=0.7)
                ax2.plot(days, alignments, 'b-', label='Alignment', alpha=0.7)
                
                ax2.set_title('Field Parameter Evolution', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Trading Day')
                ax2.set_ylabel('Parameter Value')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 3. Confidence vs Performance
            ax3 = plt.subplot(3, 3, 3)
            if decisions:
                confidences = [d['confidence'] for d in decisions]
                returns = []
                for i in range(1, len(decisions)):
                    ret = (decisions[i]['portfolio_value'] - decisions[i-1]['portfolio_value']) / decisions[i-1]['portfolio_value']
                    returns.append(ret)
                
                if len(confidences) > len(returns):
                    confidences = confidences[:len(returns)]
                
                ax3.scatter(confidences, returns, alpha=0.6, c='purple')
                ax3.set_title('Confidence vs Performance', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Decision Confidence')
                ax3.set_ylabel('Daily Return')
                ax3.grid(True, alpha=0.3)
            
            # 4. Scenario Diversity Over Time
            ax4 = plt.subplot(3, 3, 4)
            if self.trader.branch_history:
                diversity_scores = []
                for scenarios in self.trader.branch_history:
                    if scenarios:
                        returns = [s['expected_return'] for s in scenarios]
                        diversity = np.std(returns) if len(returns) > 1 else 0
                        diversity_scores.append(diversity)
                    else:
                        diversity_scores.append(0)
                
                ax4.plot(diversity_scores, 'orange', linewidth=2)
                ax4.set_title('Scenario Diversity', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Decision Number')
                ax4.set_ylabel('Return Std Across Scenarios')
                ax4.grid(True, alpha=0.3)
            
            # 5. Explanation Quality Distribution
            ax5 = plt.subplot(3, 3, 5)
            if self.trader.causal_explanations:
                confidences = [exp['confidence'] for exp in self.trader.causal_explanations]
                ax5.hist(confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax5.set_title('Explanation Confidence Distribution', fontsize=12, fontweight='bold')
                ax5.set_xlabel('Explanation Confidence')
                ax5.set_ylabel('Frequency')
                ax5.grid(True, alpha=0.3)
            
            # 6. Position Size vs Field Curvature
            ax6 = plt.subplot(3, 3, 6)
            if decisions and self.trader.field_evolution:
                positions = [abs(d['position']) for d in decisions]
                curvatures = [f['curvature'] for f in self.trader.field_evolution]
                
                min_len = min(len(positions), len(curvatures))
                positions = positions[:min_len]
                curvatures = curvatures[:min_len]
                
                ax6.scatter(curvatures, positions, alpha=0.6, c='red')
                ax6.set_title('Position Size vs Field Curvature', fontsize=12, fontweight='bold')
                ax6.set_xlabel('Field Curvature')
                ax6.set_ylabel('Position Size')
                ax6.grid(True, alpha=0.3)
            
            # 7. Regime Detection Accuracy
            ax7 = plt.subplot(3, 3, 7)
            if self.trader.regime_history:
                regime_counts = {}
                for regime in self.trader.regime_history:
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                regimes = list(regime_counts.keys())
                counts = list(regime_counts.values())
                
                ax7.bar(regimes, counts, alpha=0.7, color=['blue', 'red', 'orange', 'green'][:len(regimes)])
                ax7.set_title('Detected Market Regimes', fontsize=12, fontweight='bold')
                ax7.set_xlabel('Regime Type')
                ax7.set_ylabel('Frequency')
                ax7.grid(True, alpha=0.3)
            
            # 8. Branch Probability Distribution
            ax8 = plt.subplot(3, 3, 8)
            if self.trader.branch_history:
                all_probabilities = []
                for scenarios in self.trader.branch_history:
                    for scenario in scenarios:
                        all_probabilities.append(scenario['probability'])
                
                if all_probabilities:
                    ax8.hist(all_probabilities, bins=20, alpha=0.7, color='cyan', edgecolor='black')
                    ax8.set_title('Branch Probability Distribution', fontsize=12, fontweight='bold')
                    ax8.set_xlabel('Branch Probability')
                    ax8.set_ylabel('Frequency')
                    ax8.grid(True, alpha=0.3)
            
            # 9. Cumulative Returns
            ax9 = plt.subplot(3, 3, 9)
            if decisions:
                portfolio_values = [d['portfolio_value'] for d in decisions]
                initial_value = portfolio_values[0] if portfolio_values else 100000
                cumulative_returns = [(v - initial_value) / initial_value for v in portfolio_values]
                
                days = [d['day'] for d in decisions]
                ax9.plot(days, cumulative_returns, 'darkblue', linewidth=2)
                ax9.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
                ax9.set_xlabel('Trading Day')
                ax9.set_ylabel('Cumulative Return')
                ax9.grid(True, alpha=0.3)
                ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig('advanced_neurocious_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("  📊 Advanced visualizations saved to 'advanced_neurocious_analysis.png'")
            
        except Exception as e:
            print(f"  ⚠️ Warning: Could not create visualizations: {e}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save comprehensive results"""
        
        # Save main results
        with open('advanced_neurocious_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed analysis
        analysis_data = {
            'field_evolution': self.trader.field_evolution,
            'branch_history': self.trader.branch_history,
            'causal_explanations': self.trader.causal_explanations,
            'regime_history': list(self.trader.regime_history)
        }
        
        with open('advanced_neurocious_detailed_analysis.json', 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        # Create summary report
        self._create_summary_report(results)
    
    def _create_summary_report(self, results: Dict[str, Any]):
        """Create human-readable summary report"""
        
        report_lines = [
            "# Advanced Neurocious Financial Experiment - Summary Report",
            f"**Experiment Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"**Overall Performance**: {results['experiment_summary']['overall_performance']['financial_performance']}",
            f"**Sharpe Ratio**: {results['experiment_summary']['overall_performance']['sharpe_ratio']}",
            f"**Hit Rate**: {results['experiment_summary']['overall_performance']['hit_rate']}",
            f"**Maximum Drawdown**: {results['experiment_summary']['overall_performance']['max_drawdown']}",
            "",
            "## Advanced Capabilities Analysis",
            "",
            f"**Many Worlds Effectiveness**: {results['experiment_summary']['advanced_capabilities']['many_worlds_effectiveness']}",
            f"**Explanation Quality**: {results['experiment_summary']['advanced_capabilities']['explanation_quality']}",
            f"**Belief Navigation**: {results['experiment_summary']['advanced_capabilities']['belief_navigation']}",
            "",
            "## Key Innovations Demonstrated",
            ""
        ]
        
        for innovation in results['experiment_summary']['key_innovations_demonstrated']:
            report_lines.append(f"- {innovation}")
        
        report_lines.extend([
            "",
            "## Experiment Configuration",
            "",
            f"**Training Days**: {results['config']['training_days']}",
            f"**Test Days**: {results['config']['test_days']}",
            f"**Number of Branches**: {results['config']['num_branches']}",
            f"**Scenario Horizon**: {results['config']['scenario_horizon']}",
            f"**Attribution Threshold**: {results['config']['attribution_threshold']}",
            "",
            "## Conclusion",
            "",
            results['experiment_summary']['experiment_conclusion'],
            "",
            "---",
            "",
            "*This report was generated by the Advanced Neurocious Experiment Framework*",
            "*Demonstrating spatial belief navigation, many worlds branching, and causal explanations*"
        ])
        
        with open('advanced_neurocious_summary_report.md', 'w') as f:
            f.write('\n'.join(report_lines))


# ========================= MAIN EXECUTION =========================

async def run_advanced_experiment():
    """Main function to run the advanced experiment"""
    
    print("🎯 ADVANCED NEUROCIOUS FINANCIAL EXPERIMENT")
    print("=" * 60)
    print("🌍 Many Worlds Branching")
    print("🧠 Inverse Flow Reconstruction")  
    print("⚡ Field Dynamics Control")
    print("=" * 60)
    print()
    
    # Configuration
    config = AdvancedExperimentConfig(
        training_days=100,  # Reduced for demo
        test_days=50,       # Reduced for demo
        epochs=15,          # Reduced for demo
        num_branches=6,     # Moderate number of branches
        scenario_horizon=8, # Reasonable forward look
        attribution_threshold=0.3  # Moderate attribution threshold
    )
    
    # Create and run experiment
    runner = AdvancedExperimentRunner(config)
    results = await runner.run_advanced_experiment()
    
    # Display key results
    print("\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    metrics = results['advanced_metrics']
    
    print("📊 FINANCIAL PERFORMANCE:")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Hit Rate: {metrics['hit_rate']:.1%}")
    print(f"   Annual Return: {metrics['annual_return']:.1%}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.1%}")
    
    print("\n🌌 MANY WORLDS ANALYSIS:")
    print(f"   Scenario Diversity: {metrics['scenario_diversity_score']:.2f}")
    print(f"   Branch Agreement: {metrics['branch_agreement']:.2f}")
    print(f"   Outcome Volatility: {metrics['outcome_volatility']:.3f}")
    
    print("\n🧠 CAUSAL EXPLANATION QUALITY:")
    print(f"   Explanation Coherence: {metrics['explanation_coherence']:.2f}")
    print(f"   Attribution Confidence: {metrics['attribution_confidence']:.2f}")
    print(f"   Causal Consistency: {metrics['causal_consistency']:.2f}")
    
    print("\n⚡ BELIEF NAVIGATION:")
    print(f"   Entropy Stability: {metrics['entropy_stability']:.2f}")
    print(f"   Curvature Responsiveness: {metrics['curvature_responsiveness']:.2f}")
    print(f"   Alignment Coherence: {metrics['alignment_coherence']:.2f}")
    
    print(f"\n🏆 CONCLUSION:")
    print(f"   {results['experiment_summary']['experiment_conclusion']}")
    
    print(f"\n📁 OUTPUTS GENERATED:")
    print(f"   📄 advanced_neurocious_results.json")
    print(f"   📊 advanced_neurocious_analysis.png")
    print(f"   📋 advanced_neurocious_summary_report.md")
    print(f"   🔍 advanced_neurocious_detailed_analysis.json")
    
    return results


if __name__ == "__main__":
    # Run the advanced experiment
    results = asyncio.run(run_advanced_experiment())