"""
Realistic Market Simulation for Financial AI Testing
==================================================

This replaces the synthetic market data with realistic characteristics:
- Proper market microstructure
- Realistic return distributions  
- Non-predictable regime changes
- Proper evaluation metrics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import scipy.stats as stats
from enum import Enum

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear" 
    VOLATILE = "volatile"
    SIDEWAYS = "sideways"

@dataclass
class RealisticMarketConfig:
    """Configuration for realistic market simulation"""
    
    # Market microstructure
    tick_size: float = 0.01
    bid_ask_spread: float = 0.001  # 10 basis points
    market_impact_coefficient: float = 0.0001
    
    # Return characteristics (based on real market data)
    annual_drift: float = 0.08  # 8% expected annual return
    annual_volatility: float = 0.16  # 16% annual volatility
    fat_tail_parameter: float = 4.0  # Student t-distribution parameter
    
    # Regime characteristics
    regime_persistence: float = 0.98  # 98% chance of staying in same regime
    regime_transition_matrix: Dict = None  # Will be initialized
    
    # Market hours and holidays
    trading_days_per_year: int = 252
    intraday_periods: int = 390  # 6.5 hours * 60 minutes
    
    # Autocorrelation and clustering
    return_autocorr: float = -0.05  # Slight negative autocorr (mean reversion)
    volatility_clustering: float = 0.9  # GARCH-like volatility persistence
    
    def __post_init__(self):
        """Initialize regime transition matrix if not provided"""
        if self.regime_transition_matrix is None:
            # Realistic regime transition probabilities
            self.regime_transition_matrix = {
                MarketRegime.BULL: {
                    MarketRegime.BULL: 0.95,
                    MarketRegime.SIDEWAYS: 0.03,
                    MarketRegime.VOLATILE: 0.015,
                    MarketRegime.BEAR: 0.005
                },
                MarketRegime.BEAR: {
                    MarketRegime.BEAR: 0.92,
                    MarketRegime.VOLATILE: 0.05,
                    MarketRegime.SIDEWAYS: 0.025,
                    MarketRegime.BULL: 0.005
                },
                MarketRegime.VOLATILE: {
                    MarketRegime.VOLATILE: 0.85,
                    MarketRegime.BULL: 0.08,
                    MarketRegime.BEAR: 0.05,
                    MarketRegime.SIDEWAYS: 0.02
                },
                MarketRegime.SIDEWAYS: {
                    MarketRegime.SIDEWAYS: 0.90,
                    MarketRegime.BULL: 0.05,
                    MarketRegime.VOLATILE: 0.03,
                    MarketRegime.BEAR: 0.02
                }
            }

class RealisticMarketSimulator:
    """Generates realistic market data with proper statistical properties"""
    
    def __init__(self, config: RealisticMarketConfig, random_seed: int = None):
        self.config = config
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Current market state
        self.current_regime = MarketRegime.BULL
        self.current_price = 100.0
        self.current_volatility = config.annual_volatility / np.sqrt(config.trading_days_per_year)
        self.previous_return = 0.0
        
        # Market microstructure
        self.bid_price = self.current_price - config.bid_ask_spread/2
        self.ask_price = self.current_price + config.bid_ask_spread/2
        self.order_flow = 0.0
        
        # History tracking
        self.price_history = [self.current_price]
        self.return_history = []
        self.volatility_history = [self.current_volatility]
        self.regime_history = [self.current_regime]
        self.volume_history = []
        
        # Economic indicators (random walk with mean reversion)
        self.unemployment_rate = 4.0
        self.inflation_rate = 2.0
        self.interest_rate = 2.5
        
    def step(self) -> Dict:
        """Generate one time step of realistic market data"""
        
        # 1. Update regime (Markov chain)
        self._update_regime()
        
        # 2. Update economic indicators
        self._update_economic_indicators()
        
        # 3. Generate return with realistic properties
        regime_params = self._get_regime_parameters()
        raw_return = self._generate_return(regime_params)
        
        # 4. Apply market microstructure effects
        market_return = self._apply_microstructure_effects(raw_return)
        
        # 5. Update price and market state
        self.current_price *= (1 + market_return)
        self.previous_return = market_return
        self.return_history.append(market_return)
        self.price_history.append(self.current_price)
        
        # 6. Update volatility (GARCH-like process)
        self._update_volatility(market_return)
        
        # 7. Generate volume (correlated with volatility and |returns|)
        volume = self._generate_volume(market_return)
        self.volume_history.append(volume)
        
        # 8. Update bid/ask
        self._update_bid_ask()
        
        return self._create_market_state()
    
    def _update_regime(self):
        """Update market regime using Markov chain"""
        transition_probs = self.config.regime_transition_matrix[self.current_regime]
        regimes = list(transition_probs.keys())
        probabilities = list(transition_probs.values())
        
        self.current_regime = np.random.choice(regimes, p=probabilities)
        self.regime_history.append(self.current_regime)
    
    def _get_regime_parameters(self) -> Dict:
        """Get parameters for current regime"""
        base_drift = self.config.annual_drift / self.config.trading_days_per_year
        base_vol = self.config.annual_volatility / np.sqrt(self.config.trading_days_per_year)
        
        if self.current_regime == MarketRegime.BULL:
            return {'drift': base_drift * 1.5, 'vol_multiplier': 0.8, 'skew': 0.3}
        elif self.current_regime == MarketRegime.BEAR:
            return {'drift': base_drift * -2.0, 'vol_multiplier': 1.4, 'skew': -0.5}
        elif self.current_regime == MarketRegime.VOLATILE:
            return {'drift': base_drift * 0.5, 'vol_multiplier': 2.0, 'skew': 0.0}
        else:  # SIDEWAYS
            return {'drift': base_drift * 0.2, 'vol_multiplier': 0.6, 'skew': 0.0}
    
    def _generate_return(self, regime_params: Dict) -> float:
        """Generate return with fat tails and regime-dependent characteristics"""
        
        # Base return from Student t-distribution (fat tails)
        base_return = stats.t.rvs(df=self.config.fat_tail_parameter, 
                                 loc=regime_params['drift'],
                                 scale=self.current_volatility * regime_params['vol_multiplier'])
        
        # Add autocorrelation (mean reversion)
        autocorr_effect = self.config.return_autocorr * self.previous_return
        
        # Add skewness for regime
        if regime_params['skew'] != 0:
            skew_adjustment = stats.skewnorm.rvs(a=regime_params['skew'], scale=0.001)
            base_return += skew_adjustment
        
        return base_return + autocorr_effect
    
    def _update_volatility(self, return_val: float):
        """Update volatility using GARCH-like process"""
        # GARCH(1,1) approximation
        alpha = 1 - self.config.volatility_clustering  # News impact
        beta = self.config.volatility_clustering        # Persistence
        
        long_run_vol = self.config.annual_volatility / np.sqrt(self.config.trading_days_per_year)
        
        self.current_volatility = np.sqrt(
            (1 - alpha - beta) * long_run_vol**2 +
            alpha * return_val**2 +
            beta * self.current_volatility**2
        )
        
        self.volatility_history.append(self.current_volatility)
    
    def _apply_microstructure_effects(self, raw_return: float) -> float:
        """Apply market microstructure effects (bid-ask bounce, market impact)"""
        
        # Market impact (large moves have cost)
        impact_cost = self.config.market_impact_coefficient * abs(raw_return)
        
        # Bid-ask bounce (random noise from spread)
        bid_ask_noise = np.random.uniform(-self.config.bid_ask_spread/2, 
                                         self.config.bid_ask_spread/2)
        
        # Order flow pressure (momentum/contrarian effects)
        self.order_flow = 0.9 * self.order_flow + 0.1 * np.sign(raw_return)
        flow_effect = 0.0001 * self.order_flow
        
        return raw_return - impact_cost + bid_ask_noise + flow_effect
    
    def _generate_volume(self, return_val: float) -> float:
        """Generate trading volume correlated with volatility and |returns|"""
        base_volume = 1000000
        
        # Volume increases with volatility and absolute returns
        vol_effect = (self.current_volatility / 0.01) ** 0.5
        return_effect = (abs(return_val) / 0.01) ** 0.3
        
        # Add regime effect
        regime_multipliers = {
            MarketRegime.BULL: 1.0,
            MarketRegime.BEAR: 1.3,
            MarketRegime.VOLATILE: 1.8,
            MarketRegime.SIDEWAYS: 0.7
        }
        
        regime_effect = regime_multipliers[self.current_regime]
        
        # Log-normal noise
        noise = np.random.lognormal(0, 0.3)
        
        return base_volume * vol_effect * return_effect * regime_effect * noise
    
    def _update_bid_ask(self):
        """Update bid and ask prices"""
        half_spread = self.config.bid_ask_spread / 2
        self.bid_price = self.current_price - half_spread
        self.ask_price = self.current_price + half_spread
    
    def _update_economic_indicators(self):
        """Update economic indicators with mean reversion"""
        # Unemployment (mean revert to 4%)
        self.unemployment_rate += np.random.normal(0, 0.05) - 0.01 * (self.unemployment_rate - 4.0)
        self.unemployment_rate = max(2.0, min(12.0, self.unemployment_rate))
        
        # Inflation (mean revert to 2%)
        self.inflation_rate += np.random.normal(0, 0.1) - 0.02 * (self.inflation_rate - 2.0)
        self.inflation_rate = max(-1.0, min(6.0, self.inflation_rate))
        
        # Interest rate (follows inflation with lag)
        target_rate = max(0, self.inflation_rate + 0.5)
        self.interest_rate += 0.1 * (target_rate - self.interest_rate) + np.random.normal(0, 0.05)
        self.interest_rate = max(0.0, min(10.0, self.interest_rate))
    
    def _create_market_state(self) -> Dict:
        """Create market state dictionary with realistic features"""
        
        # Technical indicators
        prices = np.array(self.price_history[-50:])  # Last 50 periods
        returns = np.array(self.return_history[-50:]) if self.return_history else np.array([0])
        
        # Moving averages
        sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        
        # RSI approximation
        up_moves = np.sum(returns[-14:] > 0) if len(returns) >= 14 else 7
        rsi = up_moves / 14 * 100
        
        # Realized volatility
        realized_vol = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else self.current_volatility * np.sqrt(252)
        
        return {
            'price': self.current_price,
            'bid': self.bid_price,
            'ask': self.ask_price,
            'volume': self.volume_history[-1] if self.volume_history else 1000000,
            'return': self.previous_return,
            'volatility': self.current_volatility * np.sqrt(252),  # Annualized
            'regime': self.current_regime.value,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'price_to_sma10': self.current_price / sma_10 - 1,
            'price_to_sma20': self.current_price / sma_20 - 1,
            'rsi': rsi,
            'realized_volatility': realized_vol,
            'unemployment': self.unemployment_rate,
            'inflation': self.inflation_rate,
            'interest_rate': self.interest_rate,
            'order_flow': self.order_flow
        }

class RealisticEvaluationMetrics:
    """Proper evaluation metrics for financial prediction tasks"""
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = np.mean(returns) - risk_free_rate / 252
        return excess_returns / np.std(returns) * np.sqrt(252)
    
    @staticmethod
    def maximum_drawdown(prices: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return np.min(drawdown)
    
    @staticmethod
    def hit_rate(predictions: np.ndarray, actual: np.ndarray) -> float:
        """Calculate prediction hit rate (directional accuracy)"""
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actual)
        return np.mean(pred_direction == actual_direction)
    
    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        return gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if len(returns) == 0:
            return 0.0
        annual_return = np.mean(returns) * 252
        prices = np.cumprod(1 + returns)
        max_dd = abs(RealisticEvaluationMetrics.maximum_drawdown(prices))
        return annual_return / max_dd if max_dd > 0 else 0.0
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal position size using Kelly criterion"""
        if avg_loss == 0:
            return 0.0
        return win_rate - (1 - win_rate) * (avg_win / abs(avg_loss))

def create_realistic_training_data(num_days: int = 1000, config: RealisticMarketConfig = None) -> pd.DataFrame:
    """Create realistic training dataset"""
    
    if config is None:
        config = RealisticMarketConfig()
    
    simulator = RealisticMarketSimulator(config)
    
    data = []
    for day in range(num_days):
        market_state = simulator.step()
        market_state['day'] = day
        data.append(market_state)
    
    df = pd.DataFrame(data)
    
    # Add future returns for prediction targets (but make them realistic)
    # Only predict 1-5 days ahead with significant noise
    for horizon in [1, 2, 5]:
        future_returns = df['return'].shift(-horizon)
        # Add noise to make prediction harder (realistic)
        noise = np.random.normal(0, df['volatility'].std() * 0.5, len(future_returns))
        df[f'future_return_{horizon}d'] = future_returns + noise
    
    return df

if __name__ == "__main__":
    # Example usage
    config = RealisticMarketConfig()
    
    print("=== REALISTIC MARKET SIMULATION ===")
    print(f"Expected annual return: {config.annual_drift:.1%}")
    print(f"Expected annual volatility: {config.annual_volatility:.1%}")
    print(f"Fat tail parameter: {config.fat_tail_parameter}")
    
    # Generate sample data
    df = create_realistic_training_data(252)  # 1 year
    
    print(f"\nGenerated {len(df)} days of market data")
    print(f"Actual return: {df['return'].mean() * 252:.1%}")
    print(f"Actual volatility: {df['return'].std() * np.sqrt(252):.1%}")
    print(f"Sharpe ratio: {RealisticEvaluationMetrics.sharpe_ratio(df['return'].values):.2f}")
    print(f"Max drawdown: {RealisticEvaluationMetrics.maximum_drawdown(df['price'].values):.1%}")
    
    print("\nRegime distribution:")
    print(df['regime'].value_counts(normalize=True))