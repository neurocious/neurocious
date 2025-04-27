using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class BacktestEngine
    {
        private readonly FinancialGeodesicExplorer explorer;
        private readonly FinancialMetrics metrics;
        private readonly TradeExecutor executor;
        private readonly PortfolioManager portfolio;
        private readonly Dictionary<string, double> transactionCosts;
        private readonly TradingMetrics tradingMetrics;
        private readonly TechnicalAnalysis technicalAnalysis;

        public BacktestEngine(
            FinancialGeodesicExplorer explorer,
            double initialCapital = 1000000,
            Dictionary<string, double> transactionCosts = null)
        {
            this.explorer = explorer;
            this.metrics = new FinancialMetrics();
            this.executor = new TradeExecutor();
            this.portfolio = new PortfolioManager(initialCapital);
            this.transactionCosts = transactionCosts ?? new Dictionary<string, double>
            {
                ["commission"] = 0.001,  // 10 bps commission
                ["slippage"] = 0.0005,   // 5 bps slippage
                ["spread"] = 0.0002      // 2 bps spread
            };
            this.tradingMetrics = new TradingMetrics();
            this.technicalAnalysis = new TechnicalAnalysis();
        }

        public async Task<BacktestResult> RunBacktest(
            List<MarketSnapshot> historicalData,
            BacktestConfig config)
        {
            var trades = new List<Trade>();
            var portfolioHistory = new List<PortfolioSnapshot>();
            var marketStates = new List<double[]>();
            var performanceLog = new List<Dictionary<string, double>>();

            // Initialize sliding window
            var lookback = new Queue<MarketSnapshot>(config.LookbackPeriods);

            foreach (var snapshot in historicalData)
            {
                // Update market state window
                lookback.Enqueue(snapshot);
                if (lookback.Count > config.LookbackPeriods)
                    lookback.Dequeue();

                if (lookback.Count < config.LookbackPeriods)
                    continue;

                // Get trading decision
                var features = ExtractFeatures(lookback.ToList());
                marketStates.Add(features);

                var (action, confidence, metrics) = explorer.FindBestTradeDecision(features);

                // Execute trade if confidence exceeds threshold
                if (confidence > config.ConfidenceThreshold)
                {
                    var trade = executor.ExecuteTrade(
                        portfolio,
                        action,
                        snapshot.Price,
                        confidence,
                        transactionCosts);

                    if (trade != null)
                    {
                        trades.Add(trade);
                    }
                }

                // Update portfolio state
                portfolio.UpdatePortfolioValue(snapshot.Price);
                portfolioHistory.Add(portfolio.GetSnapshot());

                // Log performance metrics
                performanceLog.Add(CalculatePerformanceMetrics(
                    portfolio,
                    trades,
                    marketStates,
                    snapshot.Timestamp));

                // Check risk limits
                if (CheckRiskLimits(portfolio, config.RiskLimits))
                {
                    await HandleRiskLimitBreak(portfolio, trades);
                }
            }

            return new BacktestResult
            {
                Trades = trades,
                PortfolioHistory = portfolioHistory,
                MarketStates = marketStates,
                PerformanceLog = performanceLog,
                FinalMetrics = tradingMetrics.CalculateFinalMetrics(trades, portfolioHistory)
            };
        }

        private double[] ExtractFeatures(List<MarketSnapshot> window)
        {
            var features = new List<double>();

            // Price-based features
            features.AddRange(CalculatePriceFeatures(window));

            // Volume-based features
            features.AddRange(CalculateVolumeFeatures(window));

            // Volatility features
            features.AddRange(CalculateVolatilityFeatures(window));

            // Technical indicators
            features.AddRange(CalculateTechnicalFeatures(window));

            return features.ToArray();
        }

        private List<double> CalculatePriceFeatures(List<MarketSnapshot> window)
        {
            var prices = window.Select(w => w.Price).ToList();
            var returns = technicalAnalysis.CalculateReturns(prices);

            return new List<double>
        {
            prices.Last() / prices.Average() - 1,  // Price relative to moving average
            returns.Skip(Math.Max(0, returns.Count - 5)).Average(),  // 5-period momentum
            returns.StandardDeviation(),  // Return volatility
            prices.Last() / prices.Max() - 1,  // Distance from high
            prices.Last() / prices.Min() - 1   // Distance from low
        };
        }

        private List<double> CalculateVolumeFeatures(List<MarketSnapshot> window)
        {
            var volumes = window.Select(w => w.Volume).ToList();

            return new List<double>
        {
            volumes.Last() / volumes.Average(),  // Relative volume
            technicalAnalysis.CalculateVolumeWeightedPrice(window),  // VWAP
            technicalAnalysis.CalculateOnBalanceVolume(window)  // OBV
        };
        }

        private List<double> CalculateVolatilityFeatures(List<MarketSnapshot> window)
        {
            var prices = window.Select(w => w.Price).ToList();
            var returns = technicalAnalysis.CalculateReturns(prices);

            return new List<double>
        {
            technicalAnalysis.CalculateParkinsonVolatility(window),  // Parkinson volatility
            technicalAnalysis.CalculateGarmanKlassVolatility(window),  // Garman-Klass volatility
            returns.Select(r => Math.Abs(r)).Average(),  // Average absolute return
            returns.Select(r => r * r).Average()  // Realized variance
        };
        }

        private List<double> CalculateTechnicalFeatures(List<MarketSnapshot> window)
        {
            var prices = window.Select(w => w.Price).ToList();

            return new List<double>
        {
            technicalAnalysis.CalculateRSI(prices),
            technicalAnalysis.CalculateMACD(prices).macd,
            technicalAnalysis.CalculateBollingerPosition(prices),
            technicalAnalysis.CalculateADX(window)
        };
        }

        private Dictionary<string, double> CalculatePerformanceMetrics(
            PortfolioManager portfolio,
            List<Trade> trades,
            List<double[]> marketStates,
            DateTime timestamp)
        {
            // Create paths from market states for the explorer to analyze
            var paths = tradingMetrics.CreatePathsFromStates(marketStates);

            // Get base metrics
            var baseMetrics = metrics.CalculatePathMetrics(paths);

            // Add portfolio-specific metrics
            var portfolioMetrics = new Dictionary<string, double>
            {
                ["equity"] = portfolio.CurrentValue,
                ["drawdown"] = portfolio.CurrentDrawdown,
                ["net_exposure"] = portfolio.NetExposure,
                ["profit_factor"] = tradingMetrics.CalculateProfitFactor(trades),
                ["win_rate"] = tradingMetrics.CalculateWinRate(trades),
                ["avg_trade_return"] = tradingMetrics.CalculateAverageTradeReturn(trades),
                ["position_count"] = portfolio.OpenPositions.Count
            };

            // Combine all metrics
            return baseMetrics.Concat(portfolioMetrics)
                .ToDictionary(x => x.Key, x => x.Value);
        }

        private bool CheckRiskLimits(PortfolioManager portfolio, RiskLimits limits)
        {
            return portfolio.CurrentDrawdown > limits.MaxDrawdown ||
                   Math.Abs(portfolio.NetExposure) > limits.MaxExposure ||
                   portfolio.CurrentValue < limits.StopOutValue;
        }

        private async Task HandleRiskLimitBreak(PortfolioManager portfolio, List<Trade> trades)
        {
            // Close all positions at market
            foreach (var position in portfolio.OpenPositions)
            {
                var exitTrade = await executor.ClosePosition(position, "Risk Limit Break");
                trades.Add(exitTrade);
            }
        }
    }
}
