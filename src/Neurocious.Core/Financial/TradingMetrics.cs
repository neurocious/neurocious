using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class TradingMetrics
    {
        public double CalculateProfitFactor(List<Trade> trades)
        {
            var grossProfit = trades.Where(t => t.Size * (t.Price - t.Cost) > 0)
                                   .Sum(t => Math.Abs(t.Size * (t.Price - t.Cost)));

            var grossLoss = trades.Where(t => t.Size * (t.Price - t.Cost) < 0)
                                 .Sum(t => Math.Abs(t.Size * (t.Price - t.Cost)));

            return grossLoss == 0 ? double.MaxValue : grossProfit / grossLoss;
        }

        public double CalculateWinRate(List<Trade> trades)
        {
            if (!trades.Any()) return 0;

            int winners = trades.Count(t => t.Size * (t.Price - t.Cost) > 0);
            return (double)winners / trades.Count;
        }

        public double CalculateAverageTradeReturn(List<Trade> trades)
        {
            if (!trades.Any()) return 0;

            return trades.Average(t => t.Size * (t.Price - t.Cost) / (Math.Abs(t.Size) * t.Price));
        }

        public List<List<PradOp>> CreatePathsFromStates(List<double[]> marketStates)
        {
            var paths = new List<List<PradOp>>();
            int pathLength = 10;

            for (int i = 0; i < marketStates.Count - pathLength; i++)
            {
                var path = marketStates.Skip(i)
                                     .Take(pathLength)
                                     .Select(state => new PradOp(new Tensor(new[] { state.Length }, state)))
                                     .ToList();
                paths.Add(path);
            }

            return paths;
        }

        public Dictionary<string, double> CalculateFinalMetrics(
            List<Trade> trades,
            List<PortfolioSnapshot> portfolioHistory)
        {
            var metrics = new Dictionary<string, double>();

            // Trading metrics
            metrics["profit_factor"] = CalculateProfitFactor(trades);
            metrics["win_rate"] = CalculateWinRate(trades);
            metrics["avg_trade_return"] = CalculateAverageTradeReturn(trades);
            metrics["total_trades"] = trades.Count;

            // Portfolio metrics
            if (portfolioHistory.Any())
            {
                var returns = CalculatePortfolioReturns(portfolioHistory);
                metrics["total_return"] = (portfolioHistory.Last().Value / portfolioHistory.First().Value) - 1;
                metrics["annualized_return"] = CalculateAnnualizedReturn(metrics["total_return"], portfolioHistory.Count);
                metrics["max_drawdown"] = CalculateMaxDrawdown(portfolioHistory);
                metrics["sharpe_ratio"] = CalculateSharpeRatio(returns);
                metrics["volatility"] = CalculateVolatility(returns);
            }

            return metrics;
        }

        private List<double> CalculatePortfolioReturns(List<PortfolioSnapshot> history)
        {
            var returns = new List<double>();
            for (int i = 1; i < history.Count; i++)
            {
                returns.Add(history[i].Value / history[i - 1].Value - 1);
            }
            return returns;
        }

        private double CalculateAnnualizedReturn(double totalReturn, int periods)
        {
            return Math.Pow(1 + totalReturn, 252.0 / periods) - 1; // Assuming daily data
        }

        private double CalculateMaxDrawdown(List<PortfolioSnapshot> history)
        {
            double maxDrawdown = 0;
            double peak = history[0].Value;

            foreach (var snapshot in history)
            {
                if (snapshot.Value > peak)
                    peak = snapshot.Value;

                double drawdown = (peak - snapshot.Value) / peak;
                maxDrawdown = Math.Max(maxDrawdown, drawdown);
            }

            return maxDrawdown;
        }

        private double CalculateSharpeRatio(List<double> returns, double riskFreeRate = 0.02)
        {
            if (!returns.Any()) return 0;

            double excessReturn = returns.Average() - (riskFreeRate / 252); // Daily risk-free rate
            double volatility = CalculateVolatility(returns);

            return volatility == 0 ? 0 : (excessReturn / volatility) * Math.Sqrt(252); // Annualized
        }

        private double CalculateVolatility(List<double> returns)
        {
            if (!returns.Any()) return 0;

            double mean = returns.Average();
            double sumSquaredDeviations = returns.Sum(r => Math.Pow(r - mean, 2));
            return Math.Sqrt(sumSquaredDeviations / returns.Count) * Math.Sqrt(252); // Annualized
        }
    }
}
