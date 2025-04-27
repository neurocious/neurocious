using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class FinancialMetrics
    {
        private readonly int windowSize;
        private readonly double annualizationFactor;
        private readonly double riskFreeRate;
        private readonly double annualizationTime;

        public FinancialMetrics(
            int windowSize = 20,              // Default 20-period window for rolling calcs
            double annualizationFactor = 252,  // Default to daily data
            double riskFreeRate = 0.02)        // Annual risk-free rate
        {
            this.windowSize = windowSize;
            this.annualizationFactor = annualizationFactor;
            this.annualizationTime = annualizationFactor;
            this.riskFreeRate = riskFreeRate / annualizationFactor; // Convert to per-period
        }

        public Dictionary<string, double> CalculatePathMetrics(List<PradOp> path)
        {
            var returns = CalculateReturns(path);
            var metrics = new Dictionary<string, double>();

            // Basic Return Metrics
            metrics["total_return"] = CalculateTotalReturn(returns);
            metrics["annualized_return"] = AnnualizeReturn(metrics["total_return"], returns.Count);
            metrics["volatility"] = CalculateVolatility(returns);
            metrics["annualized_volatility"] = metrics["volatility"] * Math.Sqrt(annualizationTime);

            // Risk-Adjusted Performance Metrics
            metrics["sharpe_ratio"] = CalculateSharpeRatio(returns);
            metrics["sortino_ratio"] = CalculateSortinoRatio(returns);
            metrics["calmar_ratio"] = CalculateCalmarRatio(returns);
            metrics["information_ratio"] = CalculateInformationRatio(returns);

            // Drawdown and Risk Metrics
            var (maxDrawdown, avgDrawdown, drawdownDuration) = CalculateDrawdownMetrics(returns);
            metrics["max_drawdown"] = maxDrawdown;
            metrics["average_drawdown"] = avgDrawdown;
            metrics["max_drawdown_duration"] = drawdownDuration;
            metrics["value_at_risk"] = CalculateValueAtRisk(returns, 0.95);
            metrics["conditional_var"] = CalculateConditionalVaR(returns, 0.95);
            metrics["downside_deviation"] = CalculateDownsideDeviation(returns);

            // Advanced Statistical Metrics
            var (skewness, kurtosis) = CalculateDistributionMetrics(returns);
            metrics["return_skewness"] = skewness;
            metrics["return_kurtosis"] = kurtosis;
            metrics["jarque_bera"] = CalculateJarqueBera(skewness, kurtosis, returns.Count);

            // Market Regime Metrics
            metrics["volatility_clustering"] = CalculateVolatilityClustering(returns);
            metrics["momentum_factor"] = CalculateMomentumFactor(returns);
            metrics["reversal_tendency"] = CalculateReversalTendency(returns);
            metrics["tail_dependence"] = CalculateTailDependence(returns);

            // Path Quality Metrics
            metrics["path_smoothness"] = CalculatePathSmoothness(returns);
            metrics["information_ratio"] = CalculateInformationRatio(returns);
            metrics["gain_to_pain"] = CalculateGainToPain(returns);
            metrics["ulcer_index"] = CalculateUlcerIndex(returns);

            return metrics;
        }

        public Dictionary<string, double> CalculatePathMetrics(List<List<PradOp>> paths)
        {
            var pathQualityMetrics = new PathQualityMetrics();
            var metrics = new Dictionary<string, double>();
            List<Dictionary<string, double>> dictionaries = new List<Dictionary<string, double>>();

            // Path quality metrics
            metrics["path_smoothness"] = pathQualityMetrics.CalculatePathSmoothness(paths);
            metrics["path_consistency"] = pathQualityMetrics.CalculatePathConsistency(paths);
            metrics["path_efficiency"] = pathQualityMetrics.CalculatePathEfficiency(paths);

            // Additional financial-specific path metrics could be added here
            foreach (var list in paths)
            {
                var moreMetrics = CalculatePathMetrics(list);
                dictionaries.Add(moreMetrics);
            }

            var averaged = AverageDictionariesWithLinq(dictionaries);

            var union = metrics.Concat(averaged)
                .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

            return union;
        }

        public Dictionary<string, double> AverageDictionariesWithLinq(List<Dictionary<string, double>> dictionaries)
        {
            return dictionaries
                .SelectMany(dict => dict)
                .GroupBy(kvp => kvp.Key)
                .ToDictionary(
                    group => group.Key,
                    group => group.Average(kvp => kvp.Value)
                );
        }

        private List<double> CalculateReturns(List<PradOp> path)
        {
            var returns = new List<double>();
            for (int i = 1; i < path.Count; i++)
            {
                var pctChange = (path[i].Result.Data.Sum() - path[i - 1].Result.Data.Sum()) /
                               Math.Sqrt(path[i].Result.Data.Length);
                returns.Add(pctChange);
            }
            return returns;
        }

        private double CalculateSharpeRatio(List<double> returns)
        {
            double excessReturn = returns.Average() - riskFreeRate;
            double volatility = CalculateVolatility(returns);
            return (volatility > 0) ? (excessReturn / volatility) * Math.Sqrt(annualizationFactor) : 0;
        }

        private double CalculateSortinoRatio(List<double> returns)
        {
            double excessReturn = returns.Average() - riskFreeRate;
            double downsideDeviation = CalculateDownsideDeviation(returns);
            return (downsideDeviation > 0) ? (excessReturn / downsideDeviation) * Math.Sqrt(annualizationFactor) : 0;
        }

        private double CalculateDownsideDeviation(List<double> returns)
        {
            var negativeReturns = returns.Where(r => r < 0).ToList();
            return negativeReturns.Any()
                ? Math.Sqrt(negativeReturns.Select(r => r * r).Average())
                : 0;
        }

        private double CalculateCalmarRatio(List<double> returns)
        {
            var (maxDrawdown, _, _) = CalculateDrawdownMetrics(returns);
            double annualizedReturn = AnnualizeReturn(CalculateTotalReturn(returns), returns.Count);
            return maxDrawdown > 0 ? annualizedReturn / maxDrawdown : 0;
        }

        private (double maxDrawdown, double avgDrawdown, int maxDuration) CalculateDrawdownMetrics(List<double> returns)
        {
            double peak = 0;
            double currentValue = 0;
            double maxDrawdown = 0;
            double sumDrawdowns = 0;
            int drawdownCount = 0;
            int currentDrawdownDuration = 0;
            int maxDrawdownDuration = 0;

            for (int i = 0; i < returns.Count; i++)
            {
                currentValue += returns[i];
                peak = Math.Max(peak, currentValue);

                double drawdown = peak - currentValue;
                if (drawdown > 0)
                {
                    currentDrawdownDuration++;
                    sumDrawdowns += drawdown;
                    drawdownCount++;
                    maxDrawdown = Math.Max(maxDrawdown, drawdown);
                }
                else
                {
                    maxDrawdownDuration = Math.Max(maxDrawdownDuration, currentDrawdownDuration);
                    currentDrawdownDuration = 0;
                }
            }

            double avgDrawdown = drawdownCount > 0 ? sumDrawdowns / drawdownCount : 0;
            return (maxDrawdown, avgDrawdown, maxDrawdownDuration);
        }

        private double CalculateValueAtRisk(List<double> returns, double confidence)
        {
            var sortedReturns = returns.OrderBy(r => r).ToList();
            int index = (int)Math.Floor((1 - confidence) * returns.Count);
            return -sortedReturns[index];
        }

        private double CalculateConditionalVaR(List<double> returns, double confidence)
        {
            var var = CalculateValueAtRisk(returns, confidence);
            var tailReturns = returns.Where(r => r < -var).ToList();
            return tailReturns.Any() ? -tailReturns.Average() : var;
        }

        private (double skewness, double kurtosis) CalculateDistributionMetrics(List<double> returns)
        {
            double mean = returns.Average();
            double variance = returns.Select(r => Math.Pow(r - mean, 2)).Average();
            double stdDev = Math.Sqrt(variance);

            double skewness = returns.Select(r => Math.Pow((r - mean) / stdDev, 3)).Average();
            double kurtosis = returns.Select(r => Math.Pow((r - mean) / stdDev, 4)).Average() - 3;

            return (skewness, kurtosis);
        }

        private double CalculateJarqueBera(double skewness, double kurtosis, int n)
        {
            return n * (Math.Pow(skewness, 2) / 6 + Math.Pow(kurtosis, 2) / 24);
        }

        private double CalculateVolatilityClustering(List<double> returns)
        {
            var squaredReturns = returns.Select(r => r * r).ToList();
            return CalculateAutocorrelation(squaredReturns, 1);
        }

        private double CalculateAutocorrelation(List<double> series, int lag)
        {
            if (series.Count <= lag) return 0;

            double mean = series.Average();
            var covariance = 0.0;
            var variance = 0.0;

            for (int i = lag; i < series.Count; i++)
            {
                covariance += (series[i] - mean) * (series[i - lag] - mean);
                variance += Math.Pow(series[i] - mean, 2);
            }

            return covariance / variance;
        }

        private double CalculateMomentumFactor(List<double> returns)
        {
            if (returns.Count < windowSize) return 0;

            var momentum = new List<double>();
            for (int i = windowSize; i < returns.Count; i++)
            {
                var windowReturn = returns.Skip(i - windowSize).Take(windowSize).Sum();
                momentum.Add(Math.Sign(windowReturn) * returns[i]);
            }

            return momentum.Average();
        }

        private double CalculateReversalTendency(List<double> returns)
        {
            if (returns.Count < 2) return 0;

            int reversals = 0;
            for (int i = 1; i < returns.Count; i++)
            {
                if (Math.Sign(returns[i]) != Math.Sign(returns[i - 1]))
                    reversals++;
            }

            return (double)reversals / (returns.Count - 1);
        }

        private double CalculateTailDependence(List<double> returns)
        {
            var threshold = CalculateValueAtRisk(returns, 0.9);
            var tailEvents = returns.Select((r, i) => new { Return = r, Index = i })
                                   .Where(x => x.Return < -threshold)
                                   .Select(x => x.Index)
                                   .ToList();

            if (tailEvents.Count < 2) return 0;

            int clusteredEvents = 0;
            for (int i = 1; i < tailEvents.Count; i++)
            {
                if (tailEvents[i] - tailEvents[i - 1] <= 5) // 5-period window
                    clusteredEvents++;
            }

            return (double)clusteredEvents / (tailEvents.Count - 1);
        }

        private double CalculatePathSmoothness(List<double> returns)
        {
            if (returns.Count < 2) return 1;

            var accelerations = new List<double>();
            for (int i = 1; i < returns.Count - 1; i++)
            {
                double acceleration = (returns[i + 1] - returns[i]) - (returns[i] - returns[i - 1]);
                accelerations.Add(Math.Abs(acceleration));
            }

            return Math.Exp(-accelerations.Average());
        }

        private double CalculateInformationRatio(List<double> returns)
        {
            var excess = returns.Select(r => r - riskFreeRate).ToList();
            double trackingError = CalculateVolatility(excess);
            return trackingError > 0 ? excess.Average() / trackingError * Math.Sqrt(annualizationFactor) : 0;
        }

        private double CalculateGainToPain(List<double> returns)
        {
            var gains = returns.Where(r => r > 0).Sum();
            var pains = Math.Abs(returns.Where(r => r < 0).Sum());
            return pains > 0 ? gains / pains : double.MaxValue;
        }

        private double CalculateUlcerIndex(List<double> returns)
        {
            double peak = 0;
            double currentValue = 0;
            var squaredDrawdowns = new List<double>();

            foreach (var ret in returns)
            {
                currentValue += ret;
                peak = Math.Max(peak, currentValue);
                double drawdown = (peak - currentValue) / peak;
                squaredDrawdowns.Add(drawdown * drawdown);
            }

            return Math.Sqrt(squaredDrawdowns.Average());
        }

        private double CalculateVolatility(List<double> returns)
        {
            return Math.Sqrt(returns.Select(r => r * r).Average());
        }

        private double CalculateTotalReturn(List<double> returns)
        {
            return returns.Aggregate(1.0, (acc, r) => acc * (1 + r)) - 1;
        }

        private double AnnualizeReturn(double totalReturn, int periods)
        {
            return Math.Pow(1 + totalReturn, annualizationFactor / periods) - 1;
        }
    }
}
