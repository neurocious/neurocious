using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class RewardCalculator
    {
        private readonly Dictionary<string, double> weights;
        private readonly ExponentialMovingAverage emaCalculator;
        private readonly double baselineReturnThreshold;
        private readonly double maxDrawdownPenalty;

        public RewardCalculator(
            Dictionary<string, double> weights = null,
            double baselineReturnThreshold = 0.02,
            double maxDrawdownPenalty = 0.5)
        {
            this.weights = weights ?? DefaultWeights();
            this.emaCalculator = new ExponentialMovingAverage();
            this.baselineReturnThreshold = baselineReturnThreshold;
            this.maxDrawdownPenalty = maxDrawdownPenalty;
        }

        private Dictionary<string, double> DefaultWeights()
        {
            return new Dictionary<string, double>
            {
                ["sharpe_ratio"] = 0.25,
                ["sortino_ratio"] = 0.15,
                ["information_ratio"] = 0.10,
                ["calmar_ratio"] = 0.10,
                ["path_quality"] = 0.10,
                ["regime_accuracy"] = 0.10,
                ["exploration_quality"] = 0.10,
                ["risk_adjustment"] = 0.10
            };
        }

        public double CalculateReward(
            BacktestResult result,
            List<List<PradOp>> paths,
            Dictionary<string, double> metrics)
        {
            // Base reward components
            double baseReward = CalculateBaseReward(metrics);

            // Risk adjustments
            double riskPenalty = CalculateRiskPenalty(metrics);

            // Path quality bonus
            double pathQualityBonus = CalculatePathQualityBonus(paths);

            // Regime accuracy bonus
            double regimeBonus = CalculateRegimeAccuracyBonus(metrics);

            // Exploration quality
            double explorationBonus = CalculateExplorationBonus(paths, metrics);

            // Strategy uniqueness bonus
            double uniquenessBonus = CalculateUniquenessBonus(metrics);

            // Dynamic scaling based on market conditions
            double marketConditionScale = CalculateMarketConditionScale(result);

            // Combine components with weights
            double totalReward =
                weights["sharpe_ratio"] * baseReward +
                weights["sortino_ratio"] * (1 - riskPenalty) +
                weights["information_ratio"] * pathQualityBonus +
                weights["calmar_ratio"] * regimeBonus +
                weights["path_quality"] * explorationBonus +
                weights["regime_accuracy"] * uniquenessBonus;

            // Apply market condition scaling
            totalReward *= marketConditionScale;

            // Apply baseline threshold
            if (metrics["total_return"] < baselineReturnThreshold)
            {
                totalReward *= Math.Max(0, metrics["total_return"] / baselineReturnThreshold);
            }

            // Apply drawdown penalty
            if (metrics["max_drawdown"] > maxDrawdownPenalty)
            {
                totalReward *= Math.Max(0, 1 - (metrics["max_drawdown"] - maxDrawdownPenalty));
            }

            return Math.Max(-1, Math.Min(1, totalReward));
        }

        private double CalculateBaseReward(Dictionary<string, double> metrics)
        {
            return 0.4 * metrics["sharpe_ratio"] +
                   0.3 * metrics["sortino_ratio"] +
                   0.2 * metrics["information_ratio"] +
                   0.1 * metrics["calmar_ratio"];
        }

        private double CalculateRiskPenalty(Dictionary<string, double> metrics)
        {
            double drawdownPenalty = Math.Pow(metrics["max_drawdown"], 2);
            double volatilityPenalty = Math.Pow(metrics["downside_deviation"], 2);
            double varPenalty = metrics["value_at_risk"] * 0.5;

            return (drawdownPenalty + volatilityPenalty + varPenalty) / 3;
        }

        private double CalculatePathQualityBonus(List<List<PradOp>> paths)
        {
            var pathQuality = new PathQualityMetrics();
            double smoothnessScore = pathQuality.CalculatePathSmoothness(paths);
            double consistencyScore = pathQuality.CalculatePathConsistency(paths);
            double efficiencyScore = pathQuality.CalculatePathEfficiency(paths);

            return (smoothnessScore + consistencyScore + efficiencyScore) / 3;
        }

        private double CalculateRegimeAccuracyBonus(Dictionary<string, double> metrics)
        {
            return 0.5 * metrics["market_regime_accuracy"] +
                   0.3 * metrics["strategy_persistence"] +
                   0.2 * metrics["regime_transition_score"];
        }

        private double CalculateExplorationBonus(
            List<List<PradOp>> paths,
            Dictionary<string, double> metrics)
        {
            double explorationScore = metrics["exploration_quality"];
            double pathDiversity = CalculatePathDiversity(paths);
            double noveltyScore = CalculateNoveltyScore(paths);

            return (explorationScore + pathDiversity + noveltyScore) / 3;
        }

        private double CalculateUniquenessBonus(Dictionary<string, double> metrics)
        {
            return metrics["strategy_uniqueness"] *
                   Math.Max(0, metrics["prediction_accuracy"] - 0.5) * 2;
        }

        private double CalculateMarketConditionScale(BacktestResult result)
        {
            double volatilityScale = 1.0 / (1.0 + result.FinalMetrics["volatility"]);
            double regimeStabilityScale = result.FinalMetrics["market_regime_accuracy"];
            double liquidityScale = CalculateLiquidityScale(result);

            return (volatilityScale + regimeStabilityScale + liquidityScale) / 3;
        }

        private double CalculateLiquidityScale(BacktestResult result)
        {
            // Proxy liquidity by analyzing trade execution quality
            var tradeCosts = result.Trades.Select(t => t.Cost / (t.Size * t.Price)).Average();
            return 1.0 / (1.0 + tradeCosts * 100); // Scale to [0,1]
        }

        private double CalculatePathDiversity(List<List<PradOp>> paths)
        {
            if (paths.Count < 2) return 0;

            double totalDiversity = 0;
            int comparisons = 0;

            for (int i = 0; i < paths.Count - 1; i++)
            {
                for (int j = i + 1; j < paths.Count; j++)
                {
                    totalDiversity += CalculatePathDistance(paths[i], paths[j]);
                    comparisons++;
                }
            }

            return comparisons > 0 ? totalDiversity / comparisons : 0;
        }

        private double CalculatePathDistance(List<PradOp> path1, List<PradOp> path2)
        {
            int length = Math.Min(path1.Count, path2.Count);
            double totalDistance = 0;

            for (int i = 0; i < length; i++)
            {
                totalDistance += CalculateStateDistance(path1[i], path2[i]);
            }

            return totalDistance / length;
        }

        private double CalculateStateDistance(PradOp state1, PradOp state2)
        {
            var diff = state1.Sub(state2.Result);
            return Math.Sqrt(diff.Result.Data.Select(x => x * x).Sum());
        }

        private double CalculateNoveltyScore(List<List<PradOp>> paths)
        {
            if (paths.Count < 2) return 0;

            // Calculate average state for each timestep
            var avgStates = new List<double[]>();
            int timeSteps = paths[0].Count;

            for (int t = 0; t < timeSteps; t++)
            {
                var stateAvg = new double[paths[0][t].Result.Data.Length];

                foreach (var path in paths)
                {
                    for (int i = 0; i < stateAvg.Length; i++)
                    {
                        stateAvg[i] += path[t].Result.Data[i] / paths.Count;
                    }
                }

                avgStates.Add(stateAvg);
            }

            // Calculate novelty as average distance from mean path
            double totalNovelty = 0;
            foreach (var path in paths)
            {
                double pathNovelty = 0;
                for (int t = 0; t < timeSteps; t++)
                {
                    pathNovelty += CalculateStateDistance(
                        path[t],
                        new PradOp(new Tensor(path[t].Result.Shape, avgStates[t]))
                    );
                }
                totalNovelty += pathNovelty / timeSteps;
            }

            return totalNovelty / paths.Count;
        }
    }
}
