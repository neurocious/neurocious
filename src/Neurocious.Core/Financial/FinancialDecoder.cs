using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    /// <summary>
    /// Decodes latent states into interpretable financial signals and trading decisions.
    /// </summary>
    public class FinancialDecoder
    {
        private const double DECISION_THRESHOLD = 0.1;
        private readonly Dictionary<string, (double lower, double upper)> actionThresholds;

        public FinancialDecoder()
        {
            actionThresholds = new Dictionary<string, (double, double)>
            {
                ["Strong Buy"] = (0.5, double.MaxValue),
                ["Buy"] = (0.1, 0.5),
                ["Hold"] = (-0.1, 0.1),
                ["Sell"] = (-0.5, -0.1),
                ["Strong Sell"] = (double.MinValue, -0.5)
            };
        }

        public double[] DecodeFeatures(PradOp latentState)
        {
            return latentState.Result.Data;
        }

        public (string action, double confidence) DecodeTradeDecision(PradOp currentState, PradOp nextState)
        {
            var diff = nextState.Sub(currentState.Result);
            var moveVector = diff.Result.Data;

            // Calculate aggregate signal
            double signalStrength = moveVector.Sum() / Math.Sqrt(moveVector.Length);

            // Find appropriate action based on signal strength
            foreach (var action in actionThresholds)
            {
                if (signalStrength >= action.Value.lower && signalStrength < action.Value.upper)
                {
                    double confidence = Math.Min(1.0, Math.Abs(signalStrength) * 2);
                    return (action.Key, confidence);
                }
            }

            return ("Hold", 0.0);
        }

        public Dictionary<string, double> DecodeRiskMetrics(PradOp state)
        {
            var metrics = new Dictionary<string, double>();
            var data = state.Result.Data;

            // Calculate basic risk metrics from latent representation
            metrics["volatility_prediction"] = CalculateVolatilitySignal(data);
            metrics["tail_risk"] = CalculateTailRiskSignal(data);
            metrics["momentum"] = CalculateMomentumSignal(data);
            metrics["market_regime"] = CalculateRegimeSignal(data);

            return metrics;
        }

        private double CalculateVolatilitySignal(double[] latentVector)
        {
            // Use RMS of latent vector components as volatility proxy
            return Math.Sqrt(latentVector.Select(x => x * x).Average());
        }

        private double CalculateTailRiskSignal(double[] latentVector)
        {
            // Use higher moments of distribution as tail risk indicator
            var mean = latentVector.Average();
            var variance = latentVector.Select(x => Math.Pow(x - mean, 2)).Average();
            var skewness = latentVector.Select(x => Math.Pow(x - mean, 3)).Average() / Math.Pow(variance, 1.5);
            var kurtosis = latentVector.Select(x => Math.Pow(x - mean, 4)).Average() / Math.Pow(variance, 2);

            return (Math.Abs(skewness) + (kurtosis - 3)) / 2;
        }

        private double CalculateMomentumSignal(double[] latentVector)
        {
            // Calculate directional strength in latent space
            return latentVector.Sum() / Math.Sqrt(latentVector.Length);
        }

        private double CalculateRegimeSignal(double[] latentVector)
        {
            // Regime classification based on latent space position
            var norm = Math.Sqrt(latentVector.Sum(x => x * x));
            var angle = Math.Atan2(latentVector[1], latentVector[0]);
            return (Math.Cos(angle) * norm + 1) / 2; // Normalized to [0,1]
        }
    }
}
