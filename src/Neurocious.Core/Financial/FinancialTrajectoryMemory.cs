using Neurocious.Core.Chess;
using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    /// <summary>
    /// Stores and analyzes sequences of market states and trading decisions.
    /// </summary>
    public class FinancialTrajectoryMemory : TrajectoryMemory
    {
        public double SharpeRatio { get; }
        public double MaxDrawdown { get; }
        public double ReturnToRisk { get; }
        public Dictionary<string, double> RiskMetrics { get; }

        public FinancialTrajectoryMemory(
            List<PradOp> path,
            double reward,
            Dictionary<string, List<float>> gateActivations,
            double sharpeRatio,
            double maxDrawdown,
            Dictionary<string, double> riskMetrics) : base(path, reward, gateActivations)
        {
            SharpeRatio = sharpeRatio;
            MaxDrawdown = maxDrawdown;
            ReturnToRisk = reward / (maxDrawdown + 1e-8);
            RiskMetrics = riskMetrics;
        }

        public double CalculateQuality()
        {
            // Combine multiple performance metrics
            return 0.4 * Reward +                    // Raw returns
                   0.3 * SharpeRatio +               // Risk-adjusted returns
                   0.2 * (1.0 - MaxDrawdown) +       // Drawdown control
                   0.1 * ReturnToRisk;               // Efficiency
        }
    }
}
