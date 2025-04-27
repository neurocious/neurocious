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
    /// Represents a market regime or trading strategy as a point in latent space.
    /// </summary>
    public class MarketRegimeGate : StrategicGate
    {
        public Dictionary<string, double> RegimeCharacteristics { get; }

        public MarketRegimeGate(
            string name,
            int dim,
            float threshold,
            float weight,
            Dictionary<string, double> characteristics) : base(name, dim, threshold, weight)
        {
            RegimeCharacteristics = characteristics;
        }

        public override float CalculateActivation(PradOp state)
        {
            var baseActivation = base.CalculateActivation(state);

            // Modulate activation based on regime characteristics
            var modulation = CalculateRegimeModulation(state);

            return baseActivation * modulation;
        }

        private float CalculateRegimeModulation(PradOp state)
        {
            var decoder = new FinancialDecoder();
            var metrics = decoder.DecodeRiskMetrics(state);

            float modulation = 1.0f;
            foreach (var (characteristic, targetValue) in RegimeCharacteristics)
            {
                if (metrics.TryGetValue(characteristic, out double actualValue))
                {
                    float match = 1.0f - (float)Math.Min(1.0, Math.Abs(actualValue - targetValue));
                    modulation *= (0.5f + 0.5f * match); // Soft modulation
                }
            }

            return modulation;
        }
    }
}
