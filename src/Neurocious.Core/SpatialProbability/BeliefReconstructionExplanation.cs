using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public class BeliefReconstructionExplanation
    {
        public PradResult WarpedPriorState { get; set; }
        public float TemporalSmoothness { get; set; }
        public Dictionary<string, float> ConfidenceMetrics { get; set; }
        public string CausalJustification { get; set; }
        public List<string> CausalAntecedents { get; set; } = new();
        public Dictionary<string, float> AttributionScores { get; set; } = new();
        public float ReconstructionConfidence { get; set; }

        public string GenerateDetailedExplanation()
        {
            var sb = new StringBuilder();

            sb.AppendLine("Belief Reconstruction Analysis:");
            sb.AppendLine($"- Temporal Smoothness: {TemporalSmoothness:F3}");
            sb.AppendLine($"- Reconstruction Confidence: {ReconstructionConfidence:F3}");

            if (CausalAntecedents.Any())
            {
                sb.AppendLine("\nCausal Antecedents:");
                foreach (var antecedent in CausalAntecedents)
                {
                    var score = AttributionScores.GetValueOrDefault(antecedent, 0);
                    sb.AppendLine($"- {antecedent} (strength: {score:F3})");
                }
            }

            if (ConfidenceMetrics.Any())
            {
                sb.AppendLine("\nConfidence Metrics:");
                foreach (var (metric, value) in ConfidenceMetrics)
                {
                    sb.AppendLine($"- {metric}: {value:F3}");
                }
            }

            if (!string.IsNullOrEmpty(CausalJustification))
            {
                sb.AppendLine($"\nJustification:\n{CausalJustification}");
            }

            return sb.ToString();
        }
    }
}
