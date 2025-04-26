using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public class InverseFlowIntegration
    {
        private readonly InverseFlowField inverseFlowField;
        private readonly TemporalBuffer temporalBuffer;
        private readonly int vectorDim;
        private readonly Dictionary<string, float> causalStrengthCache;

        public InverseFlowIntegration(int[] fieldShape, int vectorDim, int bufferSize = 10)
        {
            this.vectorDim = vectorDim;
            this.inverseFlowField = new InverseFlowField(fieldShape, vectorDim);
            this.temporalBuffer = new TemporalBuffer(bufferSize);
            this.causalStrengthCache = new Dictionary<string, float>();
        }

        public BeliefReconstructionExplanation ReconstructPriorBelief(
            PradOp currentState,
            PradOp contextState,
            List<string> potentialAntecedents = null)
        {
            var regularizer = new InverseFlowField.TemporalRegularizer();
            var result = inverseFlowField.GeneratePreviousStateWithContext(
                currentState,
                contextState,
                regularizer);

            var antecedents = IdentifyCausalAntecedents(
                result.WarpedState,
                potentialAntecedents);

            var attributionScores = CalculateAttributionScores(
                result.WarpedState,
                antecedents);

            return new BeliefReconstructionExplanation
            {
                WarpedPriorState = result.WarpedState,
                TemporalSmoothness = result.TemporalSmoothness,
                ConfidenceMetrics = result.ConfidenceMetrics,
                CausalJustification = GenerateInverseJustification(result, attributionScores),
                CausalAntecedents = antecedents,
                AttributionScores = attributionScores,
                ReconstructionConfidence = CalculateReconstructionConfidence(result)
            };
        }

        public void UpdateFromForwardDynamics(PradOp forwardField, PradResult forwardRouting)
        {
            inverseFlowField.UpdateFromForwardField(forwardField, forwardRouting);
        }

        public void AddToTemporalBuffer(PradOp state)
        {
            temporalBuffer.Add(state);
        }

        private List<string> IdentifyCausalAntecedents(
            PradResult warpedState,
            List<string> potentialAntecedents)
        {
            if (potentialAntecedents == null || !potentialAntecedents.Any())
                return new List<string>();

            var antecedents = new List<string>();
            var contextWindow = temporalBuffer.GetRecentStates(5); // Look back 5 states

            foreach (var antecedent in potentialAntecedents)
            {
                // Check temporal alignment
                bool temporallyValid = ValidateTemporalAlignment(antecedent, contextWindow);

                // Check causal strength
                float causalStrength = CalculateCausalStrength(warpedState, antecedent);

                if (temporallyValid && causalStrength > 0.3f)
                {
                    antecedents.Add(antecedent);
                    causalStrengthCache[antecedent] = causalStrength;
                }
            }

            return antecedents;
        }

        private Dictionary<string, float> CalculateAttributionScores(
            PradResult warpedState,
            List<string> antecedents)
        {
            var scores = new Dictionary<string, float>();

            foreach (var antecedent in antecedents)
            {
                if (causalStrengthCache.TryGetValue(antecedent, out float cachedStrength))
                {
                    scores[antecedent] = cachedStrength;
                }
                else
                {
                    float strength = CalculateCausalStrength(warpedState, antecedent);
                    scores[antecedent] = strength;
                    causalStrengthCache[antecedent] = strength;
                }
            }

            // Normalize scores
            if (scores.Any())
            {
                float sum = scores.Values.Sum();
                foreach (var key in scores.Keys.ToList())
                {
                    scores[key] /= sum;
                }
            }

            return scores;
        }

        private bool ValidateTemporalAlignment(string antecedent, List<PradOp> contextWindow)
        {
            // Simple validation using the temporal buffer
            // Could be extended with more sophisticated temporal logic
            return contextWindow.Any();
        }

        private float CalculateCausalStrength(PradResult warpedState, string antecedent)
        {
            // Calculate similarity between warped state and potential antecedent
            // This is a simplified version - could be extended with more sophisticated causal metrics
            if (!temporalBuffer.TryGetState(antecedent, out var antecedentState))
                return 0f;

            float similarity = CosineSimilarity(
                warpedState.Result.Data,
                antecedentState.Result.Data);

            return Math.Max(0f, similarity);
        }

        private float CosineSimilarity(double[] v1, double[] v2)
        {
            double dotProduct = 0;
            double norm1 = 0;
            double norm2 = 0;

            for (int i = 0; i < v1.Length; i++)
            {
                dotProduct += v1[i] * v2[i];
                norm1 += v1[i] * v1[i];
                norm2 += v2[i] * v2[i];
            }

            return (float)(dotProduct / (Math.Sqrt(norm1) * Math.Sqrt(norm2)));
        }

        private string GenerateInverseJustification(
            InverseFlowField.InverseTransformationState result,
            Dictionary<string, float> attributionScores)
        {
            var sb = new StringBuilder();

            // Add temporal confidence
            sb.AppendLine($"Temporal coherence: {result.TemporalSmoothness:F3}");

            // Add routing confidence
            if (result.ConfidenceMetrics.TryGetValue("routing_confidence", out float routingConf))
            {
                sb.AppendLine($"Routing confidence: {routingConf:F3}");
            }

            // Add causal attribution
            if (attributionScores.Any())
            {
                sb.AppendLine("\nCausal influences:");
                foreach (var (cause, strength) in attributionScores.OrderByDescending(x => x.Value))
                {
                    sb.AppendLine($"- {cause}: {strength:F3}");
                }
            }

            return sb.ToString();
        }

        private float CalculateReconstructionConfidence(
            InverseFlowField.InverseTransformationState result)
        {
            float temporalWeight = 0.4f;
            float routingWeight = 0.4f;
            float warpingWeight = 0.2f;

            float confidence =
                temporalWeight * result.TemporalSmoothness +
                routingWeight * result.ConfidenceMetrics.GetValueOrDefault("routing_confidence", 0f) +
                warpingWeight * result.ConfidenceMetrics.GetValueOrDefault("warping_confidence", 0f);

            return confidence;
        }

        private class TemporalBuffer
        {
            private readonly Queue<(string id, PradOp state)> buffer;
            private readonly int capacity;
            private readonly Dictionary<string, PradOp> stateMap;

            public TemporalBuffer(int capacity)
            {
                this.capacity = capacity;
                this.buffer = new Queue<(string, PradOp)>(capacity);
                this.stateMap = new Dictionary<string, PradOp>();
            }

            public void Add(PradOp state)
            {
                string id = Guid.NewGuid().ToString();

                if (buffer.Count >= capacity)
                {
                    var (oldId, _) = buffer.Dequeue();
                    stateMap.Remove(oldId);
                }

                buffer.Enqueue((id, state));
                stateMap[id] = state;
            }

            public List<PradOp> GetRecentStates(int count)
            {
                return buffer
                    .Skip(Math.Max(0, buffer.Count - count))
                    .Select(x => x.state)
                    .ToList();
            }

            public bool TryGetState(string id, out PradOp state)
            {
                return stateMap.TryGetValue(id, out state);
            }
        }
    }
}
