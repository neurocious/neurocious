using Neurocious.Core.Common;
using Neurocious.Core.EnhancedVariationalAutoencoder;
using ParallelReverseAutoDiff.PRAD;
using static Neurocious.Core.SpatialProbability.InverseFlowField;
using Neurocious.Core.SpatialProbability;

namespace Neurocious.Core.Memory
{
    public class EpistemicMemorySystem
    {
        private readonly EnhancedVAE vae;
        private readonly SpatialProbabilityNetwork spn;
        private readonly InverseFlowField inverseFlow;
        private readonly BeliefMemoryStore memoryStore;
        public readonly TemporalRegularizer TemporalRegularizer;

        public EpistemicMemorySystem(
            EnhancedVAE vae,
            SpatialProbabilityNetwork spn,
            InverseFlowField inverseFlow)
        {
            this.vae = vae;
            this.spn = spn;
            this.inverseFlow = inverseFlow;
            this.memoryStore = new BeliefMemoryStore();
            this.TemporalRegularizer = new TemporalRegularizer();
        }

        // Core Memory Operations
        public async Task<BeliefMemoryCell> FormBelief(PradOp input)
        {
            // 1. Encode through VAE
            var latentVector = vae != null
        ? vae.EncodeSequence(new List<PradOp> { input }).Item1.PradOp // Get mean from tuple
        : input;

            // 2. Get field parameters
            var fieldParams = vae.ExtractFieldParameters(latentVector);

            // 3. Route through SPN for belief context
            var (routing, confidence, policy, reflexes, predictions, fieldParamsSPN, explanation, inverseExplanation) =
        spn.ProcessState(latentVector);

            // 4. Create memory cell
            var beliefCell = new BeliefMemoryCell
            {
                BeliefId = Guid.NewGuid().ToString(),
                LatentVector = latentVector.Result.Data,
                FieldParams = fieldParams,
                Conviction = CalculateInitialConviction(routing.Result, fieldParams),
                Created = DateTime.UtcNow,
                LastAccessed = DateTime.UtcNow,
                NarrativeContexts = ExtractNarrativeContexts(routing.Result, policy.Result),
                DecayRate = CalculateInitialDecayRate(fieldParams),
                MetaParameters = new Dictionary<string, float>
                {
                    ["narrative_coherence"] = CalculateNarrativeCoherence(routing.Result),
                    ["prediction_confidence"] = CalculatePredictionConfidence(predictions.Result),
                    ["reflex_activation"] = CalculateReflexActivation(reflexes.Result)
                }
            };

            // 5. Store if conviction meets threshold
            if (beliefCell.Conviction >= 0.3f)
            {
                memoryStore.Store(beliefCell);
            }

            return beliefCell;
        }

        public async Task ReinforceBeliefMemory(BeliefMemoryCell belief)
        {
            belief.Conviction = Math.Min(1.0f, belief.Conviction * 1.1f);
            belief.DecayRate *= 0.9f;
            memoryStore.Store(belief);
        }

        public async Task WeakenBeliefMemory(BeliefMemoryCell belief)
        {
            belief.Conviction *= 0.9f;
            belief.DecayRate *= 1.1f;
            memoryStore.Store(belief);
        }

        // Query Operations
        public List<BeliefMemoryCell> QueryByNarrative(string context, float minConviction = 0.3f) =>
            memoryStore.QueryByNarrative(context, minConviction);

        public BeliefMemoryCell Retrieve(string beliefId) =>
            memoryStore.Retrieve(beliefId);

        public List<BeliefMemoryCell> GetAllMemories() =>
            memoryStore.GetAllBeliefs();

        public List<BeliefMemoryCell> GetBeliefsByTimeRange(DateTime start, DateTime end) =>
            memoryStore.GetBeliefsByTimeRange(start, end);

        public float GetBeliefDrift(string beliefId) =>
            memoryStore.GetBeliefDrift(beliefId);

        // Helper Methods
        private float CalculateInitialConviction(PradResult routing, FieldParameters fieldParams)
        {
            float routingConfidence = 1 - CalculateEntropy(routing);
            float stabilityFactor = 1 / (1 + (float)fieldParams.Curvature);
            float certaintyFactor = 1 - (float)fieldParams.Entropy;

            return routingConfidence * stabilityFactor * certaintyFactor;
        }

        private float CalculateInitialDecayRate(FieldParameters fieldParams)
        {
            return 0.1f * (float)(1 + fieldParams.Entropy + fieldParams.Curvature);
        }

        private HashSet<string> ExtractNarrativeContexts(PradResult routing, PradResult policy)
        {
            var contexts = new HashSet<string>();

            // Extract top routing contexts
            var topRoutes = GetTopK(routing.Result.Data, 3);
            foreach (var (idx, strength) in topRoutes)
            {
                contexts.Add($"route_{idx}");
            }

            // Extract top policy contexts
            var topPolicies = GetTopK(policy.Result.Data, 2);
            foreach (var (idx, strength) in topPolicies)
            {
                contexts.Add($"policy_{idx}");
            }

            return contexts;
        }

        private List<(int index, float value)> GetTopK(double[] data, int k)
        {
            return data.Select((value, index) => (index, (float)value))
                      .OrderByDescending(x => x.Item2)
                      .Take(k)
                      .ToList();
        }

        private float CalculateEntropy(PradResult distribution)
        {
            float entropy = 0;
            foreach (var p in distribution.Result.Data)
            {
                if (p > 0)
                {
                    entropy -= (float)(p * Math.Log(p));
                }
            }
            return entropy;
        }

        private float CalculateNarrativeCoherence(PradResult routing)
        {
            return 1 - CalculateEntropy(routing);
        }

        private float CalculatePredictionConfidence(PradResult predictions)
        {
            return (float)predictions.Result.Data.Average();
        }

        private float CalculateReflexActivation(PradResult reflexes)
        {
            return (float)reflexes.Result.Data.Max();
        }
    }
}
