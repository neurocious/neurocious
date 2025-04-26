using Neurocious.Core.Common;
using Neurocious.Core.EnhancedVariationalAutoencoder;
using ParallelReverseAutoDiff.PRAD;
using Neurocious.Core.SpatialProbability;

namespace Neurocious.Core.Memory
{
    public class EpistemicMemoryEngine
    {
        private readonly EpistemicMemorySystem memorySystem;
        private readonly EnhancedVAE vae;
        private readonly SpatialProbabilityNetwork spn;
        private readonly InverseFlowField inverseFlow;
        private readonly AttentionalMemoryAccess attentionSystem;
        private readonly NarrativeManager narrativeManager;
        private readonly MetaBeliefSystem metaBeliefSystem;
        private readonly CancellationTokenSource consolidationCts;
        private readonly Task consolidationTask;

        public EpistemicMemoryEngine(
            EnhancedVAE vae,
            SpatialProbabilityNetwork spn,
            InverseFlowField inverseFlow)
        {
            this.vae = vae;
            this.spn = spn;
            this.inverseFlow = inverseFlow;

            // Initialize core memory system
            this.memorySystem = new EpistemicMemorySystem(vae, spn, inverseFlow);

            // Initialize higher-level components that work with the memory system
            this.attentionSystem = new AttentionalMemoryAccess(this);
            this.narrativeManager = new NarrativeManager(this);
            this.metaBeliefSystem = new MetaBeliefSystem(this);

            // Start memory consolidation process
            consolidationCts = new CancellationTokenSource();
            consolidationTask = RunMemoryConsolidation(consolidationCts.Token);
        }

        internal EpistemicMemorySystem MemorySystem
        {
            get => memorySystem;
        }

        internal SpatialProbabilityNetwork SPN
        {
            get => spn;
        }

        internal NarrativeManager NarrativeManager
        {
            get => narrativeManager;
        }

        public class MemoryConditionedGeneration
        {
            public float[] LatentVector { get; init; }
            public FieldParameters FieldParams { get; init; }
            public Dictionary<string, float> MemoryInfluences { get; init; }
            public float MemoryCoherence { get; init; }
            public List<string> ActiveNarratives { get; init; }
        }

        // High-level memory operations that coordinate between components
        public async Task<MemoryConditionedGeneration> GenerateWithMemoryContext(
            PradOp input,
            string[] relevantContexts,
            float memoryInfluence = 0.5f)
        {
            // Get relevant memories from the core system
            var activeMemories = relevantContexts
                .SelectMany(ctx => memorySystem.QueryByNarrative(ctx))
                .OrderByDescending(m => m.RetentionScore)
                .Take(5)
                .ToList();

            // Compute memory-weighted latent vector
            var baseLatent = (await vae.EncodeSequence(new List<PradOp> { input })).Item1;
            var memoryLatent = await CombineMemoryLatents(activeMemories, baseLatent);

            // Interpolate between input and memory
            var combinedLatent = InterpolateLatentVectors(
                baseLatent.Result,
                memoryLatent.Result,
                memoryInfluence);

            // Route through SPN for field parameters
            var fieldParams = vae.ExtractFieldParameters(new PradOp(combinedLatent));

            return new MemoryConditionedGeneration
            {
                LatentVector = combinedLatent.Data,
                FieldParams = fieldParams,
                MemoryInfluences = CalculateMemoryInfluences(activeMemories),
                MemoryCoherence = CalculateMemoryCoherence(activeMemories, fieldParams),
                ActiveNarratives = activeMemories
                    .SelectMany(m => m.NarrativeContexts)
                    .Distinct()
                    .ToList()
            };
        }

        public async Task<BeliefMemoryCell> FormAndStoreMemory(PradOp input)
        {
            // Use core system to form the basic belief
            var belief = await memorySystem.FormBelief(input);

            // Enhance with narrative context
            await narrativeManager.ProcessNewBelief(belief);

            // Create meta-beliefs if relevant
            await metaBeliefSystem.ProcessNewBelief(belief);

            return belief;
        }

        public async Task<List<BeliefMemoryCell>> RetrieveMemories(
            PradOp query,
            HashSet<string> activeNarratives)
        {
            // Create attentional query
            var attentionalQuery = new AttentionalMemoryAccess.AttentionalQuery
            {
                LatentQuery = query.Result.Data.Select(x => (float)x).ToArray(),
                CurrentField = vae.ExtractFieldParameters(query),
                ActiveNarratives = activeNarratives,
                QueryTime = DateTime.UtcNow
            };

            // Use attention system to retrieve relevant memories
            return await attentionSystem.QueryMemoryWithAttention(
                memorySystem,
                attentionalQuery);
        }

        private async Task RunMemoryConsolidation(CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(TimeSpan.FromMinutes(30), ct);

                    // Get recent memories from core system
                    var recentMemories = memorySystem.QueryByNarrative("*", 0)
                        .OrderByDescending(m => m.LastAccessed)
                        .Take(100)
                        .ToList();

                    // Process consolidation
                    var consolidationState = await ConsolidateMemories(recentMemories);

                    // Update narratives and meta-beliefs
                    await narrativeManager.UpdateNarratives(consolidationState);
                    await metaBeliefSystem.UpdateMetaBeliefs(consolidationState);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
        }

        public class ConsolidationState
        {
            public List<BeliefMemoryCell> RecentMemories { get; init; }
            public Dictionary<string, float> ConsolidationScores { get; init; }
            public List<string> ReinforcedNarratives { get; init; }
            public List<string> PrunedBeliefs { get; init; }
            public Dictionary<string, float> NarrativeCoherence { get; init; }
        }

        private async Task<ConsolidationState> ConsolidateMemories(
            List<BeliefMemoryCell> recentMemories)
        {
            var consolidationScores = new Dictionary<string, float>();
            var reinforcedNarratives = new HashSet<string>();
            var pruneList = new List<string>();
            var narrativeCoherence = new Dictionary<string, float>();

            foreach (var memory in recentMemories)
            {
                // Simulate memory recall
                var (coherence, reinforcement) = await SimulateMemoryReplay(memory);
                consolidationScores[memory.BeliefId] = coherence;

                if (coherence > 0.7f)
                {
                    await memorySystem.ReinforceBeliefMemory(memory);
                    reinforcedNarratives.UnionWith(memory.NarrativeContexts);
                }
                else if (coherence < 0.3f)
                {
                    pruneList.Add(memory.BeliefId);
                }

                // Generate novel associations
                await GenerateMemoryAssociations(memory, recentMemories);
            }

            return new ConsolidationState
            {
                RecentMemories = recentMemories,
                ConsolidationScores = consolidationScores,
                ReinforcedNarratives = reinforcedNarratives.ToList(),
                PrunedBeliefs = pruneList,
                NarrativeCoherence = narrativeCoherence
            };
        }

        private async Task<(float coherence, float reinforcement)> SimulateMemoryReplay(
            BeliefMemoryCell memory)
        {
            // Use core system for basic replay
            var replayState = new PradOp(new Tensor(memory.LatentVector));
            var (routing, _, _, _, predictions) = spn.ProcessState(replayState);

            // Check temporal consistency
            var inverseState = inverseFlow.GeneratePreviousStateWithContext(
                replayState,
                new PradOp(new Tensor(memory.LatentVector)),
                memorySystem.TemporalRegularizer);

            float coherence = inverseState.TemporalSmoothness;
            float predictionConfidence = CalculatePredictionConfidence(predictions.Result);

            return (coherence, predictionConfidence);
        }

        private async Task GenerateMemoryAssociations(
    BeliefMemoryCell source,
    List<BeliefMemoryCell> context)
        {
            var sourceState = new PradOp(new Tensor(source.LatentVector));
            var (routing, confidence, policy, reflexes, predictions, fieldParams, explanation, inverse) = spn.ProcessState(sourceState);

            foreach (var target in context)
            {
                if (target.BeliefId == source.BeliefId) continue;

                var similarity = CosineSimilarity(
                    source.LatentVector,
                    target.LatentVector);

                if (similarity > 0.7f)
                {
                    var sharedContexts = source.NarrativeContexts
                        .Intersect(target.NarrativeContexts)
                        .ToList();

                    narrativeManager.ConnectBeliefs(sharedContexts, source, target);
                }
            }
        }

        private float CosineSimilarity(float[] v1, float[] v2)
        {
            float dotProduct = 0, norm1 = 0, norm2 = 0;
            for (int i = 0; i < v1.Length; i++)
            {
                dotProduct += v1[i] * v2[i];
                norm1 += v1[i] * v1[i];
                norm2 += v2[i] * v2[i];
            }
            return dotProduct / (float)(Math.Sqrt(norm1) * Math.Sqrt(norm2));
        }

        private float CalculatePredictionConfidence(PradResult predictions)
        {
            return (float)predictions.Result.Data.Average();
        }

        private Tensor InterpolateLatentVectors(Tensor v1, Tensor v2, float weight)
        {
            var result = new double[v1.Data.Length];
            for (int i = 0; i < v1.Data.Length; i++)
            {
                result[i] = v1.Data[i] * (1 - weight) + v2.Data[i] * weight;
            }
            return new Tensor(v1.Shape, result);
        }

        private Dictionary<string, float> CalculateMemoryInfluences(
            List<BeliefMemoryCell> memories)
        {
            return memories.ToDictionary(
                m => m.BeliefId,
                m => m.RetentionScore);
        }

        private float CalculateMemoryCoherence(
            List<BeliefMemoryCell> memories,
            FieldParameters currentField)
        {
            if (!memories.Any()) return 1.0f;

            float totalCoherence = memories.Sum(m =>
                1.0f - (float)Math.Abs(m.FieldParams.Curvature - currentField.Curvature) -
                (float)Math.Abs(m.FieldParams.Entropy - currentField.Entropy) -
                (float)Math.Abs(m.FieldParams.Alignment - currentField.Alignment));

            return Math.Max(0, totalCoherence / memories.Count);
        }

        private async Task<PradResult> CombineMemoryLatents(
            List<BeliefMemoryCell> memories,
            PradResult baseLatent)
        {
            if (!memories.Any())
                return baseLatent;

            var weightedSum = new double[baseLatent.Result.Shape[0]];
            float totalWeight = 0;

            foreach (var memory in memories)
            {
                float weight = memory.RetentionScore;
                for (int i = 0; i < weightedSum.Length; i++)
                {
                    weightedSum[i] += memory.LatentVector[i] * weight;
                }
                totalWeight += weight;
            }

            for (int i = 0; i < weightedSum.Length; i++)
            {
                weightedSum[i] /= totalWeight;
            }

            return new PradOp(new Tensor(baseLatent.Result.Shape, weightedSum))
                .Then(PradOp.TanhOp);
        }

        public void Dispose()
        {
            consolidationCts?.Cancel();
            consolidationCts?.Dispose();
            consolidationTask?.Dispose();
        }
    }
}
