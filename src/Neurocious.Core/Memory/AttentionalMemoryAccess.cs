using Neurocious.Core.Common;

namespace Neurocious.Core.Memory
{
    public class AttentionalMemoryAccess
    {
        private readonly Dictionary<string, float> attentionWeights;
        private readonly Queue<string> workingMemory;
        private readonly int workingMemoryCapacity;
        private readonly EpistemicMemoryEngine memoryEngine;

        public AttentionalMemoryAccess(EpistemicMemoryEngine memoryEngine)
        {
            this.memoryEngine = memoryEngine;
        }

        public class AttentionalQuery
        {
            public float[] LatentQuery { get; init; }
            public FieldParameters CurrentField { get; init; }
            public HashSet<string> ActiveNarratives { get; init; }
            public DateTime QueryTime { get; init; }
        }

        public async Task<List<BeliefMemoryCell>> QueryMemoryWithAttention(
            BeliefMemoryStore store,
            AttentionalQuery query,
            int maxResults = 5)
        {
            // Calculate attention weights for all memories
            var weights = await CalculateAttentionWeights(store, query);

            // Get top K memories by attention weight
            var topMemories = weights
                .OrderByDescending(kv => kv.Value)
                .Take(maxResults)
                .Select(kv => memoryEngine.MemorySystem.Retrieve(kv.Key))
                .Where(m => m != null)
                .ToList();

            // Update working memory
            UpdateWorkingMemory(topMemories.Select(m => m.BeliefId));

            return topMemories;
        }

        private async Task<Dictionary<string, float>> CalculateAttentionWeights(
    BeliefMemoryStore memoryStore,
    AttentionalQuery query)
        {
            var weights = new Dictionary<string, float>();
            var allMemories = memoryStore.GetAllBeliefs();

            foreach (var memory in allMemories)
            {
                float weight = 0;

                // Field alignment weight
                weight += CalculateFieldAlignment(query.CurrentField, memory.FieldParams) * 0.3f;

                // Narrative overlap weight
                weight += CalculateNarrativeOverlap(query.ActiveNarratives, memory.NarrativeContexts) * 0.3f;

                // Temporal recency weight
                weight += CalculateTemporalRecency(query.QueryTime, memory.LastAccessed) * 0.2f;

                // Working memory boost
                if (workingMemory.Contains(memory.BeliefId))
                {
                    weight *= 1.2f;
                }

                weights[memory.BeliefId] = weight;
            }

            return weights;
        }

        private float CalculateFieldAlignment(FieldParameters current, FieldParameters memory)
        {
            float curvatureDiff = Math.Abs((float)(current.Curvature - memory.Curvature));
            float entropyDiff = Math.Abs((float)(current.Entropy - memory.Entropy));
            float alignmentDiff = Math.Abs((float)(current.Alignment - memory.Alignment));

            // Return similarity score (1 - normalized difference)
            return 1 - (curvatureDiff + entropyDiff + alignmentDiff) / 3;
        }

        private float CalculateTemporalRecency(DateTime current, DateTime lastAccessed)
        {
            var timeDiff = current - lastAccessed;
            // Exponential decay over 24 hours
            return (float)Math.Exp(-timeDiff.TotalHours / 24);
        }

        private float CalculateNarrativeOverlap(HashSet<string> active, HashSet<string> memory)
        {
            if (!active.Any() || !memory.Any()) return 0;

            int overlap = active.Intersect(memory).Count();
            return (float)overlap / Math.Max(active.Count, memory.Count);
        }

        private void UpdateWorkingMemory(IEnumerable<string> activeBeliefIds)
        {
            foreach (var id in activeBeliefIds)
            {
                workingMemory.Enqueue(id);
                while (workingMemory.Count > workingMemoryCapacity)
                {
                    workingMemory.Dequeue();
                }
            }
        }
    }
}
