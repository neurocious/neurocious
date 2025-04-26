using Neurocious.Core.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Memory
{
    public class AttentionalMemoryAccess
    {
        private readonly Dictionary<string, float> attentionWeights;
        private readonly Queue<string> workingMemory;
        private readonly int workingMemoryCapacity;

        public class AttentionalQuery
        {
            public float[] LatentQuery { get; init; }
            public FieldParameters CurrentField { get; init; }
            public HashSet<string> ActiveNarratives { get; init; }
            public DateTime QueryTime { get; init; }
        }

        public async Task<List<BeliefMemoryCell>> QueryMemoryWithAttention(
            AttentionalQuery query,
            int maxResults = 5)
        {
            // Calculate attention weights for all memories
            var weights = await CalculateAttentionWeights(query);

            // Get top K memories by attention weight
            var topMemories = weights
                .OrderByDescending(kv => kv.Value)
                .Take(maxResults)
                .Select(kv => memoryEngine.RetrieveMemory(kv.Key))
                .Where(m => m != null)
                .ToList();

            // Update working memory
            UpdateWorkingMemory(topMemories.Select(m => m.BeliefId));

            return topMemories;
        }

        private async Task<Dictionary<string, float>> CalculateAttentionWeights(
            AttentionalQuery query)
        {
            var weights = new Dictionary<string, float>();
            var allMemories = await memoryEngine.GetAllMemories();

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
