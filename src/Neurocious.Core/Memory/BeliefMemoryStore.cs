using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Memory
{
    public class BeliefMemoryStore
    {
        private readonly ConcurrentDictionary<string, BeliefMemoryCell> memory;
        private readonly ConcurrentDictionary<string, HashSet<string>> narrativeIndex;
        private readonly ConcurrentDictionary<string, float> beliefDriftMetrics;

        public BeliefMemoryStore()
        {
            memory = new ConcurrentDictionary<string, BeliefMemoryCell>();
            narrativeIndex = new ConcurrentDictionary<string, HashSet<string>>();
            beliefDriftMetrics = new ConcurrentDictionary<string, float>();
        }

        public List<BeliefMemoryCell> GetAllBeliefs() =>
        memory.Values.Select(b => b.Clone()).ToList();

        public List<BeliefMemoryCell> GetBeliefsByTimeRange(DateTime start, DateTime end) =>
            memory.Values
                .Where(b => b.Created >= start && b.Created <= end)
                .Select(b => b.Clone())
                .ToList();

        public void Store(BeliefMemoryCell belief)
        {
            // Store belief
            if (memory.TryGetValue(belief.BeliefId, out var existingBelief))
            {
                // Track drift if updating existing belief
                TrackBeliefDrift(existingBelief, belief);
            }

            memory[belief.BeliefId] = belief;

            // Update narrative indices
            foreach (var context in belief.NarrativeContexts)
            {
                narrativeIndex.AddOrUpdate(
                    context,
                    new HashSet<string> { belief.BeliefId },
                    (_, existing) =>
                    {
                        existing.Add(belief.BeliefId);
                        return existing;
                    });
            }
        }

        private void TrackBeliefDrift(BeliefMemoryCell old, BeliefMemoryCell @new)
        {
            float drift = CalculateVectorDrift(old.LatentVector, @new.LatentVector);
            beliefDriftMetrics[@new.BeliefId] = drift;
        }

        private float CalculateVectorDrift(float[] v1, float[] v2)
        {
            float sumSquaredDiff = 0;
            for (int i = 0; i < v1.Length; i++)
            {
                float diff = v1[i] - v2[i];
                sumSquaredDiff += diff * diff;
            }
            return (float)Math.Sqrt(sumSquaredDiff);
        }

        public BeliefMemoryCell Retrieve(string beliefId)
        {
            if (memory.TryGetValue(beliefId, out var cell))
            {
                cell.LastAccessed = DateTime.UtcNow;
                return cell.Clone();
            }
            return null;
        }

        public List<BeliefMemoryCell> QueryByNarrative(
            string context,
            float minConviction = 0.3f,
            int maxResults = 10)
        {
            if (!narrativeIndex.TryGetValue(context, out var beliefIds))
                return new List<BeliefMemoryCell>();

            return beliefIds
                .Select(id => memory[id])
                .Where(b => b.Conviction >= minConviction)
                .OrderByDescending(b => b.RetentionScore)
                .Take(maxResults)
                .Select(b => b.Clone())
                .ToList();
        }

        public void PruneMemory(float minRetention)
        {
            var toRemove = memory.Values
                .Where(b => b.RetentionScore < minRetention)
                .Select(b => b.BeliefId)
                .ToList();

            foreach (var id in toRemove)
            {
                if (memory.TryRemove(id, out var cell))
                {
                    // Remove from narrative indices
                    foreach (var context in cell.NarrativeContexts)
                    {
                        if (narrativeIndex.TryGetValue(context, out var beliefs))
                        {
                            beliefs.Remove(id);
                        }
                    }
                }
            }
        }

        public float GetBeliefDrift(string beliefId)
        {
            return beliefDriftMetrics.GetValueOrDefault(beliefId, 0f);
        }
    }

}
