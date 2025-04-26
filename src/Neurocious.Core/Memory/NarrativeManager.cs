using Neurocious.Core.Common;
using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static ManagedCuda.BasicTypes.CUarrayMapInfo;
using static Neurocious.Core.Memory.EpistemicMemoryEngine;

namespace Neurocious.Core.Memory
{
    public class NarrativeManager
    {
        private readonly EpistemicMemoryEngine engine;
        private readonly ConcurrentDictionary<string, NarrativeThread> threads;
        private readonly Dictionary<string, HashSet<string>> beliefToThreadMap;
        private readonly float threadCoherenceThreshold = 0.7f;
        private readonly int maxThreadLength = 50;

        public NarrativeManager(EpistemicMemoryEngine engine)
        {
            this.engine = engine;
            this.threads = new ConcurrentDictionary<string, NarrativeThread>();
            this.beliefToThreadMap = new Dictionary<string, HashSet<string>>();
        }

        public async Task ConnectBeliefs(List<string> sharedContexts, BeliefMemoryCell source, BeliefMemoryCell? target)
        {
            if (sharedContexts.Any())
            {
                // Reinforce existing narrative connection
                foreach (var threadId in GetSharedThreads(source.BeliefId, target.BeliefId))
                {
                    if (threads.TryGetValue(threadId, out var thread))
                    {
                        thread.Coherence *= 1.1f;  // Strengthen connection
                        thread.LastUpdate = DateTime.UtcNow;
                        await UpdateThematicWeights(thread);
                    }
                }
            }
            else
            {
                // Create new thread connecting these beliefs
                await CreateNewThread(source, target);
            }
        }

        private HashSet<string> GetSharedThreads(string beliefId1, string beliefId2)
        {
            var threads1 = beliefToThreadMap.GetValueOrDefault(beliefId1, new HashSet<string>());
            var threads2 = beliefToThreadMap.GetValueOrDefault(beliefId2, new HashSet<string>());
            return threads1.Intersect(threads2).ToHashSet();
        }

        public async Task ProcessNewBelief(BeliefMemoryCell belief)
        {
            // Find compatible threads for the new belief
            var compatibleThreads = await FindCompatibleThreads(belief);

            if (!compatibleThreads.Any())
            {
                // Start new thread if no compatible ones found
                await CreateNewThread(belief);
            }
            else
            {
                // Extend or branch existing threads
                foreach (var thread in compatibleThreads)
                {
                    await ExtendThread(thread, belief);
                }
            }

            // Update thread relationships
            await UpdateThreadRelationships();
        }

        public async Task UpdateNarratives(ConsolidationState state)
        {
            var updatedThreads = new HashSet<string>();

            // Update threads containing consolidated memories
            foreach (var belief in state.RecentMemories)
            {
                if (beliefToThreadMap.TryGetValue(belief.BeliefId, out var threadIds))
                {
                    foreach (var threadId in threadIds)
                    {
                        if (threads.TryGetValue(threadId, out var thread))
                        {
                            await UpdateThreadCoherence(thread, state.ConsolidationScores);
                            updatedThreads.Add(threadId);
                        }
                    }
                }
            }

            // Prune low coherence threads
            var threadsToPrune = threads.Values
                .Where(t => t.Coherence < threadCoherenceThreshold)
                .Select(t => t.ThreadId)
                .ToList();

            foreach (var threadId in threadsToPrune)
            {
                await PruneThread(threadId);
            }

            // Update thematic weights for all affected threads
            foreach (var threadId in updatedThreads)
            {
                if (threads.TryGetValue(threadId, out var thread))
                {
                    await UpdateThematicWeights(thread);
                }
            }
        }

        private async Task<List<NarrativeThread>> FindCompatibleThreads(
            EpistemicMemorySystem.BeliefMemoryCell belief)
        {
            var compatible = new List<NarrativeThread>();

            foreach (var thread in threads.Values)
            {
                if (await IsCompatibleWithThread(belief, thread))
                {
                    compatible.Add(thread);
                }
            }

            return compatible.OrderByDescending(t => t.Coherence).ToList();
        }

        private async Task<bool> IsCompatibleWithThread(
            EpistemicMemorySystem.BeliefMemoryCell belief,
            NarrativeThread thread)
        {
            if (!thread.BeliefSequence.Any()) return false;

            // Get the last belief in the thread
            var lastBelief = engine.memorySystem.Retrieve(thread.BeliefSequence.Last());
            if (lastBelief == null) return false;

            // Check temporal order
            if (belief.Created <= lastBelief.Created) return false;

            // Check narrative continuity using inverse flow
            var inverseState = engine.inverseFlow.GeneratePreviousStateWithContext(
                new PradOp(new Tensor(belief.LatentVector)),
                new PradOp(new Tensor(lastBelief.LatentVector)),
                engine.memorySystem.TemporalRegularizer);

            // Calculate narrative overlap
            float narrativeOverlap = CalculateNarrativeOverlap(
                belief.NarrativeContexts,
                lastBelief.NarrativeContexts);

            // Combined compatibility score
            float compatibility = inverseState.TemporalSmoothness * 0.6f + narrativeOverlap * 0.4f;

            return compatibility > threadCoherenceThreshold;
        }

        private async Task CreateNewThread(BeliefMemoryCell belief)
        {
            var thread = new NarrativeThread
            {
                ThreadId = Guid.NewGuid().ToString(),
                BeliefSequence = new List<string> { belief.BeliefId },
                Coherence = 1.0f,
                StartTime = belief.Created,
                LastUpdate = DateTime.UtcNow,
                ThematicWeights = CalculateInitialThematicWeights(belief),
                RelatedThreads = new HashSet<string>(),
                ThreadMetrics = new Dictionary<string, float>
                {
                    ["stability"] = 1.0f,
                    ["continuity"] = 1.0f,
                    ["narrative_strength"] = belief.Conviction
                }
            };

            threads[thread.ThreadId] = thread;
            UpdateBeliefThreadMapping(belief.BeliefId, thread.ThreadId);
        }

        private async Task ExtendThread(
            NarrativeThread thread,
            BeliefMemoryCell belief)
        {
            // Check if thread needs branching
            if (ShouldBranchThread(thread, belief))
            {
                await BranchThread(thread, belief);
                return;
            }

            // Extend existing thread
            thread.Sequence.Add(belief);
            thread.LastUpdate = DateTime.UtcNow;

            // Trim if exceeds max length
            if (thread.Sequence.Count > maxThreadLength)
            {
                thread.Sequence.RemoveAt(0);
            }

            // Update thread properties
            await UpdateThreadCoherence(thread);
            await UpdateThematicWeights(thread);
            UpdateBeliefThreadMapping(belief.BeliefId, thread.ThreadId);
        }

        private bool ShouldBranchThread(
            NarrativeThread thread,
            BeliefMemoryCell belief)
        {
            if (!thread.Sequence.Any()) return false;

            var lastBelief = engine.MemorySystem.Retrieve(thread.Sequence.Last().BeliefId);
            if (lastBelief == null) return false;

            // Check for significant narrative divergence
            float narrativeOverlap = CalculateNarrativeOverlap(
                belief.NarrativeContexts,
                lastBelief.NarrativeContexts);

            // Check for significant field parameter changes
            float fieldChange = CalculateFieldChange(belief.FieldParams, lastBelief.FieldParams);

            return narrativeOverlap < 0.3f || fieldChange > 0.7f;
        }

        private async Task BranchThread(
            NarrativeThread parentThread,
            EpistemicMemorySystem.BeliefMemoryCell belief)
        {
            // Create new thread starting from the branch point
            var branchedThread = new NarrativeThread
            {
                ThreadId = Guid.NewGuid().ToString(),
                BeliefSequence = new List<string> { belief.BeliefId },
                Coherence = 1.0f,
                StartTime = belief.Created,
                LastUpdate = DateTime.UtcNow,
                ThematicWeights = CalculateInitialThematicWeights(belief),
                RelatedThreads = new HashSet<string> { parentThread.ThreadId },
                ThreadMetrics = new Dictionary<string, float>
                {
                    ["stability"] = 1.0f,
                    ["continuity"] = 1.0f,
                    ["narrative_strength"] = belief.Conviction,
                    ["branch_confidence"] = 0.8f
                }
            };

            threads[branchedThread.ThreadId] = branchedThread;
            parentThread.RelatedThreads.Add(branchedThread.ThreadId);
            UpdateBeliefThreadMapping(belief.BeliefId, branchedThread.ThreadId);
        }

        private void UpdateBeliefThreadMapping(string beliefId, string threadId)
        {
            if (!beliefToThreadMap.ContainsKey(beliefId))
            {
                beliefToThreadMap[beliefId] = new HashSet<string>();
            }
            beliefToThreadMap[beliefId].Add(threadId);
        }

        private async Task UpdateThreadCoherence(
            NarrativeThread thread,
            Dictionary<string, float> consolidationScores = null)
        {
            float totalCoherence = 0;
            int pairs = 0;

            for (int i = 1; i < thread.BeliefSequence.Count; i++)
            {
                var current = engine.memorySystem.Retrieve(thread.BeliefSequence[i]);
                var previous = engine.memorySystem.Retrieve(thread.BeliefSequence[i - 1]);

                if (current == null || previous == null) continue;

                var inverseState = engine.inverseFlow.GeneratePreviousStateWithContext(
                    new PradOp(new Tensor(current.LatentVector)),
                    new PradOp(new Tensor(previous.LatentVector)),
                    engine.memorySystem.TemporalRegularizer);

                float pairCoherence = inverseState.TemporalSmoothness;

                // Apply consolidation scores if available
                if (consolidationScores != null)
                {
                    if (consolidationScores.TryGetValue(current.BeliefId, out float score))
                    {
                        pairCoherence *= score;
                    }
                }

                totalCoherence += pairCoherence;
                pairs++;
            }

            thread.Coherence = pairs > 0 ? totalCoherence / pairs : 1.0f;
        }

        private Dictionary<string, float> CalculateInitialThematicWeights(
            BeliefMemoryCell belief)
        {
            return belief.NarrativeContexts.ToDictionary(
                context => context,
                context => 1.0f);
        }

        private async Task UpdateThematicWeights(NarrativeThread thread)
        {
            var weights = new Dictionary<string, float>();
            var beliefs = thread.BeliefSequence
                .Select(id => engine.memorySystem.Retrieve(id))
                .Where(b => b != null)
                .ToList();

            foreach (var belief in beliefs)
            {
                foreach (var context in belief.NarrativeContexts)
                {
                    weights[context] = weights.GetValueOrDefault(context, 0) + belief.Conviction;
                }
            }

            // Normalize weights
            float total = weights.Values.Sum();
            if (total > 0)
            {
                foreach (var key in weights.Keys.ToList())
                {
                    weights[key] /= total;
                }
            }

            thread.ThematicWeights = weights;
        }

        private async Task UpdateThreadRelationships()
        {
            foreach (var thread in threads.Values)
            {
                foreach (var otherThread in threads.Values)
                {
                    if (thread.ThreadId == otherThread.ThreadId) continue;

                    float relationship = CalculateThreadRelationship(thread, otherThread);
                    if (relationship > 0.7f)
                    {
                        thread.RelatedThreads.Add(otherThread.ThreadId);
                        otherThread.RelatedThreads.Add(thread.ThreadId);
                    }
                }
            }
        }

        private float CalculateThreadRelationship(NarrativeThread t1, NarrativeThread t2)
        {
            // Calculate thematic overlap
            float thematicOverlap = CalculateThematicOverlap(t1.ThematicWeights, t2.ThematicWeights);

            // Calculate temporal overlap
            float temporalOverlap = CalculateTemporalOverlap(t1, t2);

            return 0.7f * thematicOverlap + 0.3f * temporalOverlap;
        }

        private float CalculateThematicOverlap(
            Dictionary<string, float> weights1,
            Dictionary<string, float> weights2)
        {
            var allThemes = weights1.Keys.Union(weights2.Keys);
            float overlap = 0;

            foreach (var theme in allThemes)
            {
                float w1 = weights1.GetValueOrDefault(theme, 0);
                float w2 = weights2.GetValueOrDefault(theme, 0);
                overlap += Math.Min(w1, w2);
            }

            return overlap;
        }

        private float CalculateTemporalOverlap(NarrativeThread t1, NarrativeThread t2)
        {
            var t1Start = t1.StartTime;
            var t1End = engine.memorySystem.Retrieve(t1.BeliefSequence.Last())?.Created ?? t1.LastUpdate;
            var t2Start = t2.StartTime;
            var t2End = engine.memorySystem.Retrieve(t2.BeliefSequence.Last())?.Created ?? t2.LastUpdate;

            var overlapStart = t1Start > t2Start ? t1Start : t2Start;
            var overlapEnd = t1End < t2End ? t1End : t2End;

            if (overlapEnd <= overlapStart) return 0;

            var overlap = (overlapEnd - overlapStart).TotalHours;
            var totalSpan = Math.Max(
                (t1End - t1Start).TotalHours,
                (t2End - t2Start).TotalHours);

            return (float)(overlap / totalSpan);
        }

        internal float CalculateNarrativeOverlap(
            HashSet<string> contexts1,
            HashSet<string> contexts2)
        {
            if (!contexts1.Any() || !contexts2.Any()) return 0;

            float overlap = contexts1.Intersect(contexts2).Count() /
                          (float)Math.Max(contexts1.Count, contexts2.Count);

            return overlap;
        }

        private float CalculateFieldChange(FieldParameters f1, FieldParameters f2)
        {
            float curvatureDiff = Math.Abs((float)(f1.Curvature - f2.Curvature));
            float entropyDiff = Math.Abs((float)(f1.Entropy - f2.Entropy));
            float alignmentDiff = Math.Abs((float)(f1.Alignment - f2.Alignment));

            return (curvatureDiff + entropyDiff + alignmentDiff) / 3;
        }

        private async Task PruneThread(string threadId)
        {
            if (threads.TryRemove(threadId, out var thread))
            {
                // Remove thread references
                foreach (var beliefId in thread.BeliefSequence)
                {
                    if (beliefToThreadMap.TryGetValue(beliefId, out var threads))
                    {
                        threads.Remove(threadId);
                    }
                }

                // Remove from related threads
                foreach (var relatedId in thread.RelatedThreads)
                {
                    if (threads.TryGetValue(relatedId, out var relatedThread))
                    {
                        relatedThread.RelatedThreads.Remove(threadId);
                    }
                }
            }
        }

        public List<NarrativeThread> GetActiveThreads()
        {
            return threads.Values
                .Where(t => t.Coherence >= threadCoherenceThreshold)
                .OrderByDescending(t => t.LastUpdate)
                .ToList();
        }

        public List<NarrativeThread> GetThreadsForBelief(string beliefId)
        {
            if (beliefToThreadMap.TryGetValue(beliefId, out var threadIds))
            {
                return threadIds
                    .Select(id => threads.TryGetValue(id, out var thread) ? thread : null)
                    .Where(t => t != null)
                    .ToList();
            }
            return new List<NarrativeThread>();
        }
    }
}
