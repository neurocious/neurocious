using Neurocious.Core.SpatialProbability;
using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Neurocious.Core.Memory.EpistemicMemoryEngine;

namespace Neurocious.Core.Memory
{
    public class MetaBeliefSystem
    {
        private readonly EpistemicMemoryEngine engine;
        private readonly ConcurrentDictionary<string, MetaBelief> metaBeliefs;
        private readonly Dictionary<string, HashSet<string>> beliefToMetaMap;

        public MetaBeliefSystem(EpistemicMemoryEngine engine)
        {
            this.engine = engine;
            this.metaBeliefs = new ConcurrentDictionary<string, MetaBelief>();
            this.beliefToMetaMap = new Dictionary<string, HashSet<string>>();
        }

        public async Task ProcessNewBelief(BeliefMemoryCell belief)
        {
            // Find related beliefs that might have influenced this one
            var relatedBeliefs = await FindRelatedBeliefs(belief);

            foreach (var related in relatedBeliefs)
            {
                var transition = await AnalyzeBeliefTransition(related, belief);
                if (transition != null)
                {
                    await CreateMetaBelief(related.BeliefId, belief.BeliefId, transition);
                }
            }
        }

        public async Task<List<MetaBelief>> TraceBeliefEvolution(string beliefId)
        {
            if (!beliefToMetaMap.TryGetValue(beliefId, out var metaBeliefIds))
                return new List<MetaBelief>();

            return metaBeliefIds
                .Select(id => metaBeliefs[id])
                .OrderBy(mb => mb.Created)
                .ToList();
        }

        private async Task<BeliefTransition> AnalyzeBeliefTransition(
            BeliefMemoryCell from,
            BeliefMemoryCell to)
        {
            // Use the inverse flow field to validate transition
            var inverseState = engine.inverseFlow.GeneratePreviousStateWithContext(
                new PradOp(new Tensor(to.LatentVector)),
                new PradOp(new Tensor(from.LatentVector)),
                engine.memorySystem.TemporalRegularizer);

            if (inverseState.TemporalSmoothness < 0.3f)
                return null; // Not a valid transition

            // Analyze the type of transition
            var metrics = CalculateTransitionMetrics(from, to);
            var type = DetermineTransitionType(metrics);

            return new BeliefTransition
            {
                FromBeliefId = from.BeliefId,
                ToBeliefId = to.BeliefId,
                Type = type,
                Strength = inverseState.TemporalSmoothness,
                Metrics = metrics
            };
        }

        private Dictionary<string, float> CalculateTransitionMetrics(
            BeliefMemoryCell from,
            BeliefMemoryCell to)
        {
            return new Dictionary<string, float>
            {
                ["conviction_delta"] = to.Conviction - from.Conviction,
                ["entropy_delta"] = (float)(to.FieldParams.Entropy - from.FieldParams.Entropy),
                ["curvature_delta"] = (float)(to.FieldParams.Curvature - from.FieldParams.Curvature),
                ["narrative_overlap"] = CalculateNarrativeOverlap(from.NarrativeContexts, to.NarrativeContexts),
                ["temporal_gap"] = (float)(to.Created - from.Created).TotalHours
            };
        }

        private TransitionType DetermineTransitionType(Dictionary<string, float> metrics)
        {
            float convictionDelta = metrics["conviction_delta"];
            float entropyDelta = metrics["entropy_delta"];
            float narrativeOverlap = metrics["narrative_overlap"];

            if (entropyDelta < -0.3f && convictionDelta > 0.1f)
                return TransitionType.Refinement;
            else if (Math.Abs(entropyDelta) > 0.5f || narrativeOverlap < 0.3f)
                return TransitionType.Revision;
            else if (convictionDelta > 0.2f)
                return TransitionType.Reinforcement;
            else if (convictionDelta < -0.2f)
                return TransitionType.Weakening;
            else
                return TransitionType.Refinement;
        }

        private float CalculateNarrativeOverlap(
            HashSet<string> contexts1,
            HashSet<string> contexts2)
        {
            if (!contexts1.Any() || !contexts2.Any()) return 0;

            float overlap = contexts1.Intersect(contexts2).Count() /
                          (float)Math.Max(contexts1.Count, contexts2.Count);

            return overlap;
        }

        private async Task<List<BeliefMemoryCell>> FindRelatedBeliefs(
            BeliefMemoryCell belief)
        {
            // Get beliefs sharing narrative contexts
            var narrativeRelated = belief.NarrativeContexts
                .SelectMany(ctx => engine.memorySystem.QueryByNarrative(ctx))
                .Where(b => b.BeliefId != belief.BeliefId)
                .ToList();

            // Filter to recent beliefs that could be causal antecedents
            return narrativeRelated
                .Where(b => b.Created < belief.Created &&
                           (belief.Created - b.Created).TotalHours < 24)
                .OrderByDescending(b => b.Created)
                .Take(5)
                .ToList();
        }

        private async Task CreateMetaBelief(
            string fromId,
            string toId,
            BeliefTransition transition)
        {
            var metaBelief = new MetaBelief
            {
                MetaBeliefId = Guid.NewGuid().ToString(),
                TargetBeliefId = toId,
                TransitionConfidence = transition.Strength,
                ReasoningChain = GenerateReasoningChain(transition),
                ContextualFactors = GenerateContextualFactors(transition),
                Transition = transition,
                Created = DateTime.UtcNow
            };

            metaBeliefs[metaBelief.MetaBeliefId] = metaBelief;

            // Update belief to meta-belief mapping
            if (!beliefToMetaMap.ContainsKey(fromId))
                beliefToMetaMap[fromId] = new HashSet<string>();
            if (!beliefToMetaMap.ContainsKey(toId))
                beliefToMetaMap[toId] = new HashSet<string>();

            beliefToMetaMap[fromId].Add(metaBelief.MetaBeliefId);
            beliefToMetaMap[toId].Add(metaBelief.MetaBeliefId);
        }

        private List<string> GenerateReasoningChain(BeliefTransition transition)
        {
            var reasoning = new List<string>();

            // Add transition type explanation
            reasoning.Add($"Belief underwent {transition.Type} transition");

            // Add metric-based reasons
            foreach (var (metric, value) in transition.Metrics)
            {
                if (Math.Abs(value) > 0.3f)
                {
                    reasoning.Add($"Significant change in {metric}: {value:F2}");
                }
            }

            return reasoning;
        }

        private Dictionary<string, float> GenerateContextualFactors(BeliefTransition transition)
        {
            return new Dictionary<string, float>
            {
                ["transition_strength"] = transition.Strength,
                ["temporal_proximity"] = Math.Max(0, 1 - transition.Metrics["temporal_gap"] / 24),
                ["narrative_continuity"] = transition.Metrics["narrative_overlap"]
            };
        }

        public async Task UpdateMetaBeliefs(ConsolidationState state)
        {
            foreach (var belief in state.RecentMemories)
            {
                if (beliefToMetaMap.TryGetValue(belief.BeliefId, out var metaBeliefIds))
                {
                    foreach (var metaBeliefId in metaBeliefIds)
                    {
                        if (metaBeliefs.TryGetValue(metaBeliefId, out var metaBelief))
                        {
                            // Update transition confidence based on consolidation score
                            if (state.ConsolidationScores.TryGetValue(belief.BeliefId, out float score))
                            {
                                metaBelief.TransitionConfidence *= score;
                                metaBelief.Transition.Strength *= score;
                            }
                        }
                    }
                }
            }
        }
    }
}
