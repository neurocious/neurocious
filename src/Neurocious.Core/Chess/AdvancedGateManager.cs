using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Chess
{
    public class AdvancedGateManager : IAdvancedGateManager
    {
        private readonly Dictionary<string, StrategicGate> gates;
        private readonly Dictionary<string, GateMetrics> gateMetrics;
        private readonly Queue<TrajectoryMemory> recentTrajectories;
        private readonly Dictionary<string, HierarchicalGate> hierarchicalGates;
        private readonly CorrelationTracker correlationTracker;

        private const int MAX_MEMORY_SIZE = 1000;
        private const float CORRELATION_THRESHOLD = 0.85f;
        private const float PRUNING_THRESHOLD = 0.05f;
        private const int PRUNING_WINDOW = 500;
        private int gateCounter = 0;

        public AdvancedGateManager(Dictionary<string, StrategicGate> initialGates)
        {
            gates = new Dictionary<string, StrategicGate>(initialGates);
            gateMetrics = new Dictionary<string, GateMetrics>();
            recentTrajectories = new Queue<TrajectoryMemory>();
            hierarchicalGates = new Dictionary<string, HierarchicalGate>();
            correlationTracker = new CorrelationTracker();

            foreach (var gate in gates.Values)
            {
                gateMetrics[gate.Name] = new GateMetrics();
            }
        }

        public Dictionary<string, StrategicGate> Gates
        {
            get => gates;
        }

        public void ProcessTrajectory(List<PradOp> path, double reward)
        {
            var trajectory = new TrajectoryMemory(path, reward, GetGateActivations(path));
            recentTrajectories.Enqueue(trajectory);

            if (recentTrajectories.Count > MAX_MEMORY_SIZE)
            {
                recentTrajectories.Dequeue();
            }

            UpdateMetrics(trajectory);
            correlationTracker.UpdateCorrelations(trajectory.GateActivations);

            if (recentTrajectories.Count >= MAX_MEMORY_SIZE)
            {
                AnalyzeForGateChanges();
            }
        }

        public List<(string Gate1, string Gate2)> GetHighlyCorrelatedPairs(float threshold = 0.85f)
        {
            return correlationTracker.GetHighlyCorrelatedPairs(threshold);
        }

        public List<string> GetInactiveGates(int window = 500, float activationThreshold = 0.05f)
        {
            return gateMetrics
                .Where(m => m.Value.GetRecentActivationRate(window) < activationThreshold)
                .Select(m => m.Key)
                .ToList();
        }

        public List<string> GetCurrentHierarchicalGates()
        {
            return hierarchicalGates.Keys.ToList();
        }

        private void UpdateMetrics(TrajectoryMemory trajectory)
        {
            // Update metrics for each gate
            foreach (var (gateName, activations) in trajectory.GateActivations)
            {
                if (!gateMetrics.ContainsKey(gateName))
                {
                    gateMetrics[gateName] = new GateMetrics();
                }

                // Record peak activation for this trajectory
                float peakActivation = activations.Max();
                gateMetrics[gateName].RecordActivation(peakActivation);

                // If this was a successful trajectory, update the gate's effectiveness
                if (trajectory.Reward > 0.7)
                {
                    // Find critical activation points (where gate activation exceeded threshold)
                    var criticalPoints = FindCriticalActivationPoints(activations);
                    if (criticalPoints.Any())
                    {
                        foreach (var point in criticalPoints)
                        {
                            gateMetrics[gateName].RecordCriticalPoint(
                                point.activation,
                                point.timeStep,
                                trajectory.Path.Count,
                                trajectory.Reward
                            );
                        }
                    }
                }
            }

            // Update temporal correlations between gates
            UpdateTemporalCorrelations(trajectory);
        }

        private List<(float activation, int timeStep)> FindCriticalActivationPoints(List<float> activations)
        {
            const float ACTIVATION_THRESHOLD = 0.7f;
            var criticalPoints = new List<(float activation, int timeStep)>();

            for (int i = 0; i < activations.Count; i++)
            {
                // Look for activation peaks
                bool isPeak = activations[i] > ACTIVATION_THRESHOLD;

                if (i > 0)
                    isPeak &= activations[i] > activations[i - 1];
                if (i < activations.Count - 1)
                    isPeak &= activations[i] >= activations[i + 1];

                if (isPeak)
                {
                    criticalPoints.Add((activations[i], i));
                }
            }

            return criticalPoints;
        }

        private void UpdateTemporalCorrelations(TrajectoryMemory trajectory)
        {
            var gates = trajectory.GateActivations.Keys.ToList();
            int trajectoryLength = trajectory.Path.Count;

            for (int i = 0; i < gates.Count; i++)
            {
                for (int j = i + 1; j < gates.Count; j++)
                {
                    var gate1 = gates[i];
                    var gate2 = gates[j];

                    var activations1 = trajectory.GateActivations[gate1];
                    var activations2 = trajectory.GateActivations[gate2];

                    // Calculate temporal relationship
                    for (int t = 0; t < trajectoryLength - 1; t++)
                    {
                        // Look for cases where one gate activates before another
                        if (activations1[t] > 0.5f && activations2[t + 1] > 0.5f)
                        {
                            gateMetrics[gate1].RecordLeadsTo(gate2);
                        }
                        else if (activations2[t] > 0.5f && activations1[t + 1] > 0.5f)
                        {
                            gateMetrics[gate2].RecordLeadsTo(gate1);
                        }
                    }
                }
            }
        }

        private Dictionary<string, List<float>> GetGateActivations(List<PradOp> path)
        {
            var activations = new Dictionary<string, List<float>>();

            foreach (var state in path)
            {
                foreach (var gate in gates.Values)
                {
                    if (!activations.ContainsKey(gate.Name))
                    {
                        activations[gate.Name] = new List<float>();
                    }
                    activations[gate.Name].Add(gate.CalculateActivation(state));
                }

                foreach (var hGate in hierarchicalGates.Values)
                {
                    if (!activations.ContainsKey(hGate.Name))
                    {
                        activations[hGate.Name] = new List<float>();
                    }
                    activations[hGate.Name].Add(hGate.CalculateActivation(state, gates));
                }
            }

            return activations;
        }

        private void AnalyzeForGateChanges()
        {
            // Check for gate fusion opportunities
            var correlatedPairs = correlationTracker.GetHighlyCorrelatedPairs(CORRELATION_THRESHOLD);
            foreach (var (gate1, gate2) in correlatedPairs)
            {
                if (gates.ContainsKey(gate1) && gates.ContainsKey(gate2))
                {
                    ConsiderGateFusion(gate1, gate2);
                }
            }

            // Check for gate pruning
            var inactiveGates = gateMetrics
                .Where(m => m.Value.GetRecentActivationRate(PRUNING_WINDOW) < PRUNING_THRESHOLD)
                .Select(m => m.Key)
                .ToList();

            foreach (var gateName in inactiveGates)
            {
                if (!IsBaseGate(gateName)) // Don't prune fundamental gates
                {
                    PruneGate(gateName);
                }
            }

            // Consider creating hierarchical gates
            AnalyzeForHierarchicalPatterns();
        }

        private void ConsiderGateFusion(string gate1Name, string gate2Name)
        {
            var gate1 = gates[gate1Name];
            var gate2 = gates[gate2Name];

            // Create fused gate vector
            var fusedVector = new float[gate1.LatentVector.Length];
            for (int i = 0; i < fusedVector.Length; i++)
            {
                fusedVector[i] = (gate1.LatentVector[i] + gate2.LatentVector[i]) / 2;
            }

            // Create new gate with combined properties
            string fusedName = $"fused_gate_{++gateCounter}";
            var fusedGate = new StrategicGate(
                name: fusedName,
                dim: fusedVector.Length,
                threshold: (gate1.ActivationThreshold + gate2.ActivationThreshold) / 2,
                weight: Math.Max(gate1.Weight, gate2.Weight)
            );

            fusedGate.UpdateVector(fusedVector, 1.0f);

            // Add new gate and remove old ones
            gates[fusedName] = fusedGate;
            gateMetrics[fusedName] = new GateMetrics();
            gates.Remove(gate1Name);
            gates.Remove(gate2Name);
            gateMetrics.Remove(gate1Name);
            gateMetrics.Remove(gate2Name);

            Console.WriteLine($"Fused gates {gate1Name} and {gate2Name} into {fusedName}");
        }

        private void PruneGate(string gateName)
        {
            gates.Remove(gateName);
            gateMetrics.Remove(gateName);

            // Remove any hierarchical gates that depended on this gate
            var dependentGates = hierarchicalGates
                .Where(g => g.Value.DependsOn(gateName))
                .Select(g => g.Key)
                .ToList();

            foreach (var dependent in dependentGates)
            {
                hierarchicalGates.Remove(dependent);
            }

            Console.WriteLine($"Pruned gate {gateName} and its dependents");
        }

        private void AnalyzeForHierarchicalPatterns()
        {
            var recentSuccesses = recentTrajectories
                .Where(t => t.Reward > 0.7)
                .ToList();

            if (recentSuccesses.Count < 50) return; // Need enough data

            foreach (var traj in recentSuccesses)
            {
                // Find gates that frequently activate together
                var coActivations = FindCoActivatingGates(traj.GateActivations);

                foreach (var group in coActivations)
                {
                    if (group.Count >= 2 && !HasHierarchicalGate(group))
                    {
                        CreateHierarchicalGate(group);
                    }
                }
            }
        }

        private List<HashSet<string>> FindCoActivatingGates(Dictionary<string, List<float>> activations)
        {
            var result = new List<HashSet<string>>();
            var gateNames = activations.Keys.ToList();

            for (int i = 0; i < gateNames.Count; i++)
            {
                var coActivating = new HashSet<string> { gateNames[i] };

                for (int j = i + 1; j < gateNames.Count; j++)
                {
                    if (AreConsistentlyCoActivating(
                        activations[gateNames[i]],
                        activations[gateNames[j]]))
                    {
                        coActivating.Add(gateNames[j]);
                    }
                }

                if (coActivating.Count > 1)
                {
                    result.Add(coActivating);
                }
            }

            return result;
        }

        private bool AreConsistentlyCoActivating(List<float> activations1, List<float> activations2)
        {
            const float ACTIVATION_THRESHOLD = 0.7f;
            const float CO_ACTIVATION_RATE = 0.8f;

            int coActivations = 0;
            for (int i = 0; i < activations1.Count; i++)
            {
                if (activations1[i] > ACTIVATION_THRESHOLD && activations2[i] > ACTIVATION_THRESHOLD)
                {
                    coActivations++;
                }
            }

            return (float)coActivations / activations1.Count > CO_ACTIVATION_RATE;
        }

        private void CreateHierarchicalGate(HashSet<string> componentGates)
        {
            string name = $"hierarchical_gate_{++gateCounter}";
            var hGate = new HierarchicalGate(name, componentGates.ToList());
            hierarchicalGates[name] = hGate;

            Console.WriteLine($"Created hierarchical gate {name} from {string.Join(", ", componentGates)}");
        }

        private bool HasHierarchicalGate(HashSet<string> components)
        {
            return hierarchicalGates.Values.Any(g => g.HasSameComponents(components));
        }

        private bool IsBaseGate(string gateName)
        {
            return gateName.StartsWith("material_") ||
                   gateName.StartsWith("position_") ||
                   gateName.StartsWith("king_");
        }

        public class HierarchicalGate
        {
            public string Name { get; }
            private readonly List<string> componentGates;
            private readonly float threshold;

            public HierarchicalGate(string name, List<string> components, float threshold = 0.7f)
            {
                Name = name;
                componentGates = components;
                this.threshold = threshold;
            }

            public float CalculateActivation(PradOp state, Dictionary<string, StrategicGate> gates)
            {
                float totalActivation = 0;
                int validComponents = 0;

                foreach (var gateName in componentGates)
                {
                    if (gates.TryGetValue(gateName, out var gate))
                    {
                        totalActivation += gate.CalculateActivation(state);
                        validComponents++;
                    }
                }

                if (validComponents == 0) return 0;

                float meanActivation = totalActivation / validComponents;
                return meanActivation > threshold ? meanActivation : 0;
            }

            public bool DependsOn(string gateName)
            {
                return componentGates.Contains(gateName);
            }

            public bool HasSameComponents(HashSet<string> components)
            {
                return components.SetEquals(componentGates);
            }
        }

        private class CorrelationTracker
        {
            private readonly Dictionary<(string, string), Queue<(float, float)>> correlationHistory;
            private const int HISTORY_SIZE = 1000;

            public CorrelationTracker()
            {
                correlationHistory = new Dictionary<(string, string), Queue<(float, float)>>();
            }

            public void UpdateCorrelations(Dictionary<string, List<float>> activations)
            {
                var gates = activations.Keys.ToList();

                for (int i = 0; i < gates.Count; i++)
                {
                    for (int j = i + 1; j < gates.Count; j++)
                    {
                        var pair = (gates[i], gates[j]);
                        if (!correlationHistory.ContainsKey(pair))
                        {
                            correlationHistory[pair] = new Queue<(float, float)>();
                        }

                        for (int k = 0; k < activations[gates[i]].Count; k++)
                        {
                            var history = correlationHistory[pair];
                            history.Enqueue((activations[gates[i]][k], activations[gates[j]][k]));

                            while (history.Count > HISTORY_SIZE)
                            {
                                history.Dequeue();
                            }
                        }
                    }
                }
            }

            public List<(string Gate1, string Gate2)> GetHighlyCorrelatedPairs(float threshold)
            {
                var correlatedPairs = new List<(string, string)>();

                foreach (var entry in correlationHistory)
                {
                    var correlation = CalculateCorrelation(entry.Value.ToList());
                    if (correlation > threshold)
                    {
                        correlatedPairs.Add(entry.Key);
                    }
                }

                return correlatedPairs;
            }

            private float CalculateCorrelation(List<(float x, float y)> points)
            {
                if (points.Count < 2) return 0;

                float meanX = points.Average(p => p.x);
                float meanY = points.Average(p => p.y);

                float covariance = points.Sum(p => (p.x - meanX) * (p.y - meanY));
                float varX = points.Sum(p => (p.x - meanX) * (p.x - meanX));
                float varY = points.Sum(p => (p.y - meanY) * (p.y - meanY));

                if (varX < float.Epsilon || varY < float.Epsilon) return 0;

                return covariance / MathF.Sqrt(varX * varY);
            }
        }
    }
}
