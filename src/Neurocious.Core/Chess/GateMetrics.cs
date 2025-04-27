using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Chess
{
    public class GateMetrics
    {
        private readonly Queue<(DateTime time, float activation)> recentActivations;
        private readonly Dictionary<string, Queue<float>> temporalRelations;
        private readonly List<CriticalPoint> criticalPoints;
        private const int MAX_HISTORY = 1000;
        private const int MAX_CRITICAL_POINTS = 100;

        public GateMetrics()
        {
            recentActivations = new Queue<(DateTime, float)>();
            temporalRelations = new Dictionary<string, Queue<float>>();
            criticalPoints = new List<CriticalPoint>();
        }

        public void RecordActivation(float activation)
        {
            recentActivations.Enqueue((DateTime.UtcNow, activation));
            if (recentActivations.Count > MAX_HISTORY)
            {
                recentActivations.Dequeue();
            }
        }

        public void RecordCriticalPoint(float activation, int timeStep, int totalSteps, double reward)
        {
            criticalPoints.Add(new CriticalPoint
            {
                Activation = activation,
                RelativePosition = timeStep / (float)totalSteps,
                Reward = reward,
                Timestamp = DateTime.UtcNow
            });

            if (criticalPoints.Count > MAX_CRITICAL_POINTS)
            {
                criticalPoints.RemoveAt(0);
            }
        }

        public void RecordLeadsTo(string otherGate)
        {
            if (!temporalRelations.ContainsKey(otherGate))
            {
                temporalRelations[otherGate] = new Queue<float>();
            }

            temporalRelations[otherGate].Enqueue(1.0f);
            if (temporalRelations[otherGate].Count > MAX_HISTORY)
            {
                temporalRelations[otherGate].Dequeue();
            }
        }

        public float GetRecentActivationRate(int windowSize)
        {
            var recent = recentActivations
                .TakeLast(windowSize)
                .ToList();

            if (!recent.Any()) return 0;

            return recent.Count(a => a.activation > 0.5f) / (float)recent.Count;
        }

        public float GetLeadsToStrength(string otherGate)
        {
            if (!temporalRelations.ContainsKey(otherGate))
                return 0;

            return temporalRelations[otherGate].Average();
        }

        public List<(float position, float importance)> GetCriticalActivationProfile()
        {
            // Only consider recent critical points
            var recentPoints = criticalPoints
                .Where(p => (DateTime.UtcNow - p.Timestamp).TotalHours < 24)
                .ToList();

            if (!recentPoints.Any())
                return new List<(float, float)>();

            // Group by position in trajectory (rounded to nearest 0.1)
            var groupedPoints = recentPoints
                .GroupBy(p => Math.Round(p.RelativePosition, 1))
                .Select(g => (
                    position: (float)g.Key,
                    importance: (float)(g.Average(p => p.Activation * p.Reward))
                ))
                .OrderBy(x => x.position)
                .ToList();

            return groupedPoints;
        }

        private class CriticalPoint
        {
            public float Activation { get; set; }
            public float RelativePosition { get; set; }
            public double Reward { get; set; }
            public DateTime Timestamp { get; set; }
        }
    }
}
