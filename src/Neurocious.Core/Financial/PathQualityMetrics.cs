using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class PathQualityMetrics
    {
        private const double SMOOTHNESS_THRESHOLD = 0.1;
        private const double CONSISTENCY_THRESHOLD = 0.2;
        private const double EFFICIENCY_RADIUS = 0.1;

        public double CalculatePathSmoothness(List<List<PradOp>> paths)
        {
            if (paths.Count == 0) return 0;

            double totalSmoothness = 0;

            foreach (var path in paths)
            {
                if (path.Count < 3) continue;

                double pathSmoothness = 0;
                for (int i = 1; i < path.Count - 1; i++)
                {
                    // Calculate acceleration (second derivative) at each point
                    var prevState = path[i - 1].Result.Data;
                    var currState = path[i].Result.Data;
                    var nextState = path[i + 1].Result.Data;

                    var velocity1 = CalculateVelocity(prevState, currState);
                    var velocity2 = CalculateVelocity(currState, nextState);
                    var acceleration = CalculateVelocity(velocity1, velocity2);

                    // Penalize high acceleration (sudden changes in direction)
                    double accelerationMagnitude = Math.Sqrt(acceleration.Sum(a => a * a));
                    double smoothnessPenalty = Math.Exp(-accelerationMagnitude / SMOOTHNESS_THRESHOLD);
                    pathSmoothness += smoothnessPenalty;
                }

                totalSmoothness += pathSmoothness / (path.Count - 2);
            }

            return totalSmoothness / paths.Count;
        }

        public double CalculatePathConsistency(List<List<PradOp>> paths)
        {
            if (paths.Count < 2) return 0;

            double totalConsistency = 0;
            int comparisons = 0;

            // Compare each path with every other path
            for (int i = 0; i < paths.Count - 1; i++)
            {
                for (int j = i + 1; j < paths.Count; j++)
                {
                    double pathSimilarity = CalculatePathSimilarity(paths[i], paths[j]);
                    totalConsistency += pathSimilarity;
                    comparisons++;
                }
            }

            return comparisons > 0 ? totalConsistency / comparisons : 0;
        }

        public double CalculatePathEfficiency(List<List<PradOp>> paths)
        {
            if (paths.Count == 0) return 0;

            double totalEfficiency = 0;

            foreach (var path in paths)
            {
                if (path.Count < 2) continue;

                // Calculate total path length
                double pathLength = 0;
                for (int i = 1; i < path.Count; i++)
                {
                    var distance = CalculateStateDistance(
                        path[i - 1].Result.Data,
                        path[i].Result.Data);
                    pathLength += distance;
                }

                // Calculate direct distance between start and end
                var directDistance = CalculateStateDistance(
                    path.First().Result.Data,
                    path.Last().Result.Data);

                // Efficiency is ratio of direct distance to path length
                // (adjusted to handle very small distances)
                double efficiency = pathLength > EFFICIENCY_RADIUS
                    ? directDistance / pathLength
                    : 1.0;

                totalEfficiency += efficiency;
            }

            return totalEfficiency / paths.Count;
        }

        private double[] CalculateVelocity(double[] state1, double[] state2)
        {
            var velocity = new double[state1.Length];
            for (int i = 0; i < state1.Length; i++)
            {
                velocity[i] = state2[i] - state1[i];
            }
            return velocity;
        }

        private double CalculatePathSimilarity(List<PradOp> path1, List<PradOp> path2)
        {
            int length = Math.Min(path1.Count, path2.Count);
            if (length < 2) return 0;

            double totalSimilarity = 0;

            // Compare path segments
            for (int i = 1; i < length; i++)
            {
                var direction1 = CalculateVelocity(
                    path1[i - 1].Result.Data,
                    path1[i].Result.Data);

                var direction2 = CalculateVelocity(
                    path2[i - 1].Result.Data,
                    path2[i].Result.Data);

                // Calculate cosine similarity between directions
                double similarity = CalculateCosineSimilarity(direction1, direction2);

                // Apply threshold to focus on significant movements
                if (VectorMagnitude(direction1) > CONSISTENCY_THRESHOLD &&
                    VectorMagnitude(direction2) > CONSISTENCY_THRESHOLD)
                {
                    totalSimilarity += similarity;
                }
            }

            return totalSimilarity / (length - 1);
        }

        private double CalculateStateDistance(double[] state1, double[] state2)
        {
            double sumSquaredDiff = 0;
            for (int i = 0; i < state1.Length; i++)
            {
                double diff = state1[i] - state2[i];
                sumSquaredDiff += diff * diff;
            }
            return Math.Sqrt(sumSquaredDiff);
        }

        private double CalculateCosineSimilarity(double[] v1, double[] v2)
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

            double denominator = Math.Sqrt(norm1) * Math.Sqrt(norm2);
            return denominator > 1e-10 ? dotProduct / denominator : 0;
        }

        private double VectorMagnitude(double[] vector)
        {
            return Math.Sqrt(vector.Sum(x => x * x));
        }
    }
}
