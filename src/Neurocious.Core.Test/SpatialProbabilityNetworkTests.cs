using Neurocious.Core.SpatialProbability;
using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.PRAD.SpatialProbabilityNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Test
{
    public class SpatialProbabilityNetworkTests
    {
        [Fact]
        public void ConsistentRouting_ProducesStableFlow()
        {
            // Arrange
            var state = CreateRandomState(10);
            var patterns = new List<FlowPattern>();

            // Act
            for (int i = 0; i < 5; i++)
            {
                patterns.Add(spn.AnalyzeFieldFlow(state));
            }

            // Assert
            var stabilityVariance = CalculateVariance(patterns.Select(p => p.Stability));
            Assert.True(stabilityVariance < 0.1); // Flow patterns should be relatively consistent
        }

        private PradOp CreateRandomState(int dim, float strength = 1.0f)
        {
            var random = new Random();
            var data = new double[dim];
            for (int i = 0; i < dim; i++)
            {
                data[i] = (random.NextDouble() * 2 - 1) * strength;
            }
            return new PradOp(new Tensor(new[] { dim }, data));
        }

        private double CalculateVariance(IEnumerable<float> values)
        {
            var list = values.ToList();
            var mean = list.Average();
            var sumSquares = list.Sum(x => Math.Pow(x - mean, 2));
            return sumSquares / list.Count;
        }
    }
}
