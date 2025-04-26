using Neurocious.Core.SpatialProbability;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Test
{
    public class FlowPatternTests
    {
        [Fact]
        public void FlowPattern_PropertiesInitializeCorrectly()
        {
            // Arrange & Act
            var pattern = new FlowPattern
            {
                Position = new float[] { 1, 2, 3 },
                FlowDirection = new float[] { 0.1f, 0.2f, 0.3f },
                LocalCurvature = 0.5f,
                LocalEntropy = 0.3f,
                LocalAlignment = 0.8f,
                Stability = 0.9f
            };

            // Assert
            Assert.NotNull(pattern.Position);
            Assert.NotNull(pattern.FlowDirection);
            Assert.Equal(3, pattern.Position.Length);
            Assert.Equal(3, pattern.FlowDirection.Length);
            Assert.Equal(0.5f, pattern.LocalCurvature);
            Assert.Equal(0.3f, pattern.LocalEntropy);
            Assert.Equal(0.8f, pattern.LocalAlignment);
            Assert.Equal(0.9f, pattern.Stability);
        }
    }
}
