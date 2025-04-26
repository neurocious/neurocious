using Neurocious.Core.SpatialProbability;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Test
{
    public class GeometricFieldTests
    {
        [Fact]
        public void GeometricField_PropertiesInitializeCorrectly()
        {
            // Arrange & Act
            var field = new GeometricField
            {
                Direction = new float[] { 0.1f, 0.2f, 0.3f },
                Strength = 0.8f,
                LocalCurvature = 0.4f,
                LocalDivergence = 0.2f,
                LocalRotation = 0.1f
            };

            // Assert
            Assert.NotNull(field.Direction);
            Assert.Equal(3, field.Direction.Length);
            Assert.Equal(0.8f, field.Strength);
            Assert.Equal(0.4f, field.LocalCurvature);
            Assert.Equal(0.2f, field.LocalDivergence);
            Assert.Equal(0.1f, field.LocalRotation);
        }
    }
}
