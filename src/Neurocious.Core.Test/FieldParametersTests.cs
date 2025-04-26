using Neurocious.Core.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Test
{
    public class FieldParametersTests
    {
        [Fact]
        public void FieldParameters_PropertiesInitializeCorrectly()
        {
            // Arrange & Act
            var parameters = new FieldParameters
            {
                Curvature = 0.5f,
                Entropy = 0.3f,
                Alignment = 0.8f
            };

            // Assert
            Assert.Equal(0.5f, parameters.Curvature);
            Assert.Equal(0.3f, parameters.Entropy);
            Assert.Equal(0.8f, parameters.Alignment);
        }
    }
}
