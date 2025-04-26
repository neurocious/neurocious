using Neurocious.Core.SpatialProbability;

namespace Neurocious.Core.Test;

public class FieldMetricsTests
{
    [Fact]
    public void FieldMetrics_PropertiesInitializeCorrectly()
    {
        // Arrange & Act
        var metrics = new FieldMetrics
        {
            GlobalEntropy = 0.5f,
            GlobalCurvature = 0.4f,
            GlobalAlignment = 0.7f,
            BeliefStability = 0.8f,
            CoherenceScore = 0.9f
        };

        // Assert
        Assert.Equal(0.5f, metrics.GlobalEntropy);
        Assert.Equal(0.4f, metrics.GlobalCurvature);
        Assert.Equal(0.7f, metrics.GlobalAlignment);
        Assert.Equal(0.8f, metrics.BeliefStability);
        Assert.Equal(0.9f, metrics.CoherenceScore);
    }
}
