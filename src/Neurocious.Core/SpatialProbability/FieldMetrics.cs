using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public class FieldMetrics
    {
        public float AverageFlowStrength { get; init; }
        public float DirectionalCoherence { get; init; }
        public float BackwardConfidence { get; init; }
        public Dictionary<string, float> FieldStatistics { get; init; }
    }
}
