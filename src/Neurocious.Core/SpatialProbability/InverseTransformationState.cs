using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public class InverseTransformationState
    {
        public PradResult WarpedState { get; init; }
        public PradResult ContextualRouting { get; init; }
        public float TemporalSmoothness { get; init; }
        public Dictionary<string, float> ConfidenceMetrics { get; init; }
        public float[] FlowDirection { get; init; }
    }
}
