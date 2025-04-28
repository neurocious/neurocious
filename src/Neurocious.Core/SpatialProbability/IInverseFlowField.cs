using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public interface IInverseFlowField
    {
        InverseTransformationState GeneratePreviousStateWithContext(
            PradOp currentState,
            PradOp context,
            TemporalRegularizer temporalRegularizer);

        void UpdateFromForwardField(PradOp forwardField, PradResult forwardRouting);

        FieldMetrics CalculateMetrics();
    }
}
