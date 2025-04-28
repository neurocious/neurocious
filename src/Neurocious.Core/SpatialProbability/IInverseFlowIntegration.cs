using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public interface IInverseFlowIntegration
    {
        BeliefReconstructionExplanation ReconstructPriorBelief(
            PradOp currentState,
            PradOp contextState,
            List<string> potentialAntecedents = null);

        void UpdateFromForwardDynamics(PradOp forwardField, PradResult forwardRouting);

        void AddToTemporalBuffer(PradOp state);
    }
}
