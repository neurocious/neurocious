using Neurocious.Core.Common;
using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public interface ISpatialProbabilityNetwork
    {
        // Core usage
        (PradResult routing, PradResult confidence, PradResult policy, PradResult reflexes, PradResult predictions,
         FieldParameters fieldParams, BeliefExplanation explanation, BeliefReconstructionExplanation inverseExplanation)
         ProcessState(PradOp state);

        void UpdateFields(PradResult route, PradResult reward, List<PradOp> sequence);
        void UpdateFieldParameters(FieldParameters fieldParams);
        void Back(PradResult loss);
        void ClearTemporalBuffer();

        // Branching
        List<WorldBranch> SimulateWorldBranches(FieldParameters currentState, int numBranches = 3);

        // Diagnostics and analysis
        Dictionary<string, float> GetDiagnostics();
        FlowPattern AnalyzeFieldFlow(PradOp state, int steps = 10);
        BeliefReconstructionExplanation ReconstructPriorBelief(PradOp currentState, List<string> potentialAntecedents = null);

        // Properties
        PradOp VectorField { get; set; }
        PradOp EntropyField { get; set; }
    }
}
