using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Training
{
    public interface IFieldAwareKLDivergence
    {
        PradResult CalculateKL(
            PradResult mean,
            PradResult logVar,
            PradOp latentState);

        FieldKLMetrics AnalyzeKLContribution(
            PradResult mean,
            PradResult logVar,
            PradOp latentState);
    }
}
