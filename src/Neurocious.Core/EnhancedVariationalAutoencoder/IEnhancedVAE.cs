using Neurocious.Core.Common;
using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.EnhancedVariationalAutoencoder
{
    public interface IEnhancedVAE
    {
        int LatentDimension { get; }

        (PradResult mean, PradResult logVar) EncodeSequence(List<PradOp> sequence);

        (PradResult reconstruction, FieldParameters fieldParams) DecodeWithField(PradOp latentVector);

        (PradResult reconstruction, FieldParameters fieldParams, PradResult mean, PradResult logVar) ForwardSequence(List<PradOp> sequence);

        double EstimateLatentCurvature(Tensor z);

        double EstimateLatentEntropy(Tensor z);

        FieldParameters ExtractFieldParameters(PradOp state);
    }
}
