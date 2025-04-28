using Neurocious.Core.Common;
using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.EnhancedVariationalAutoencoder
{
    public interface IFieldRegularizer
    {
        PradResult ComputeLoss(FieldParameters fieldParams);
    }
}
