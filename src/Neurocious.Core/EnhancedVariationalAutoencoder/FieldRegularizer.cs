using Neurocious.Core.Common;
using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.EnhancedVariationalAutoencoder
{
    public class FieldRegularizer : IFieldRegularizer
    {
        public PradResult ComputeLoss(FieldParameters fieldParams)
        {
            var losses = new List<double>();

            // Curvature smoothness
            losses.Add(Math.Pow(fieldParams.Curvature, 2) * 0.1);

            // Entropy bounds
            losses.Add(Math.Max(0, fieldParams.Entropy - 1) * 10);
            losses.Add(Math.Max(0, -fieldParams.Entropy) * 10);

            // Alignment regularization
            losses.Add(Math.Pow(fieldParams.Alignment, 2) * 0.05);

            return new PradOp(new Tensor(new[] { 1 },
                new[] { losses.Average() }));
        }
    }
}
