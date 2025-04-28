using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Training
{
    public class FieldKLMetrics
    {
        public float BaseKL { get; init; }
        public float FieldAlignmentScore { get; init; }
        public float UncertaintyAdaptation { get; init; }
        public Dictionary<string, float> DetailedMetrics { get; init; }
    }
}
