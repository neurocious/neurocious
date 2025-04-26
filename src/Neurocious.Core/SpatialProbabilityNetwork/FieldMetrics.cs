using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbabilityNetwork
{
    public class FieldMetrics
    {
        public float GlobalEntropy { get; set; }
        public float GlobalCurvature { get; set; }
        public float GlobalAlignment { get; set; }
        public float BeliefStability { get; set; }
        public float CoherenceScore { get; set; }
    }
}
