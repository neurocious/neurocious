using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public class GeometricField
    {
        public float[] Direction { get; set; }
        public float Strength { get; set; }
        public float LocalCurvature { get; set; }
        public float LocalDivergence { get; set; }
        public float LocalRotation { get; set; }
    }
}
