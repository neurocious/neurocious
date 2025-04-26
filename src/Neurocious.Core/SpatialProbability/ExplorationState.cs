using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public class ExplorationState
    {
        public float NoveltyScore { get; set; }
        public float UncertaintyScore { get; set; }
        public float ExplorationRate { get; set; }
    }
}
