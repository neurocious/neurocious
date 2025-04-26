using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Memory
{
    public class MemoryRetrievalResult
    {
        public BeliefMemoryCell MainBelief { get; init; }
        public List<BeliefMemoryCell> RelatedBeliefs { get; init; }
        public float TemporalCoherence { get; init; }
        public Dictionary<string, float> RetrievalMetrics { get; init; }
    }
}
