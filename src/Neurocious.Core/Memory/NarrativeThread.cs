using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Memory
{
    public class NarrativeThread
    {
        public string ThreadId { get; init; }
        public List<BeliefMemoryCell> Sequence { get; init; }
        public float Coherence { get; set; }
        public DateTime StartTime { get; init; }
        public DateTime LastUpdate { get; set; }
        public Dictionary<string, float> ThematicWeights { get; set; }
        public HashSet<string> RelatedThreads { get; set; }
        public Dictionary<string, List<string>> CausalLinks { get; set; }
    }
}
