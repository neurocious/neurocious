using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Memory
{
    public class MetaBelief
    {
        public string MetaBeliefId { get; init; }
        public string TargetBeliefId { get; init; }
        public BeliefMemoryCell CurrentBelief { get; set; }
        public BeliefMemoryCell PriorBelief { get; set; }
        public DateTime TransitionTime { get; init; }
        public float TransitionConfidence { get; set; }
        public List<string> ReasoningChain { get; set; }
        public Dictionary<string, float> ContextualFactors { get; set; }
    }
}
