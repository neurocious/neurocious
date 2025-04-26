using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Neurocious.Core.Memory.MetaBeliefSystem;

namespace Neurocious.Core.Memory
{
    public class BeliefTransition
    {
        public string FromBeliefId { get; init; }
        public string ToBeliefId { get; init; }
        public TransitionType Type { get; init; }
        public float Strength { get; set; }
        public Dictionary<string, float> Metrics { get; init; }
    }
}
