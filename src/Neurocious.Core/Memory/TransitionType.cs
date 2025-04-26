using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Memory
{
    public enum TransitionType
    {
        Refinement,    // Belief becomes more precise
        Revision,      // Belief changes significantly
        Reinforcement, // Belief becomes stronger
        Weakening,     // Belief becomes weaker
        Merge,         // Multiple beliefs combine
        Split         // Belief splits into multiple
    }
}
