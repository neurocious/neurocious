using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Chess
{
    public interface IAdvancedGateManager
    {
        // Core
        void ProcessTrajectory(List<PradOp> path, double reward);

        // Access gates
        Dictionary<string, StrategicGate> Gates { get; }

        // Diagnostics / Management
        List<(string Gate1, string Gate2)> GetHighlyCorrelatedPairs(float threshold = 0.85f);
        List<string> GetInactiveGates(int window = 500, float threshold = 0.05f);
        List<string> GetCurrentHierarchicalGates();
    }
}
