using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Chess
{
    public class TrajectoryMemory
    {
        public List<PradOp> Path { get; }
        public double Reward { get; }
        public DateTime Timestamp { get; }
        public Dictionary<string, List<float>> GateActivations { get; }

        public TrajectoryMemory(
            List<PradOp> path,
            double reward,
            Dictionary<string, List<float>> gateActivations)
        {
            Path = path;
            Reward = reward;
            Timestamp = DateTime.UtcNow;
            GateActivations = gateActivations;
        }
    }
}
