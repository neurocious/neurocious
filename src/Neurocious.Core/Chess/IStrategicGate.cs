using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Chess
{
    public interface IStrategicGate
    {
        string Name { get; }
        float ActivationThreshold { get; }
        float Weight { get; }
        float[] LatentVector { get; }

        float CalculateActivation(PradOp state);
        void UpdateVector(float[] newVector, float learningRate);
    }
}
