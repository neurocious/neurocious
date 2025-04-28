using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Chess
{
    public interface IGateMetrics
    {
        // Records
        void RecordActivation(float activation);
        void RecordCriticalPoint(float activation, int timeStep, int totalSteps, double reward);
        void RecordLeadsTo(string otherGate);

        // Queries
        float GetRecentActivationRate(int windowSize);
        float GetLeadsToStrength(string otherGate);
        List<(float position, float importance)> GetCriticalActivationProfile();
    }
}
