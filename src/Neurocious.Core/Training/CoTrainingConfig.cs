using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Training
{
    public class CoTrainingConfig
    {
        public float Beta { get; init; } = 0.7f;      // Reduced KL weight (was 1.0)
        public float Gamma { get; init; } = 0.5f;     // Increased narrative continuity (was 0.3)
        public float Delta { get; init; } = 0.3f;     // Field alignment
        public float Eta { get; init; } = 1.0f;       // SPN loss weight
        public float EntropyScaling { get; init; } = 2.0f;  // How much entropy affects variance
        public float CurvatureScaling { get; init; } = 1.5f;  // How much curvature affects variance
        public int BatchSize { get; init; } = 32;
        public float LearningRate { get; init; } = 0.001f;
    }
}
