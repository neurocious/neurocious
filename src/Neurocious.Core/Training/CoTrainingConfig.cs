using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Training
{
    public class CoTrainingConfig
    {
        // Original weights
        public float Beta { get; init; } = 0.7f;      // KL weight (reduced from 1.0)
        public float Gamma { get; init; } = 0.5f;     // Narrative continuity weight
        public float Delta { get; init; } = 0.3f;     // Field alignment weight
        public float Eta { get; init; } = 1.0f;       // SPN loss weight

        // New weights for full epistemic training
        public float PolicyWeight { get; init; } = 0.5f;     // Policy prediction weight
        public float ReflexWeight { get; init; } = 0.3f;     // Reflex behavior weight
        public float PredictionWeight { get; init; } = 0.4f; // Future prediction weight

        // Uncertainty scaling factors
        public float EntropyScaling { get; init; } = 2.0f;   // How much entropy affects variance
        public float CurvatureScaling { get; init; } = 1.5f; // How much curvature affects variance

        // Training parameters
        public int BatchSize { get; init; } = 32;
        public float LearningRate { get; init; } = 0.001f;
    }
}
