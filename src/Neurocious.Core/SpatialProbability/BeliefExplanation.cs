using Neurocious.Core.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public class BeliefExplanation
    {
        public string BeliefLabel { get; set; }
        public Dictionary<string, float> FeatureContributions { get; set; } = new();
        public float Confidence { get; set; }
        public FieldParameters FieldParams { get; set; }
        public string Justification { get; set; }
        public List<string> TopContributingFeatures { get; set; } = new();
        public Dictionary<string, float> CounterfactualShifts { get; set; } = new();
        public List<float[]> TrajectoryPath { get; set; } = new();
    }
}
