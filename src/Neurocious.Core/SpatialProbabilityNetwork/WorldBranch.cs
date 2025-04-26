using Neurocious.Core.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbabilityNetwork
{
    public class WorldBranch
    {
        public SpatialProbabilityNetwork Network { get; }
        public float Value { get; private set; }
        public float Probability { get; private set; }
        public FieldParameters InitialState { get; }

        public WorldBranch(SpatialProbabilityNetwork network, FieldParameters state, float probability)
        {
            Network = network;
            InitialState = state;
            Probability = probability;
            Value = 0;
        }

        public void UpdateValue(float newValue)
        {
            Value = newValue;
            Probability *= BRANCH_DECAY_RATE;
        }
    }
}
