using Neurocious.Core.Common;

namespace Neurocious.Core.SpatialProbability
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
            Probability *= SpatialProbabilityNetwork.BRANCH_DECAY_RATE;
        }
    }
}
