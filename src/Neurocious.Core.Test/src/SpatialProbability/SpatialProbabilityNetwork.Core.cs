using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.SpatialProbability
{
    public partial class SpatialProbabilityNetwork
    {
        private readonly int stateDim;
        private readonly int[] fieldShape;
        private readonly int vectorDim;
        private readonly int bufferSize;

        // Core fields
        private PradOp vectorField;        // Learned directional fields for belief tendency
        private PradOp curvatureField;     // Field curvature for belief stability
        private PradOp entropyField;       // Field entropy for uncertainty
        private PradOp alignmentField;     // Field alignment for belief coherence
        private readonly Queue<Tensor> temporalBuffer;  // Historical state buffer

        // World branching and exploration
        private readonly List<SpatialProbabilityNetwork> branches;
        private readonly Dictionary<string, int> routeVisits;
        private readonly Random random;
        private readonly List<PradOp> trainableParameters;

        // Field parameters
        private const float LEARNING_RATE = 0.01f;
        private const float FIELD_DECAY = 0.999f;
        private const float MIN_FIELD_STRENGTH = 1e-6f;
        private const float NOVELTY_WEIGHT = 0.1f;
        internal const float BRANCH_DECAY_RATE = 0.95f;
    }
}
