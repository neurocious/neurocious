using Neurocious.Core.SpatialProbability;
using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.Chess
{
    public static class SpatialProbabilityNetworkExtensions
    {
        public static void UpdateFieldsWithEntropyAwareness(
            this SpatialProbabilityNetwork spn,
            PradResult route,
            PradResult reward,
            List<PradOp> sequence)
        {
            // Get current field parameters
            var (_, _, fieldParams) = spn.RouteStateInternal(sequence);

            // Calculate adaptive learning rate
            float adaptiveLearningRate = 0.01f *
                (1 - (float)fieldParams.Entropy) *        // Learn less in high-entropy regions
                (1 / (1 + (float)fieldParams.Curvature)); // Learn less in unstable regions

            // Calculate entropy-aware update scaling
            float entropyScaling = 1.0f - (float)fieldParams.Entropy;

            // Update vector field with entropy awareness
            var fieldUpdate = route.Then(r => {
                var learningRateTensor = new Tensor(
                    r.Result.Shape,
                    adaptiveLearningRate * entropyScaling
                );
                return r.Mul(reward.Result).Mul(learningRateTensor);
            });

            // Apply weighted update to vector field
            var alignmentWeight = Math.Abs(fieldParams.Alignment);
            spn.VectorField = new PradOp(
                spn.VectorField.Mul(new Tensor(
                    spn.VectorField.CurrentShape,
                    1 - alignmentWeight * adaptiveLearningRate * entropyScaling)
                ).Result.Add(fieldUpdate.Result)
            );

            // Normalize vector field
            spn.VectorField = new PradOp(spn.NormalizeVectorField(spn.VectorField.CurrentTensor));

            // Update probability field with entropy regulation
            var probabilityUpdate = route.Then(r => {
                var learningRateTensor = new Tensor(r.Result.Shape, adaptiveLearningRate);
                return r.Mul(reward.Result).Mul(learningRateTensor);
            });

            // Update entropy field with adaptive decay
            float entropyDecay = 0.999f + 0.001f * entropyScaling; // Slower decay in high-entropy regions
            spn.EntropyField = new PradOp(
                spn.EntropyField.Mul(new Tensor(spn.EntropyField.CurrentShape, entropyDecay)).Result
                .Add(probabilityUpdate.Result)
            );

            // Renormalize probability field
            spn.EntropyField = spn.EntropyField.Then(PradOp.SoftmaxOp);

            // Apply entropy regularization
            float regularizationStrength = 0.03f * (1.0f - entropyScaling);
            var uniformDistribution = new Tensor(
                spn.EntropyField.CurrentShape,
                Enumerable.Repeat(
                    1.0 / spn.EntropyField.CurrentShape[0],
                    spn.EntropyField.CurrentShape[0]
                ).ToArray()
            );

            spn.EntropyField = new PradOp(
                spn.EntropyField.Result.ElementwiseMultiply(
                    new Tensor(spn.EntropyField.CurrentShape, 1 - regularizationStrength)
                ).Add(
                    uniformDistribution.ElementwiseMultiply(
                        new Tensor(uniformDistribution.Shape, regularizationStrength)
                    )
                )
            );
        }
    }
}
