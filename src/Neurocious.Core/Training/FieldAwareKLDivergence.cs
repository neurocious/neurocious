using Neurocious.Core.SpatialProbability;
using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.Training
{
    public class FieldAwareKLDivergence
    {
        private readonly SpatialProbabilityNetwork spn;

        public FieldAwareKLDivergence(SpatialProbabilityNetwork spn)
        {
            this.spn = spn;
        }

        public PradResult CalculateKL(
            PradResult mean,
            PradResult logVar,
            PradOp latentState)
        {
            // Get field-based prior parameters
            var (muField, sigmaFieldSquared) = GetFieldBasedPrior(latentState);

            // Calculate field-aware KL divergence
            // KL(q(z|x) || N(mu_field, sigma_field))
            var deviation = mean.Sub(muField.Result);
            var normalizedDeviation = deviation.Then(d =>
                d.ElementwiseMultiply(d.Result))  // (mean - mu_field)^2
                .Then(d => d.Div(sigmaFieldSquared.Result));  // Divide by sigma_field^2

            var varianceRatio = logVar.Then(PradOp.ExpOp)  // exp(logVar)
                .Then(v => v.Div(sigmaFieldSquared.Result));  // Divide by sigma_field^2

            var logSigmaRatio = sigmaFieldSquared.Then(PradOp.LnOp)
                .Sub(logVar.Result);  // log(sigma_field^2) - logVar

            // Combine terms:
            // 0.5 * Sum((exp(logVar) + (mean - mu_field)^2)/sigma_field^2 - 1 - logVar + log(sigma_field^2))
            var kl = varianceRatio
                .Add(normalizedDeviation.Result)
                .Sub(new Tensor(mean.Result.Shape, 1.0))
                .Sub(logVar.Result)
                .Add(logSigmaRatio.Result)
                .Then(x => x.Mul(new Tensor(x.Result.Shape, 0.5)));

            return kl.Then(PradOp.MeanOp);
        }

        private (PradResult muField, PradResult sigmaFieldSquared) GetFieldBasedPrior(
            PradOp latentState)
        {
            // Get expected direction from vector field
            var fieldDirection = spn.VectorField.MatMul(latentState);

            // Project current state along field direction to get expected mean
            var muField = fieldDirection;

            // Calculate sigma based on field uncertainty
            var (routing, _, fieldParams) = spn.RouteStateInternal(
                new List<PradOp> { latentState });

            // Base variance on field entropy and curvature
            var baseVariance = 1.0f + (float)fieldParams.Entropy;
            var curvatureScaling = 1.0f + (float)fieldParams.Curvature;

            // Create tensor for sigma squared
            var sigmaFieldSquared = new PradOp(new Tensor(
                latentState.Result.Shape,
                Enumerable.Repeat(baseVariance * curvatureScaling, latentState.Result.Data.Length)
                    .ToArray()));

            return (muField, sigmaFieldSquared);
        }

        public class FieldKLMetrics
        {
            public float BaseKL { get; init; }
            public float FieldAlignmentScore { get; init; }
            public float UncertaintyAdaptation { get; init; }
            public Dictionary<string, float> DetailedMetrics { get; init; }
        }

        public FieldKLMetrics AnalyzeKLContribution(
            PradResult mean,
            PradResult logVar,
            PradOp latentState)
        {
            var (muField, sigmaFieldSquared) = GetFieldBasedPrior(latentState);

            // Calculate standard KL (against N(0,1))
            var standardKL = CalculateStandardKL(mean, logVar);

            // Calculate field-based KL
            var fieldKL = CalculateKL(mean, logVar, latentState);

            // Calculate alignment between mean and field direction
            var alignmentScore = CalculateFieldAlignment(mean, muField);

            // Calculate uncertainty adaptation
            var uncertaintyScore = CalculateUncertaintyAdaptation(
                logVar, sigmaFieldSquared);

            return new FieldKLMetrics
            {
                BaseKL = standardKL.Result.Data[0],
                FieldAlignmentScore = alignmentScore,
                UncertaintyAdaptation = uncertaintyScore,
                DetailedMetrics = new Dictionary<string, float>
                {
                    ["mean_field_deviation"] = CalculateMeanDeviation(mean, muField),
                    ["variance_adaptation"] = uncertaintyScore,
                    ["field_kl"] = fieldKL.Result.Data[0],
                    ["kl_reduction"] = standardKL.Result.Data[0] - fieldKL.Result.Data[0]
                }
            };
        }

        private PradResult CalculateStandardKL(PradResult mean, PradResult logVar)
        {
            var kl = logVar.Then(PradOp.ExpOp)
                .Add(mean.Then(PradOp.SquareOp).Result)
                .Sub(new Tensor(mean.Result.Shape, 1.0))
                .Sub(logVar.Result);

            return kl.Then(PradOp.MeanOp)
                .Mul(new Tensor(new[] { 1 }, new[] { 0.5 }));
        }

        private float CalculateFieldAlignment(PradResult mean, PradResult fieldMean)
        {
            var dotProduct = mean.ElementwiseMultiply(fieldMean.Result)
                .Then(PradOp.SumOp);

            var norm1 = mean.Then(PradOp.SquareOp)
                .Then(PradOp.SumOp)
                .Then(PradOp.SquareRootOp);

            var norm2 = fieldMean.Then(PradOp.SquareOp)
                .Then(PradOp.SumOp)
                .Then(PradOp.SquareRootOp);

            return dotProduct.Result.Data[0] /
                   (norm1.Result.Data[0] * norm2.Result.Data[0]);
        }

        private float CalculateUncertaintyAdaptation(
            PradResult logVar,
            PradResult fieldSigmaSquared)
        {
            var predictedVar = logVar.Then(PradOp.ExpOp);
            var ratio = predictedVar.Div(fieldSigmaSquared.Result);
            return 1.0f - Math.Abs(1.0f - (float)ratio.Result.Data.Average());
        }

        private float CalculateMeanDeviation(PradResult mean, PradResult fieldMean)
        {
            var diff = mean.Sub(fieldMean.Result);
            return (float)diff.Then(PradOp.SquareOp)
                .Then(PradOp.MeanOp)
                .Result.Data[0];
        }
    }
}
