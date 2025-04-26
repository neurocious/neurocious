using Neurocious.Core.Common;
using Neurocious.Core.EnhancedVariationalAutoencoder;
using Neurocious.Core.SpatialProbability;
using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.Training
{
    public class EpistemicCoTraining
    {
        private readonly EnhancedVAE vae;
        private readonly SpatialProbabilityNetwork spn;
        private readonly AdamOptimizer optimizerVAE;
        private readonly AdamOptimizer optimizerSPN;
        private readonly FieldAwareKLDivergence fieldKL;

        public EpistemicCoTraining(
            EnhancedVAE vae,
            SpatialProbabilityNetwork spn,
            CoTrainingConfig config)
        {
            this.vae = vae;
            this.spn = spn;
            this.fieldKL = new FieldAwareKLDivergence(spn);
            this.optimizerVAE = new AdamOptimizer(config.LearningRate, 0.9, 0.999);
            this.optimizerSPN = new AdamOptimizer(config.LearningRate, 0.9, 0.999);
        }

        public async Task TrainStep(
            List<PradOp> inputSequence,
            List<float> rewards,
            CoTrainingConfig config)
        {
            // Build full latent sequence by encoding each input separately
            var latentSequence = new List<PradOp>();
            var means = new List<PradResult>();
            var logVars = new List<PradResult>();

            foreach (var input in inputSequence)
            {
                var (mean, logVar) = vae.Encode(input);  // Encode each input individually
                var latent = Reparameterize(mean, logVar);
                latentSequence.Add(latent);
                means.Add(mean);
                logVars.Add(logVar);
            }

            // === SPN Forward Pass with full latent sequence ===
            var (routing, confidence, fieldParams) = spn.RouteStateInternal(latentSequence);

            // === VAE reconstruction from final latent state ===
            var reconOutput = vae.DecodeWithField(latentSequence.Last());
            var reconLoss = CalculateReconstructionLoss(inputSequence.Last(), reconOutput.reconstruction);

            // Use field-aware KL with dynamic uncertainty scaling
            var entropyScale = 1.0f + (float)fieldParams.Entropy * config.EntropyScaling;
            var curvatureScale = 1.0f + (float)fieldParams.Curvature * config.CurvatureScaling;
            var varianceScale = entropyScale * curvatureScale;

            // KL loss on final state
            var klLoss = fieldKL.CalculateKL(means.Last(), logVars.Last(), latentSequence.Last());

            // Calculate SPN base losses
            var spnRewardLoss = CalculateSPNRewardLoss(routing, rewards);
            var spnEntropyLoss = CalculateSPNEntropyLoss(routing);

            // === Sequence-based losses using full latent trajectory ===
            var narrativeContinuityLoss = CalculateNarrativeContinuityLoss(latentSequence);
            var fieldAlignmentLoss = CalculateFieldAlignmentLoss(latentSequence, spn.VectorField);

            // === Total Loss ===
            var totalLoss = reconLoss
                + config.Beta * klLoss
                + config.Gamma * narrativeContinuityLoss
                + config.Delta * fieldAlignmentLoss
                + config.Eta * (spnRewardLoss + spnEntropyLoss);

            // === Backward and Optimize ===
            totalLoss.Back();
            optimizerVAE.Optimize(vae.GetTrainableParameters());
            optimizerSPN.Optimize(spn.GetTrainableParameters());

            // Optional: Log training metrics
            if (Random.Shared.NextDouble() < 0.01) // Log occasionally
            {
                var sequenceMetrics = new List<FieldAwareKLDivergence.FieldKLMetrics>();

                // Analyze KL contribution for each step in sequence
                for (int i = 0; i < latentSequence.Count; i++)
                {
                    var stepMetrics = fieldKL.AnalyzeKLContribution(
                        means[i],
                        logVars[i],
                        latentSequence[i]);
                    sequenceMetrics.Add(stepMetrics);
                }

                LogTrainingMetrics(sequenceMetrics, fieldParams, varianceScale);
            }
        }

        private void LogTrainingMetrics(
            List<FieldAwareKLDivergence.FieldKLMetrics> sequenceMetrics,
            FieldParameters fieldParams,
            float varianceScale)
        {
            Console.WriteLine($"Training Metrics (Sequence Average):");
            Console.WriteLine($"Field KL: {sequenceMetrics.Average(m => m.DetailedMetrics["field_kl"]):F3}");
            Console.WriteLine($"KL Reduction: {sequenceMetrics.Average(m => m.DetailedMetrics["kl_reduction"]):F3}");
            Console.WriteLine($"Field Alignment: {sequenceMetrics.Average(m => m.FieldAlignmentScore):F3}");
            Console.WriteLine($"Uncertainty Adaptation: {sequenceMetrics.Average(m => m.UncertaintyAdaptation):F3}");
            Console.WriteLine($"Field Entropy: {fieldParams.Entropy:F3}");
            Console.WriteLine($"Field Curvature: {fieldParams.Curvature:F3}");
            Console.WriteLine($"Variance Scale: {varianceScale:F3}");

            // Add sequence-specific metrics
            Console.WriteLine($"Sequence Length: {sequenceMetrics.Count}");
            Console.WriteLine($"KL Variance: {sequenceMetrics.Select(m => m.DetailedMetrics["field_kl"]).Variance():F3}");
            Console.WriteLine($"Field Alignment Stability: {1 - sequenceMetrics.Select(m => m.FieldAlignmentScore).Variance():F3}");
        }

        private PradResult Reparameterize(PradResult mean, PradResult logVar)
        {
            var std = logVar.Then(v => v.Mul(new Tensor(v.Result.Shape, 0.5)))
                           .Then(PradOp.ExpOp);

            var epsilon = new Tensor(mean.Result.Shape,
                Enumerable.Range(0, mean.Result.Data.Length)
                    .Select(_ => Random.Shared.NextGaussian())
                    .ToArray());

            return mean.Add(std.Result.ElementwiseMultiply(epsilon));
        }

        private PradResult CalculateNarrativeContinuityLoss(List<PradOp> latentSequence)
        {
            var losses = new List<PradResult>();

            for (int t = 1; t < latentSequence.Count; t++)
            {
                var current = latentSequence[t];
                var previous = latentSequence[t - 1];

                // Calculate L2 distance between consecutive states
                var diff = current.Sub(previous.Result);
                var squaredNorm = diff.Then(PradOp.SquareOp)
                                    .Then(PradOp.SumOp);

                losses.Add(squaredNorm);
            }

            // Sum all transition losses
            return losses.Aggregate((a, b) => a.Add(b.Result));
        }

        private PradResult CalculateFieldAlignmentLoss(
            List<PradOp> latentSequence,
            PradOp vectorField)
        {
            var losses = new List<PradResult>();

            for (int t = 0; t < latentSequence.Count - 1; t++)
            {
                var current = latentSequence[t];
                var next = latentSequence[t + 1];

                // Get actual transition vector
                var transitionVector = next.Sub(current.Result);

                // Get field-predicted direction
                var fieldDirection = vectorField.MatMul(current.Result);

                // Calculate cosine similarity
                var cosineSimilarity = CalculateCosineSimilarity(
                    transitionVector,
                    fieldDirection);

                // Loss is 1 - cosine similarity
                losses.Add(new PradOp(new Tensor(new[] { 1 }, new[] { 1.0 }))
                    .Sub(cosineSimilarity.Result));
            }

            // Sum all alignment losses
            return losses.Aggregate((a, b) => a.Add(b.Result));
        }

        private PradResult CalculateCosineSimilarity(PradResult v1, PradResult v2)
        {
            var dotProduct = v1.Then(x => x.ElementwiseMultiply(v2.Result))
                              .Then(PradOp.SumOp);

            var norm1 = v1.Then(PradOp.SquareOp)
                         .Then(PradOp.SumOp)
                         .Then(PradOp.SquareRootOp);

            var norm2 = v2.Then(PradOp.SquareOp)
                         .Then(PradOp.SumOp)
                         .Then(PradOp.SquareRootOp);

            return dotProduct.Then(d => d.Div(
                norm1.Result.ElementwiseMultiply(norm2.Result)));
        }

        private PradResult CalculateReconstructionLoss(
            List<PradOp> input,
            PradResult output)
        {
            // MSE loss for simplicity
            var diff = output.Sub(input.Last().Result);
            return diff.Then(PradOp.SquareOp)
                      .Then(PradOp.MeanOp);
        }

        private PradResult CalculateKLDivergence(PradResult mean, PradResult logVar)
        {
            // KL divergence with standard normal prior
            var kl = logVar.Then(PradOp.ExpOp)
                          .Add(mean.Then(PradOp.SquareOp).Result)
                          .Sub(new Tensor(mean.Result.Shape, 1.0))
                          .Sub(logVar.Result);

            return kl.Then(PradOp.MeanOp)
                     .Mul(new Tensor(new[] { 1 }, new[] { 0.5 }));
        }

        private PradResult CalculateSPNRewardLoss(PradResult routing, List<float> rewards)
        {
            // Negative log likelihood weighted by rewards
            var negLogLikelihood = routing.Then(PradOp.LnOp)
                                        .Then(x => x.ElementwiseMultiply(
                                            new Tensor(x.Result.Shape, rewards.ToArray())));

            return negLogLikelihood.Then(PradOp.MeanOp)
                                 .Mul(new Tensor(new[] { 1 }, new[] { -1.0 }));
        }

        private PradResult CalculateSPNEntropyLoss(PradResult routing)
        {
            // Entropy of routing distribution
            return routing.Then(PradOp.LnOp)
                        .Then(ln => ln.ElementwiseMultiply(routing.Result))
                        .Then(PradOp.MeanOp)
                        .Mul(new Tensor(new[] { 1 }, new[] { -1.0 }));
        }
    }
}
