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

        public async Task Train(
        List<List<PradOp>> trainingSequences,       // Sequences of input states
        List<List<float>> rewardSequences,          // Corresponding rewards
        List<List<PradOp>> expectedActionSequences, // Expected policy actions
        List<List<PradOp>> observedReactionSequences, // Observed reflexes
        List<List<PradOp>> futureStateSequences,    // Future states for prediction
        int epochs,
        CoTrainingConfig config)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float epochLoss = 0;
                int batchCount = 0;

                // Process sequences in batches
                for (int i = 0; i < trainingSequences.Count; i += config.BatchSize)
                {
                    var batchSize = Math.Min(config.BatchSize, trainingSequences.Count - i);
                    var batchLoss = 0.0f;

                    // Process each sequence in the batch
                    for (int j = 0; j < batchSize; j++)
                    {
                        var sequenceIndex = i + j;
                        await TrainStep(
                            trainingSequences[sequenceIndex],
                            rewardSequences[sequenceIndex],
                            expectedActionSequences[sequenceIndex],
                            observedReactionSequences[sequenceIndex],
                            futureStateSequences[sequenceIndex],
                            config
                        );
                        // Note: Need to capture and accumulate loss from TrainStep
                    }

                    epochLoss += batchLoss;
                    batchCount++;

                    if (batchCount % 10 == 0)
                    {
                        Console.WriteLine($"Epoch {epoch}, Batch {batchCount}, Average Loss: {batchLoss / batchSize:F4}");
                    }
                }

                Console.WriteLine($"Epoch {epoch} completed. Average Loss: {epochLoss / batchCount:F4}");

                // Optional: Validation step
                if (epoch % 5 == 0)
                {
                    await ValidateModel(/* validation data */);
                }
            }
        }

        private async Task ValidateModel(/* validation parameters */)
        {
            // TODO: Implement validation
            // - Check belief coherence
            // - Evaluate prediction accuracy
            // - Assess policy performance
            // - Measure reflex responsiveness
        }

        public async Task TrainStep(
    List<PradOp> inputSequence,
    List<float> rewards,
    List<PradOp> expectedActions,  // For policy supervision
    List<PradOp> observedReactions,  // For reflex supervision
    List<PradOp> futureStates,  // For prediction supervision
    CoTrainingConfig config)
        {
            // Build full latent sequence
            var latentSequence = new List<PradOp>();
            var means = new List<PradResult>();
            var logVars = new List<PradResult>();

            for (int i = 0; i < inputSequence.Count; i++)
            {
                var currentSubsequence = inputSequence.Take(i + 1).ToList();
                var (mean, logVar) = vae.EncodeSequence(currentSubsequence);
                var latent = Reparameterize(mean, logVar);
                latentSequence.Add(latent.PradOp);
                means.Add(mean);
                logVars.Add(logVar);
            }

            var totalLoss = new PradOp(new Tensor(new[] { 1 }, new[] { 0.0 }));

            // Process each timestep
            for (int t = 0; t < latentSequence.Count; t++)
            {
                // Full epistemic processing
                var (routing, confidence, policy, reflexes, predictions,
                     fieldParams, explanation, inverseExplanation) =
                    spn.ProcessState(latentSequence[t]);

                // === Core Losses ===
                // Routing and reward
                var routingLoss = CalculateRoutingLoss(routing, rewards[t]);

                // Policy prediction
                var policyLoss = CalculatePolicyLoss(policy, expectedActions[t]);

                // Reflex behavior
                var reflexLoss = CalculateReflexLoss(reflexes, observedReactions[t]);

                // Future prediction
                var predictionLoss = CalculatePredictionLoss(predictions, futureStates[t]);

                // === VAE Losses ===
                // Reconstruction
                var reconOutput = vae.DecodeWithField(latentSequence[t]);
                var reconLoss = CalculateReconstructionLoss(inputSequence[t], reconOutput.reconstruction);

                // Field-aware KL
                var entropyScale = 1.0f + (float)fieldParams.Entropy * config.EntropyScaling;
                var curvatureScale = 1.0f + (float)fieldParams.Curvature * config.CurvatureScaling;
                var klLoss = fieldKL.CalculateKL(means[t], logVars[t], latentSequence[t]);

                // === Sequential Losses ===
                var narrativeContinuityLoss = t > 0
                    ? CalculateNarrativeContinuityLoss(latentSequence.Take(t + 1).ToList()).PradOp
                    : new PradOp(new Tensor(new[] { 1 }, new[] { 0.0 }));

                var fieldAlignmentLoss = t > 0
                    ? CalculateFieldAlignmentLoss(latentSequence.Take(t + 1).ToList(), spn.VectorField).PradOp
                    : new PradOp(new Tensor(new[] { 1 }, new[] { 0.0 }));

                // Combine losses
                var stepLoss = routingLoss
                    + config.PolicyWeight * policyLoss
                    + config.ReflexWeight * reflexLoss
                    + config.PredictionWeight * predictionLoss
                    + config.Beta * klLoss
                    + reconLoss
                    + config.Gamma * narrativeContinuityLoss
                    + config.Delta * fieldAlignmentLoss;

                totalLoss = totalLoss.Add(stepLoss.Result);

                // Optional: Log detailed metrics for this timestep
                if (Random.Shared.NextDouble() < 0.01)
                {
                    LogStepMetrics(t, routing, policy, reflexes, predictions,
                                  explanation, inverseExplanation, fieldParams);
                }
            }

            // Backward pass and optimization
            totalLoss.Back();
            optimizerVAE.Optimize(vae.GetTrainableParameters());
            optimizerSPN.Optimize(spn.GetTrainableParameters());
        }

        private PradResult CalculateRoutingLoss(PradResult routing, float reward)
        {
            // Create a tensor from the reward
            var rewardTensor = new Tensor(routing.Result.Shape, new[] { reward });

            // Weight routing probabilities by reward
            // Negative sign because we want to maximize reward
            return routing.Then(r => r.ElementwiseMultiply(rewardTensor))
                         .Then(PradOp.MeanOp)
                         .Mul(new Tensor(new[] { 1 }, new[] { -1.0 }));
        }

        private PradResult CalculatePolicyLoss(PradResult policy, PradOp expectedAction)
        {
            // Cross entropy between policy and expected action
            return policy.Then(p => p.Sub(expectedAction.Result))
                        .Then(PradOp.SquareOp)
                        .Then(PradOp.MeanOp);
        }

        private PradResult CalculateReflexLoss(PradResult reflexes, PradOp observedReaction)
        {
            // Binary cross entropy for reflex triggers
            return reflexes.Then(r => r.Sub(observedReaction.Result))
                          .Then(PradOp.SquareOp)
                          .Then(PradOp.MeanOp);
        }

        private PradResult CalculatePredictionLoss(PradResult predictions, PradOp futureState)
        {
            // MSE between predicted and actual future state
            return predictions.Then(p => p.Sub(futureState.Result))
                             .Then(PradOp.SquareOp)
                             .Then(PradOp.MeanOp);
        }

        private void LogStepMetrics(
            int timestep,
            PradResult routing,
            PradResult policy,
            PradResult reflexes,
            PradResult predictions,
            BeliefExplanation explanation,
            BeliefReconstructionExplanation inverseExplanation,
            FieldParameters fieldParams)
        {
            Console.WriteLine($"Timestep {timestep} Metrics:");
            Console.WriteLine($"Routing confidence: {routing.Result.Data.Max():F3}");
            Console.WriteLine($"Policy certainty: {policy.Result.Data.Max():F3}");
            Console.WriteLine($"Active reflexes: {reflexes.Result.Data.Count(x => x > 0.5)}");
            Console.WriteLine($"Prediction confidence: {predictions.Result.Data.Average():F3}");
            Console.WriteLine($"Field params - Entropy: {fieldParams.Entropy:F3}, Curvature: {fieldParams.Curvature:F3}");
            Console.WriteLine($"Explanation quality: {explanation.Confidence:F3}");
            Console.WriteLine($"Inverse reconstruction smoothness: {inverseExplanation.TemporalSmoothness:F3}");
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
            PradOp input,
            PradResult output)
        {
            var diff = output.Sub(input.Result);
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
