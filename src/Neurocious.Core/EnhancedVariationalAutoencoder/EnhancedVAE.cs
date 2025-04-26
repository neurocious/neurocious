using Neurocious.Core.Common;
using Neurocious.Core.SpatialProbability;
using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.EnhancedVariationalAutoencoder
{
    public class EnhancedVAE
    {
        protected const int INPUT_DIM = 784;
        protected const int HIDDEN_DIM = 256;
        protected const int LATENT_DIM = 32;
        protected const int SEQUENCE_LENGTH = 16;  // For sequential inputs
        protected const int NUM_HEADS = 4;
        protected const double DROPOUT_RATE = 0.1;

        // Sequential encoder components 
        protected PradOp encoderInputProj;  // Project input to hidden dimension
        protected List<AttentionBlock> encoderAttentionBlocks;
        protected PradOp encoderLayerNorm;
        protected PradOp encoderMean;
        protected PradOp encoderLogVar;

        // Decoder components with field outputs
        protected PradOp decoderFC1;
        protected PradOp decoderFC2;
        protected PradOp decoderOutput;        // Regular reconstruction
        protected PradOp decoderFieldOutput;   // Field parameter output
        protected FieldRegularizer fieldRegularizer;

        protected double klWeight = 0.0;  // For KL annealing
        protected readonly Random random = new Random();

        public EnhancedVAE()
        {
            InitializeNetworks();
        }

        protected virtual void InitializeNetworks()
        {
            // Initialize sequential encoder
            encoderInputProj = new PradOp(InitializeWeights(INPUT_DIM, HIDDEN_DIM));

            // Initialize multi-head attention blocks
            encoderAttentionBlocks = new List<AttentionBlock>();
            for (int i = 0; i < 3; i++)
            {
                encoderAttentionBlocks.Add(new AttentionBlock(HIDDEN_DIM));
            }

            encoderLayerNorm = new PradOp(InitializeWeights(HIDDEN_DIM, HIDDEN_DIM));
            encoderMean = new PradOp(InitializeWeights(HIDDEN_DIM, LATENT_DIM));
            encoderLogVar = new PradOp(InitializeWeights(HIDDEN_DIM, LATENT_DIM));

            // Initialize decoder
            decoderFC1 = new PradOp(InitializeWeights(LATENT_DIM, HIDDEN_DIM));
            decoderFC2 = new PradOp(InitializeWeights(HIDDEN_DIM, HIDDEN_DIM));
            decoderOutput = new PradOp(InitializeWeights(HIDDEN_DIM, INPUT_DIM));
            decoderFieldOutput = new PradOp(InitializeWeights(HIDDEN_DIM, 3));  // Field parameters (e.g., curvature, entropy, alignment)

            // Initialize field regularizer
            fieldRegularizer = new FieldRegularizer();
        }

        private Tensor InitializeWeights(int inputDim, int outputDim)
        {
            var stddev = Math.Sqrt(2.0 / (inputDim + outputDim));
            var weights = new double[inputDim * outputDim];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextGaussian(0, stddev);
            }
            return new Tensor(new[] { inputDim, outputDim }, weights);
        }

        public virtual (PradResult mean, PradResult logVar) EncodeSequence(List<PradOp> sequence)
        {
            // Project each input to hidden dimension
            var projectedSequence = sequence.Select(input =>
                encoderInputProj.MatMul(input).Then(PradOp.LeakyReLUOp)).ToList();

            // Apply multi-head attention blocks
            var processedSequence = projectedSequence;
            foreach (var attentionBlock in encoderAttentionBlocks)
            {
                processedSequence = processedSequence.Select(x =>
                    attentionBlock.Forward(x, training: true)).ToList();
            }

            // Layer normalization and mean pooling
            var normalized = processedSequence.Select(x =>
                encoderLayerNorm.MatMul(x.Result).Then(PradOp.LeakyReLUOp));

            var pooled = normalized.Aggregate((a, b) =>
                new PradOp(a.Result).Add(b.Result))
                .Then(x => x.Div(new Tensor(x.Result.Shape, sequence.Count)));

            // Project to latent parameters
            var mean = encoderMean.MatMul(pooled.Result);
            var logVar = encoderLogVar.MatMul(pooled.Result);

            return (mean, logVar);
        }

        protected virtual (PradResult reconstruction, FieldParameters fieldParams) DecodeWithField(PradOp latentVector)
        {
            var hidden1 = decoderFC1.MatMul(latentVector)
                .Then(PradOp.LeakyReLUOp);

            var hidden2 = decoderFC2.MatMul(hidden1.Result)
                .Then(PradOp.LeakyReLUOp);

            // Regular reconstruction
            var reconstruction = decoderOutput.MatMul(hidden2.Result)
                .Then(PradOp.SigmoidOp);

            // Field parameters with specific activation functions
            var rawFieldParams = decoderFieldOutput.MatMul(hidden2.Result);

            // Apply specific activation functions for each field parameter
            var curvature = new PradOp(rawFieldParams.Result.Indexer("0"))
                .Then(PradOp.ReluOp);  // Non-negative curvature

            var entropy = new PradOp(rawFieldParams.Result.Indexer("1"))
                .Then(PradOp.SigmoidOp);  // Bounded [0,1]

            var alignment = new PradOp(rawFieldParams.Result.Indexer("2"))
                .Then(PradOp.TanhOp);  // Directional [-1,1]

            var fieldParams = new FieldParameters(new Tensor(
                new[] { 3 },
                new[] {
                curvature.Result.Data[0],
                entropy.Result.Data[0],
                alignment.Result.Data[0]
                }));

            return (reconstruction, fieldParams);
        }

        public virtual (PradResult reconstruction, FieldParameters fieldParams, PradResult mean, PradResult logVar)
            ForwardSequence(List<PradOp> sequence)
        {
            // Encode sequence to latent distribution
            var (mean, logVar) = EncodeSequence(sequence);

            // Sample latent vector using reparameterization trick
            var latentVector = ReparameterizationTrick(mean, logVar);

            // Decode with field parameters
            var (reconstruction, fieldParams) = DecodeWithField(new PradOp(latentVector.Result));

            return (reconstruction, fieldParams, mean, logVar);
        }

        public double EstimateLatentCurvature(Tensor z)
        {
            // Estimate local curvature in latent space using second derivatives
            var gradients = new double[z.Data.Length];
            var epsilon = 1e-5;

            for (int i = 0; i < z.Data.Length; i++)
            {
                var zPlus = z.Data.ToArray();
                var zMinus = z.Data.ToArray();
                zPlus[i] += epsilon;
                zMinus[i] -= epsilon;

                var reconstructionPlus = DecodeWithField(new PradOp(new Tensor(z.Shape, zPlus))).reconstruction;
                var reconstructionMinus = DecodeWithField(new PradOp(new Tensor(z.Shape, zMinus))).reconstruction;

                gradients[i] = (reconstructionPlus.Result.Data[0] - 2 * z.Data[i] +
                    reconstructionMinus.Result.Data[0]) / (epsilon * epsilon);
            }

            // Return mean absolute curvature
            return gradients.Select(Math.Abs).Average();
        }

        public double EstimateLatentEntropy(Tensor z)
        {
            // Estimate entropy using local neighborhood sampling
            var samples = new List<double>();
            var numSamples = 100;
            var radius = 0.1;

            for (int i = 0; i < numSamples; i++)
            {
                var noise = new double[z.Data.Length];
                for (int j = 0; j < noise.Length; j++)
                {
                    noise[j] = random.NextGaussian(0, radius);
                }

                var neighborZ = z.Add(new Tensor(z.Shape, noise));
                var (recon, _) = DecodeWithField(new PradOp(neighborZ));
                samples.Add(recon.Result.Data.Average());
            }

            // Compute entropy estimate using histogram method
            var histogram = new Dictionary<int, int>();
            var binSize = 0.1;
            foreach (var sample in samples)
            {
                var bin = (int)(sample / binSize);
                if (!histogram.ContainsKey(bin))
                    histogram[bin] = 0;
                histogram[bin]++;
            }

            var entropy = 0.0;
            foreach (var count in histogram.Values)
            {
                var p = count / (double)samples.Count;
                entropy -= p * Math.Log(p);
            }

            return entropy;
        }

        protected virtual PradResult ComputeLoss(
            List<Tensor> inputs,
            PradResult reconstruction,
            FieldParameters fieldParams,
            PradResult mean,
            PradResult logVar,
            int epoch)
        {
            // Compute reconstruction loss
            var reconLoss = ComputeBatchReconstructionLoss(inputs, reconstruction);

            // Compute KL loss with annealing
            var klLoss = ComputeKLDivergenceLoss(mean, logVar);

            // Anneal KL weight from 0 to 1 over epochs
            klWeight = Math.Min(1.0, epoch / 50.0);  // Reach full weight at epoch 50

            // Add field regularization losses
            var fieldRegLoss = fieldRegularizer.ComputeLoss(fieldParams);

            // Add contrastive loss for regime transitions
            var contrastiveLoss = ComputeContrastiveLoss(mean, inputs);

            // Combine losses with weights
            return new PradOp(reconLoss.Result)
                .Add(klLoss.Result.ElementwiseMultiply(new Tensor(klLoss.Result.Shape, klWeight)))
                .Add(fieldRegLoss.Result.Mul(new Tensor(fieldRegLoss.Result.Shape, 0.1)))
                .Add(contrastiveLoss.Result.Mul(new Tensor(contrastiveLoss.Result.Shape, 0.05)));
        }

        protected PradResult ReparameterizationTrick(PradResult mean, PradResult logVar)
        {
            // Scale and shift the standard deviation
            var std = logVar.Then(v => v.Mul(new Tensor(v.Result.Shape, 0.5)))
                           .Then(PradOp.ExpOp);

            // Generate random normal samples
            var epsilon = new Tensor(mean.Result.Shape,
                Enumerable.Range(0, mean.Result.Data.Length)
                    .Select(_ => random.NextGaussian())
                    .ToArray());

            // Combine mean and scaled noise
            return mean.Add(std.Result.ElementwiseMultiply(epsilon));
        }

        public FieldParameters ExtractFieldParameters(PradOp state)
        {
            // Get latent field representation
            var latentState = state.MatMul(decoderFieldOutput.CurrentTensor);

            // Extract individual field parameters with appropriate activation functions
            var curvature = new PradOp(latentState.Result.Indexer("0"))
                .Then(PradOp.ReluOp);  // Non-negative curvature

            var entropy = new PradOp(latentState.Result.Indexer("1"))
                .Then(PradOp.SigmoidOp);  // Bounded [0,1]

            var alignment = new PradOp(latentState.Result.Indexer("2"))
                .Then(PradOp.TanhOp);  // Directional [-1,1]

            return new FieldParameters(new Tensor(
                new[] { 3 },
                new[] {
            curvature.Result.Data[0],
            entropy.Result.Data[0],
            alignment.Result.Data[0]
                }));
        }

        protected PradResult ComputeKLDivergenceLoss(PradResult mean, PradResult logVar)
        {
            // KL divergence loss between approximate posterior and standard normal prior
            // KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
            var kl = logVar.Then(lv => {
                var variance = lv.Then(PradOp.ExpOp);
                var meanSquared = mean.Then(m => m.ElementwiseMultiply(m.Result));

                return new PradOp(new Tensor(lv.Result.Shape, 1.0))
                    .Add(lv.Result)
                    .Sub(meanSquared.Result)
                    .Sub(variance.Result)
                    .Then(sum => sum.Mul(new Tensor(sum.Result.Shape, -0.5)));
            });

            return kl.Then(PradOp.MeanOp);
        }

        protected PradResult ComputeBCELoss(Tensor predicted, Tensor target)
        {
            // Binary Cross Entropy Loss
            // BCE = -Σ(y * log(p) + (1-y) * log(1-p))
            var epsilon = 1e-7;  // Prevent log(0)

            // Clip predictions to prevent numerical instability
            var clippedPred = predicted.Data.Select(p => Math.Max(Math.Min(p, 1 - epsilon), epsilon));

            var bce = new double[predicted.Data.Length];
            for (int i = 0; i < predicted.Data.Length; i++)
            {
                double y = target.Data[i];
                double p = clippedPred.ElementAt(i);
                bce[i] = -(y * Math.Log(p) + (1 - y) * Math.Log(1 - p));
            }

            return new PradOp(new Tensor(predicted.Shape, bce))
                .Then(PradOp.MeanOp);
        }

        private PradResult ComputeBatchReconstructionLoss(List<Tensor> inputs, PradResult reconstruction)
        {
            var batchLoss = new List<PradResult>();
            for (int i = 0; i < inputs.Count; i++)
            {
                batchLoss.Add(ComputeBCELoss(reconstruction.Result, inputs[i]));
            }

            // Average losses across batch
            return batchLoss.Aggregate((a, b) =>
                new PradOp(a.Result).Add(b.Result))
                .Then(x => x.Div(new Tensor(x.Result.Shape, inputs.Count)));
        }

        protected PradResult ComputeContrastiveLoss(PradResult latentMean, List<Tensor> inputs)
        {
            // Simple contrastive loss using cosine similarity
            var similarities = new List<PradResult>();

            for (int i = 0; i < inputs.Count; i++)
            {
                for (int j = i + 1; j < inputs.Count; j++)
                {
                    var similarity = new PradOp(latentMean.Result)
                        .MatMul(new PradOp(inputs[j]).Transpose().Result)
                        .Then(x => x.Div(
                            new PradOp(latentMean.Result).Then(PradOp.SquareRootOp).Result
                                .ElementwiseMultiply(
                                    new PradOp(inputs[j]).Then(PradOp.SquareRootOp).Result
                                )
                        ));

                    similarities.Add(similarity);
                }
            }

            // Average similarity
            return similarities.Aggregate((a, b) =>
                new PradOp(a.Result).Add(b.Result))
                .Then(x => x.Div(new Tensor(x.Result.Shape, similarities.Count)));
        }

        public virtual void Train(List<List<Tensor>> sequenceData, int epochs, int batchSize = 32)
        {
            var optimizer = new AdamOptimizer(0.001, 0.9, 0.999);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalLoss = 0;

                // Process mini-batches of sequences
                for (int i = 0; i < sequenceData.Count; i += batchSize)
                {
                    var batch = sequenceData.Skip(i).Take(batchSize).ToList();

                    foreach (var sequence in batch)
                    {
                        // Convert sequence to PradOp
                        var sequenceOps = sequence.Select(x => new PradOp(x)).ToList();

                        // Forward pass
                        var (reconstruction, fieldParams, mean, logVar) = ForwardSequence(sequenceOps);

                        // Compute loss
                        var loss = ComputeLoss(sequence, reconstruction, fieldParams, mean, logVar, epoch);
                        totalLoss += loss.Result.Data[0];

                        // Backward pass
                        loss.Back();

                        // Update weights
                        optimizer.Optimize(encoderInputProj);
                        foreach (var block in encoderAttentionBlocks)
                        {
                            optimizer.Optimize(block.QueryProj);
                            optimizer.Optimize(block.KeyProj);
                            optimizer.Optimize(block.ValueProj);
                            optimizer.Optimize(block.OutputProj);
                            optimizer.Optimize(block.LayerNorm);
                        }
                        optimizer.Optimize(encoderLayerNorm);
                        optimizer.Optimize(encoderMean);
                        optimizer.Optimize(encoderLogVar);
                        optimizer.Optimize(decoderFC1);
                        optimizer.Optimize(decoderFC2);
                        optimizer.Optimize(decoderOutput);
                        optimizer.Optimize(decoderFieldOutput);
                    }
                }

                if (epoch % 10 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}, Average Loss: {totalLoss / sequenceData.Count}, KL Weight: {klWeight:F3}");
                }
            }
        }
    }
}
