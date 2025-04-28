using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.SpatialProbability
{
    public class InverseFlowField : IInverseFlowField
    {
        private readonly int[] fieldShape;
        private readonly int vectorDim;
        private readonly int contextDim;

        // Core fields
        private PradOp inverseVectorField;
        private PradOp flowStrengthField;

        // Dynamic state warping components
        private readonly WarpingModule warpingNetwork;
        private readonly ContextEncoder contextEncoder;
        private readonly float learningRate;

        public InverseFlowField(int[] fieldShape, int vectorDim, int contextDim = -1, float learningRate = 0.01f)
        {
            this.fieldShape = fieldShape;
            this.vectorDim = vectorDim;
            this.contextDim = contextDim > 0 ? contextDim : vectorDim;
            this.learningRate = learningRate;

            InitializeFields();
            warpingNetwork = new WarpingModule(vectorDim, vectorDim * 2);
            contextEncoder = new ContextEncoder(this.contextDim, vectorDim);
        }

        private class WarpingModule
        {
            private readonly PradOp fc1;
            private readonly PradOp fc2;
            private readonly PradOp attention;

            public WarpingModule(int inputDim, int hiddenDim)
            {
                fc1 = new PradOp(InitializeWeights(inputDim, hiddenDim));
                fc2 = new PradOp(InitializeWeights(hiddenDim, inputDim));
                attention = new PradOp(InitializeWeights(hiddenDim, hiddenDim));
            }

            public PradResult WarpState(PradOp state, PradOp context)
            {
                // Non-linear transformation with attention
                var hidden = state.MatMul(fc1)
                    .Then(PradOp.LeakyReLUOp);

                // Apply context-aware attention
                var attentionWeights = hidden.MatMul(attention)
                    .Then(weights => weights.MatMul(context.Transpose().Result))
                    .Then(PradOp.SoftmaxOp);

                var attentionOutput = attentionWeights.MatMul(context.Result);

                // Combine and project back
                return hidden.Add(attentionOutput.Result)
                    .Then(combined => combined.MatMul(fc2.Result))
                    .Then(PradOp.TanhOp);  // Bounded output
            }

            public IEnumerable<PradOp> GetTrainableParameters()
            {
                yield return fc1;
                yield return fc2;
                yield return attention;
            }
        }

        private class ContextEncoder
        {
            private readonly PradOp encoder;
            private readonly PradOp projector;

            public ContextEncoder(int contextDim, int outputDim)
            {
                encoder = new PradOp(InitializeWeights(contextDim, outputDim * 2));
                projector = new PradOp(InitializeWeights(outputDim * 2, outputDim));
            }

            public PradResult Encode(PradOp context)
            {
                return context.MatMul(encoder)
                    .Then(PradOp.LeakyReLUOp)
                    .Then(x => x.MatMul(projector.Result))
                    .Then(PradOp.TanhOp);
            }

            public IEnumerable<PradOp> GetTrainableParameters()
            {
                yield return encoder;
                yield return projector;
            }
        }

        private void InitializeFields()
        {
            // Initialize inverse vector field
            var vectorFieldData = new double[fieldShape[0] * fieldShape[1] * vectorDim];
            Random rand = new Random();
            for (int i = 0; i < vectorFieldData.Length; i++)
            {
                vectorFieldData[i] = rand.NextGaussian(0, 0.01);
            }
            var inverseTensor = new Tensor(fieldShape.Concat(new[] { vectorDim }).ToArray(), vectorFieldData);
            inverseVectorField = new PradOp(NormalizeVectorField(inverseTensor));

            // Initialize flow strength field uniformly
            var strengthData = new double[fieldShape[0] * fieldShape[1]];
            for (int i = 0; i < strengthData.Length; i++)
            {
                strengthData[i] = 1.0 / (fieldShape[0] * fieldShape[1]);
            }
            flowStrengthField = new PradOp(new Tensor(fieldShape, strengthData));
        }

        public InverseTransformationState GeneratePreviousStateWithContext(
            PradOp currentState,
            PradOp context,
            TemporalRegularizer temporalRegularizer)
        {
            // 1. Apply dynamic state warping
            var warpedState = warpingNetwork.WarpState(currentState, context);

            // 2. Get context-aware routing
            var contextualRouting = ComputeContextualRouting(warpedState, context);

            // 3. Apply inverse transformation
            var previousState = ApplyInverseTransformation(warpedState, contextualRouting);

            // 4. Check temporal smoothness
            var (smoothness, confidence) = temporalRegularizer.AnalyzeTransition(
                previousState.Result,
                currentState.Result);

            // 5. Calculate confidence metrics
            var metrics = new Dictionary<string, float>
            {
                ["warping_confidence"] = CalculateWarpingConfidence(warpedState),
                ["routing_confidence"] = CalculateRoutingConfidence(contextualRouting),
                ["temporal_confidence"] = confidence,
                ["overall_confidence"] = CombineConfidences(warpedState, contextualRouting, confidence)
            };

            // 6. Extract flow direction
            var flowDirection = CalculateFlowDirection(previousState, currentState);

            return new InverseTransformationState
            {
                WarpedState = warpedState,
                ContextualRouting = contextualRouting,
                TemporalSmoothness = smoothness,
                ConfidenceMetrics = metrics,
                FlowDirection = flowDirection
            };
        }

        private PradResult ComputeContextualRouting(PradResult warpedState, PradOp context)
        {
            // Encode context
            var encodedContext = contextEncoder.Encode(context);

            // Compute attention between warped state and context
            var attention = warpedState.Then(state =>
                state.MatMul(encodedContext.Result.Transpose()))
                .Then(PradOp.SoftmaxOp);

            // Weight inverse vectors by attention
            return attention.Then(a =>
                a.MatMul(inverseVectorField.Result));
        }

        private PradResult ApplyInverseTransformation(
            PradResult warpedState,
            PradResult contextualRouting)
        {
            // Combine warped state with routing
            var combined = warpedState.Then(state =>
                state.ElementwiseMultiply(contextualRouting.Result));

            // Apply non-linear transformation
            return combined.Then(PradOp.LeakyReLUOp)
                .Then(transformed => transformed.MatMul(
                    inverseVectorField.Transpose().Result));
        }

        public void UpdateFromForwardField(PradOp forwardField, PradResult forwardRouting)
        {
            // Compute inverse vectors from forward field
            var inverseUpdate = forwardField.Then(field => {
                // Negate and normalize vectors
                var negated = field.ElementwiseMultiply(
                    new Tensor(field.Result.Shape, -1.0));
                return new PradOp(NormalizeVectorField(negated.Result));
            });

            // Update inverse vector field
            inverseVectorField = new PradOp(
                inverseVectorField.Add(
                    inverseUpdate.Result.ElementwiseMultiply(
                        new Tensor(inverseUpdate.Result.Shape, learningRate)))
                .Result);

            // Update flow strength based on forward routing
            var strengthUpdate = forwardRouting.Then(r => {
                var momentum = new Tensor(r.Result.Shape, 0.9);
                return flowStrengthField.Mul(momentum)
                    .Add(r.Mul(new Tensor(r.Result.Shape, 0.1)).Result);
            });

            flowStrengthField = new PradOp(strengthUpdate.Result);
        }

        private Tensor NormalizeVectorField(Tensor field)
        {
            var shape = field.Shape;
            var result = new double[field.Data.Length];

            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    double sumSquares = 0;
                    for (int k = 0; k < vectorDim; k++)
                    {
                        var idx = (i * shape[1] * vectorDim) + (j * vectorDim) + k;
                        sumSquares += field.Data[idx] * field.Data[idx];
                    }
                    var norm = Math.Sqrt(sumSquares);

                    for (int k = 0; k < vectorDim; k++)
                    {
                        var idx = (i * shape[1] * vectorDim) + (j * vectorDim) + k;
                        result[idx] = field.Data[idx] / (norm + 1e-6);
                    }
                }
            }

            return new Tensor(shape, result);
        }

        private float CalculateWarpingConfidence(PradResult warpedState)
        {
            return warpedState.Then(state => {
                var variance = state.Then(PradOp.SquareOp)
                    .Then(PradOp.MeanOp)
                    .Then(mean => mean.Sub(
                        state.Then(PradOp.MeanOp)
                        .Then(PradOp.SquareOp).Result));
                return new PradOp(new Tensor(variance.Result.Shape, 1.0))
                    .Sub(variance.Result);
            }).Result.Data[0];
        }

        private float CalculateRoutingConfidence(PradResult routing)
        {
            return routing.Then(r => {
                var entropy = r.Then(PradOp.LnOp)
                    .Then(ln => ln.ElementwiseMultiply(r.Result))
                    .Then(PradOp.MeanOp);
                return new PradOp(new Tensor(entropy.Result.Shape, 1.0))
                    .Sub(entropy.Result);
            }).Result.Data[0];
        }

        private float CombineConfidences(
            PradResult warpedState,
            PradResult routing,
            float temporalConfidence)
        {
            float warpingConf = CalculateWarpingConfidence(warpedState);
            float routingConf = CalculateRoutingConfidence(routing);

            return 0.4f * warpingConf +
                   0.4f * routingConf +
                   0.2f * temporalConfidence;
        }

        private float[] CalculateFlowDirection(PradResult previousState, PradOp currentState)
        {
            var flow = new float[vectorDim];
            for (int i = 0; i < vectorDim; i++)
            {
                flow[i] = (float)(currentState.Result.Data[i] - previousState.Result.Data[i]);
            }

            // Normalize flow vector
            float norm = (float)Math.Sqrt(flow.Sum(x => x * x));
            if (norm > 1e-6f)
            {
                for (int i = 0; i < flow.Length; i++)
                {
                    flow[i] /= norm;
                }
            }

            return flow;
        }

        private static Tensor InitializeWeights(int inputDim, int outputDim)
        {
            var weights = new double[inputDim * outputDim];
            var random = new Random();
            var stddev = Math.Sqrt(2.0 / (inputDim + outputDim));

            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextGaussian(0, stddev);
            }

            return new Tensor(new[] { inputDim, outputDim }, weights);
        }

        public FieldMetrics CalculateMetrics()
        {
            // Calculate average flow strength
            var avgStrength = flowStrengthField.Then(PradOp.MeanOp);

            // Calculate directional coherence
            var coherence = inverseVectorField.Then(field => {
                var meanVector = field.Then(PradOp.MeanOp);
                return field.MatMul(meanVector.Result);
            });

            // Calculate field statistics
            var fieldStats = CalculateFieldStatistics();

            return new FieldMetrics
            {
                AverageFlowStrength = avgStrength.Result.Data[0],
                DirectionalCoherence = coherence.Result.Data[0],
                BackwardConfidence = fieldStats["confidence"],
                FieldStatistics = fieldStats
            };
        }

        private Dictionary<string, float> CalculateFieldStatistics()
        {
            var stats = new Dictionary<string, float>();
            var vectorMagnitudes = new double[fieldShape[0] * fieldShape[1]];
            var maxMagnitude = double.MinValue;
            var minMagnitude = double.MaxValue;

            for (int i = 0; i < fieldShape[0]; i++)
            {
                for (int j = 0; j < fieldShape[1]; j++)
                {
                    double sumSquares = 0;
                    for (int k = 0; k < vectorDim; k++)
                    {
                        var idx = (i * fieldShape[1] * vectorDim) + (j * vectorDim) + k;
                        sumSquares += Math.Pow(inverseVectorField.Result.Data[idx], 2);
                    }
                    var magnitude = Math.Sqrt(sumSquares);
                    vectorMagnitudes[i * fieldShape[1] + j] = magnitude;

                    maxMagnitude = Math.Max(maxMagnitude, magnitude);
                    minMagnitude = Math.Min(minMagnitude, magnitude);
                }
            }

            stats["max_magnitude"] = (float)maxMagnitude;
            stats["min_magnitude"] = (float)minMagnitude;
            stats["mean_magnitude"] = (float)vectorMagnitudes.Average();
            stats["magnitude_std"] = (float)Math.Sqrt(
                vectorMagnitudes.Select(m => Math.Pow(m - stats["mean_magnitude"], 2)).Average());
            stats["confidence"] = (float)(1.0 - stats["magnitude_std"] / stats["mean_magnitude"]);

            return stats;
        }
    }
}
