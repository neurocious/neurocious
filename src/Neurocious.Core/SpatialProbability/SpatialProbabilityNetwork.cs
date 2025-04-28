using Neurocious.Core.Common;
using Neurocious.Core.EnhancedVariationalAutoencoder;
using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.SpatialProbability
{
    public class SpatialProbabilityNetwork : ISpatialProbabilityNetwork
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

        // Neural components
        private readonly PolicyNetwork policyNetwork;
        private readonly ReflexNetwork reflexNetwork;
        private readonly PredictionNetwork predictionNetwork;

        // World branching and exploration
        private readonly List<SpatialProbabilityNetwork> branches;
        private readonly Dictionary<string, int> routeVisits;
        private readonly Random random;
        private readonly List<PradOp> trainableParameters;
        private readonly EnhancedVAE vaeModel;

        // Field parameters
        private const float LEARNING_RATE = 0.01f;
        private const float FIELD_DECAY = 0.999f;
        private const float MIN_FIELD_STRENGTH = 1e-6f;
        private const float NOVELTY_WEIGHT = 0.1f;
        internal const float BRANCH_DECAY_RATE = 0.95f;

        private readonly InverseFlowIntegration inverseFlowIntegration;

        public SpatialProbabilityNetwork(
            EnhancedVAE vae = null,
            int stateDim = 20,
            int[] fieldShape = null,
            int vectorDim = 8,
            int bufferSize = 10,
            SpatialProbabilityNetwork parent = null)
        {
            this.stateDim = stateDim;
            this.fieldShape = fieldShape ?? new[] { 32, 32 };
            this.vectorDim = vectorDim;
            this.bufferSize = bufferSize;
            this.vaeModel = vae;

            // Initialize fields
            InitializeFields();
            temporalBuffer = new Queue<Tensor>(bufferSize);

            // Initialize neural components
            policyNetwork = new PolicyNetwork(stateDim);
            reflexNetwork = new ReflexNetwork(stateDim);
            predictionNetwork = new PredictionNetwork(stateDim);

            // Initialize branching and exploration components
            branches = new List<SpatialProbabilityNetwork>();
            routeVisits = new Dictionary<string, int>();
            random = new Random();
            trainableParameters = new List<PradOp>();

            // Track trainable parameters
            RegisterTrainableParameters();

            // Initialize inverse flow integration
            inverseFlowIntegration = new InverseFlowIntegration(
                fieldShape: this.fieldShape,
                vectorDim: this.vectorDim,
                bufferSize: this.bufferSize);
        }

        public PradOp VectorField
        {
            get => vectorField;
            set => vectorField = value;
        }

        public PradOp EntropyField
        {
            get => entropyField;
            set => entropyField = value;
        }

        public PradResult CalculateFieldAlignment()
        {
            return vectorField.Then(field => {
                var meanVector = field.Then(PradOp.MeanOp);
                return field.MatMul(meanVector.Result)
                    .Then(PradOp.MeanOp);
            });
        }

        private void InitializeFields()
        {
            // Initialize vector field with normalized random values
            var vectorFieldData = new double[fieldShape[0] * fieldShape[1] * vectorDim];
            for (int i = 0; i < vectorFieldData.Length; i++)
            {
                vectorFieldData[i] = random.NextDouble() * 2 - 1;
            }
            var vectorFieldTensor = new Tensor(fieldShape.Concat(new[] { vectorDim }).ToArray(), vectorFieldData);
            vectorField = new PradOp(NormalizeVectorField(vectorFieldTensor));

            // Initialize field metrics
            var fieldSize = fieldShape[0] * fieldShape[1];
            curvatureField = new PradOp(new Tensor(fieldShape, new double[fieldSize]));
            entropyField = new PradOp(new Tensor(fieldShape, Enumerable.Repeat(1.0 / fieldSize, fieldSize).ToArray()));
            alignmentField = new PradOp(new Tensor(fieldShape, new double[fieldSize]));
        }

        private void RegisterTrainableParameters()
        {
            trainableParameters.Add(vectorField);
            trainableParameters.Add(curvatureField);
            trainableParameters.Add(entropyField);
            trainableParameters.Add(alignmentField);
            // Add neural network parameters
            trainableParameters.AddRange(policyNetwork.GetTrainableParameters());
            trainableParameters.AddRange(reflexNetwork.GetTrainableParameters());
            trainableParameters.AddRange(predictionNetwork.GetTrainableParameters());
        }

        // Core field operations
        public (PradResult routing, PradResult confidence, PradResult policy, PradResult reflexes, PradResult predictions, FieldParameters fieldParams, BeliefExplanation explanation, BeliefReconstructionExplanation inverseExplanation)
        ProcessState(PradOp state)
        {
            // Add to temporal buffer
            if (temporalBuffer.Count >= bufferSize)
            {
                temporalBuffer.Dequeue();
            }
            temporalBuffer.Enqueue(state.CurrentTensor);

            var sequence = temporalBuffer.Select(x => new PradOp(x)).ToList();

            // Get base routing with exploration
            var (routing, confidence, fieldParams) = RouteStateInternal(sequence);

            // Get temporal context
            var historyTensor = GetHistoryTensor();

            // Generate policy and value
            var (policy, _) = policyNetwork.Forward(state, new PradOp(historyTensor));

            // Check reflexes
            var reflexes = reflexNetwork.Forward(state);

            // Make predictions
            var predictions = predictionNetwork.Forward(new PradOp(historyTensor));

            // Generate belief explanation
            var latent = vaeModel != null ? vaeModel.Encode(state) : state;
            var explanation = GenerateBeliefExplanation(latent, routing, fieldParams, confidence);

            var inverseContext = new PradOp(GetHistoryTensor());
            var inverseExplanation = inverseFlowIntegration.ReconstructPriorBelief(
                state,
                inverseContext,
                explanation?.TopContributingFeatures);

            return (routing, confidence, policy, reflexes, predictions,
                    fieldParams, explanation, inverseExplanation);
        }

        public BeliefReconstructionExplanation ReconstructPriorBelief(
    PradOp currentState,
    List<string> potentialAntecedents = null)
        {
            var context = new PradOp(GetHistoryTensor());
            return inverseFlowIntegration.ReconstructPriorBelief(
                currentState,
                context,
                potentialAntecedents);
        }

        public void UpdateFieldParameters(FieldParameters fieldParams)
        {
            // Scale base field magnitudes by the field parameters
            double curvatureScaling = 1.0 + fieldParams.Curvature;
            double entropyScaling = fieldParams.Entropy;
            double alignmentScaling = Math.Abs(fieldParams.Alignment);

            // Update curvature field
            var newCurvatureField = curvatureField.Then(field => {
                var scaledField = field.ElementwiseMultiply(
                    new Tensor(field.Result.Shape, curvatureScaling)
                );
                return scaledField;
            });
            curvatureField = new PradOp(newCurvatureField.Result);

            // Update entropy field with regularization
            var newEntropyField = entropyField.Then(field => {
                // Blend current entropy with target entropy
                var currentValues = field.Result.Data;
                var targetValues = new double[currentValues.Length];
                for (int i = 0; i < targetValues.Length; i++)
                {
                    targetValues[i] = entropyScaling;
                }
                var targetTensor = new Tensor(field.Result.Shape, targetValues);

                // Smooth transition (0.8 current + 0.2 target)
                return field.Mul(new Tensor(field.Result.Shape, 0.8))
                    .Add(targetTensor.ElementwiseMultiply(new Tensor(targetTensor.Shape, 0.2)));
            });
            entropyField = new PradOp(newEntropyField.Result);

            // Update alignment field based on desired alignment
            var newAlignmentField = alignmentField.Then(field => {
                var direction = fieldParams.Alignment > 0 ? 1.0 : -1.0;
                var magnitude = alignmentScaling;

                return field.Mul(new Tensor(field.Result.Shape, direction * magnitude));
            });
            alignmentField = new PradOp(newAlignmentField.Result);

            // Update vector field to respect new field parameters
            var newVectorField = vectorField.Then(field => {
                // Scale magnitude by curvature
                var scaledField = field.ElementwiseMultiply(
                    new Tensor(field.Result.Shape, 1.0 / (1.0 + curvatureScaling))
                );

                // Apply alignment influence
                if (Math.Abs(fieldParams.Alignment) > 0.1)
                {
                    var alignmentDirection = alignmentField.Result;
                    scaledField = scaledField.Add(
                        alignmentDirection.ElementwiseMultiply(
                            new Tensor(alignmentDirection.Shape, alignmentScaling * 0.3)
                        )
                    );
                }

                return scaledField;
            });

            // Normalize the vector field
            vectorField = new PradOp(NormalizeVectorField(newVectorField.Result));

            // Update the field coupling strengths based on parameters
            UpdateFieldCouplingStrengths(fieldParams);
        }

        private void UpdateFieldCouplingStrengths(FieldParameters fieldParams)
        {
            // Calculate coupling strengths between fields
            double curvatureToEntropyStrength = 0.3 * (1.0 - Math.Abs(fieldParams.Alignment));
            double entropyToAlignmentStrength = 0.2 * fieldParams.Curvature;
            double alignmentToVectorStrength = 0.4 * (1.0 - fieldParams.Entropy);

            // Update curvature-entropy coupling
            if (curvatureToEntropyStrength > 0.1)
            {
                var entropyCoupling = curvatureField.Then(field => {
                    return field.ElementwiseMultiply(
                        new Tensor(field.Result.Shape, curvatureToEntropyStrength)
                    );
                });
                entropyField = new PradOp(
                    entropyField.Add(entropyCoupling.Result).Then(PradOp.SoftmaxOp).Result
                );
            }

            // Update entropy-alignment coupling
            if (entropyToAlignmentStrength > 0.1)
            {
                var alignmentCoupling = entropyField.Then(field => {
                    return field.ElementwiseMultiply(
                        new Tensor(field.Result.Shape, entropyToAlignmentStrength)
                    );
                });
                alignmentField = new PradOp(
                    alignmentField.Add(alignmentCoupling.Result)
                        .Then(x => x.Mul(new Tensor(x.Result.Shape, 0.5))).Result
                );
            }

            // Update alignment-vector coupling
            if (alignmentToVectorStrength > 0.1)
            {
                var vectorCoupling = alignmentField.Then(field => {
                    return field.ElementwiseMultiply(
                        new Tensor(field.Result.Shape, alignmentToVectorStrength)
                    );
                });
                vectorField = new PradOp(NormalizeVectorField(
                    vectorField.Add(vectorCoupling.Result).Result
                ));
            }
        }

        internal (PradResult routing, PradResult confidence, FieldParameters fieldParams) RouteStateInternal(List<PradOp> sequence)
        {
            // Project through VAE if available
            var routingState = vaeModel != null
               ? ProcessVAESequence(sequence)  // New helper method
               : sequence.Last();  // Use last state if no VAE

            // Calculate base routing
            var similarity = routingState.MatMul(vectorField.Transpose());
            var baseRouting = similarity.Then(PradOp.SoftmaxOp);

            // Calculate field parameters
            var fieldParams = CalculateFieldParameters(routingState, baseRouting);

            // Update field metrics
            UpdateFieldMetrics(routingState, baseRouting, fieldParams);

            // Add exploration
            var exploration = UpdateExploration(routingState);
            var routing = AddExplorationNoise(baseRouting, exploration.ExplorationRate);

            // Calculate confidence
            var confidence = CalculateRoutingConfidence(fieldParams);

            return (routing, confidence, fieldParams);
        }

        private PradOp ProcessVAESequence(List<PradOp> sequence)
        {
            var (mean, logVar) = vaeModel.EncodeSequence(sequence);
            return ReparameterizationTrick(mean, logVar);
        }

        private PradOp ReparameterizationTrick(PradResult mean, PradResult logVar)
        {
            var std = logVar.Then(v => v.Mul(new Tensor(v.Result.Shape, 0.5)))
                           .Then(PradOp.ExpOp);

            var epsilon = new Tensor(mean.Result.Shape,
                Enumerable.Range(0, mean.Result.Data.Length)
                    .Select(_ => Random.Shared.NextGaussian())
                    .ToArray());

            return mean.Add(std.Result.ElementwiseMultiply(epsilon));
        }

        public void UpdateFields(PradResult route, PradResult reward, List<PradOp> sequence)
        {
            // Get current field parameters
            var (_, _, fieldParams) = RouteStateInternal(sequence);

            // Calculate adaptive learning rate
            float adaptiveLearningRate = LEARNING_RATE *
                (1 - fieldParams.Entropy) *         // Learn more when certain
                (1 / (1 + fieldParams.Curvature));  // Learn less in unstable regions

            // Update vector field
            var fieldUpdate = route.Then(r => {
                var learningRateTensor = new Tensor(r.Result.Shape, adaptiveLearningRate);
                return r.Mul(reward.Result).Mul(learningRateTensor);
            });

            // Apply weighted update
            var alignmentWeight = Math.Abs(fieldParams.Alignment);
            vectorField = new PradOp(
                vectorField.Mul(new Tensor(vectorField.CurrentShape, 1 - alignmentWeight * LEARNING_RATE)).Result
                .Add(fieldUpdate.Result)
            );

            // Normalize field
            vectorField = new PradOp(NormalizeVectorField(vectorField.CurrentTensor));

            // Update metrics and apply coupling
            UpdateFieldMetrics(state, route, fieldParams);
            ApplyBeliefCoupling(route, fieldParams);

            // Update neural components with TD learning
            UpdateWithReward(reward.Result.Data[0]);

            // Update inverse flow field
            inverseFlowIntegration.UpdateFromForwardDynamics(vectorField, route);
        }

        // World branching support
        public List<WorldBranch> SimulateWorldBranches(FieldParameters currentState, int numBranches = 3)
        {
            var branches = new List<WorldBranch>();
            for (int i = 0; i < numBranches; i++)
            {
                var perturbedState = PerturbFieldParameters(currentState);
                var branchNetwork = CloneNetwork();
                float branchProb = CalculateBranchProbability(currentState, perturbedState);
                branches.Add(new WorldBranch(branchNetwork, perturbedState, branchProb));
            }
            return branches;
        }

        public FlowPattern AnalyzeFieldFlow(PradOp state, int steps = 10)
        {
            var currentState = state;
            var history = new List<GeometricField>();

            for (int i = 0; i < steps; i++)
            {
                // Get current routing and field parameters
                var (routing, _, fieldParams) = RouteStateInternal(currentState);
                var geometry = CalculateFieldGeometry(currentState, routing);
                history.Add(geometry);

                // Update state following field flow
                currentState = new PradOp(routing.Result);
            }

            // Analyze flow stability and patterns
            return new FlowPattern
            {
                Position = state.Result.Data,
                FlowDirection = history.Last().Direction,
                LocalCurvature = history.Average(g => g.LocalCurvature),
                LocalEntropy = history.Average(g => -Math.Log(g.Strength)),
                LocalAlignment = history.Average(g => 1 - Math.Abs(g.LocalRotation)),
                Stability = CalculateFlowStability(history)
            };
        }

        // Backpropagation support
        public void Back(PradResult loss)
        {
            loss.Back();
            foreach (var param in trainableParameters)
            {
                if (param.SeedGradient != null)
                {
                    var update = param.SeedGradient.ElementwiseMultiply(
                        new Tensor(param.SeedGradient.Shape, LEARNING_RATE)
                    );
                    param.Add(update);
                }
            }
        }

        public void ClearTemporalBuffer()
        {
            temporalBuffer.Clear();
        }

        // Diagnostic methods
        public Dictionary<string, float> GetDiagnostics()
        {
            var metrics = new Dictionary<string, float>();

            // Field metrics
            var fieldMetrics = CalculateFieldMetrics();
            metrics["global_entropy"] = fieldMetrics.GlobalEntropy;
            metrics["global_curvature"] = fieldMetrics.GlobalCurvature;
            metrics["global_alignment"] = fieldMetrics.GlobalAlignment;
            metrics["belief_stability"] = fieldMetrics.BeliefStability;
            metrics["coherence_score"] = fieldMetrics.CoherenceScore;

            // Branch metrics
            metrics["active_branches"] = branches.Count;
            metrics["mean_branch_value"] = branches.Any() ? branches.Average(b => b.Value) : 0;

            // Neural metrics
            metrics["reflex_rate"] = GetReflexActivations();

            // Exploration metrics
            if (temporalBuffer.Any())
            {
                var exploration = UpdateExploration(new PradOp(temporalBuffer.Last()));
                metrics["novelty_score"] = exploration.NoveltyScore;
                metrics["uncertainty_score"] = exploration.UncertaintyScore;
                metrics["exploration_rate"] = exploration.ExplorationRate;
            }

            return metrics;
        }

        // Neural network components
        private class PolicyNetwork
        {
            private readonly PradOp stateEncoder;
            private readonly PradOp historyEncoder;
            private readonly PradOp attention;
            private readonly PradOp policyHead;
            private readonly PradOp valueHead;

            public PolicyNetwork(int stateDim)
            {
                stateEncoder = new PradOp(InitializeWeights(stateDim, 64));
                historyEncoder = new PradOp(InitializeWeights(stateDim, 64));
                attention = new PradOp(InitializeWeights(64, 64));
                policyHead = new PradOp(InitializeWeights(128, 10));
                valueHead = new PradOp(InitializeWeights(128, 1));
            }

            public (PradResult policy, PradResult value) Forward(PradOp currentState, PradOp history)
            {
                var stateEncoded = stateEncoder.MatMul(currentState)
                    .Then(PradOp.LeakyReLUOp);

                var historyEncoded = historyEncoder.MatMul(history)
                    .Then(PradOp.LeakyReLUOp);

                var attentionWeights = stateEncoded.MatMul(attention)
                    .Then(x => x.MatMul(historyEncoded.Transpose().Result))
                    .Then(PradOp.SoftmaxOp);

                var attentionOutput = attentionWeights.MatMul(historyEncoded.Result);

                var combined = ConcatFeatures(stateEncoded.Result, attentionOutput.Result);

                var policy = combined.MatMul(policyHead)
                    .Then(PradOp.SigmoidOp);

                var value = combined.MatMul(valueHead);

                return (policy, value);
            }

            public IEnumerable<PradOp> GetTrainableParameters()
            {
                yield return stateEncoder;
                yield return historyEncoder;
                yield return attention;
                yield return policyHead;
                yield return valueHead;
            }

            private PradOp ConcatFeatures(Tensor a, Tensor b)
            {
                var concat = new double[a.Data.Length + b.Data.Length];
                a.Data.CopyTo(concat, 0);
                b.Data.CopyTo(concat, a.Data.Length);
                return new PradOp(new Tensor(new[] { concat.Length }, concat));
            }
        }

        private class ReflexNetwork
        {
            private readonly PradOp layer1;
            private readonly PradOp layer2;
            private readonly PradOp outputLayer;

            public ReflexNetwork(int stateDim)
            {
                layer1 = new PradOp(InitializeWeights(stateDim, 32));
                layer2 = new PradOp(InitializeWeights(32, 16));
                outputLayer = new PradOp(InitializeWeights(16, 5)); // 5 reflex controls
            }

            public PradResult Forward(PradOp state)
            {
                return state.MatMul(layer1)
                    .Then(PradOp.LeakyReLUOp)
                    .Then(x => x.MatMul(layer2.Result))
                    .Then(PradOp.LeakyReLUOp)
                    .Then(x => x.MatMul(outputLayer.Result))
                    .Then(PradOp.SigmoidOp);
            }

            public IEnumerable<PradOp> GetTrainableParameters()
            {
                yield return layer1;
                yield return layer2;
                yield return outputLayer;
            }
        }

        private class PredictionNetwork
        {
            private readonly PradOp sequenceEncoder;
            private readonly PradOp hidden;
            private readonly PradOp outputLayer;

            public PredictionNetwork(int stateDim)
            {
                sequenceEncoder = new PradOp(InitializeWeights(stateDim * 10, 64));
                hidden = new PradOp(InitializeWeights(64, 32));
                outputLayer = new PradOp(InitializeWeights(32, 4)); // [value, confidence, upper, lower]
            }

            public PradResult Forward(PradOp sequence)
            {
                return sequence.MatMul(sequenceEncoder)
                    .Then(PradOp.LeakyReLUOp)
                    .Then(x => x.MatMul(hidden.Result))
                    .Then(PradOp.LeakyReLUOp)
                    .Then(x => x.MatMul(outputLayer.Result));
            }

            public IEnumerable<PradOp> GetTrainableParameters()
            {
                yield return sequenceEncoder;
                yield return hidden;
                yield return outputLayer;
            }
        }

        // Core field operations
        internal Tensor NormalizeVectorField(Tensor field)
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
                        result[idx] = field.Data[idx] / norm;
                    }
                }
            }

            return new Tensor(shape, result);
        }

        private void UpdateFieldMetrics(PradOp state, PradResult routing, FieldParameters localParams)
        {
            // Calculate geometric field properties
            var geometry = CalculateFieldGeometry(state, routing);

            // Update curvature field incorporating geometric properties
            var curvatureUpdate = localParams.Curvature * (1 - FIELD_DECAY) +
                                geometry.LocalCurvature * LEARNING_RATE;
            curvatureField = new PradOp(
                curvatureField.Mul(new Tensor(curvatureField.CurrentShape, FIELD_DECAY)).Result
                .Add(new Tensor(curvatureField.CurrentShape, curvatureUpdate))
            );

            // Update entropy field with local divergence influence
            var entropyUpdate = localParams.Entropy * (1 - FIELD_DECAY) +
                              Math.Abs(geometry.LocalDivergence) * LEARNING_RATE;
            entropyField = new PradOp(
                entropyField.Mul(new Tensor(entropyField.CurrentShape, FIELD_DECAY)).Result
                .Add(new Tensor(entropyField.CurrentShape, entropyUpdate))
            );

            // Update alignment field considering field rotation
            var alignmentUpdate = localParams.Alignment * (1 - FIELD_DECAY) +
                                (1 - Math.Abs(geometry.LocalRotation)) * LEARNING_RATE;
            alignmentField = new PradOp(
                alignmentField.Mul(new Tensor(alignmentField.CurrentShape, FIELD_DECAY)).Result
                .Add(new Tensor(alignmentField.CurrentShape, alignmentUpdate))
            );
        }

        private FieldParameters CalculateFieldParameters(PradOp state, PradResult routing)
        {
            return new FieldParameters
            {
                Curvature = (float)CalculateLocalCurvature(state, routing).Result.Data[0],
                Entropy = (float)CalculateLocalEntropy(routing).Result.Data[0],
                Alignment = (float)CalculateLocalAlignment(state, routing).Result.Data[0],
            };
        }

        private void UpdateWithReward(float reward, float discountFactor = 0.99f)
        {
            if (temporalBuffer.Count < 2) return;

            var currentState = new PradOp(temporalBuffer.Last());
            var previousState = new PradOp(temporalBuffer.ElementAt(temporalBuffer.Count - 2));

            var (_, currentValue) = policyNetwork.Forward(currentState, GetHistoryTensor());
            var (_, previousValue) = policyNetwork.Forward(previousState, GetHistoryTensor());

            var tdError = reward + discountFactor * currentValue.Result.Data[0] - previousValue.Result.Data[0];
            var tdLoss = new PradResult(new Tensor(new[] { 1 }, new[] { tdError }));

            Back(tdLoss);
        }

        private Tensor GetHistoryTensor()
        {
            var history = temporalBuffer.ToArray();
            var historyData = new double[bufferSize * stateDim];

            for (int i = 0; i < history.Length; i++)
            {
                Array.Copy(history[i].Data, 0, historyData, i * stateDim, stateDim);
            }

            if (history.Length < bufferSize)
            {
                Array.Clear(historyData, history.Length * stateDim, (bufferSize - history.Length) * stateDim);
            }

            return new Tensor(new[] { bufferSize, stateDim }, historyData);
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

        private SpatialProbabilityNetwork CloneNetwork()
        {
            return new SpatialProbabilityNetwork(
                vae: vaeModel,
                stateDim: stateDim,
                fieldShape: fieldShape,
                vectorDim: vectorDim,
                bufferSize: bufferSize,
                parent: this
            );
        }

        private FieldParameters PerturbFieldParameters(FieldParameters state)
        {
            return new FieldParameters
            {
                Curvature = state.Curvature + (float)random.NextGaussian(0, 0.1),
                Entropy = state.Entropy + (float)random.NextGaussian(0, 0.1),
                Alignment = state.Alignment + (float)random.NextGaussian(0, 0.1)
            };
        }

        private float CalculateBranchProbability(FieldParameters original, FieldParameters perturbed)
        {
            float distance = (float)Math.Sqrt(
                Math.Pow(original.Curvature - perturbed.Curvature, 2) +
                Math.Pow(original.Entropy - perturbed.Entropy, 2) +
                Math.Pow(original.Alignment - perturbed.Alignment, 2));

            return (float)Math.Exp(-distance);
        }

        private ExplorationState UpdateExploration(PradOp state)
        {
            string routeSignature = CalculateRouteSignature(state);
            routeVisits[routeSignature] = routeVisits.GetValueOrDefault(routeSignature, 0) + 1;

            float noveltyScore = CalculateNoveltyScore(routeSignature);
            float uncertaintyScore = (float)CalculateFieldEntropy().Result.Data[0];
            float explorationRate = CombineExplorationFactors(noveltyScore, uncertaintyScore);

            return new ExplorationState
            {
                NoveltyScore = noveltyScore,
                UncertaintyScore = uncertaintyScore,
                ExplorationRate = explorationRate
            };
        }

        private PradResult CalculateStructuralFieldEntropy()
        {
            // Calculate field divergence
            var divergence = vectorField.Then(field => {
                // Compute spatial derivatives
                var dx = field.Then(f => f.Diff(axis: 0));
                var dy = field.Then(f => f.Diff(axis: 1));

                // Sum divergence
                return dx.Add(dy.Result);
            });

            // Convert divergence to probability distribution
            var probabilities = divergence.Then(PradOp.SoftmaxOp);

            // Calculate entropy
            return probabilities.Then(p => {
                return p.Then(PradOp.LnOp)
                        .Then(ln => ln.ElementwiseMultiply(p.Result))
                        .Then(prod => prod.Mean(axis: 0))
                        .Then(mean => mean.Mul(new Tensor(mean.Result.Shape, -1.0)));
            });
        }

        private PradResult CalculateDirectionalFieldEntropy()
        {
            // For a vector field, entropy can be calculated in several ways:

            // 1. Flow Directional Entropy
            var flowDirections = vectorField.Then(field => {
                // Normalize vectors
                var norm = field.Then(PradOp.SquareOp)
                               .Then(PradOp.SumOp)
                               .Then(PradOp.SquareRootOp);
                return field.Div(norm.Result);
            });

            // 2. Calculate angular distribution of vectors
            var angles = flowDirections.Then(directions => {
                // Convert vectors to angles (assuming 2D field)
                return directions.Then(d =>
                    d.Then(PradOp.Atan2Op, directions.Transpose().Result));
            });

            // 3. Compute entropy of angular distribution
            var entropy = angles.Then(a => {
                // Use softmax to get probability distribution
                var probs = a.Then(PradOp.SoftmaxOp);

                // Calculate entropy: -Σ p_i * log(p_i)
                return probs.Then(PradOp.LnOp)
                           .Then(ln => ln.ElementwiseMultiply(probs.Result))
                           .Then(PradOp.MeanOp)
                           .Then(mean => mean.Mul(new Tensor(mean.Result.Shape, -1.0)));
            });

            return entropy;
        }

        private PradResult CalculateFieldEntropy()
        {
            var directionalEntropy = CalculateDirectionalFieldEntropy();
            var structuralEntropy = CalculateStructuralFieldEntropy();

            // Combine both entropy measures
            return directionalEntropy.Then(d =>
                d.Add(structuralEntropy.Result).Then(sum =>
                    sum.Div(new Tensor(sum.Result.Shape, 2.0d))));
        }

        private string CalculateRouteSignature(PradOp state)
        {
            return string.Join(",",
                state.CurrentTensor.Data.Select(x => Math.Round(x, 2)));
        }

        private float CalculateNoveltyScore(string routeSignature)
        {
            int visits = routeVisits[routeSignature];
            return (float)Math.Exp(-visits * NOVELTY_WEIGHT);
        }

        private float CombineExplorationFactors(float novelty, float uncertainty)
        {
            float baseRate = 0.1f;
            float noveltyFactor = NOVELTY_WEIGHT * novelty;
            float uncertaintyFactor = (1 - NOVELTY_WEIGHT) * uncertainty;
            return baseRate * (noveltyFactor + uncertaintyFactor);
        }

        private PradResult AddExplorationNoise(PradResult probs, float explorationRate)
        {
            var noise = new Tensor(probs.Result.Shape,
                Enumerable.Range(0, probs.Result.Data.Length)
                    .Select(_ => random.NextGaussian(0, explorationRate))
                    .ToArray());

            return probs.Then(p => p.Add(noise))
                .Then(PradOp.SoftmaxOp);
        }

        private void ApplyBeliefCoupling(PradResult route, FieldParameters fieldParams)
        {
            float couplingStrength = Math.Max(0, fieldParams.Alignment) * (1 - fieldParams.Entropy);
            var coupledRegions = route.Then(r => {
                return r.Result.Data
                    .Select(p => p > MIN_FIELD_STRENGTH ? p * couplingStrength : 0)
                    .ToArray();
            });

            var couplingUpdate = new PradOp(new Tensor(route.Result.Shape, coupledRegions));
            vectorField = new PradOp(
                vectorField.Add(couplingUpdate.Mul(new Tensor(couplingUpdate.CurrentShape, LEARNING_RATE)).Result).Result
            );
        }

        private float GetReflexActivations()
        {
            var recentStates = temporalBuffer.TakeLast(10);
            var activations = recentStates
                .Select(state => reflexNetwork.Forward(new PradOp(state)))
                .Count(result => result.Result.Data.Any(x => x > 0.5));

            return activations / (float)Math.Max(1, recentStates.Count());
        }

        private BeliefExplanation GenerateBeliefExplanation(
            PradOp latent, PradResult routing,
            FieldParameters fieldParams, PradResult confidence)
        {
            var contributions = new Dictionary<string, float>();
            var latentData = latent.Result.Data;
            var vectorFieldTensor = vectorField.CurrentTensor;

            // Calculate feature contributions through field alignment
            int vectorCount = fieldShape[0] * fieldShape[1];
            for (int i = 0; i < vectorDim; i++)
            {
                double sum = 0;
                for (int j = 0; j < vectorCount; j++)
                {
                    int idx = j * vectorDim + i;
                    sum += vectorFieldTensor.Data[idx];
                }
                float contribution = (float)(latentData[i] * (sum / vectorCount));
                contributions[$"feature_{i}"] = contribution;
            }

            // Get top contributing features
            var topContributors = contributions
                .OrderByDescending(kv => Math.Abs(kv.Value))
                .Take(3)
                .ToList();

            // Generate counterfactual shifts
            var counterfactuals = new Dictionary<string, float>();
            foreach (var (feature, value) in topContributors)
            {
                var perturbedLatent = latent.Result.Data.ToArray();
                perturbedLatent[int.Parse(feature.Split('_')[1])] = 0; // Zero out the feature
                var perturbedRouting = RouteStateInternal(new PradOp(new Tensor(latent.Result.Shape, perturbedLatent))).routing;
                float shift = CalculateRoutingShift(routing.Result, perturbedRouting.Result);
                counterfactuals[feature] = shift;
            }

            return new BeliefExplanation
            {
                BeliefLabel = GetTopAttractorRegion(routing.Result),
                FeatureContributions = contributions,
                Confidence = confidence.Result.Data[0],
                FieldParams = fieldParams,
                TopContributingFeatures = topContributors.Select(tc => tc.Key).ToList(),
                CounterfactualShifts = counterfactuals,
                TrajectoryPath = temporalBuffer.Select(t => t.Data).ToList(),
                Justification = GenerateJustification(topContributors, counterfactuals, fieldParams)
            };
        }

        private string GetTopAttractorRegion(Tensor routing)
        {
            int topRegion = 0;
            double maxProb = routing.Data[0];
            for (int i = 1; i < routing.Data.Length; i++)
            {
                if (routing.Data[i] > maxProb)
                {
                    maxProb = routing.Data[i];
                    topRegion = i;
                }
            }
            return $"Region_{topRegion}";
        }

        private float CosineSimilarity(float[] v1, float[] v2)
        {
            float dot = v1.Zip(v2, (a, b) => a * b).Sum();
            float norm1 = (float)Math.Sqrt(v1.Sum(x => x * x));
            float norm2 = (float)Math.Sqrt(v2.Sum(x => x * x));
            return dot / (norm1 * norm2);
        }

        private float CalculateFlowStability(List<GeometricField> history)
        {
            // Analyze flow consistency over time
            float directionStability = 0;
            float strengthStability = 0;
            float geometricStability = 0;

            for (int i = 1; i < history.Count; i++)
            {
                // Direction consistency
                directionStability += CosineSimilarity(
                    history[i].Direction,
                    history[i - 1].Direction);

                // Strength consistency
                strengthStability += 1 - Math.Abs(
                    history[i].Strength - history[i - 1].Strength);

                // Geometric consistency (curvature, divergence, rotation)
                geometricStability += 1 - (
                    Math.Abs(history[i].LocalCurvature - history[i - 1].LocalCurvature) +
                    Math.Abs(history[i].LocalDivergence - history[i - 1].LocalDivergence) +
                    Math.Abs(history[i].LocalRotation - history[i - 1].LocalRotation)
                ) / 3;
            }

            // Combine stability measures
            return (directionStability + strengthStability + geometricStability) /
                   (3 * (history.Count - 1));
        }

        private float CalculateRoutingShift(Tensor original, Tensor perturbed)
        {
            float sum = 0;
            for (int i = 0; i < original.Data.Length; i++)
            {
                sum += Math.Abs((float)(original.Data[i] - perturbed.Data[i]));
            }
            return sum / original.Data.Length;
        }

        private string GenerateJustification(
            List<KeyValuePair<string, float>> topContributors,
            Dictionary<string, float> counterfactuals,
            FieldParameters fieldParams)
        {
            var justification = new System.Text.StringBuilder();

            // Add top contributors
            justification.AppendLine("Primary factors:");
            foreach (var (feature, contribution) in topContributors)
            {
                string impact = contribution > 0 ? "supporting" : "opposing";
                justification.AppendLine($"- {feature}: {Math.Abs(contribution):F3} ({impact})");
            }

            // Add counterfactual analysis
            justification.AppendLine("\nCounterfactual impacts:");
            foreach (var (feature, shift) in counterfactuals)
            {
                justification.AppendLine($"- Without {feature}: {shift:F3} belief shift");
            }

            // Add field configuration impact
            justification.AppendLine($"\nField state: " +
                $"stability={1 - fieldParams.Curvature:F2}, " +
                $"certainty={1 - fieldParams.Entropy:F2}, " +
                $"coherence={fieldParams.Alignment:F2}");

            return justification.ToString();
        }

        private PradResult CalculateRoutingConfidence(FieldParameters fieldParams)
        {
            // Calculate confidence incorporating flow stability
            var state = new PradOp(new Tensor(vectorField.CurrentShape, fieldParams.Alignment));
            var flowPattern = AnalyzeFieldFlow(state);

            float confidence = (1 - fieldParams.Entropy) *                // High confidence when entropy is low
                             (1 / (1 + fieldParams.Curvature)) *         // High confidence when curvature is low
                             Math.Abs(fieldParams.Alignment) *           // High confidence when alignment is strong
                             flowPattern.Stability;                      // High confidence when flow is stable

            return new PradOp(new Tensor(new[] { 1 }, confidence));
        }

        private GeometricField CalculateFieldGeometry(PradOp state, PradResult routing)
        {
            // Calculate field direction and strength
            var fieldDirection = vectorField.MatMul(routing.Result);
            var fieldStrength = routing.Result.Data.Max();

            // Calculate local curvature using finite differences
            var curvature = CalculateLocalCurvature(state, routing);

            // Calculate divergence (spreading/converging of belief flow)
            var divergence = CalculateLocalDivergence(state, routing);

            // Calculate rotation (circular flow patterns)
            var rotation = CalculateLocalRotation(state, routing);

            return new GeometricField
            {
                Direction = fieldDirection.Result.Data,
                Strength = fieldStrength,
                LocalCurvature = curvature.Result.Data[0],
                LocalDivergence = divergence.Result.Data[0],
                LocalRotation = rotation.Result.Data[0]
            };
        }

        private PradResult CalculateLocalCurvature(PradOp state, PradResult routing)
        {
            // Estimate local field curvature through finite differences
            var epsilon = 1e-5f;
            var deltaCurvature = state.Then(s => {
                var perturbed = new PradOp(s.Result.Add(new Tensor(s.Result.Shape, epsilon)));
                var perturbedRouting = perturbed.MatMul(vectorField.Transpose())
                    .Then(PradOp.SoftmaxOp);

                // Second derivative approximation
                return routing.Sub(perturbedRouting.Result).Then(PradOp.SquareOp)
                    .Then(PradOp.MeanOp).Mul(new Tensor(new[] { 1 }, 1.0f / (epsilon * epsilon)));
            });

            return deltaCurvature;
        }

        private PradResult CalculateLocalDivergence(PradOp state, PradResult routing)
        {
            // Compute divergence using central differences
            var epsilon = 1e-5f;
            return state.Then(s => {
                var gradients = new List<PradResult>();
                for (int i = 0; i < s.Result.Shape[0]; i++)
                {
                    var delta = new float[s.Result.Shape[0]];
                    delta[i] = epsilon;

                    var forward = new PradOp(s.Result.Add(new Tensor(delta)));
                    var backward = new PradOp(s.Result.Sub(new Tensor(delta)));

                    var forwardRoute = forward.MatMul(vectorField.Transpose()).Then(PradOp.SoftmaxOp);
                    var backwardRoute = backward.MatMul(vectorField.Transpose()).Then(PradOp.SoftmaxOp);

                    gradients.Add(forwardRoute.Sub(backwardRoute.Result).Mul(new Tensor(new[] { 1 }, 1.0f / (2 * epsilon))));
                }

                return gradients.Aggregate((a, b) => a.Add(b.Result)).Then(PradOp.MeanOp);
            });
        }

        private PradResult CalculateLocalRotation(PradOp state, PradResult routing)
        {
            // Calculate curl-like measure of rotational flow
            var epsilon = 1e-5f;
            return state.Then(s => {
                var rotations = new List<PradResult>();
                for (int i = 0; i < s.Result.Shape[0] - 1; i++)
                {
                    for (int j = i + 1; j < s.Result.Shape[0]; j++)
                    {
                        var deltaI = new float[s.Result.Shape[0]];
                        var deltaJ = new float[s.Result.Shape[0]];
                        deltaI[i] = epsilon;
                        deltaJ[j] = epsilon;

                        var plusI = new PradOp(s.Result.Add(new Tensor(deltaI)));
                        var plusJ = new PradOp(s.Result.Add(new Tensor(deltaJ)));

                        var routeI = plusI.MatMul(vectorField.Transpose()).Then(PradOp.SoftmaxOp);
                        var routeJ = plusJ.MatMul(vectorField.Transpose()).Then(PradOp.SoftmaxOp);

                        rotations.Add(routeI.Sub(routeJ.Result).Then(PradOp.MeanOp));
                    }
                }

                return rotations.Aggregate((a, b) => a.Add(b.Result)).Then(PradOp.MeanOp);
            });
        }

        private PradResult CalculateLocalEntropy(PradResult routing)
        {
            // Calculate entropy of routing distribution
            return routing.Then(r => {
                var logProbs = r.Then(PradOp.LnOp);
                return r.Mul(logProbs.Result).Then(PradOp.MeanOp).Mul(new Tensor(new[] { 1 }, -1.0));
            });
        }

        private PradResult CalculateLocalAlignment(PradOp state, PradResult routing)
        {
            // Calculate alignment through vector field coherence
            var fieldDirection = vectorField.Then(v => v.MatMul(routing.Result));
            return state.MatMul(fieldDirection.Result).Then(PradOp.MeanOp);
        }
    }
}
