using Neurocious.Core.Common;
using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.SpatialProbability
{
    public partial class SpatialProbabilityNetwork
    {
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

        private (PradResult routing, PradResult confidence, FieldParameters fieldParams) RouteStateInternal(PradOp state)
        {
            // Project through VAE if available
            var routingState = vaeModel != null ?
                vaeModel.Encode(state) : state;

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

        private FieldParameters CalculateFieldParameters(PradOp state, PradResult routing)
        {
            return new FieldParameters
            {
                Curvature = (float)CalculateLocalCurvature(state, routing).Result.Data[0],
                Entropy = (float)CalculateLocalEntropy(routing).Result.Data[0],
                Alignment = (float)CalculateLocalAlignment(state, routing).Result.Data[0],
            };
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

        private float CosineSimilarity(float[] v1, float[] v2)
        {
            float dot = v1.Zip(v2, (a, b) => a * b).Sum();
            float norm1 = (float)Math.Sqrt(v1.Sum(x => x * x));
            float norm2 = (float)Math.Sqrt(v2.Sum(x => x * x));
            return dot / (norm1 * norm2);
        }
    }
}
