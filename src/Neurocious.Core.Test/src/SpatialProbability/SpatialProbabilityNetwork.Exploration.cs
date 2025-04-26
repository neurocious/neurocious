using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public partial class SpatialProbabilityNetwork
    {
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

        private PradResult AddExplorationNoise(PradResult probs, float explorationRate)
        {
            var noise = new Tensor(probs.Result.Shape,
                Enumerable.Range(0, probs.Result.Data.Length)
                    .Select(_ => random.NextGaussian(0, explorationRate))
                    .ToArray());

            return probs.Then(p => p.Add(noise))
                .Then(PradOp.SoftmaxOp);
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
    }
}
