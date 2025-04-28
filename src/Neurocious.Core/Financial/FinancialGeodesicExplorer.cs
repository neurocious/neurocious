using Neurocious.Core.Chess;
using Neurocious.Core.EnhancedVariationalAutoencoder;
using Neurocious.Core.SpatialProbability;
using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.Financial
{
    public class FinancialGeodesicExplorer : IFinancialGeodesicExplorer
    {
        private readonly EnhancedVAE architectVAE;
        private readonly EnhancedVAE explorerVAE;
        private readonly SpatialProbabilityNetwork spn;
        private readonly FinancialEncoder encoder;
        private readonly FinancialDecoder decoder;
        private readonly AdvancedGateManager gateManager;

        // Hyperparameters for loss function components
        private readonly double pathEnergyWeight = 0.4;     // α: Path smoothness and stability
        private readonly double gateFailureWeight = 0.3;    // β: Market regime alignment
        private readonly double riskWeight = 0.2;           // γ: Risk-adjusted returns
        private readonly double volatilityWeight = 0.1;     // δ: Volatility clustering

        // Exploration parameters
        private readonly int pathSteps = 20;                // Forward-looking steps
        private readonly double baseStepSize = 0.02;        // Conservative base step size
        private readonly int candidatePaths = 12;           // Number of candidate paths
        private readonly double baseExplorationRate = 0.15; // Higher for market uncertainty

        public FinancialGeodesicExplorer(
            EnhancedVAE architectVAE,
            EnhancedVAE explorerVAE,
            SpatialProbabilityNetwork spn,
            FinancialEncoder encoder,
            FinancialDecoder decoder)
        {
            this.architectVAE = architectVAE;
            this.explorerVAE = explorerVAE;
            this.spn = spn;
            this.encoder = encoder;
            this.decoder = decoder;
            this.gateManager = new AdvancedGateManager(InitializeFinancialGates());
        }

        private Dictionary<string, StrategicGate> InitializeFinancialGates()
        {
            return new Dictionary<string, StrategicGate>
            {
                ["momentum"] = new MarketRegimeGate(
                    name: "momentum",
                    dim: explorerVAE.LatentDimension,
                    threshold: 0.2f,
                    weight: 0.4f,
                    characteristics: new Dictionary<string, double>
                    {
                        ["momentum"] = 0.8,
                        ["volatility_prediction"] = 0.4
                    }),

                ["mean_reversion"] = new MarketRegimeGate(
                    name: "mean_reversion",
                    dim: explorerVAE.LatentDimension,
                    threshold: 0.3f,
                    weight: 0.3f,
                    characteristics: new Dictionary<string, double>
                    {
                        ["momentum"] = -0.5,
                        ["market_regime"] = 0.5
                    }),

                ["volatility_regime"] = new MarketRegimeGate(
                    name: "volatility_regime",
                    dim: explorerVAE.LatentDimension,
                    threshold: 0.25f,
                    weight: 0.2f,
                    characteristics: new Dictionary<string, double>
                    {
                        ["volatility_prediction"] = 0.8,
                        ["tail_risk"] = 0.6
                    }),

                ["risk_off"] = new MarketRegimeGate(
                    name: "risk_off",
                    dim: explorerVAE.LatentDimension,
                    threshold: 0.4f,
                    weight: 0.3f,
                    characteristics: new Dictionary<string, double>
                    {
                        ["tail_risk"] = 0.7,
                        ["market_regime"] = 0.2
                    }),

                ["regime_transition"] = new MarketRegimeGate(
                    name: "regime_transition",
                    dim: explorerVAE.LatentDimension,
                    threshold: 0.35f,
                    weight: 0.15f,
                    characteristics: new Dictionary<string, double>
                    {
                        ["volatility_prediction"] = 0.6,
                        ["market_regime"] = 0.5
                    })
            };
        }

        public (string action, double confidence, Dictionary<string, double> metrics)
            FindBestTradeDecision(double[] marketSnapshot)
        {
            // Encode market state to latent space
            var encodedState = encoder.EncodeSnapshot(marketSnapshot);
            var latentState = explorerVAE.EncodeSequence(
                new List<PradOp> { encodedState }).Item1.PradOp;

            // Shape the field based on macro conditions
            ShapeFieldForMarket(latentState);

            // Generate and evaluate trading paths
            var candidatePaths = GenerateCandidatePaths(latentState);
            var (bestPath, bestReward, metrics) = EvaluatePaths(candidatePaths);

            // Record trajectory for learning
            var memory = new FinancialTrajectoryMemory(
                bestPath,
                bestReward,
                GetGateActivations(bestPath),
                metrics["sharpe_ratio"],
                metrics["max_drawdown"],
                metrics);

            gateManager.ProcessTrajectory(bestPath, bestReward);

            // Reinforce successful strategy
            ReinforceBestPath(bestPath, bestReward);

            // Decode next state into trading decision
            var nextState = bestPath[1];
            var (action, confidence) = decoder.DecodeTradeDecision(latentState, nextState);

            return (action, confidence, metrics);
        }

        private void ShapeFieldForMarket(PradOp marketState)
        {
            // Get macro view from architect
            var architectEncoding = architectVAE.EncodeSequence(
                new List<PradOp> { marketState }).Item1.PradOp;
            var fieldParams = architectVAE.ExtractFieldParameters(architectEncoding);

            // Modify field parameters based on market conditions
            var riskMetrics = decoder.DecodeRiskMetrics(marketState);
            fieldParams.Entropy = Math.Max(fieldParams.Entropy, riskMetrics["volatility_prediction"]);
            fieldParams.Curvature = Math.Max(fieldParams.Curvature, riskMetrics["tail_risk"]);

            // Update SPN field parameters
            spn.UpdateFieldParameters(fieldParams);
        }

        private List<List<PradOp>> GenerateCandidatePaths(PradOp startState)
        {
            var paths = new List<List<PradOp>>();
            var riskMetrics = decoder.DecodeRiskMetrics(startState);

            // Generate paths with risk-aware exploration
            for (int i = 0; i < candidatePaths; i++)
            {
                var explorationScale = baseExplorationRate * (1 + riskMetrics["volatility_prediction"]);
                paths.Add(FollowMarketPath(startState, explorationScale));
            }

            return paths;
        }

        private List<PradOp> FollowMarketPath(PradOp startState, double explorationScale)
        {
            var path = new List<PradOp> { startState };
            var current = startState;
            var random = new Random();

            for (int i = 0; i < pathSteps; i++)
            {
                var (routing, confidence, fieldParams) = spn.RouteStateInternal(
                    new List<PradOp> { current });

                // Get vector field direction with volatility scaling
                var moveDirection = routing.MatMul(spn.VectorField);
                var riskMetrics = decoder.DecodeRiskMetrics(current);
                var volatilityScale = 1.0 / (1.0 + riskMetrics["volatility_prediction"]);

                // Combine field guidance with market momentum
                var probabilitySample = SampleFromProbabilityField(current, routing);
                var fieldWeight = 0.6 + 0.2 * confidence.Result.Data[0];
                var momentumWeight = 1.0 - fieldWeight;

                var combinedDirection = moveDirection.Result.ElementwiseMultiply(
                    new Tensor(moveDirection.Result.Shape, fieldWeight * volatilityScale))
                    .Add(probabilitySample.ElementwiseMultiply(
                        new Tensor(probabilitySample.Shape, momentumWeight)));

                // Add adaptive noise based on market conditions
                var noiseScale = explorationScale * (1.0 + riskMetrics["tail_risk"]);
                var noise = new double[combinedDirection.Data.Length];
                for (int j = 0; j < noise.Length; j++)
                {
                    noise[j] = random.NextGaussian(0, noiseScale);
                }

                // Risk-aware step size
                var adaptiveStepSize = baseStepSize * volatilityScale;

                // Move to next state
                var nextState = current.Add(
                    combinedDirection.Add(new Tensor(combinedDirection.Shape, noise))
                    .Mul(new Tensor(combinedDirection.Shape, adaptiveStepSize)));

                path.Add(nextState);
                current = nextState;
            }

            return path;
        }

        private Tensor SampleFromProbabilityField(PradOp state, PradResult routing)
        {
            var probabilities = routing.Result;
            var random = new Random();
            var cumulative = 0.0;
            var sample = random.NextDouble();
            int selectedIndex = probabilities.Data.Length - 1;

            for (int i = 0; i < probabilities.Data.Length; i++)
            {
                cumulative += probabilities.Data[i];
                if (sample <= cumulative)
                {
                    selectedIndex = i;
                    break;
                }
            }

            var result = new double[probabilities.Data.Length];
            result[selectedIndex] = 1.0;

            return new Tensor(probabilities.Shape, result);
        }

        private (List<PradOp> path, double reward, Dictionary<string, double> metrics)
            EvaluatePaths(List<List<PradOp>> paths)
        {
            List<PradOp> bestPath = null;
            double bestScore = double.MinValue;
            Dictionary<string, double> bestMetrics = null;

            foreach (var path in paths)
            {
                var (metrics, totalScore) = CalculatePathMetrics(path);

                if (totalScore > bestScore)
                {
                    bestScore = totalScore;
                    bestPath = path;
                    bestMetrics = metrics;
                }
            }

            return (bestPath, bestScore, bestMetrics);
        }

        private (Dictionary<string, double> metrics, double totalScore)
            CalculatePathMetrics(List<PradOp> path)
        {
            var metrics = new Dictionary<string, double>();

            // Calculate returns and volatility
            var returns = new List<double>();
            for (int i = 1; i < path.Count; i++)
            {
                var pctChange = CalculateReturn(path[i - 1], path[i]);
                returns.Add(pctChange);
            }

            // Basic metrics
            metrics["total_return"] = returns.Sum();
            metrics["volatility"] = Math.Sqrt(returns.Select(r => r * r).Average());
            metrics["sharpe_ratio"] = metrics["total_return"] / (metrics["volatility"] + 1e-8);

            // Drawdown analysis
            double peak = 0;
            double currentValue = 0;
            metrics["max_drawdown"] = 0;

            foreach (var ret in returns)
            {
                currentValue += ret;
                peak = Math.Max(peak, currentValue);
                metrics["max_drawdown"] = Math.Max(metrics["max_drawdown"],
                    peak - currentValue);
            }

            // Path smoothness
            metrics["path_energy"] = CalculatePathEnergy(path);
            metrics["regime_alignment"] = CalculateRegimeAlignment(path);
            metrics["risk_adjusted_return"] = metrics["total_return"] /
                (1 + metrics["max_drawdown"]);

            // Combine into final score
            double totalScore =
                pathEnergyWeight * (1 - metrics["path_energy"]) +
                gateFailureWeight * metrics["regime_alignment"] +
                riskWeight * metrics["risk_adjusted_return"] +
                volatilityWeight * (1 - metrics["volatility"]);

            return (metrics, totalScore);
        }

        private double CalculateReturn(PradOp state1, PradOp state2)
        {
            // Simple return calculation in latent space
            var diff = state2.Sub(state1.Result);
            return diff.Result.Data.Sum() / Math.Sqrt(diff.Result.Data.Length);
        }

        private double CalculatePathEnergy(List<PradOp> path)
        {
            double energy = 0.0;

            for (int i = 0; i < path.Count - 1; i++)
            {
                var current = path[i];
                var (routing, _, fieldParams) = spn.RouteStateInternal(
                    new List<PradOp> { current });

                // Local energy components
                double curvatureEnergy = fieldParams.Curvature * 0.5;
                double entropyEnergy = fieldParams.Entropy * 0.3;
                double alignmentEnergy = (1.0 - Math.Abs(fieldParams.Alignment)) * 0.2;

                energy += curvatureEnergy + entropyEnergy + alignmentEnergy;
            }

            return energy / (path.Count - 1);
        }

        private double CalculateRegimeAlignment(List<PradOp> path)
        {
            double totalAlignment = 0.0;

            for (int i = 0; i < path.Count; i++)
            {
                var state = path[i];
                double pathProgress = i / (double)(path.Count - 1);

                double maxActivation = 0.0;
                foreach (var gate in gateManager.Gates.Values)
                {
                    var activation = gate.CalculateActivation(state);
                    maxActivation = Math.Max(maxActivation, activation);
                }

                totalAlignment += maxActivation;
            }

            return totalAlignment / path.Count;
        }

        private void ReinforceBestPath(List<PradOp> path, double reward)
        {
            var rewardScalar = Math.Max(-1.0, Math.Min(1.0, reward));

            for (int i = 0; i < path.Count; i++)
            {
                var current = path[i];
                var discount = Math.Pow(0.95, path.Count - i - 1);
                var discountedReward = new PradOp(
                    new Tensor(new[] { 1 }, rewardScalar * discount));

                var (routing, _, _) = spn.RouteStateInternal(new List<PradOp> { current });

                // Use entropy-aware field updates
                spn.UpdateFieldsWithEntropyAwareness(
                    routing,
                    discountedReward,
                    path.Take(i + 1).ToList());
            }
        }

        private Dictionary<string, List<float>> GetGateActivations(List<PradOp> path)
        {
            var activations = new Dictionary<string, List<float>>();

            foreach (var state in path)
            {
                foreach (var gate in gateManager.Gates.Values)
                {
                    if (!activations.ContainsKey(gate.Name))
                    {
                        activations[gate.Name] = new List<float>();
                    }
                    activations[gate.Name].Add(gate.CalculateActivation(state));
                }
            }

            return activations;
        }
    }
}
