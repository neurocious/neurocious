using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.Financial
{
    public class FinancialGeodesicTrainer
    {
        private readonly FinancialGeodesicExplorer explorer;
        private readonly BacktestEngine backtester;
        private readonly FinancialMetrics metrics;
        private readonly double learningRate;
        private readonly int batchSize;
        private readonly int epochSamples;

        public FinancialGeodesicTrainer(
            FinancialGeodesicExplorer explorer,
            BacktestEngine backtester,
            double learningRate = 0.001,
            int batchSize = 32,
            int epochSamples = 1000)
        {
            this.explorer = explorer;
            this.backtester = backtester;
            this.metrics = new FinancialMetrics();
            this.learningRate = learningRate;
            this.batchSize = batchSize;
            this.epochSamples = epochSamples;
        }

        public async Task TrainEpoch(List<MarketSnapshot> historicalData)
        {
            var random = new Random();
            var totalLoss = 0.0;
            var batches = 0;

            // Break historical data into training sequences
            var sequences = CreateTrainingSequences(historicalData);

            for (int i = 0; i < epochSamples; i += batchSize)
            {
                var batchSequences = sequences
                    .OrderBy(x => random.Next())
                    .Take(batchSize)
                    .ToList();

                var batchLoss = await TrainBatch(batchSequences);

                totalLoss += batchLoss;
                batches++;

                if (batches % 10 == 0)
                {
                    Console.WriteLine($"Batch {batches}, Average Loss: {batchLoss / batchSize:F4}");
                    await ValidateBatch(batchSequences);
                }
            }

            Console.WriteLine($"Epoch completed. Average Loss: {totalLoss / batches:F4}");
        }

        private List<List<MarketSnapshot>> CreateTrainingSequences(List<MarketSnapshot> data)
        {
            var sequences = new List<List<MarketSnapshot>>();
            int sequenceLength = 50; // Length of each training sequence
            int stride = 10;         // How many steps to move forward for next sequence

            for (int i = 0; i < data.Count - sequenceLength; i += stride)
            {
                sequences.Add(data.Skip(i).Take(sequenceLength).ToList());
            }

            return sequences;
        }

        private async Task<double> TrainBatch(List<List<MarketSnapshot>> sequences)
        {
            double batchLoss = 0;

            foreach (var sequence in sequences)
            {
                // Run backtest on sequence
                var result = await backtester.RunBacktest(sequence, new BacktestConfig
                {
                    LookbackPeriods = 20,
                    ConfidenceThreshold = 0.5  // Lower threshold during training
                });

                // Get paths and performance metrics
                var paths = ExtractPaths(result.MarketStates);
                var performance = CalculatePerformance(result);

                // Update models based on results
                await UpdateModels(paths, performance);

                batchLoss += -performance.sharpeRatio; // Minimize negative Sharpe ratio
            }

            return batchLoss;
        }

        private async Task ValidateBatch(List<List<MarketSnapshot>> sequences)
        {
            var validationMetrics = new Dictionary<string, double>();

            foreach (var sequence in sequences)
            {
                var result = await backtester.RunBacktest(sequence, new BacktestConfig
                {
                    LookbackPeriods = 20,
                    ConfidenceThreshold = 0.6  // Higher threshold for validation
                });

                // Aggregate validation metrics
                foreach (var (key, value) in result.FinalMetrics)
                {
                    if (!validationMetrics.ContainsKey(key))
                        validationMetrics[key] = 0;
                    validationMetrics[key] += value;
                }
            }

            // Average metrics across sequences
            foreach (var key in validationMetrics.Keys.ToList())
            {
                validationMetrics[key] /= sequences.Count;
            }

            Console.WriteLine("Validation Metrics:");
            Console.WriteLine($"Sharpe Ratio: {validationMetrics["sharpe_ratio"]:F3}");
            Console.WriteLine($"Max Drawdown: {validationMetrics["max_drawdown"]:F3}");
            Console.WriteLine($"Win Rate: {validationMetrics["win_rate"]:F3}");
        }

        private List<List<PradOp>> ExtractPaths(List<double[]> marketStates)
        {
            var paths = new List<List<PradOp>>();
            int pathLength = 10;

            for (int i = 0; i < marketStates.Count - pathLength; i++)
            {
                var path = marketStates
                    .Skip(i)
                    .Take(pathLength)
                    .Select(state => explorer.encoder.EncodeSnapshot(state))
                    .ToList();

                paths.Add(path);
            }

            return paths;
        }

        private (double sharpeRatio, double totalReturn, double maxDrawdown)
            CalculatePerformance(BacktestResult result)
        {
            var returns = CalculateReturns(result.PortfolioHistory);

            return (
                sharpeRatio: metrics.CalculateSharpeRatio(returns),
                totalReturn: result.PortfolioHistory.Last().Value / result.PortfolioHistory.First().Value - 1,
                maxDrawdown: result.PortfolioHistory.Max(s => s.Drawdown)
            );
        }

        private List<double> CalculateReturns(List<PortfolioSnapshot> history)
        {
            var returns = new List<double>();
            for (int i = 1; i < history.Count; i++)
            {
                returns.Add(history[i].Value / history[i - 1].Value - 1);
            }
            return returns;
        }

        private async Task UpdateModels(
            List<List<PradOp>> paths,
            (double sharpeRatio, double totalReturn, double maxDrawdown) performance)
        {
            // Calculate rewards for paths
            var baseReward = performance.sharpeRatio;
            var drawdownPenalty = Math.Pow(performance.maxDrawdown, 2);
            var adjustedReward = baseReward * (1 - drawdownPenalty);

            foreach (var path in paths)
            {
                // Update VAE
                await UpdateVAE(path);

                // Update SPN
                UpdateSPN(path, adjustedReward);

                // Update gates
                UpdateGates(path, adjustedReward);
            }
        }

        private async Task UpdateVAE(List<PradOp> path)
        {
            // Train both VAEs on the sequence
            await explorer.architectVAE.TrainOnSequence(path);
            await explorer.explorerVAE.TrainOnSequence(path);
        }

        private void UpdateSPN(List<PradOp> path, double reward)
        {
            var rewardScalar = Math.Max(-1.0, Math.Min(1.0, reward));

            for (int i = 0; i < path.Count; i++)
            {
                var discountFactor = Math.Pow(0.95, path.Count - i - 1);
                var discountedReward = new PradOp(new Tensor(new[] { 1 }, rewardScalar * discountFactor));

                var routing = explorer.spn.RouteStateInternal(new List<PradOp> { path[i] });
                explorer.spn.UpdateFieldsWithEntropyAwareness(
                    routing.routing,
                    discountedReward,
                    path.Take(i + 1).ToList());
            }
        }

        private void UpdateGates(List<PradOp> path, double reward)
        {
            foreach (var state in path)
            {
                // Find most active gates
                var activations = explorer.gateManager.Gates.Values
                    .Select(gate => (gate, activation: gate.CalculateActivation(state)))
                    .OrderByDescending(x => x.activation)
                    .Take(2);

                // Update gate vectors
                foreach (var (gate, activation) in activations)
                {
                    if (activation > gate.ActivationThreshold * 0.5)
                    {
                        var stateVector = state.Result.Data.Select(x => (float)x).ToArray();
                        gate.UpdateVector(stateVector, learningRate * activation * (float)reward);
                    }
                }
            }
        }

        public Dictionary<string, double> GetTrainingStats()
        {
            return new Dictionary<string, double>
            {
                ["learning_rate"] = learningRate,
                ["batch_size"] = batchSize,
                ["epoch_samples"] = epochSamples
                // Add more stats as needed
            };
        }
    }
}
