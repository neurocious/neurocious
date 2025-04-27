using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.Chess
{
    /// <summary>
    /// Manages the training process for the chess geodesic explorer.
    /// </summary>
    public class ChessGeodesicTrainer
    {
        private readonly ChessGeodesicExplorer explorer;
        private readonly string[] testPositions;
        private readonly double learningRate;
        private readonly int batchSize;
        private readonly int epochSamples;

        public ChessGeodesicTrainer(
            ChessGeodesicExplorer explorer,
            string[] testPositions,
            double learningRate = 0.001,
            int batchSize = 32,
            int epochSamples = 1000)
        {
            this.explorer = explorer;
            this.testPositions = testPositions;
            this.learningRate = learningRate;
            this.batchSize = batchSize;
            this.epochSamples = epochSamples;
        }

        public async Task TrainEpoch()
        {
            var random = new Random();
            var totalLoss = 0.0;
            var batches = 0;

            for (int i = 0; i < epochSamples; i += batchSize)
            {
                var batchLoss = await TrainBatch(
                    testPositions.OrderBy(x => random.Next()).Take(batchSize));

                totalLoss += batchLoss;
                batches++;

                if (batches % 10 == 0)
                {
                    Console.WriteLine($"Batch {batches}, Average Loss: {batchLoss / batchSize:F4}");
                }
            }

            Console.WriteLine($"Epoch completed. Average Loss: {totalLoss / batches:F4}");
        }

        private async Task<double> TrainBatch(IEnumerable<string> positions)
        {
            double batchLoss = 0;

            foreach (var position in positions)
            {
                // Generate and evaluate paths
                var boardState = explorer.chessEncoder.EncodeFEN(position);
                var latentState = explorer.explorerVAE.EncodeSequence(
                    new List<PradOp> { boardState }).Item1.PradOp;

                var paths = explorer.GenerateCandidatePaths(latentState);
                var (bestPath, pathEnergy) = explorer.EvaluatePaths(paths, position);

                // Update models based on path results
                await UpdateModels(bestPath, -pathEnergy);

                batchLoss += pathEnergy;
            }

            return batchLoss;
        }

        private async Task UpdateModels(List<PradOp> path, double reward)
        {
            // Update VAE
            await UpdateVAE(path);

            // Update SPN with path reinforcement
            UpdateSPN(path, reward);

            // Update strategic gates
            UpdateStrategicGates(path);
        }

        private async Task UpdateVAE(List<PradOp> path)
        {
            // Train both VAEs on the sequence
            await explorer.architectVAE.TrainOnSequence(path);
            await explorer.explorerVAE.TrainOnSequence(path);
        }

        private void UpdateSPN(List<PradOp> path, double reward)
        {
            // Scale reward and apply temporal discounting
            var rewardScalar = Math.Max(-1.0, Math.Min(1.0, reward));

            for (int i = 0; i < path.Count; i++)
            {
                var discountFactor = Math.Pow(0.95, path.Count - i - 1);
                var discountedReward = new PradOp(new Tensor(new[] { 1 }, rewardScalar * discountFactor));

                var routing = explorer.spn.RouteStateInternal(new List<PradOp> { path[i] });
                explorer.spn.UpdateFields(routing.routing, discountedReward, path.Take(i + 1).ToList());
            }
        }

        private void UpdateStrategicGates(List<PradOp> path)
        {
            foreach (var state in path)
            {
                // Find most active gates for this state
                var activations = explorer.strategicGates.Values
                    .Select(gate => (gate, activation: gate.CalculateActivation(state)))
                    .OrderByDescending(x => x.activation)
                    .Take(2);

                // Update gate vectors based on state
                foreach (var (gate, activation) in activations)
                {
                    if (activation > gate.ActivationThreshold * 0.5)
                    {
                        var stateVector = state.Result.Data.Select(x => (float)x).ToArray();
                        gate.UpdateVector(stateVector, learningRate * activation);
                    }
                }
            }
        }
    }
}
