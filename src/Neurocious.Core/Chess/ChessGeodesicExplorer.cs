using Neurocious.Core.EnhancedVariationalAutoencoder;
using Neurocious.Core.SpatialProbability;
using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.Chess
{
    /// <summary>
    /// Explores chess positions through high-dimensional latent space using field theory.
    /// </summary>
    public class ChessGeodesicExplorer : IChessGeodesicExplorer
    {
        private readonly EnhancedVAE architectVAE;
        private readonly EnhancedVAE explorerVAE;
        private readonly SpatialProbabilityNetwork spn;
        private readonly ChessEncoder chessEncoder;
        private readonly ChessDecoder chessDecoder;
        private readonly AdvancedGateManager gateManager;

        // Hyperparameters for loss function components
        private readonly double pathEnergyWeight = 0.6;     // α: Path energy importance
        private readonly double gateFailureWeight = 0.3;    // β: Strategic gate alignment
        private readonly double goalDeviationWeight = 0.1;  // γ: Goal state achievement

        // Exploration parameters
        private readonly int pathSteps = 12;                // Steps to look ahead
        private readonly double baseStepSize = 0.05;        // Base step size for path following
        private readonly int candidatePaths = 8;            // Number of candidate paths to generate
        private readonly double baseExplorationRate = 0.1;  // Base exploration noise scale

        public ChessGeodesicExplorer(
            EnhancedVAE architectVAE,
            EnhancedVAE explorerVAE,
            SpatialProbabilityNetwork spn,
            ChessEncoder chessEncoder,
            ChessDecoder chessDecoder)
        {
            this.architectVAE = architectVAE;
            this.explorerVAE = explorerVAE;
            this.spn = spn;
            this.chessEncoder = chessEncoder;
            this.chessDecoder = chessDecoder;

            // Initialize gate manager with base strategic gates
            this.gateManager = new AdvancedGateManager(InitializeBaseGates());
        }

        private Dictionary<string, StrategicGate> InitializeBaseGates()
        {
            return new Dictionary<string, StrategicGate>
            {
                ["material_advantage"] = new StrategicGate(
                    name: "material_advantage",
                    dim: explorerVAE.LatentDimension,
                    threshold: 0.2f,
                    weight: 0.4f),

                ["center_control"] = new StrategicGate(
                    name: "center_control",
                    dim: explorerVAE.LatentDimension,
                    threshold: 0.3f,
                    weight: 0.3f),

                ["development"] = new StrategicGate(
                    name: "development",
                    dim: explorerVAE.LatentDimension,
                    threshold: 0.25f,
                    weight: 0.2f),

                ["king_safety"] = new StrategicGate(
                    name: "king_safety",
                    dim: explorerVAE.LatentDimension,
                    threshold: 0.4f,
                    weight: 0.3f),

                ["pawn_structure"] = new StrategicGate(
                    name: "pawn_structure",
                    dim: explorerVAE.LatentDimension,
                    threshold: 0.35f,
                    weight: 0.15f)
            };
        }

        public string FindBestMove(string fenPosition)
        {
            // Encode chess position to latent state
            var boardState = chessEncoder.EncodeFEN(fenPosition);
            var latentState = explorerVAE.EncodeSequence(new List<PradOp> { boardState }).Item1.PradOp;

            // Generate multiple candidate paths following the field
            var candidateLatentPaths = GenerateCandidatePaths(latentState);

            // Evaluate paths and select the best
            var (bestPath, bestEnergy) = EvaluatePaths(candidateLatentPaths, fenPosition);

            // Record trajectory for gate learning
            gateManager.ProcessTrajectory(bestPath, -bestEnergy);

            // Reinforce the best path
            ReinforceBestPath(bestPath, -bestEnergy);

            // Decode the first move from the best path
            var firstMoveState = bestPath[1]; // Skip the starting position
            return chessDecoder.DecodeLatentMove(latentState, firstMoveState);
        }

        private List<List<PradOp>> GenerateCandidatePaths(PradOp startState)
        {
            var paths = new List<List<PradOp>>();

            // Shape the field based on architect's guidance
            ShapeFieldForPosition(startState);

            // Generate multiple candidate paths with different samplings
            for (int i = 0; i < candidatePaths; i++)
            {
                paths.Add(FollowProbabilisticFieldPath(startState));
            }

            return paths;
        }

        private void ShapeFieldForPosition(PradOp position)
        {
            // Get architect's guidance
            var architectEncoding = architectVAE.EncodeSequence(new List<PradOp> { position }).Item1.PradOp;
            var fieldParams = architectVAE.ExtractFieldParameters(architectEncoding);

            // Update SPN's field parameters
            spn.UpdateFieldParameters(fieldParams);
        }

        private List<PradOp> FollowProbabilisticFieldPath(PradOp startState)
        {
            var path = new List<PradOp> { startState };
            var current = startState;
            var random = new Random();

            for (int i = 0; i < pathSteps; i++)
            {
                var (routing, confidence, fieldParams) = spn.RouteStateInternal(new List<PradOp> { current });

                // Get vector field direction
                var moveDirection = routing.PradOp.MatMul(spn.VectorField.CurrentTensor);

                // Sample from probability field
                var probabilitySample = SampleFromProbabilityField(current, routing);

                // Combine directional guidance with probability-based sampling
                var vectorWeight = 0.7;
                var probabilityWeight = 0.3;

                var combinedDirection = moveDirection.Result.ElementwiseMultiply(
                    new Tensor(moveDirection.Result.Shape, vectorWeight))
                    .Add(probabilitySample.ElementwiseMultiply(
                        new Tensor(probabilitySample.Shape, probabilityWeight)));

                // Calculate entropy-aware exploration rate
                float explorationRate = this.CalculateEntropyAwareExplorationRate(
                    fieldParams,
                    (float)baseExplorationRate);

                // Add adaptive noise
                var noise = new double[combinedDirection.Data.Length];
                for (int j = 0; j < noise.Length; j++)
                {
                    noise[j] = random.NextGaussian(0, explorationRate);
                }

                // Calculate adaptive step size
                var adaptiveStepSize = baseStepSize * (1.0 / (1.0 + fieldParams.Curvature));

                // Move to next state
                var nextState = current.Add(
                    combinedDirection.Add(new Tensor(combinedDirection.Shape, noise))
                    .Mul(new Tensor(combinedDirection.Shape, adaptiveStepSize)));

                path.Add(nextState.PradOp);
                current = nextState.PradOp;
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

        private (List<PradOp> path, double energy) EvaluatePaths(
            List<List<PradOp>> paths,
            string startFEN)
        {
            List<PradOp> bestPath = null;
            double bestScore = double.MaxValue;

            foreach (var path in paths)
            {
                double pathEnergy = CalculatePathEnergy(path);
                double gateFailure = CalculateGateFailure(path);
                double goalDeviation = CalculateGoalDeviation(path, startFEN);

                double totalEnergy =
                    pathEnergyWeight * pathEnergy +
                    gateFailureWeight * gateFailure +
                    goalDeviationWeight * goalDeviation;

                if (totalEnergy < bestScore)
                {
                    bestScore = totalEnergy;
                    bestPath = path;
                }
            }

            return (bestPath, bestScore);
        }

        private double CalculatePathEnergy(List<PradOp> path)
        {
            double energy = 0.0;

            for (int i = 0; i < path.Count - 1; i++)
            {
                var current = path[i];
                var (routing, _, fieldParams) = spn.RouteStateInternal(new List<PradOp> { current });

                // Calculate local energy components
                double curvatureEnergy = fieldParams.Curvature * 0.5;
                double alignmentEnergy = (1.0 - Math.Abs(fieldParams.Alignment)) * 0.3;
                double entropyEnergy = fieldParams.Entropy * 0.2;

                energy += curvatureEnergy + alignmentEnergy + entropyEnergy;
            }

            return energy / (path.Count - 1);
        }

        private double CalculateGateFailure(List<PradOp> path)
        {
            double totalFailure = 0.0;

            for (int i = 0; i < path.Count; i++)
            {
                double pathProgress = i / (double)(path.Count - 1);
                var state = path[i];

                foreach (var gate in gateManager.Gates.Values)
                {
                    double activation = gate.CalculateActivation(state);
                    double targetActivation = GetTargetGateActivation(gate.Name, pathProgress);
                    double failureContribution = Math.Pow(targetActivation - activation, 2);

                    totalFailure += failureContribution * gate.Weight;
                }
            }

            return totalFailure / path.Count;
        }

        private double GetTargetGateActivation(string gateName, double pathProgress)
        {
            return gateName switch
            {
                "material_advantage" => 0.3 + 0.7 * pathProgress,
                "center_control" => Math.Min(0.8, 0.4 + 0.6 * pathProgress),
                "development" => Math.Min(0.9, 0.5 + 0.5 * pathProgress),
                "king_safety" => 0.7 + 0.3 * pathProgress,
                "pawn_structure" => 0.4 + 0.4 * pathProgress,
                _ when gateName.StartsWith("hierarchical_") => 0.5 + 0.5 * pathProgress,
                _ when gateName.StartsWith("fused_") => 0.4 + 0.6 * pathProgress,
                _ => 0.5 + 0.5 * pathProgress
            };
        }

        private double CalculateGoalDeviation(List<PradOp> path, string startFEN)
        {
            var finalState = path[^1];
            string finalFEN = chessDecoder.DecodeToFEN(finalState);

            // Calculate strategic metrics
            double materialScore = EvaluateMaterialAdvantage(finalFEN);
            double kingSafetyScore = EvaluateKingSafety(finalFEN);
            double positionScore = EvaluatePosition(finalFEN);

            return -(materialScore * 0.5 + kingSafetyScore * 0.3 + positionScore * 0.2);
        }

        private void ReinforceBestPath(List<PradOp> path, double reward)
        {
            var rewardScalar = Math.Max(-1.0, Math.Min(1.0, reward));

            for (int i = 0; i < path.Count; i++)
            {
                var current = path[i];
                var discount = Math.Pow(0.95, path.Count - i);
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

        // Chess evaluation helper methods
        private double EvaluateMaterialAdvantage(string fen)
        {
            var pieceValues = new Dictionary<char, int>
        {
            {'P', 1}, {'N', 3}, {'B', 3}, {'R', 5}, {'Q', 9}, {'K', 0},
            {'p', -1}, {'n', -3}, {'b', -3}, {'r', -5}, {'q', -9}, {'k', 0}
        };

            return fen.Where(c => pieceValues.ContainsKey(c))
                     .Sum(c => pieceValues[c]) / 39.0; // Normalize by max possible advantage
        }

        private double EvaluateKingSafety(string fen)
        {
            // Simplified king safety evaluation
            var board = fen.Split(' ')[0];
            var whiteKingPos = FindPiece(board, 'K');
            var blackKingPos = FindPiece(board, 'k');

            return (EvaluateKingPosition(whiteKingPos, true) -
                    EvaluateKingPosition(blackKingPos, false)) / 2.0;
        }

        private (int rank, int file) FindPiece(string board, char piece)
        {
            var ranks = board.Split('/');
            for (int rank = 0; rank < 8; rank++)
            {
                int file = 0;
                foreach (char c in ranks[rank])
                {
                    if (char.IsDigit(c))
                    {
                        file += c - '0';
                    }
                    else if (c == piece)
                    {
                        return (rank, file);
                    }
                    else
                    {
                        file++;
                    }
                }
            }
            return (-1, -1);
        }

        private double EvaluateKingPosition(
            (int rank, int file) kingPos,
            bool isWhite)
        {
            if (kingPos.rank == -1) return 0;

            // Prefer back rank in early/midgame
            double rankSafety = isWhite
                ? (7 - kingPos.rank) / 7.0
                : kingPos.rank / 7.0;

            // Prefer castled positions (kingside or queenside)
            double fileSafety = Math.Min(
                Math.Abs(kingPos.file - 1),  // Queenside
                Math.Abs(kingPos.file - 6)   // Kingside
            ) / 6.0;

            return (rankSafety * 0.7 + fileSafety * 0.3);
        }

        private double EvaluatePosition(string fen)
        {
            var board = fen.Split(' ')[0];

            // Evaluate center control
            double centerControl = EvaluateCenterControl(board);

            // Evaluate piece development
            double development = EvaluateDevelopment(board);

            // Evaluate pawn structure
            double pawnStructure = EvaluatePawnStructure(board);

            return (centerControl * 0.4 + development * 0.3 + pawnStructure * 0.3);
        }

        private double EvaluateCenterControl(string board)
        {
            // Center squares: e4, e5, d4, d5
            var centerSquares = new[] { (3, 4), (4, 4), (3, 3), (4, 3) };
            var ranks = board.Split('/');

            double control = 0;
            foreach (var (rank, file) in centerSquares)
            {
                char piece = GetPieceAt(ranks, rank, file);
                if (char.IsUpper(piece)) control += 1;  // White piece
                else if (char.IsLower(piece)) control -= 1;  // Black piece
            }

            return control / 4.0;  // Normalize to [-1, 1]
        }

        private double EvaluateDevelopment(string board)
        {
            var ranks = board.Split('/');
            double development = 0;

            // Check if pieces have moved from their starting squares
            // White pieces
            if (GetPieceAt(ranks, 7, 1) != 'N') development += 1;  // b1 knight
            if (GetPieceAt(ranks, 7, 6) != 'N') development += 1;  // g1 knight
            if (GetPieceAt(ranks, 7, 2) != 'B') development += 1;  // c1 bishop
            if (GetPieceAt(ranks, 7, 5) != 'B') development += 1;  // f1 bishop

            // Black pieces
            if (GetPieceAt(ranks, 0, 1) != 'n') development -= 1;  // b8 knight
            if (GetPieceAt(ranks, 0, 6) != 'n') development -= 1;  // g8 knight
            if (GetPieceAt(ranks, 0, 2) != 'b') development -= 1;  // c8 bishop
            if (GetPieceAt(ranks, 0, 5) != 'b') development -= 1;  // f8 bishop

            return development / 8.0;  // Normalize to [-1, 1]
        }

        private double EvaluatePawnStructure(string board)
        {
            var ranks = board.Split('/');
            double structure = 0;

            // Evaluate doubled pawns (penalize)
            for (int file = 0; file < 8; file++)
            {
                int whitePawns = 0;
                int blackPawns = 0;
                for (int rank = 0; rank < 8; rank++)
                {
                    char piece = GetPieceAt(ranks, rank, file);
                    if (piece == 'P') whitePawns++;
                    else if (piece == 'p') blackPawns++;
                }
                if (whitePawns > 1) structure -= 0.5;
                if (blackPawns > 1) structure += 0.5;
            }

            // Evaluate isolated pawns (penalize)
            for (int file = 0; file < 8; file++)
            {
                bool hasWhitePawn = false;
                bool hasBlackPawn = false;
                bool hasNeighborWhitePawn = false;
                bool hasNeighborBlackPawn = false;

                // Check for pawns in current file
                for (int rank = 0; rank < 8; rank++)
                {
                    char piece = GetPieceAt(ranks, rank, file);
                    if (piece == 'P') hasWhitePawn = true;
                    else if (piece == 'p') hasBlackPawn = true;
                }

                // Check neighboring files
                if (file > 0)
                {
                    for (int rank = 0; rank < 8; rank++)
                    {
                        char piece = GetPieceAt(ranks, rank, file - 1);
                        if (piece == 'P') hasNeighborWhitePawn = true;
                        else if (piece == 'p') hasNeighborBlackPawn = true;
                    }
                }
                if (file < 7)
                {
                    for (int rank = 0; rank < 8; rank++)
                    {
                        char piece = GetPieceAt(ranks, rank, file + 1);
                        if (piece == 'P') hasNeighborWhitePawn = true;
                        else if (piece == 'p') hasNeighborBlackPawn = true;
                    }
                }

                if (hasWhitePawn && !hasNeighborWhitePawn) structure -= 0.5;
                if (hasBlackPawn && !hasNeighborBlackPawn) structure += 0.5;
            }

            // Normalize pawn structure score
            return structure / 8.0;  // Typical max score considering doubled and isolated pawns
        }

        private char GetPieceAt(string[] ranks, int rank, int file)
        {
            int currentFile = 0;
            string rankStr = ranks[rank];

            foreach (char c in rankStr)
            {
                if (char.IsDigit(c))
                {
                    currentFile += c - '0';
                    if (currentFile > file) return '.';
                }
                else
                {
                    if (currentFile == file) return c;
                    currentFile++;
                }
            }

            return '.';
        }
    }
}
