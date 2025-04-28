using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.Chess
{
    /// <summary>
    /// Encodes chess positions into tensor representations suitable for neural processing.
    /// </summary>
    public class ChessEncoder : IChessEncoder
    {
        private const int BOARD_SIZE = 8;
        private const int CHANNELS = 12; // 6 piece types * 2 colors
        private const int TOTAL_FEATURES = BOARD_SIZE * BOARD_SIZE * CHANNELS;

        // Piece type indices (0-5 for white, 6-11 for black)
        private static readonly Dictionary<char, int> PIECE_INDICES = new()
    {
        {'P', 0}, {'N', 1}, {'B', 2}, {'R', 3}, {'Q', 4}, {'K', 5},
        {'p', 6}, {'n', 7}, {'b', 8}, {'r', 9}, {'q', 10}, {'k', 11}
    };

        public PradOp EncodeFEN(string fen)
        {
            var parts = fen.Split(' ');
            var boardFen = parts[0];
            var sideToMove = parts[1];
            var castlingRights = parts[2];
            var enPassant = parts[3];

            // Initialize board tensor (12 channels, 8x8 board)
            var boardTensor = new double[TOTAL_FEATURES];

            // Parse board position
            var ranks = boardFen.Split('/');
            for (int rank = 0; rank < BOARD_SIZE; rank++)
            {
                int file = 0;
                foreach (char c in ranks[rank])
                {
                    if (char.IsDigit(c))
                    {
                        file += c - '0';
                    }
                    else
                    {
                        // Set piece presence in appropriate channel
                        int pieceIndex = PIECE_INDICES[c];
                        int flatIndex = (pieceIndex * BOARD_SIZE * BOARD_SIZE) + (rank * BOARD_SIZE) + file;
                        boardTensor[flatIndex] = 1.0;
                        file++;
                    }
                }
            }

            // Add extra features beyond piece positions
            var extraFeatures = EncodeExtraFeatures(sideToMove, castlingRights, enPassant);
            var fullTensor = new double[TOTAL_FEATURES + extraFeatures.Length];
            Array.Copy(boardTensor, fullTensor, TOTAL_FEATURES);
            Array.Copy(extraFeatures, 0, fullTensor, TOTAL_FEATURES, extraFeatures.Length);

            return new PradOp(new Tensor(new[] { fullTensor.Length }, fullTensor));
        }

        private double[] EncodeExtraFeatures(string sideToMove, string castlingRights, string enPassant)
        {
            var features = new List<double>();

            // Side to move (1 for white, 0 for black)
            features.Add(sideToMove == "w" ? 1.0 : 0.0);

            // Castling rights (4 bits)
            features.Add(castlingRights.Contains('K') ? 1.0 : 0.0);
            features.Add(castlingRights.Contains('Q') ? 1.0 : 0.0);
            features.Add(castlingRights.Contains('k') ? 1.0 : 0.0);
            features.Add(castlingRights.Contains('q') ? 1.0 : 0.0);

            // En passant square (if any)
            if (enPassant != "-")
            {
                int file = enPassant[0] - 'a';
                int rank = enPassant[1] - '1';
                int squareIndex = rank * 8 + file;
                var enPassantFeatures = new double[64]; // One-hot encoding of possible squares
                enPassantFeatures[squareIndex] = 1.0;
                features.AddRange(enPassantFeatures);
            }
            else
            {
                features.AddRange(new double[64]); // No en passant square
            }

            return features.ToArray();
        }
    }
}
