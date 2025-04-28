using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.Chess
{
    /// <summary>
    /// Decodes latent representations back into chess positions and moves.
    /// </summary>
    public class ChessDecoder : IChessDecoder
    {
        private const int BOARD_SIZE = 8;
        private const int CHANNELS = 12;
        private readonly string[] FILES = { "a", "b", "c", "d", "e", "f", "g", "h" };
        private readonly string[] RANKS = { "1", "2", "3", "4", "5", "6", "7", "8" };

        private static readonly Dictionary<int, char> PIECE_CHARS = new()
    {
        {0, 'P'}, {1, 'N'}, {2, 'B'}, {3, 'R'}, {4, 'Q'}, {5, 'K'},
        {6, 'p'}, {7, 'n'}, {8, 'b'}, {9, 'r'}, {10, 'q'}, {11, 'k'}
    };

        public string DecodeToFEN(PradOp latentState)
        {
            var boardData = latentState.Result.Data;
            var fen = new System.Text.StringBuilder();

            // Decode board position
            for (int rank = 0; rank < BOARD_SIZE; rank++)
            {
                int emptySquares = 0;

                for (int file = 0; file < BOARD_SIZE; file++)
                {
                    bool pieceFound = false;

                    // Check each channel for a piece at this square
                    for (int channel = 0; channel < CHANNELS; channel++)
                    {
                        int idx = (channel * BOARD_SIZE * BOARD_SIZE) + (rank * BOARD_SIZE) + file;
                        if (boardData[idx] > 0.5) // Threshold for piece presence
                        {
                            if (emptySquares > 0)
                            {
                                fen.Append(emptySquares);
                                emptySquares = 0;
                            }
                            fen.Append(PIECE_CHARS[channel]);
                            pieceFound = true;
                            break;
                        }
                    }

                    if (!pieceFound)
                    {
                        emptySquares++;
                    }
                }

                if (emptySquares > 0)
                {
                    fen.Append(emptySquares);
                }

                if (rank < BOARD_SIZE - 1)
                {
                    fen.Append('/');
                }
            }

            // Add extra position details using extra features from the latent state
            var extraFeatures = DecodeExtraFeatures(boardData);
            fen.Append($" {extraFeatures.sideToMove} {extraFeatures.castling} {extraFeatures.enPassant} 0 1");

            return fen.ToString();
        }

        public string DecodeLatentMove(PradOp currentState, PradOp nextState)
        {
            // Calculate difference between states to identify the move
            var diff = nextState.Sub(currentState.Result);
            var moveData = diff.Result.Data;

            // Find the strongest changes in the board representation
            var changes = new List<(int channel, int rank, int file, double value)>();

            for (int channel = 0; channel < CHANNELS; channel++)
            {
                for (int rank = 0; rank < BOARD_SIZE; rank++)
                {
                    for (int file = 0; file < BOARD_SIZE; file++)
                    {
                        int idx = (channel * BOARD_SIZE * BOARD_SIZE) + (rank * BOARD_SIZE) + file;
                        if (Math.Abs(moveData[idx]) > 0.5)
                        {
                            changes.Add((channel, rank, file, moveData[idx]));
                        }
                    }
                }
            }

            // Sort changes by absolute magnitude
            changes.Sort((a, b) => Math.Abs(b.value).CompareTo(Math.Abs(a.value)));

            // Find 'from' and 'to' squares based on strongest negative and positive changes
            var from = changes.First(c => c.value < 0);
            var to = changes.First(c => c.value > 0);

            // Convert to algebraic notation
            return $"{FILES[from.file]}{RANKS[from.rank]}{FILES[to.file]}{RANKS[to.rank]}";
        }

        private (string sideToMove, string castling, string enPassant) DecodeExtraFeatures(double[] data)
        {
            int extraOffset = BOARD_SIZE * BOARD_SIZE * CHANNELS;

            // Decode side to move
            string sideToMove = data[extraOffset] > 0.5 ? "w" : "b";

            // Decode castling rights
            var castling = new System.Text.StringBuilder();
            if (data[extraOffset + 1] > 0.5) castling.Append('K');
            if (data[extraOffset + 2] > 0.5) castling.Append('Q');
            if (data[extraOffset + 3] > 0.5) castling.Append('k');
            if (data[extraOffset + 4] > 0.5) castling.Append('q');
            string castlingStr = castling.Length > 0 ? castling.ToString() : "-";

            // Decode en passant square
            string enPassant = "-";
            var epOffset = extraOffset + 5;
            for (int i = 0; i < 64; i++)
            {
                if (data[epOffset + i] > 0.5)
                {
                    int rank = i / 8;
                    int file = i % 8;
                    enPassant = $"{FILES[file]}{RANKS[rank]}";
                    break;
                }
            }

            return (sideToMove, castlingStr, enPassant);
        }
    }
}
