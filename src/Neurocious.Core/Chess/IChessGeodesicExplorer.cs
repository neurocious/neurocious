using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Chess
{
    public interface IChessGeodesicExplorer
    {
        /// <summary>
        /// Finds the best move from a given FEN position using latent space exploration.
        /// </summary>
        /// <param name="fenPosition">The FEN string of the current board position.</param>
        /// <returns>The best move in algebraic notation (e.g., e2e4).</returns>
        string FindBestMove(string fenPosition);
    }
}
