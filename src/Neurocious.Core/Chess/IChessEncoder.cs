using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Chess
{
    public interface IChessEncoder
    {
        /// <summary>
        /// Encodes a FEN string into a latent tensor representation.
        /// </summary>
        /// <param name="fen">The FEN string representing the chess board.</param>
        /// <returns>A PradOp representing the encoded board state.</returns>
        PradOp EncodeFEN(string fen);
    }
}
