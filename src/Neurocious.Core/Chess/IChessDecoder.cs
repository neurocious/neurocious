using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Chess
{
    public interface IChessDecoder
    {
        /// <summary>
        /// Decodes a latent state into a FEN string.
        /// </summary>
        /// <param name="latentState">The latent tensor representing the board.</param>
        /// <returns>FEN representation of the board.</returns>
        string DecodeToFEN(PradOp latentState);

        /// <summary>
        /// Decodes the difference between two latent states into an algebraic move.
        /// </summary>
        /// <param name="currentState">The current board state in latent space.</param>
        /// <param name="nextState">The next board state in latent space.</param>
        /// <returns>Algebraic notation of the move.</returns>
        string DecodeLatentMove(PradOp currentState, PradOp nextState);
    }
}
