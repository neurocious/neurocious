using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.EnhancedVariationalAutoencoder
{
    public interface IAttentionBlock
    {
        /// <summary>
        /// Forward pass through the attention block.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <param name="training">Whether to apply causal masking (True during training).</param>
        /// <returns>Output tensor after attention and normalization.</returns>
        PradResult Forward(PradOp input, bool training = true);
    }
}
