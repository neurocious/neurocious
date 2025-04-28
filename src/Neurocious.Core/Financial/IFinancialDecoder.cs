using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public interface IFinancialDecoder
    {
        /// <summary>
        /// Decodes the full latent vector into interpretable features.
        /// </summary>
        double[] DecodeFeatures(PradOp latentState);

        /// <summary>
        /// Decodes the trading decision (action + confidence) between two states.
        /// </summary>
        (string action, double confidence) DecodeTradeDecision(PradOp currentState, PradOp nextState);

        /// <summary>
        /// Decodes risk-related metrics from a latent state.
        /// </summary>
        Dictionary<string, double> DecodeRiskMetrics(PradOp state);
    }
}
