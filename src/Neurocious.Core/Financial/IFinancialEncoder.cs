using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public interface IFinancialEncoder
    {
        /// <summary>
        /// Encodes a financial snapshot into a tensor suitable for neural processing.
        /// </summary>
        PradOp EncodeSnapshot(double[] features);

        /// <summary>
        /// Updates normalization statistics based on recent snapshots.
        /// </summary>
        void UpdateNormalizationStats(List<double[]> recentSnapshots);
    }
}
