using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public interface IFinancialGeodesicExplorer
    {
        /// <summary>
        /// Analyzes a market snapshot and proposes the best trade decision.
        /// </summary>
        (string action, double confidence, Dictionary<string, double> metrics) FindBestTradeDecision(double[] marketSnapshot);
    }
}
