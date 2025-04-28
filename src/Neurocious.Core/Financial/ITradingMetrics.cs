using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public interface ITradingMetrics
    {
        // Trading performance metrics
        double CalculateProfitFactor(List<Trade> trades);
        double CalculateWinRate(List<Trade> trades);
        double CalculateAverageTradeReturn(List<Trade> trades);

        // Path construction for model input
        List<List<PradOp>> CreatePathsFromStates(List<double[]> marketStates);

        // Full portfolio evaluation
        Dictionary<string, double> CalculateFinalMetrics(List<Trade> trades, List<PortfolioSnapshot> portfolioHistory);
    }
}
