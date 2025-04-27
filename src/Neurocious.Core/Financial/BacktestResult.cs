using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class BacktestResult
    {
        public List<Trade> Trades { get; set; }
        public List<PortfolioSnapshot> PortfolioHistory { get; set; }
        public List<double[]> MarketStates { get; set; }
        public List<Dictionary<string, double>> PerformanceLog { get; set; }
        public Dictionary<string, double> FinalMetrics { get; set; }
    }
}
