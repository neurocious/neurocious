using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public interface IPortfolioManager
    {
        double InitialCapital { get; }
        double CurrentValue { get; }
        double NetExposure { get; }
        double CurrentDrawdown { get; }

        void ProcessTrade(Trade trade);
        void UpdatePortfolioValue(double currentPrice);
        PortfolioSnapshot GetSnapshot();
    }
}
