using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public interface IBacktestEngine
    {
        Task<BacktestResult> RunBacktest(List<MarketSnapshot> historicalData, BacktestConfig config);
    }
}
}
