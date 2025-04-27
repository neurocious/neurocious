using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class PortfolioSnapshot
    {
        public DateTime Timestamp { get; set; }
        public double Value { get; set; }
        public double Drawdown { get; set; }
        public double NetExposure { get; set; }
        public List<Position> OpenPositions { get; set; }
    }
}
