using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class BacktestConfig
    {
        public int LookbackPeriods { get; set; } = 20;
        public double ConfidenceThreshold { get; set; } = 0.6;
        public RiskLimits RiskLimits { get; set; }
    }
}
