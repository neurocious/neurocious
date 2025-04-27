using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class RiskLimits
    {
        public double MaxDrawdown { get; set; } = 0.20;
        public double MaxExposure { get; set; } = 1.0;
        public double StopOutValue { get; set; }
    }
}
