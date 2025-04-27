using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class ExponentialMovingAverage
    {
        private readonly double alpha;
        private double? currentValue;

        public ExponentialMovingAverage(double alpha = 0.1)
        {
            this.alpha = alpha;
            this.currentValue = null;
        }

        public double Update(double newValue)
        {
            if (currentValue == null)
            {
                currentValue = newValue;
            }
            else
            {
                currentValue = alpha * newValue + (1 - alpha) * currentValue.Value;
            }

            return currentValue.Value;
        }

        public void Reset()
        {
            currentValue = null;
        }
    }
}
