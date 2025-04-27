using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class Trade
    {
        public string Action { get; set; }
        public double Price { get; set; }
        public double Size { get; set; }
        public double Cost { get; set; }
        public DateTime Timestamp { get; set; }
        public double Confidence { get; set; }
        public string Reason { get; set; }
    }
}
