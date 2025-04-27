using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class Position
    {
        public bool IsLong { get; set; }
        public double Size { get; set; }
        public double EntryPrice { get; set; }
        public double CurrentPrice { get; set; }
        public double UnrealizedPnL => (CurrentPrice - EntryPrice) * Size * (IsLong ? 1 : -1);

        public void UpdateSize(double additionalSize)
        {
            Size += additionalSize;
        }

        public void UpdateValue(double price)
        {
            CurrentPrice = price;
        }

        public Position Clone()
        {
            return new Position
            {
                IsLong = IsLong,
                Size = Size,
                EntryPrice = EntryPrice,
                CurrentPrice = CurrentPrice
            };
        }
    }
}
