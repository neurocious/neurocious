using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class PortfolioManager : IPortfolioManager
    {
        public double InitialCapital { get; }
        public double CurrentValue { get; private set; }
        public List<Position> OpenPositions { get; private set; }
        public double NetExposure => OpenPositions.Sum(p => p.Size);
        public double PeakValue { get; private set; }
        public double CurrentDrawdown => (PeakValue - CurrentValue) / PeakValue;

        public PortfolioManager(double initialCapital)
        {
            InitialCapital = initialCapital;
            CurrentValue = initialCapital;
            PeakValue = initialCapital;
            OpenPositions = new List<Position>();
        }

        public void ProcessTrade(Trade trade)
        {
            // Update positions
            if (trade.Action == "Buy" || trade.Action == "Sell")
            {
                UpdatePositions(trade);
            }

            // Update portfolio value
            CurrentValue -= trade.Cost;
            PeakValue = Math.Max(PeakValue, CurrentValue);
        }

        public void UpdatePortfolioValue(double currentPrice)
        {
            foreach (var position in OpenPositions)
            {
                position.UpdateValue(currentPrice);
            }

            CurrentValue = InitialCapital + OpenPositions.Sum(p => p.UnrealizedPnL);
            PeakValue = Math.Max(PeakValue, CurrentValue);
        }

        private void UpdatePositions(Trade trade)
        {
            bool isLong = trade.Action == "Buy";
            var existingPosition = OpenPositions
                .FirstOrDefault(p => p.IsLong == isLong);

            if (existingPosition != null)
            {
                existingPosition.UpdateSize(trade.Size);
                if (Math.Abs(existingPosition.Size) < 1e-6)
                {
                    OpenPositions.Remove(existingPosition);
                }
            }
            else
            {
                OpenPositions.Add(new Position
                {
                    IsLong = isLong,
                    Size = trade.Size,
                    EntryPrice = trade.Price
                });
            }
        }

        public PortfolioSnapshot GetSnapshot()
        {
            return new PortfolioSnapshot
            {
                Timestamp = DateTime.UtcNow,
                Value = CurrentValue,
                Drawdown = CurrentDrawdown,
                NetExposure = NetExposure,
                OpenPositions = OpenPositions.Select(p => p.Clone()).ToList()
            };
        }
    }
}
