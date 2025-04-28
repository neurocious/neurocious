using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class TradeExecutor : ITradeExecutor
    {
        public Trade ExecuteTrade(
            PortfolioManager portfolio,
            string action,
            double price,
            double confidence,
            Dictionary<string, double> costs)
        {
            // Calculate position size based on confidence and portfolio value
            double positionSize = CalculatePositionSize(portfolio, confidence);

            // Apply transaction costs
            double totalCost = CalculateTransactionCosts(positionSize, price, costs);

            // Execute the trade
            var trade = new Trade
            {
                Action = action,
                Price = price,
                Size = positionSize,
                Cost = totalCost,
                Timestamp = DateTime.UtcNow,
                Confidence = confidence
            };

            // Update portfolio
            portfolio.ProcessTrade(trade);

            return trade;
        }

        private double CalculatePositionSize(PortfolioManager portfolio, double confidence)
        {
            // Kelly criterion with confidence adjustment
            double kelly = confidence * 2 - 1; // Transform [0.5, 1] to [0, 1]
            double fraction = Math.Max(0.0, Math.Min(0.2, kelly * 0.5)); // Conservative Kelly

            return portfolio.CurrentValue * fraction;
        }

        private double CalculateTransactionCosts(
            double size,
            double price,
            Dictionary<string, double> costs)
        {
            double totalCost = 0;
            double notional = size * price;

            totalCost += notional * costs["commission"];
            totalCost += notional * costs["slippage"];
            totalCost += notional * costs["spread"] / 2;

            return totalCost;
        }

        public async Task<Trade> ClosePosition(Position position, string reason)
        {
            return new Trade
            {
                Action = position.IsLong ? "Sell" : "Buy",
                Price = position.CurrentPrice,
                Size = -position.Size,
                Timestamp = DateTime.UtcNow,
                Reason = reason
            };
        }
    }
}
