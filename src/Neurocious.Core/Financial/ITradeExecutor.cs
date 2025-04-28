using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public interface ITradeExecutor
    {
        /// <summary>
        /// Executes a trade based on portfolio status, market price, confidence, and transaction costs.
        /// </summary>
        /// <param name="portfolio">The current portfolio manager instance.</param>
        /// <param name="action">"Buy" or "Sell".</param>
        /// <param name="price">The market price at which to execute the trade.</param>
        /// <param name="confidence">Confidence level between 0.5 and 1.0.</param>
        /// <param name="costs">Dictionary of transaction cost percentages (commission, slippage, spread).</param>
        /// <returns>The executed trade.</returns>
        Trade ExecuteTrade(PortfolioManager portfolio, string action, double price, double confidence, Dictionary<string, double> costs);

        /// <summary>
        /// Closes an open position asynchronously, generating an exit trade.
        /// </summary>
        /// <param name="position">The position to be closed.</param>
        /// <param name="reason">The reason for closing (e.g., "stop loss", "take profit").</param>
        /// <returns>The closing trade.</returns>
        Task<Trade> ClosePosition(Position position, string reason);
    }
}
