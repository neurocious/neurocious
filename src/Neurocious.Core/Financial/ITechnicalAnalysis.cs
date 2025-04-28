using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public interface ITechnicalAnalysis
    {
        double CalculateRSI(List<double> prices, int period = 14);
        (double macd, double signal, double histogram) CalculateMACD(List<double> prices, int fastPeriod = 12, int slowPeriod = 26, int signalPeriod = 9);
        (double upper, double middle, double lower) CalculateBollingerBands(List<double> prices, int period = 20, double stdDev = 2.0);
        double CalculateBollingerPosition(List<double> prices);
        double CalculateADX(List<MarketSnapshot> window, int period = 14);
        double CalculateParkinsonVolatility(List<MarketSnapshot> window, int period = 14);
        double CalculateGarmanKlassVolatility(List<MarketSnapshot> window, int period = 14);
        double CalculateVolumeWeightedPrice(List<MarketSnapshot> window);
        double CalculateOnBalanceVolume(List<MarketSnapshot> window);
    }
}
