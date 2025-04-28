using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class TechnicalAnalysis : ITechnicalAnalysis
    {
        public double CalculateRSI(List<double> prices, int period = 14)
        {
            if (prices.Count < period + 1) return 50;

            var returns = CalculateReturns(prices);
            var gains = new List<double>();
            var losses = new List<double>();

            for (int i = 0; i < returns.Count; i++)
            {
                gains.Add(Math.Max(0, returns[i]));
                losses.Add(Math.Max(0, -returns[i]));
            }

            double avgGain = gains.TakeLast(period).Average();
            double avgLoss = losses.TakeLast(period).Average();

            if (avgLoss == 0) return 100;
            double rs = avgGain / avgLoss;
            return 100 - (100 / (1 + rs));
        }

        public (double macd, double signal, double histogram) CalculateMACD(
            List<double> prices,
            int fastPeriod = 12,
            int slowPeriod = 26,
            int signalPeriod = 9)
        {
            var fastEMA = CalculateEMA(prices, fastPeriod);
            var slowEMA = CalculateEMA(prices, slowPeriod);
            var macd = fastEMA - slowEMA;
            var signal = CalculateEMA(new List<double> { macd }, signalPeriod);
            var histogram = macd - signal;

            return (macd, signal, histogram);
        }

        public (double upper, double middle, double lower) CalculateBollingerBands(
            List<double> prices,
            int period = 20,
            double stdDev = 2.0)
        {
            var sma = prices.TakeLast(period).Average();
            var std = CalculateStandardDeviation(prices.TakeLast(period).ToList());

            return (
                upper: sma + (stdDev * std),
                middle: sma,
                lower: sma - (stdDev * std)
            );
        }

        public double CalculateBollingerPosition(List<double> prices)
        {
            var (upper, middle, lower) = CalculateBollingerBands(prices);
            var currentPrice = prices.Last();
            var bandWidth = upper - lower;

            if (bandWidth == 0) return 0;
            return (currentPrice - lower) / bandWidth;
        }

        public double CalculateADX(List<MarketSnapshot> window, int period = 14)
        {
            if (window.Count < period + 1) return 0;

            var trueRanges = new List<double>();
            var posDMs = new List<double>();
            var negDMs = new List<double>();

            for (int i = 1; i < window.Count; i++)
            {
                var current = window[i];
                var prev = window[i - 1];

                // True Range
                double tr = Math.Max(current.High - current.Low,
                    Math.Max(Math.Abs(current.High - prev.Close),
                            Math.Abs(current.Low - prev.Close)));
                trueRanges.Add(tr);

                // Directional Movement
                double upMove = current.High - prev.High;
                double downMove = prev.Low - current.Low;

                if (upMove > downMove && upMove > 0)
                {
                    posDMs.Add(upMove);
                    negDMs.Add(0);
                }
                else if (downMove > upMove && downMove > 0)
                {
                    posDMs.Add(0);
                    negDMs.Add(downMove);
                }
                else
                {
                    posDMs.Add(0);
                    negDMs.Add(0);
                }
            }

            // Calculate smoothed averages
            double atr = CalculateWilder(trueRanges, period);
            double posDI = CalculateWilder(posDMs, period) / atr * 100;
            double negDI = CalculateWilder(negDMs, period) / atr * 100;

            // Calculate ADX
            double dx = Math.Abs(posDI - negDI) / (posDI + negDI) * 100;
            return CalculateWilder(new List<double> { dx }, period);
        }

        public double CalculateParkinsonVolatility(List<MarketSnapshot> window, int period = 14)
        {
            if (window.Count < period) return 0;

            var logRanges = window.Select(w =>
                Math.Log(w.High / w.Low));

            return Math.Sqrt(logRanges.Sum(x => x * x) / (4 * period * Math.Log(2)));
        }

        public double CalculateGarmanKlassVolatility(List<MarketSnapshot> window, int period = 14)
        {
            if (window.Count < period) return 0;

            double sum = 0;
            for (int i = 0; i < period; i++)
            {
                var w = window[i];
                double logHLSquared = Math.Pow(Math.Log(w.High / w.Low), 2);
                double logCOSquared = Math.Pow(Math.Log(w.Close / w.Open), 2);

                sum += 0.5 * logHLSquared - (2 * Math.Log(2) - 1) * logCOSquared;
            }

            return Math.Sqrt(sum / period);
        }

        public double CalculateVolumeWeightedPrice(List<MarketSnapshot> window)
        {
            if (!window.Any()) return 0;

            double volumeSum = window.Sum(w => w.Volume);
            if (volumeSum == 0) return window.Last().Price;

            return window.Sum(w => w.Price * w.Volume) / volumeSum;
        }

        public double CalculateOnBalanceVolume(List<MarketSnapshot> window)
        {
            if (window.Count < 2) return 0;

            double obv = 0;
            for (int i = 1; i < window.Count; i++)
            {
                if (window[i].Price > window[i - 1].Price)
                    obv += window[i].Volume;
                else if (window[i].Price < window[i - 1].Price)
                    obv -= window[i].Volume;
            }
            return obv;
        }

        private double CalculateEMA(List<double> values, int period)
        {
            if (!values.Any()) return 0;

            double multiplier = 2.0 / (period + 1);
            double ema = values.First();

            for (int i = 1; i < values.Count; i++)
            {
                ema = (values[i] - ema) * multiplier + ema;
            }

            return ema;
        }

        private double CalculateWilder(List<double> values, int period)
        {
            if (!values.Any()) return 0;

            double sum = values.Take(period).Sum();
            double wilder = sum;

            for (int i = period; i < values.Count; i++)
            {
                wilder = (wilder * (period - 1) + values[i]) / period;
            }

            return wilder;
        }

        private double CalculateStandardDeviation(List<double> values)
        {
            if (!values.Any()) return 0;

            double avg = values.Average();
            double sumOfSquares = values.Sum(x => Math.Pow(x - avg, 2));
            return Math.Sqrt(sumOfSquares / values.Count);
        }

        internal List<double> CalculateReturns(List<double> prices)
        {
            var returns = new List<double>();
            for (int i = 1; i < prices.Count; i++)
            {
                returns.Add((prices[i] / prices[i - 1]) - 1);
            }
            return returns;
        }
    }

}
