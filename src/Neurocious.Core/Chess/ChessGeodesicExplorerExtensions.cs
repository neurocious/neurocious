using Neurocious.Core.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Chess
{
    public static class ChessGeodesicExplorerExtensions
    {
        public static float CalculateEntropyAwareExplorationRate(
            this ChessGeodesicExplorer explorer,
            FieldParameters fieldParams,
            float baseRate)
        {
            // Dynamic scaling based on entropy
            float entropyScaling = 1.0f + (float)fieldParams.Entropy * 2.0f;

            // Inverse scaling based on curvature (avoid too much noise in high-curvature regions)
            float curvatureScaling = 1.0f / (1.0f + (float)fieldParams.Curvature);

            // Scale base exploration rate
            float adjustedRate = baseRate * entropyScaling * curvatureScaling;

            // Add oscillating component for more natural exploration
            float time = (float)(DateTime.UtcNow.Ticks / TimeSpan.TicksPerSecond);
            float oscillation = MathF.Sin(time * 0.1f) * 0.1f + 1.0f;

            return adjustedRate * oscillation;
        }
    }
}
