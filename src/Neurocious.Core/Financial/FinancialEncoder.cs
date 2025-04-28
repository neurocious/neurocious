using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.Financial
{
    /// <summary>
    /// Encodes financial market snapshots into tensor representations.
    /// </summary>
    public class FinancialEncoder : IFinancialEncoder
    {
        private readonly int inputSize;
        private readonly bool normalizeFeatures;
        private double[] featureMeans;
        private double[] featureStds;

        public FinancialEncoder(int featureCount, bool normalizeFeatures = true)
        {
            inputSize = featureCount;
            this.normalizeFeatures = normalizeFeatures;
            featureMeans = new double[featureCount];
            featureStds = new double[featureCount];
        }

        public PradOp EncodeSnapshot(double[] features)
        {
            if (features.Length != inputSize)
                throw new ArgumentException($"Expected {inputSize} features, got {features.Length}");

            var processedFeatures = features;
            if (normalizeFeatures)
            {
                processedFeatures = NormalizeFeatures(features);
            }

            return new PradOp(new Tensor(new[] { inputSize }, processedFeatures));
        }

        public void UpdateNormalizationStats(List<double[]> recentSnapshots)
        {
            if (!normalizeFeatures) return;

            for (int i = 0; i < inputSize; i++)
            {
                var values = recentSnapshots.Select(s => s[i]).ToList();
                featureMeans[i] = values.Average();
                featureStds[i] = Math.Sqrt(values.Select(v => Math.Pow(v - featureMeans[i], 2)).Average() + 1e-8);
            }
        }

        private double[] NormalizeFeatures(double[] features)
        {
            var normalized = new double[features.Length];
            for (int i = 0; i < features.Length; i++)
            {
                normalized[i] = (features[i] - featureMeans[i]) / (featureStds[i] + 1e-8);
            }
            return normalized;
        }
    }
}
