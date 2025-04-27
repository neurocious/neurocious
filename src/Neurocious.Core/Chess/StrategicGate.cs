using Neurocious.Core.SpatialProbability;
using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.Chess
{
    /// <summary>
    /// Represents a strategic concept in chess as a point in latent space.
    /// </summary>
    public class StrategicGate
    {
        public string Name { get; }
        public float[] LatentVector { get; private set; }
        public float ActivationThreshold { get; }
        public float Weight { get; }

        private const float EPSILON = 1e-8f;
        private readonly Random random = new Random();

        public StrategicGate(string name, int dim, float threshold, float weight)
        {
            Name = name;
            ActivationThreshold = threshold;
            Weight = weight;
            LatentVector = InitializeLatentVector(dim);
        }

        private float[] InitializeLatentVector(int dim)
        {
            var vector = new float[dim];
            for (int i = 0; i < dim; i++)
            {
                vector[i] = (float)random.NextGaussian(0, 1);
            }

            NormalizeVector(vector);
            return vector;
        }

        public float CalculateActivation(PradOp state)
        {
            var stateVector = state.Result.Data.Select(x => (float)x).ToArray();
            NormalizeVector(stateVector);

            float distance = 1f - CosineSimilarity(stateVector, LatentVector);

            // RBF activation for smoother gradients
            return MathF.Exp(-MathF.Pow(distance / ActivationThreshold, 2));
        }

        private float CosineSimilarity(float[] a, float[] b)
        {
            float dot = 0, normA = 0, normB = 0;

            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            return dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB) + EPSILON);
        }

        private void NormalizeVector(float[] vector)
        {
            float magnitude = MathF.Sqrt(vector.Sum(v => v * v) + EPSILON);
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] /= magnitude;
            }
        }

        public void UpdateVector(float[] newVector, float learningRate)
        {
            NormalizeVector(newVector);

            for (int i = 0; i < LatentVector.Length; i++)
            {
                LatentVector[i] = LatentVector[i] * (1 - learningRate) + newVector[i] * learningRate;
            }

            NormalizeVector(LatentVector);
        }
    }
}
