using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public class TemporalRegularizer
    {
        private readonly Queue<Tensor> stateHistory;
        private readonly int historyLength;
        private readonly float smoothnessThreshold;

        public TemporalRegularizer(int historyLength = 10, float smoothnessThreshold = 0.5f)
        {
            this.historyLength = historyLength;
            this.smoothnessThreshold = smoothnessThreshold;
            stateHistory = new Queue<Tensor>();
        }

        public (float smoothness, float confidence) AnalyzeTransition(
            Tensor currentState,
            Tensor previousState)
        {
            // Add to history
            stateHistory.Enqueue(currentState);
            if (stateHistory.Count > historyLength)
                stateHistory.Dequeue();

            // Calculate transition smoothness
            float transitionMagnitude = CalculateTransitionMagnitude(currentState, previousState);
            float historicalAverage = CalculateHistoricalAverage();
            float smoothness = 1.0f / (1.0f + Math.Abs(transitionMagnitude - historicalAverage));

            // Calculate confidence based on smoothness
            float confidence = smoothness > smoothnessThreshold ?
                smoothness : smoothness * (smoothness / smoothnessThreshold);

            return (smoothness, confidence);
        }

        private float CalculateTransitionMagnitude(Tensor current, Tensor previous)
        {
            float sumSquaredDiff = 0;
            for (int i = 0; i < current.Data.Length; i++)
            {
                float diff = (float)(current.Data[i] - previous.Data[i]);
                sumSquaredDiff += diff * diff;
            }
            return (float)Math.Sqrt(sumSquaredDiff);
        }

        private float CalculateHistoricalAverage()
        {
            if (stateHistory.Count < 2) return 0;

            float sum = 0;
            var states = stateHistory.ToArray();
            for (int i = 1; i < states.Length; i++)
            {
                sum += CalculateTransitionMagnitude(states[i], states[i - 1]);
            }
            return sum / (states.Length - 1);
        }
    }
}
