using Neurocious.Core.SpatialProbabilityNetwork;
using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.EnhancedVariationalAutoencoder
{
    public class AttentionBlock
    {
        public PradOp QueryProj { get; private set; }
        public PradOp KeyProj { get; private set; }
        public PradOp ValueProj { get; private set; }
        public PradOp OutputProj { get; private set; }
        public PradOp LayerNorm { get; private set; }

        private readonly int hiddenDim;
        private readonly int numHeads;
        private readonly double dropoutRate;

        public AttentionBlock(int hiddenDim, int numHeads = 4, double dropoutRate = 0.1)
        {
            this.hiddenDim = hiddenDim;
            this.numHeads = numHeads;
            this.dropoutRate = dropoutRate;

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            var dimPerHead = hiddenDim / numHeads;

            QueryProj = new PradOp(InitializeWeights(hiddenDim, hiddenDim));
            KeyProj = new PradOp(InitializeWeights(hiddenDim, hiddenDim));
            ValueProj = new PradOp(InitializeWeights(hiddenDim, hiddenDim));
            OutputProj = new PradOp(InitializeWeights(hiddenDim, hiddenDim));
            LayerNorm = new PradOp(InitializeWeights(hiddenDim, hiddenDim));
        }

        private Tensor InitializeWeights(int inputDim, int outputDim)
        {
            var stddev = Math.Sqrt(2.0 / (inputDim + outputDim));
            var weights = new double[inputDim * outputDim];
            var random = new Random();
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextGaussian(0, stddev);
            }
            return new Tensor(new[] { inputDim, outputDim }, weights);
        }

        public PradResult Forward(PradOp input, bool training = true)
        {
            // Project to Q, K, V
            var queries = QueryProj.MatMul(input);
            var keys = KeyProj.MatMul(input);
            var values = ValueProj.MatMul(input);

            // Split into heads
            var queryHeads = SplitHeads(queries.Result);
            var keyHeads = SplitHeads(keys.Result);
            var valueHeads = SplitHeads(values.Result);

            // Scaled dot-product attention for each head
            var attentionHeads = new List<PradResult>();
            for (int h = 0; h < numHeads; h++)
            {
                var scaledDotProduct = new PradOp(queryHeads[h])
                    .MatMul(new PradOp(keyHeads[h]).Transpose().Result)
                    .Then(x => x.Mul(new Tensor(x.Result.Shape, 1.0 / Math.Sqrt(hiddenDim / numHeads))));

                // Apply causal mask if training
                if (training)
                {
                    scaledDotProduct = ApplyCausalMask(scaledDotProduct);
                }

                var attention = scaledDotProduct
                    .Then(PradOp.SoftmaxOp)
                    .Then(attn => attn.MatMul(new PradOp(valueHeads[h]).Result));

                attentionHeads.Add(attention);
            }

            // Concatenate heads and project
            var concatenated = ConcatenateHeads(attentionHeads);
            var output = OutputProj.MatMul(concatenated.Result);

            // Add residual connection and layer norm
            return new PradOp(output.Result)
                .Add(input.Result)
                .Then(x => LayerNorm.MatMul(x.Result));
        }

        private List<Tensor> SplitHeads(Tensor input)
        {
            var dimPerHead = hiddenDim / numHeads;
            var heads = new List<Tensor>();

            for (int h = 0; h < numHeads; h++)
            {
                var head = new double[input.Shape[0] * dimPerHead];
                for (int i = 0; i < input.Shape[0]; i++)
                {
                    for (int j = 0; j < dimPerHead; j++)
                    {
                        head[i * dimPerHead + j] = input.Data[i * hiddenDim + h * dimPerHead + j];
                    }
                }
                heads.Add(new Tensor(new[] { input.Shape[0], dimPerHead }, head));
            }

            return heads;
        }

        private PradResult ConcatenateHeads(List<PradResult> heads)
        {
            var dimPerHead = hiddenDim / numHeads;
            var concatenated = new double[heads[0].Result.Shape[0] * hiddenDim];

            for (int h = 0; h < numHeads; h++)
            {
                for (int i = 0; i < heads[h].Result.Shape[0]; i++)
                {
                    for (int j = 0; j < dimPerHead; j++)
                    {
                        concatenated[i * hiddenDim + h * dimPerHead + j] =
                            heads[h].Result.Data[i * dimPerHead + j];
                    }
                }
            }

            return new PradOp(new Tensor(new[] { heads[0].Result.Shape[0], hiddenDim }, concatenated));
        }

        private PradResult ApplyCausalMask(PradResult attention)
        {
            var shape = attention.Result.Shape;
            var mask = new double[shape[0] * shape[1]];

            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    mask[i * shape[1] + j] = j <= i ? 1 : float.NegativeInfinity;
                }
            }

            return new PradOp(attention.Result)
                .Add(new Tensor(shape, mask));
        }
    }
}
