using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Common
{
    public class FieldParameters
    {
        public double Curvature { get; set; }  // Non-negative, measures regime instability
        public double Entropy { get; set; }    // [0,1], measures narrative uncertainty
        public double Alignment { get; set; }  // [-1,1], measures directional coherence

        public FieldParameters(Tensor fieldParams)
        {
            Curvature = fieldParams.Data[0];  // Applied ReLU
            Entropy = fieldParams.Data[1];     // Applied Sigmoid
            Alignment = fieldParams.Data[2];   // Applied Tanh
        }

        public Tensor ToTensor()
        {
            return new Tensor(new[] { 3 }, new[] { Curvature, Entropy, Alignment });
        }
    }
}
